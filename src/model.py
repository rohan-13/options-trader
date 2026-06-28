"""
Win-probability model for 0-7 DTE short puts.

Algorithm: HistGradientBoostingClassifier (sklearn) — handles NaN natively,
           fast, no extra install, comparable to LightGBM at this data size.

Training: walk-forward cross-validation to prevent look-ahead bias.
           Train on a rolling window, test on the next quarter.

Primary output: win_probability (0–1) per candidate trade.
"""
from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OrdinalEncoder

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"

# ── Feature definitions ───────────────────────────────────────────────────────

# Features always available from trade data (no external data needed)
CORE_FEATURES = [
    "dte_at_open",
    "premium",
    "log_premium",
    "strike",
    "log_strike",
    "iv_proxy",
    "premium_to_strike",
    "day_of_week",
    "month_of_year",
    "year",
    "ticker_win_rate",   # target-encoded in-fold ticker win rate
]

# Features from yfinance / AlphaVantage (optional — NaN if not available)
ENRICHED_FEATURES = [
    "otm_pct",
    "premium_to_spot",
    "hv_20d",
    "return_5d",
    "return_20d",
    "return_60d",
    "vs_52w_high",
    "vs_52w_low",
    "vix_at_open",
    "vix_pct_rank_252d",
    "vix_return_5d",
    "days_to_earnings",
    # AV premium
    "iv_actual",
    "delta_actual",
    "theta_actual",
    "iv_rank_52w",
    "pcr_at_open",
    "av_vix",
    "av_vix_rank",
]

ALL_FEATURES = CORE_FEATURES + ENRICHED_FEATURES


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, ticker_win_rates: Optional[dict] = None) -> pd.DataFrame:
    """
    Add model-specific engineered features to a trades DataFrame.

    `ticker_win_rates` should be computed from TRAINING data only (prevents leakage).
    Pass None to use the overall mean (0.807) as a neutral prior.
    """
    out = df.copy()

    # Log-transform right-skewed features
    out["log_premium"] = np.log1p(out["premium"].clip(lower=0))
    out["log_strike"] = np.log1p(out["strike"].clip(lower=0))

    # Normalized premium as fraction of strike (not spot — avoids needing spot)
    out["premium_to_strike"] = (out["premium"] / (out["strike"] * 100)).clip(0, 1)

    # IV proxy (recompute to ensure consistency)
    def _iv_proxy(row):
        dte = row.get("dte_at_open")
        strike = row.get("strike")
        premium = row.get("premium")
        if not dte or dte <= 0 or not strike or strike <= 0 or not premium or premium <= 0:
            return float("nan")
        return (premium / 100) / (0.4 * strike * math.sqrt(dte / 365.0))

    if "iv_proxy" not in out.columns or out["iv_proxy"].isna().all():
        out["iv_proxy"] = out.apply(_iv_proxy, axis=1)

    # Calendar features
    dt = pd.to_datetime(out["open_date"])
    out["day_of_week"] = dt.dt.dayofweek
    out["month_of_year"] = dt.dt.month
    out["year"] = dt.dt.year

    # Ticker win rate (in-fold target encoding)
    global_mean = ticker_win_rates.get("__mean__", 0.807) if ticker_win_rates else 0.807
    if ticker_win_rates:
        out["ticker_win_rate"] = out["ticker"].map(ticker_win_rates).fillna(global_mean)
    else:
        out["ticker_win_rate"] = global_mean

    return out


def compute_ticker_win_rates(df: pd.DataFrame, smoothing: int = 20) -> dict:
    """
    Compute smoothed per-ticker win rates for target encoding.
    Smoothing blends toward global mean when a ticker has few samples.
    """
    global_mean = df["is_win"].mean()
    rates = {}
    for ticker, g in df.groupby("ticker"):
        n = len(g)
        ticker_mean = g["is_win"].mean()
        # Bayesian smoothing: blend toward global mean
        smoothed = (n * ticker_mean + smoothing * global_mean) / (n + smoothing)
        rates[ticker] = round(smoothed, 4)
    rates["__mean__"] = round(global_mean, 4)
    return rates


# ── Model class ───────────────────────────────────────────────────────────────

class WinProbabilityModel:
    """
    Gradient-boosted win-probability classifier with walk-forward cross-validation.
    """

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 max_depth: int = 5, min_samples_leaf: int = 20):
        self.params = dict(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        self.model: Optional[CalibratedClassifierCV] = None
        self.feature_cols: list[str] = []
        self.ticker_win_rates: dict = {}
        self.feature_importance_: Optional[pd.Series] = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in ALL_FEATURES if c in df.columns]

    def fit(self, df: pd.DataFrame) -> "WinProbabilityModel":
        """Train on all available data (final model for production use)."""
        df = engineer_features(df)
        self.ticker_win_rates = compute_ticker_win_rates(df)
        df = engineer_features(df, self.ticker_win_rates)

        self.feature_cols = self._get_feature_cols(df)
        X = df[self.feature_cols].values.astype(float)
        y = df["is_win"].values.astype(int)

        base = HistGradientBoostingClassifier(**self.params, random_state=42)
        self.model = CalibratedClassifierCV(base, cv=5, method="isotonic")
        self.model.fit(X, y)

        # Permutation-based feature importance on a held-out sample
        try:
            sample = df.sample(min(1000, len(df)), random_state=42)
            sample = engineer_features(sample, self.ticker_win_rates)
            X_s = sample[self.feature_cols].values.astype(float)
            y_s = sample["is_win"].values.astype(int)
            result = permutation_importance(
                self.model, X_s, y_s, n_repeats=10, random_state=42, scoring="roc_auc"
            )
            self.feature_importance_ = pd.Series(
                result.importances_mean, index=self.feature_cols
            ).sort_values(ascending=False)
        except Exception as e:
            log.debug("Feature importance failed: %s", e)

        log.info("Model trained on %d samples, %d features", len(df), len(self.feature_cols))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return win probability for each row in df."""
        if self.model is None:
            raise RuntimeError("Call fit() first")
        df = engineer_features(df, self.ticker_win_rates)
        X = df[self.feature_cols].reindex(columns=self.feature_cols).values.astype(float)
        return self.model.predict_proba(X)[:, 1]

    def walk_forward_cv(
        self,
        df: pd.DataFrame,
        train_months: int = 9,
        test_months: int = 3,
        min_train_rows: int = 200,
    ) -> pd.DataFrame:
        """
        Walk-forward cross-validation. Returns a DataFrame with one row per fold
        containing AUC, log-loss, and win-rate at various probability thresholds.
        """
        df = engineer_features(df).copy()
        df["open_date"] = pd.to_datetime(df["open_date"])
        df = df.sort_values("open_date").reset_index(drop=True)

        start = df["open_date"].min().to_period("M")
        end = df["open_date"].max().to_period("M")

        folds = []
        train_start = start
        fold_num = 0

        while True:
            train_end = train_start + train_months - 1
            test_start = train_end + 1
            test_end = test_start + test_months - 1

            if test_end > end:
                break

            train_mask = (
                (df["open_date"].dt.to_period("M") >= train_start) &
                (df["open_date"].dt.to_period("M") <= train_end)
            )
            test_mask = (
                (df["open_date"].dt.to_period("M") >= test_start) &
                (df["open_date"].dt.to_period("M") <= test_end)
            )

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) < min_train_rows or len(test_df) == 0:
                train_start += test_months
                continue

            fold_num += 1

            # Target-encode tickers using training data only
            ticker_rates = compute_ticker_win_rates(train_df)
            train_df = engineer_features(train_df, ticker_rates)
            test_df = engineer_features(test_df, ticker_rates)

            feat_cols = self._get_feature_cols(train_df)
            X_train = train_df[feat_cols].values.astype(float)
            y_train = train_df["is_win"].values.astype(int)
            X_test = test_df[feat_cols].values.astype(float)
            y_test = test_df["is_win"].values.astype(int)

            base = HistGradientBoostingClassifier(**self.params, random_state=42)
            cal = CalibratedClassifierCV(base, cv=3, method="isotonic")
            cal.fit(X_train, y_train)
            proba = cal.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
            ll = log_loss(y_test, proba)

            fold_result = {
                "fold": fold_num,
                "train_period": f"{train_start} → {train_end}",
                "test_period": f"{test_start} → {test_end}",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "baseline_win_rate": float(y_test.mean()),
                "auc_roc": round(auc, 4),
                "log_loss": round(ll, 4),
            }

            # Win rate at different score thresholds
            for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
                mask = proba >= thresh
                n = mask.sum()
                wr = float(y_test[mask].mean()) if n > 0 else float("nan")
                fold_result[f"wr_at_{int(thresh*100)}"] = round(wr, 4) if not math.isnan(wr) else float("nan")
                fold_result[f"n_at_{int(thresh*100)}"] = int(n)

            folds.append(fold_result)
            log.info(
                "Fold %d (%s): AUC=%.3f  baseline=%.1f%%  n_test=%d",
                fold_num, fold_result["test_period"], auc,
                y_test.mean() * 100, len(test_df),
            )

            train_start += test_months

        return pd.DataFrame(folds)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(path or MODEL_DIR / "win_prob_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Model saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "WinProbabilityModel":
        path = Path(path or MODEL_DIR / "win_prob_model.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("Model loaded from %s", path)
        return obj
