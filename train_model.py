#!/usr/bin/env python3
"""
Phase 2: Train the win-probability model.

Loads strategy trades directly from E*Trade CSVs (no external data needed),
runs walk-forward cross-validation, trains a final model on all data,
and saves it to models/win_prob_model.pkl.

Usage:
  python train_model.py
  python train_model.py --dataset output/dataset_phase1.csv   # if you have enriched data
  python train_model.py --train-months 12 --test-months 3

Output:
  models/win_prob_model.pkl       — saved model (load with WinProbabilityModel.load())
  output/cv_results.csv           — per-fold cross-validation metrics
  output/feature_importance.csv   — ranked feature importances
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import pandas as pd

from src.loader import load_all
from src.parser import enrich_options
from src.matcher import match_trades
from src.model import WinProbabilityModel, engineer_features, ALL_FEATURES


def load_trades(dataset_path: str | None = None) -> pd.DataFrame:
    """Load strategy trades either from a pre-built dataset or raw CSVs."""
    if dataset_path and Path(dataset_path).exists():
        log.info("Loading pre-built dataset from %s", dataset_path)
        df = pd.read_csv(dataset_path)
        log.info("  %d rows, %d cols", len(df), len(df.columns))
        return df

    log.info("Building trades from raw E*Trade CSVs…")
    raw = load_all()
    parsed = enrich_options(raw)
    trades = match_trades(parsed)

    strategy = trades[
        trades["is_closed"] &
        (trades["strategy"] == "PUT_short") &
        (trades["dte_at_open"].notna()) &
        (trades["dte_at_open"] >= 0) &
        (trades["dte_at_open"] <= 7) &
        (trades["premium"] > 0)
    ].copy()
    strategy["is_win"] = (strategy["net_pnl"] > 0).astype(int)
    log.info("Strategy trades: %d rows, %.1f%% win rate",
             len(strategy), strategy["is_win"].mean() * 100)
    return strategy


def print_cv_summary(cv: pd.DataFrame) -> None:
    print(f"\n{'=' * 75}")
    print("  WALK-FORWARD CROSS-VALIDATION RESULTS")
    print(f"{'=' * 75}")
    print(f"  {'Fold':<4} {'Test Period':<22} {'N_test':>7} {'Baseline':>9} "
          f"{'AUC':>7} {'WR@70':>8} {'N@70':>7}")
    print(f"  {'─' * 70}")
    for _, r in cv.iterrows():
        wr70 = r.get("wr_at_70", float("nan"))
        n70 = r.get("n_at_70", 0)
        wr_str = f"{wr70:.1%}" if wr70 == wr70 else "  n/a"
        print(f"  {int(r['fold']):<4} {r['test_period']:<22} {int(r['test_rows']):>7} "
              f"{r['baseline_win_rate']:.1%}    {r['auc_roc']:>6.3f}  {wr_str:>8} {int(n70):>7}")

    print(f"  {'─' * 70}")
    print(f"  {'MEAN':<4} {'':22} {int(cv['test_rows'].sum()):>7} "
          f"{cv['baseline_win_rate'].mean():.1%}    {cv['auc_roc'].mean():>6.3f}")

    print(f"\n  Threshold analysis (averaged across folds):")
    print(f"  {'Threshold':>12} {'Win Rate':>10} {'Trades/fold':>12} {'Lift':>8}")
    print(f"  {'─' * 45}")
    baseline = cv["baseline_win_rate"].mean()
    for thresh in [60, 65, 70, 75, 80]:
        wr_col = f"wr_at_{thresh}"
        n_col = f"n_at_{thresh}"
        if wr_col not in cv.columns:
            continue
        valid = cv[wr_col].dropna()
        if valid.empty:
            continue
        avg_wr = valid.mean()
        avg_n = cv[n_col].mean()
        lift = avg_wr - baseline
        print(f"  {thresh/100:.0%} score    {avg_wr:>9.1%}  {avg_n:>11.1f}  {lift:>+7.1%}")
    print(f"{'=' * 75}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train win-probability model")
    ap.add_argument("--dataset", default=None,
                    help="Pre-built dataset CSV (default: build from raw CSVs)")
    ap.add_argument("--train-months", type=int, default=9,
                    help="Training window size in months (default 9)")
    ap.add_argument("--test-months", type=int, default=3,
                    help="Test window size in months (default 3)")
    ap.add_argument("--no-cv", action="store_true",
                    help="Skip cross-validation, just train final model")
    args = ap.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_trades(args.dataset)

    if len(df) < 100:
        log.error("Too few trades (%d) to train a meaningful model.", len(df))
        sys.exit(1)

    log.info("Feature columns available: %s",
             [c for c in ALL_FEATURES if c in engineer_features(df.head(1)).columns])

    # ── Walk-forward cross-validation ─────────────────────────────────────────
    model = WinProbabilityModel()

    if not args.no_cv:
        log.info("Running walk-forward CV (train=%dm, test=%dm)…",
                 args.train_months, args.test_months)
        cv = model.walk_forward_cv(
            df,
            train_months=args.train_months,
            test_months=args.test_months,
        )

        if cv.empty:
            log.warning("No CV folds generated — not enough data span. "
                        "Try --train-months 6 --test-months 2")
        else:
            print_cv_summary(cv)
            cv_path = Path("output/cv_results.csv")
            cv_path.parent.mkdir(exist_ok=True)
            cv.to_csv(cv_path, index=False)
            log.info("CV results saved → %s", cv_path)
    else:
        log.info("Skipping CV (--no-cv)")

    # ── Train final model on all data ─────────────────────────────────────────
    log.info("Training final model on all %d trades…", len(df))
    model.fit(df)
    model_path = model.save()

    # Feature importance
    if model.feature_importance_ is not None:
        fi = model.feature_importance_.reset_index()
        fi.columns = ["feature", "importance"]
        fi_path = Path("output/feature_importance.csv")
        fi.to_csv(fi_path, index=False)

        print("\n  Feature Importances (top 15):")
        print(f"  {'Feature':<25} {'Importance':>12}")
        print(f"  {'─' * 40}")
        for _, row in fi.head(15).iterrows():
            print(f"  {row['feature']:<25} {row['importance']:>12.4f}")
        print()

    log.info("Done.")
    log.info("  Model   → %s", model_path)
    log.info("  CV      → output/cv_results.csv")
    log.info("  Next    → python algo.py --scan  (uses model for scoring)")


if __name__ == "__main__":
    main()
