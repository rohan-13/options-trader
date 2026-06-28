"""
Feature engineering for ML model training.

Attaches market-context features at the trade open date to each closed strategy
trade. Uses yfinance for historical price/VIX data; reads the earnings cache for
earnings-proximity. All features are point-in-time — no look-ahead.

Primary output columns added to the trades DataFrame:
  spot_at_open       — underlying closing price on open date
  otm_pct            — (spot - strike) / spot * 100
  iv_proxy           — rough implied vol: (premium/100) / (0.4 * strike * sqrt(dte/365))
  premium_to_spot    — (premium/100) / spot — normalized credit as fraction of stock price
  hv_20d             — 20-day annualised historical vol of underlying at open
  return_5d          — underlying 5-day return prior to open
  return_20d         — 20-day return prior to open
  return_60d         — 60-day return prior to open
  vs_52w_high        — % below 52-week high at open (0 = at high, negative = below)
  vs_52w_low         — % above 52-week low at open (positive = above trough)
  vix_at_open        — VIX close on open date
  vix_pct_rank_252d  — VIX percentile rank in trailing 252 trading days (0–100)
  vix_return_5d      — VIX 5-day return prior to open (positive = fear rising)
  days_to_earnings   — calendar days to next confirmed earnings date (999 if unknown)
  day_of_week        — 0=Mon … 4=Fri
  month_of_year      — 1–12
  is_win             — target label (1 = net P&L > 0)
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_YF_OK = False
try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    pass


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ffill_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
    return s.reindex(full_idx).ffill()


def _fetch_via_av(sym: str) -> pd.Series:
    """Try AlphaVantage (uses cache if already downloaded)."""
    try:
        from src.av_fetcher import AlphaVantageFetcher, PremiumRequired, RateLimitError
        fetcher = AlphaVantageFetcher()
        if sym == "^VIX":
            return fetcher.get_vix_history()
        df = fetcher.get_daily_prices(sym)
        s = df["close"].dropna() if "close" in df.columns else pd.Series(dtype=float)
        return _ffill_series(s)
    except Exception:
        return pd.Series(dtype=float)


def _download_history(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.Series]:
    """
    Download daily close prices per ticker.
    Priority: AlphaVantage cache/API → yfinance → empty series.
    Returns {ticker: pd.Series(close, index=DatetimeIndex)}.
    """
    all_symbols = tickers + ["^VIX"]
    log.info("Fetching historical data for %d symbols (%s → %s)…", len(all_symbols), start, end)

    result: dict[str, pd.Series] = {}
    yf_warned = False

    # Build one shared fetcher so the per-call throttle is respected across tickers
    av_fetcher = None
    av_available = False
    try:
        from src.av_fetcher import AlphaVantageFetcher, RateLimitError as _RLE
        av_fetcher = AlphaVantageFetcher()
        av_available = True
    except Exception:
        pass

    for sym in all_symbols:
        # ── Try AlphaVantage first (checks cache, then API) ───────────────────
        if av_available and av_fetcher is not None:
            try:
                if sym == "^VIX":
                    s = av_fetcher.get_vix_history()
                else:
                    df = av_fetcher.get_daily_prices(sym)
                    s = df["close"].dropna() if "close" in df.columns else pd.Series(dtype=float)
                    s = _ffill_series(s)
                if not s.empty:
                    result[sym] = s
                    continue
            except RateLimitError:
                log.warning("AlphaVantage rate limit hit — switching to yfinance for remaining tickers")
                av_available = False
            except Exception:
                pass  # premium endpoint or other error — fall through to yfinance

        # ── Fall back to yfinance ─────────────────────────────────────────────
        if _YF_OK:
            try:
                raw = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
                col = "Close" if "Close" in raw.columns else (raw.columns[0] if not raw.empty else None)
                if col and not raw.empty:
                    s = raw[col].dropna()
                    result[sym] = _ffill_series(s)
                    continue
            except Exception:
                if not yf_warned:
                    log.warning("yfinance also unavailable — some tickers will have NaN price features")
                    yf_warned = True

        result[sym] = pd.Series(dtype=float)

    return result


def _lookup(series: pd.Series, dt: pd.Timestamp) -> Optional[float]:
    """Return the most recent value in series on or before dt."""
    if series.empty:
        return None
    idx = series.index[series.index <= dt]
    if idx.empty:
        return None
    return float(series.loc[idx[-1]])


def _rolling_return(series: pd.Series, dt: pd.Timestamp, days: int) -> Optional[float]:
    """Percentage return over the prior `days` calendar days ending at dt."""
    v_now = _lookup(series, dt)
    v_then = _lookup(series, dt - timedelta(days=days))
    if v_now is None or v_then is None or v_then == 0:
        return None
    return (v_now - v_then) / v_then


def _hv_20d(series: pd.Series, dt: pd.Timestamp) -> Optional[float]:
    """20-day annualised historical volatility of log returns ending at dt."""
    window_end = series.index[series.index <= dt]
    if len(window_end) < 22:
        return None
    window = series.loc[window_end[-22:]]
    log_ret = np.log(window / window.shift(1)).dropna()
    if len(log_ret) < 5:
        return None
    return float(log_ret.std() * math.sqrt(252))


def _vix_pct_rank(vix_series: pd.Series, dt: pd.Timestamp) -> Optional[float]:
    """VIX percentile rank within the trailing 252-trading-day window ending at dt."""
    window_end = vix_series.index[vix_series.index <= dt]
    if len(window_end) < 10:
        return None
    window = vix_series.loc[window_end[-252:]]
    v = _lookup(vix_series, dt)
    if v is None:
        return None
    return float((window < v).mean() * 100)


def _days_to_next_earnings(ticker: str, open_dt: date) -> int:
    """
    Return calendar days to the next confirmed earnings date on or after open_dt.
    Returns 999 if no upcoming earnings found in the cache.
    """
    try:
        from src.earnings import get_earnings_dates
        future = [d for d in get_earnings_dates(ticker) if d >= open_dt]
        if future:
            return (min(future) - open_dt).days
    except Exception:
        pass
    return 999


def _iv_proxy(premium: float, strike: float, dte: float) -> Optional[float]:
    """
    Rough implied-vol proxy derived from the premium collected.
    Approximation from ATM Black-Scholes: C ≈ 0.4 * S * σ * sqrt(T).
    We use strike ≈ S for OTM puts and solve for σ (annualised).
    """
    if dte is None or dte <= 0 or strike <= 0 or premium <= 0:
        return None
    T = dte / 365.0
    return round((premium / 100) / (0.4 * strike * math.sqrt(T)), 4)


# ── Public API ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "spot_at_open",
    "otm_pct",
    "iv_proxy",
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
    "day_of_week",
    "month_of_year",
]


def build_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Attach market-context features to each closed short-put 0-7 DTE trade.

    Open trades are excluded (no known outcome). Rows with no price data
    (delisted tickers, data gaps) are dropped with a warning.

    Parameters
    ----------
    trades : pd.DataFrame
        Output of src.matcher.match_trades() — one row per round-trip trade.

    Returns
    -------
    pd.DataFrame
        Filtered strategy trades with FEATURE_COLS appended plus `is_win` target.
        All original columns are preserved for traceability.
    """
    df = trades[
        trades["is_closed"] &
        (trades["strategy"] == "PUT_short") &
        (trades["dte_at_open"].notna()) &
        (trades["dte_at_open"] >= 0) &
        (trades["dte_at_open"] <= 7) &
        (trades["premium"] > 0)
    ].copy()

    if df.empty:
        log.warning("No closed short-put 0-7 DTE trades found — returning empty DataFrame.")
        return df

    df["open_date"] = pd.to_datetime(df["open_date"])
    log.info("Building features for %d strategy trades.", len(df))

    tickers = df["ticker"].dropna().unique().tolist()

    # Need ~400 days of history before earliest trade for 252d rolling windows
    earliest = df["open_date"].min()
    latest = df["open_date"].max()
    hist_start = (earliest - timedelta(days=400)).strftime("%Y-%m-%d")
    hist_end = (latest + timedelta(days=2)).strftime("%Y-%m-%d")

    price_data = _download_history(tickers, hist_start, hist_end)
    vix_series = price_data.get("^VIX", pd.Series(dtype=float))

    records: list[dict] = []
    skipped = 0

    for _, row in df.iterrows():
        ticker = row["ticker"]
        open_dt = pd.Timestamp(row["open_date"])
        strike = float(row["strike"])
        premium = float(row["premium"])
        dte = float(row["dte_at_open"]) if row["dte_at_open"] is not None else None

        price_series = price_data.get(ticker, pd.Series(dtype=float))
        spot = _lookup(price_series, open_dt)

        if spot is None or spot <= 0:
            log.debug("No price data for %s on %s — skipping.", ticker, open_dt.date())
            skipped += 1
            continue

        # 52-week high/low
        w52_end = price_series.index[price_series.index <= open_dt]
        if len(w52_end) >= 2:
            w52 = price_series.loc[w52_end[-252:]]
            high52 = float(w52.max())
            low52 = float(w52.min())
            vs_high = round((spot - high52) / high52 * 100, 2) if high52 > 0 else None
            vs_low = round((spot - low52) / low52 * 100, 2) if low52 > 0 else None
        else:
            vs_high = vs_low = None

        vix = _lookup(vix_series, open_dt)
        vix_rank = _vix_pct_rank(vix_series, open_dt)
        vix_r5 = _rolling_return(vix_series, open_dt, 5)
        hv = _hv_20d(price_series, open_dt)
        r5 = _rolling_return(price_series, open_dt, 5)
        r20 = _rolling_return(price_series, open_dt, 20)
        r60 = _rolling_return(price_series, open_dt, 60)

        feat: dict = {**row.to_dict()}
        feat.update({
            "spot_at_open":      round(spot, 4),
            "otm_pct":           round((spot - strike) / spot * 100, 2) if spot > 0 else None,
            "iv_proxy":          _iv_proxy(premium, strike, dte),
            "premium_to_spot":   round((premium / 100) / spot, 6) if spot > 0 else None,
            "hv_20d":            round(hv, 4) if hv is not None else None,
            "return_5d":         round(r5, 4) if r5 is not None else None,
            "return_20d":        round(r20, 4) if r20 is not None else None,
            "return_60d":        round(r60, 4) if r60 is not None else None,
            "vs_52w_high":       vs_high,
            "vs_52w_low":        vs_low,
            "vix_at_open":       round(vix, 2) if vix is not None else None,
            "vix_pct_rank_252d": round(vix_rank, 1) if vix_rank is not None else None,
            "vix_return_5d":     round(vix_r5, 4) if vix_r5 is not None else None,
            "days_to_earnings":  _days_to_next_earnings(ticker, open_dt.date()),
            "day_of_week":       open_dt.dayofweek,
            "month_of_year":     open_dt.month,
            "is_win":            int(row["is_win"]),
        })
        records.append(feat)

    if skipped:
        log.warning(
            "Skipped %d trades with no price data (%.1f%%).",
            skipped, skipped / len(df) * 100,
        )

    if not records:
        log.error("No trades enriched — check yfinance connectivity.")
        return pd.DataFrame()

    result = pd.DataFrame(records).reset_index(drop=True)

    for col in FEATURE_COLS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    log.info("Feature build complete: %d trades enriched, %d skipped.", len(result), skipped)
    _log_feature_summary(result)
    return result


def _log_feature_summary(df: pd.DataFrame) -> None:
    log.info("─" * 60)
    log.info("  %-22s  %6s  %8s  %8s  %6s", "Feature", "N", "Mean", "Std", "NaN%")
    log.info("─" * 60)
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        n_nan = s.isna().sum()
        pct_nan = n_nan / len(df) * 100
        valid = s.dropna()
        mean = valid.mean() if len(valid) else float("nan")
        std = valid.std() if len(valid) else float("nan")
        log.info("  %-22s  %6d  %8.3f  %8.3f  %5.1f%%", col, len(valid), mean, std, pct_nan)
    log.info("─" * 60)
    log.info(
        "  Target: is_win — wins=%d / total=%d  (%.1f%%)",
        int(df["is_win"].sum()), len(df), df["is_win"].mean() * 100,
    )
