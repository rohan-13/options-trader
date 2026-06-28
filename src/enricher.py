"""
Enriches the strategy-trades DataFrame with AlphaVantage market data.

New columns added (NaN if unavailable):
  av_vix          — VIX close from AlphaVantage (replaces yfinance VIX)
  av_vix_rank     — VIX percentile rank in trailing 252 trading days (0–100)
  av_spot         — underlying close from AV TIME_SERIES_DAILY
  iv_actual       — implied volatility of the traded contract [PREMIUM]
  delta_actual    — delta at entry [PREMIUM]
  theta_actual    — theta at entry (daily, per contract × 100) [PREMIUM]
  iv_rank_52w     — where today's ATM IV sits vs the past 52 weeks (0–100) [PREMIUM]
  pcr_at_open     — whole-chain put-call ratio on the open date [PREMIUM]
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from src.av_fetcher import AlphaVantageFetcher, PremiumRequired, RateLimitError

log = logging.getLogger(__name__)

_AV_COLS = [
    "av_vix", "av_vix_rank", "av_spot",
    "iv_actual", "delta_actual", "theta_actual", "iv_rank_52w", "pcr_at_open",
]


# ── Point-in-time helpers ─────────────────────────────────────────────────────

def _lookup(series: pd.Series, dt: pd.Timestamp) -> Optional[float]:
    idx = series.index[series.index <= dt]
    return float(series.loc[idx[-1]]) if not idx.empty else None


def _pct_rank_trailing(series: pd.Series, dt: pd.Timestamp, window: int = 252) -> Optional[float]:
    past = series[series.index <= dt].iloc[-window:]
    v = _lookup(series, dt)
    if v is None or len(past) < 10:
        return None
    return float((past < v).mean() * 100)


def _find_contract(
    chain: pd.DataFrame, strike: float, expiry: str, opt_type: str
) -> Optional[pd.Series]:
    """Locate the closest-matching row in an options chain."""
    if chain.empty or "strike" not in chain.columns:
        return None

    sub = chain
    if "type" in chain.columns:
        sub = chain[chain["type"].str.upper() == opt_type.upper()]
    if sub.empty:
        return None

    if "expiration" in sub.columns and expiry:
        try:
            exp_ts = pd.Timestamp(expiry)
            matched = sub[pd.to_datetime(sub["expiration"]) == exp_ts]
            if not matched.empty:
                sub = matched
        except Exception:
            pass

    idx = (sub["strike"] - strike).abs().idxmin()
    return sub.loc[idx]


# ── Main entry point ──────────────────────────────────────────────────────────

def enrich_with_av(
    trades: pd.DataFrame,
    fetcher: AlphaVantageFetcher,
    fetch_options: bool = False,
    fetch_pcr: bool = False,
) -> pd.DataFrame:
    """
    Attach AlphaVantage features to a strategy-trades DataFrame.

    Parameters
    ----------
    trades       : output of build_features() — one row per strategy trade
    fetcher      : configured AlphaVantageFetcher instance
    fetch_options: pull HISTORICAL_OPTIONS per (ticker, date) [PREMIUM]
    fetch_pcr    : pull HISTORICAL_PUT_CALL_RATIO per (ticker, date) [PREMIUM]

    Returns a new DataFrame with AV columns appended.
    """
    df = trades.copy()
    for col in _AV_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    tickers = df["ticker"].dropna().unique().tolist()

    # ── 1. VIX from AlphaVantage ──────────────────────────────────────────────
    vix_series: pd.Series = pd.Series(dtype=float)
    try:
        vix_series = fetcher.get_vix_history()
        log.info("VIX history loaded: %d days (%s → %s)",
                 len(vix_series), vix_series.index.min().date(), vix_series.index.max().date())
    except RateLimitError as e:
        log.warning("AV rate limit hit fetching VIX: %s", e)
    except PremiumRequired:
        log.warning("INDEX_DATA appears to require premium — skipping av_vix")
    except Exception as e:
        log.warning("Could not fetch VIX from AV: %s", e)

    # ── 2. Underlying prices from AlphaVantage ────────────────────────────────
    price_cache: dict[str, pd.Series] = {}
    for sym in tickers:
        try:
            price_df = fetcher.get_daily_prices(sym)
            s = price_df["close"].dropna()
            full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
            price_cache[sym] = s.reindex(full_idx).ffill()
        except RateLimitError as e:
            log.warning("Rate limit hit fetching prices for %s: %s", sym, e)
            break
        except Exception as e:
            log.warning("Could not fetch prices for %s: %s", sym, e)

    # ── 3. Historical options chains (PREMIUM) ────────────────────────────────
    iv_history: dict[str, dict[str, float]] = {t: {} for t in tickers}
    if fetch_options:
        _collect_options(df, fetcher, price_cache, iv_history)

    # ── 4. Put-call ratio (PREMIUM) ───────────────────────────────────────────
    if fetch_pcr:
        _collect_pcr(df, fetcher)

    # ── 5. Fill scalar features row by row ───────────────────────────────────
    iv_series: dict[str, pd.Series] = _build_iv_series(iv_history)

    for i, row in df.iterrows():
        dt = pd.Timestamp(row["open_date"])
        sym = row["ticker"]

        # VIX
        if not vix_series.empty:
            v = _lookup(vix_series, dt)
            if v:
                df.at[i, "av_vix"] = round(v, 2)
            rank = _pct_rank_trailing(vix_series, dt)
            if rank is not None:
                df.at[i, "av_vix_rank"] = round(rank, 1)

        # Underlying spot
        ps = price_cache.get(sym, pd.Series(dtype=float))
        if not ps.empty:
            spot = _lookup(ps, dt)
            if spot:
                df.at[i, "av_spot"] = round(spot, 4)

        # Options features (filled in _collect_options)
        # IV rank — computed after all chains collected
        ivs = iv_series.get(sym)
        if ivs is not None and not ivs.empty:
            rank = _pct_rank_trailing(ivs, dt)
            if rank is not None:
                df.at[i, "iv_rank_52w"] = round(rank, 1)

    log.info("AV enrichment complete — %d trades", len(df))
    _log_fill_rates(df)
    return df


# ── Sub-routines ──────────────────────────────────────────────────────────────

def _collect_options(
    df: pd.DataFrame,
    fetcher: AlphaVantageFetcher,
    price_cache: dict[str, pd.Series],
    iv_history: dict[str, dict[str, float]],
) -> None:
    pairs = (
        df[["ticker", "open_date"]].dropna()
        .assign(open_date=lambda x: pd.to_datetime(x["open_date"]).dt.strftime("%Y-%m-%d"))
        .drop_duplicates()
        .values.tolist()
    )
    log.info("Fetching HISTORICAL_OPTIONS for %d (ticker, date) pairs…", len(pairs))

    premium_failed = False
    fetched = 0

    for ticker, date_str in pairs:
        if premium_failed:
            break
        try:
            chain = fetcher.get_historical_options(ticker, date_str)
        except PremiumRequired:
            log.warning(
                "HISTORICAL_OPTIONS requires premium — "
                "run with a premium API key to unlock iv_actual, delta_actual, theta_actual, iv_rank_52w"
            )
            premium_failed = True
            break
        except RateLimitError as e:
            log.warning("Rate limit: %s — stopping options fetch", e)
            break
        except Exception as e:
            log.debug("Options fetch failed %s %s: %s", ticker, date_str, e)
            continue

        if chain.empty:
            continue

        fetched += 1

        # Record ATM IV for this (ticker, date) to build IV rank series
        dt = pd.Timestamp(date_str)
        ps = price_cache.get(ticker, pd.Series(dtype=float))
        spot = _lookup(ps, dt) if not ps.empty else None
        if spot and "implied_volatility" in chain.columns and "strike" in chain.columns:
            puts = chain[chain["type"].str.upper() == "PUT"] if "type" in chain.columns else chain
            if not puts.empty:
                atm = puts.iloc[(puts["strike"] - spot).abs().argsort().iloc[:1]]
                iv_val = atm["implied_volatility"].iloc[0] if not atm.empty else None
                if iv_val and not math.isnan(float(iv_val)):
                    iv_history[ticker][date_str] = float(iv_val)

        # Fill contract-level Greeks for matching rows
        mask = (
            (df["ticker"] == ticker) &
            (pd.to_datetime(df["open_date"]).dt.strftime("%Y-%m-%d") == date_str)
        )
        for idx in df[mask].index:
            row = df.loc[idx]
            contract = _find_contract(
                chain,
                float(row["strike"]),
                str(row.get("expiry", "")),
                str(row.get("opt_type", "PUT")),
            )
            if contract is None:
                continue
            for feat, col in [
                ("implied_volatility", "iv_actual"),
                ("delta", "delta_actual"),
                ("theta", "theta_actual"),
            ]:
                val = contract.get(feat)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    df.at[idx, col] = round(float(val), 6)

    if fetched:
        log.info("Options chains fetched/loaded: %d unique (ticker, date) pairs", fetched)


def _collect_pcr(df: pd.DataFrame, fetcher: AlphaVantageFetcher) -> None:
    pairs = (
        df[["ticker", "open_date"]].dropna()
        .assign(open_date=lambda x: pd.to_datetime(x["open_date"]).dt.strftime("%Y-%m-%d"))
        .drop_duplicates()
        .values.tolist()
    )
    log.info("Fetching HISTORICAL_PUT_CALL_RATIO for %d (ticker, date) pairs…", len(pairs))

    premium_failed = False
    for ticker, date_str in pairs:
        if premium_failed:
            break
        try:
            data = fetcher.get_historical_pcr(ticker, date_str)
        except PremiumRequired:
            log.warning(
                "HISTORICAL_PUT_CALL_RATIO requires premium — skipping pcr_at_open"
            )
            premium_failed = True
            break
        except RateLimitError as e:
            log.warning("Rate limit: %s — stopping PCR fetch", e)
            break
        except Exception as e:
            log.debug("PCR fetch failed %s %s: %s", ticker, date_str, e)
            continue

        # The API returns: {"put_call_ratio": "0.89", "expiration_dates": [...]}
        raw = data.get("put_call_ratio") or data.get("overall_put_call_ratio")
        if raw is None:
            continue
        try:
            pcr_float = round(float(raw), 4)
        except (ValueError, TypeError):
            continue

        mask = (
            (df["ticker"] == ticker) &
            (pd.to_datetime(df["open_date"]).dt.strftime("%Y-%m-%d") == date_str)
        )
        df.loc[mask, "pcr_at_open"] = pcr_float


def _build_iv_series(iv_history: dict[str, dict[str, float]]) -> dict[str, pd.Series]:
    result = {}
    for ticker, iv_dict in iv_history.items():
        if iv_dict:
            s = pd.Series(iv_dict)
            s.index = pd.to_datetime(s.index)
            result[ticker] = s.sort_index()
    return result


def _log_fill_rates(df: pd.DataFrame) -> None:
    log.info("AV feature fill rates:")
    for col in _AV_COLS:
        if col not in df.columns:
            continue
        n = df[col].notna().sum()
        log.info("  %-20s  %d / %d  (%.0f%%)", col, n, len(df), n / len(df) * 100)
