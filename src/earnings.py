"""
Earnings date utilities — data sourced from Nasdaq's public calendar API.

Persistent cache (.earnings_cache.json) is populated by running:
  python fetch_earnings.py   (takes ~3-5 min, covers full trade history + 90d ahead)

The dashboard and live scanner both read from cache. The live scanner skips
the earnings filter for tickers not yet in cache (log warning emitted).
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

_CACHE_FILE    = Path(".earnings_cache.json")
_CACHE_TTL     = 7           # days before the whole cache is considered stale
_NASDAQ_URL    = "https://api.nasdaq.com/api/calendar/earnings"
_NASDAQ_HDRS   = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nasdaq.com/",
}

_cache: dict = {}   # process-level memory cache


# ── Disk cache ────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        log.debug("Could not write earnings cache: %s", e)


# ── Nasdaq live fetch (used only when cache misses a ticker) ──────────────────

def _nasdaq_fetch_date(query_date: date) -> list[dict]:
    """Return all earnings rows for one date from Nasdaq."""
    try:
        r = requests.get(
            _NASDAQ_URL,
            params={"date": str(query_date)},
            headers=_NASDAQ_HDRS,
            timeout=10,
        )
        if r.ok:
            return r.json().get("data", {}).get("rows", []) or []
    except Exception as e:
        log.debug("Nasdaq fetch failed for %s: %s", query_date, e)
    return []


def _fetch_upcoming_for_ticker(symbol: str, days: int = 10) -> list[str]:
    """
    Scan the next `days` weekdays on Nasdaq for the given ticker.
    Used as a fast fallback when the ticker is missing from the cache.
    """
    today = date.today()
    dates: list[str] = []
    current = today
    checked = 0
    while checked < days:
        if current.weekday() < 5:
            rows = _nasdaq_fetch_date(current)
            for row in rows:
                if row.get("symbol", "").upper() == symbol.upper():
                    dates.append(str(current))
            checked += 1
            time.sleep(0.4 + random.uniform(0, 0.2))
        current += timedelta(days=1)
    return dates


# ── Public API ────────────────────────────────────────────────────────────────

def get_earnings_dates(symbol: str) -> tuple[date, ...]:
    """
    Return all known earnings dates for a symbol (from cache).
    If the ticker is not in the cache, does a quick 10-day Nasdaq scan
    to check for any imminent earnings, then caches the result.
    """
    global _cache
    if not _cache:
        _cache = _load_cache()

    if symbol in _cache and symbol != "__dates_fetched__":
        return tuple(date.fromisoformat(d) for d in _cache[symbol].get("dates", []))

    # Cache miss — do a quick near-term scan so the live scanner works even if
    # fetch_earnings.py hasn't been run yet.
    log.info("%s: not in earnings cache — scanning next 10 days via Nasdaq", symbol)
    upcoming = _fetch_upcoming_for_ticker(symbol, days=10)
    _cache[symbol] = {"dates": upcoming, "fetched": str(date.today())}
    _save_cache(_cache)
    return tuple(date.fromisoformat(d) for d in upcoming)


def straddles_earnings(symbol: str, start: date, end: date) -> bool:
    """Return True if any earnings date falls within [start, end] inclusive."""
    return any(start <= ed <= end for ed in get_earnings_dates(symbol))


def next_earnings_date(symbol: str) -> date | None:
    """Return the next upcoming earnings date on or after today, or None."""
    today = date.today()
    future = [d for d in get_earnings_dates(symbol) if d >= today]
    return min(future) if future else None


_QUARTERLY_DAYS = 91   # typical days between earnings

def all_cached_earnings() -> pd.DataFrame:
    """
    Return a DataFrame of every ticker in the cache that has earnings data.
    Columns: ticker, next_earnings, est_next_earnings, last_earnings, total_dates,
             days_away, est_days_away, confirmed
    Sorted by effective next date ascending (confirmed first within each day bucket).
    """
    cache = _load_cache()
    today = date.today()
    rows = []
    for symbol, entry in cache.items():
        if symbol.startswith("__"):   # skip meta keys
            continue
        raw_dates = entry.get("dates", [])
        if not raw_dates:
            continue
        parsed = sorted(date.fromisoformat(d) for d in raw_dates)
        future = [d for d in parsed if d >= today]
        past   = [d for d in parsed if d < today]
        next_e = min(future) if future else None
        last_e = max(past)   if past   else None

        # Estimate next date from last + one quarter when Nasdaq hasn't announced yet
        from datetime import timedelta
        est_next = (last_e + timedelta(days=_QUARTERLY_DAYS)) if (last_e and next_e is None) else None

        effective_next = next_e if next_e else est_next
        rows.append({
            "ticker":           symbol,
            "next_earnings":    next_e,
            "est_next_earnings": est_next,
            "last_earnings":    last_e,
            "total_dates":      len(parsed),
            "days_away":        (next_e - today).days if next_e else None,
            "est_days_away":    (est_next - today).days if est_next else None,
            "confirmed":        next_e is not None,
            "_sort_key":        (effective_next or date(9999, 1, 1)),
        })

    if not rows:
        return pd.DataFrame(columns=["ticker", "next_earnings", "est_next_earnings",
                                     "last_earnings", "total_dates", "days_away",
                                     "est_days_away", "confirmed"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["_sort_key", "confirmed"], ascending=[True, False]).drop(columns="_sort_key")
    return df.reset_index(drop=True)


def earnings_data_available() -> bool:
    """Return True if the cache has real earnings data."""
    cache = _load_cache()
    return any(
        len(v.get("dates", [])) > 0
        for k, v in cache.items()
        if not k.startswith("__")
    )


def flag_earnings_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean 'straddles_earnings' column to a trades DataFrame."""
    global _cache
    if not _cache:
        _cache = _load_cache()

    df = trades.copy()
    flags: list[bool] = []
    for _, row in df.iterrows():
        try:
            open_dt   = pd.Timestamp(row["open_date"]).date()
            expiry_dt = pd.Timestamp(row["expiry"]).date()
            flags.append(straddles_earnings(row["ticker"], open_dt, expiry_dt))
        except Exception:
            flags.append(False)
    df["straddles_earnings"] = flags
    return df


def earnings_impact_report(strategy_trades: pd.DataFrame) -> None:
    """Print a comparison: all strategy trades vs earnings-free vs earnings-straddling."""
    from src.patterns import backtest_metrics

    print("\nRunning earnings filter analysis...")

    if not earnings_data_available():
        print("\n  [!] No earnings data in cache.")
        print("      Run: python fetch_earnings.py   (~3-5 min, only needed once a week)\n")
        return

    flagged = flag_earnings_trades(strategy_trades)
    clean   = flagged[~flagged["straddles_earnings"]]
    risky   = flagged[flagged["straddles_earnings"]]

    m_all   = backtest_metrics(flagged)
    m_clean = backtest_metrics(clean)
    m_risky = backtest_metrics(risky)

    def _sign(v):
        return f"+${v:,.2f}" if v and v >= 0 else f"-${abs(v):,.2f}" if v else "N/A"
    def _pct(v):
        return f"{v*100:.1f}%" if v is not None else "N/A"
    def _pf(v):
        return f"{v:.2f}" if v is not None else "N/A"

    print(f"\n{'='*74}")
    print(f"  EARNINGS FILTER IMPACT")
    print(f"{'='*74}")
    print(f"  {'Metric':<28} {'All Strategy':>14} {'No Earnings':>14} {'Earnings Only':>14}")
    print(f"{'─'*74}")

    rows = [
        ("Trades",
         f"{m_all.get('trades',0):,}", f"{m_clean.get('trades',0):,}", f"{m_risky.get('trades',0):,}"),
        ("Win rate",
         _pct(m_all.get("win_rate")), _pct(m_clean.get("win_rate")), _pct(m_risky.get("win_rate"))),
        ("Total P&L",
         _sign(m_all.get("total_pnl",0)), _sign(m_clean.get("total_pnl",0)), _sign(m_risky.get("total_pnl",0))),
        ("Avg P&L/trade",
         _sign(m_all.get("avg_pnl",0)), _sign(m_clean.get("avg_pnl",0)), _sign(m_risky.get("avg_pnl",0))),
        ("Profit factor",
         _pf(m_all.get("profit_factor")), _pf(m_clean.get("profit_factor")), _pf(m_risky.get("profit_factor"))),
        ("Max drawdown",
         _sign(m_all.get("max_drawdown",0)), _sign(m_clean.get("max_drawdown",0)), _sign(m_risky.get("max_drawdown",0))),
        ("Sharpe ratio",
         f"{m_all.get('sharpe',0):.2f}", f"{m_clean.get('sharpe',0):.2f}", f"{m_risky.get('sharpe',0):.2f}"),
        ("Worst trade",
         _sign(m_all.get("worst_trade",0)), _sign(m_clean.get("worst_trade",0)), _sign(m_risky.get("worst_trade",0))),
    ]
    for label, a, c, r in rows:
        print(f"  {label:<28} {a:>14} {c:>14} {r:>14}")
    print(f"{'─'*74}")

    n_risky = len(risky)
    n_total = len(flagged)
    if n_total:
        risky_pnl = m_risky.get("total_pnl", 0) or 0
        print(f"\n  Earnings filter excludes {n_risky:,} of {n_total:,} trades ({n_risky/n_total*100:.1f}%)")
        print(f"  Those {n_risky} trades generated: {_sign(risky_pnl)}")

    if n_risky > 0 and not risky.empty:
        print(f"\n  Worst earnings-straddling trades:")
        worst = risky.nsmallest(5, "net_pnl")[["ticker", "open_date", "expiry", "net_pnl"]]
        for _, r in worst.iterrows():
            print(f"    {r['ticker']:<8}  opened {str(r['open_date'])[:10]}"
                  f"  expiry {str(r['expiry'])[:10]}  P&L {_sign(r['net_pnl'])}")

    print(f"{'='*74}\n")
