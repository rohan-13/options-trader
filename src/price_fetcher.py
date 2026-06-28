"""
Direct Yahoo Finance price fetcher (bypasses yfinance header issues).

Uses Yahoo Finance v8/v11 APIs directly with proper headers.
Caches daily OHLCV to data/cache/prices/ so each ticker downloads once.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "prices"
_LAST_CALL = 0.0
_CALL_GAP = 0.5
_SESSION: requests.Session | None = None
_CRUMB: str | None = None


def _get_session() -> tuple[requests.Session, str]:
    """Return (session, crumb) — initialises cookie jar on first call."""
    global _SESSION, _CRUMB
    if _SESSION is not None and _CRUMB is not None:
        return _SESSION, _CRUMB

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com",
    })

    # Step 1: Visit Yahoo Finance to acquire A3 cookie
    for seed_url in ["https://finance.yahoo.com", "https://fc.yahoo.com"]:
        try:
            s.get(seed_url, timeout=8)
            break
        except Exception:
            continue

    time.sleep(0.3)

    # Step 2: Get crumb
    crumb = ""
    try:
        r = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=10)
        if r.status_code == 200 and r.text and "<" not in r.text:
            crumb = r.text.strip()
            log.debug("Got Yahoo crumb: %s…", crumb[:6])
        else:
            log.debug("Crumb endpoint: status=%d text=%r", r.status_code, r.text[:50])
    except Exception as e:
        log.debug("Crumb fetch failed: %s", e)

    _SESSION = s
    _CRUMB = crumb
    return s, crumb


def _throttle() -> None:
    global _LAST_CALL
    wait = _CALL_GAP - (time.time() - _LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL = time.time()


def get_daily_prices(symbol: str, start: str, end: str, force: bool = False) -> pd.Series:
    """
    Return daily close prices for `symbol` between `start` and `end` (YYYY-MM-DD).
    Returns a pd.Series indexed by date, forward-filled over weekends/holidays.
    Returns empty Series on failure (caller skips the trade).
    """
    cache_path = _CACHE_DIR / f"{symbol}.csv"

    # Check cache; if it covers the requested range, use it
    if not force and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index)
        s = cached["close"].dropna()
        if not s.empty and s.index.min() <= pd.Timestamp(start) and s.index.max() >= pd.Timestamp(end):
            return _ffill(s, start, end)
        # Cache exists but doesn't cover the range — re-fetch with extended range

    series = _fetch_yahoo(symbol, start, end)
    if not series.empty:
        _save_cache(symbol, series)
    return _ffill(series, start, end)


def get_daily_prices_full(symbol: str, force: bool = False) -> pd.Series:
    """Fetch maximum available history (20+ years) for a symbol."""
    cache_path = _CACHE_DIR / f"{symbol}.csv"
    if not force and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index)
        s = cached["close"].dropna()
        if not s.empty:
            return s

    start = "2010-01-01"
    end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    series = _fetch_yahoo(symbol, start, end)
    if not series.empty:
        _save_cache(symbol, series)
    return series


def _fetch_yahoo(symbol: str, start: str, end: str) -> pd.Series:
    period1 = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    period2 = int(datetime.strptime(end, "%Y-%m-%d").timestamp()) + 86400

    session, crumb = _get_session()
    params = f"interval=1d&period1={period1}&period2={period2}"
    if crumb:
        params += f"&crumb={crumb}"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"

    _throttle()
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 429:
            log.debug("Rate limited for %s — waiting 2s", symbol)
            time.sleep(2)
            r = session.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("Yahoo fetch failed for %s: %s", symbol, e)
        return pd.Series(dtype=float)

    try:
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
        dates = pd.to_datetime(timestamps, unit="s").normalize()
        s = pd.Series(closes, index=dates, name="close", dtype=float)
        return s.dropna()
    except (KeyError, IndexError, TypeError) as e:
        log.debug("Yahoo parse failed for %s: %s", symbol, e)
        return pd.Series(dtype=float)


def _ffill(series: pd.Series, start: str, end: str) -> pd.Series:
    if series.empty:
        return series
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
    return series.reindex(full_idx).ffill()


def _save_cache(symbol: str, series: pd.Series) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = series.to_frame(name="close")
    df.index.name = "date"
    df.to_csv(_CACHE_DIR / f"{symbol}.csv")
    log.debug("Cached %d price rows for %s", len(df), symbol)
