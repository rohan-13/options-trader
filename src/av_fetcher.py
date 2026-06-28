"""
AlphaVantage data fetcher with aggressive disk caching.

Free-tier endpoints (25 req/day):
  TIME_SERIES_DAILY  → underlying daily OHLCV
  INDEX_DATA(VIX)    → VIX daily history

Premium endpoints (require paid plan):
  HISTORICAL_OPTIONS          → full chain with IV & Greeks per (symbol, date)
  HISTORICAL_PUT_CALL_RATIO   → daily PCR per (symbol, date)

Environment:
    ALPHAVANTAGE_API_KEY — your API key (required)

Cache layout (data/cache/av/):
    prices/{SYMBOL}.csv
    vix.csv
    options/{SYMBOL}/{DATE}.json
    pcr/{SYMBOL}/{DATE}.json
"""
from __future__ import annotations

import json
import logging
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"
_DEFAULT_CACHE = Path(__file__).parent.parent / "data" / "cache" / "av"


class PremiumRequired(Exception):
    pass


class RateLimitError(Exception):
    pass


class AlphaVantageFetcher:
    """Thin wrapper around the AlphaVantage REST API with file-based caching."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        call_gap_sec: float = 13.0,
    ):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Set ALPHAVANTAGE_API_KEY in your .env file. "
                "Get a free key at https://www.alphavantage.co/support/#api-key"
            )
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE)
        self.call_gap_sec = call_gap_sec
        self._last_call: float = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        wait = self.call_gap_sec - (time.time() - self._last_call)
        if wait > 0:
            log.debug("AV throttle: waiting %.1fs", wait)
            time.sleep(wait)
        self._last_call = time.time()

    # ── Raw HTTP call ─────────────────────────────────────────────────────────

    def _get(self, params: dict) -> dict | str:
        params = {**params, "apikey": self.api_key}
        self._throttle()
        log.debug("AV call: function=%s symbol=%s date=%s",
                  params.get("function"), params.get("symbol"), params.get("date", ""))

        r = requests.get(_BASE_URL, params=params, timeout=30)
        r.raise_for_status()

        text = r.text.strip()

        # AV sometimes returns a JSON error even when CSV was requested —
        # detect by content-type or leading brace and always check for errors.
        ct = r.headers.get("Content-Type", "")
        is_json_response = "json" in ct or text.startswith("{")

        if is_json_response:
            try:
                data = r.json()
            except Exception:
                return text  # not actually JSON despite appearances
            msg = data.get("Information", "") or data.get("Note", "") or data.get("Error Message", "")
            if msg:
                if "premium" in msg.lower():
                    raise PremiumRequired(msg)
                if "rate limit" in msg.lower() or "25 requests per day" in msg.lower():
                    raise RateLimitError(msg)
                raise ValueError(f"AV API error: {msg}")
            return data  # legitimate JSON response

        want_json = params.get("datatype", "json") == "json"
        if want_json:
            # Requested JSON and got non-JSON text — shouldn't happen but handle it
            return text
        return text  # CSV

    # ── Daily price series ────────────────────────────────────────────────────

    def get_daily_prices(self, symbol: str, force: bool = False) -> pd.DataFrame:
        """Daily OHLCV for symbol (free tier). Returns DataFrame indexed by date."""
        cache = self.cache_dir / "prices" / f"{symbol}.csv"
        if not force and cache.exists():
            df = pd.read_csv(cache, index_col="date", parse_dates=True)
            df.index = pd.to_datetime(df.index)
            return df

        log.info("Fetching daily prices: %s", symbol)
        csv_text = self._get({
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "csv",
        })
        df = pd.read_csv(StringIO(csv_text))
        df = df.rename(columns={"timestamp": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        log.info("Cached %d days of %s prices", len(df), symbol)
        return df

    # ── VIX history ───────────────────────────────────────────────────────────

    def get_vix_history(self, force: bool = False) -> pd.Series:
        """Daily VIX close prices as a forward-filled Series (free tier)."""
        cache = self.cache_dir / "vix.csv"
        if not force and cache.exists():
            df = pd.read_csv(cache, index_col="date", parse_dates=True)
            df.index = pd.to_datetime(df.index)
            s = df["close"].dropna()
        else:
            log.info("Fetching VIX history (INDEX_DATA)")
            data = self._get({
                "function": "INDEX_DATA",
                "symbol": "VIX",
                "interval": "daily",
                "datatype": "json",
            })
            records = data.get("data", [])
            if not records:
                raise ValueError("Empty VIX data returned by INDEX_DATA")
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache)
            log.info("Cached %d days of VIX history", len(df))
            s = df["close"].dropna()

        full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
        return s.reindex(full_idx).ffill().rename("vix")

    # ── Historical options chain (PREMIUM) ────────────────────────────────────

    def get_historical_options(
        self, symbol: str, date: str, force: bool = False
    ) -> pd.DataFrame:
        """
        Full options chain for (symbol, date) including IV and Greeks.
        Raises PremiumRequired if the API key doesn't include this endpoint.
        `date` must be YYYY-MM-DD format.
        """
        cache = self.cache_dir / "options" / symbol / f"{date}.json"
        if not force and cache.exists():
            with open(cache) as f:
                records = json.load(f)
            return pd.DataFrame(records) if records else pd.DataFrame()

        log.info("Fetching historical options: %s  %s", symbol, date)
        csv_text = self._get({
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "date": date,
            "datatype": "csv",
        })
        if not isinstance(csv_text, str) or not csv_text.strip():
            log.warning("Empty options response for %s on %s", symbol, date)
            return pd.DataFrame()

        df = pd.read_csv(StringIO(csv_text))
        if df.empty:
            return df

        _numeric = [
            "strike", "last", "mark", "bid", "bid_size", "ask", "ask_size",
            "volume", "open_interest", "implied_volatility",
            "delta", "gamma", "theta", "vega", "rho",
        ]
        for col in _numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(df.to_dict(orient="records"), f)
        log.info("Cached %d contracts for %s on %s", len(df), symbol, date)
        return df

    # ── Historical put-call ratio (PREMIUM) ───────────────────────────────────

    def get_historical_pcr(
        self, symbol: str, date: str, force: bool = False
    ) -> dict:
        """
        Put-call ratio for (symbol, date).
        Returns raw API dict; caller extracts the relevant PCR value.
        Raises PremiumRequired if plan doesn't include this endpoint.
        """
        cache = self.cache_dir / "pcr" / symbol / f"{date}.json"
        if not force and cache.exists():
            with open(cache) as f:
                return json.load(f)

        log.info("Fetching PCR: %s  %s", symbol, date)
        data = self._get({
            "function": "HISTORICAL_PUT_CALL_RATIO",
            "symbol": symbol,
            "date": date,
        })
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(data, f)
        return data

    # ── Cache status ──────────────────────────────────────────────────────────

    def cache_status(self, tickers: list[str]) -> dict:
        """Returns a summary dict of what's already cached."""
        status: dict = {
            "vix": (self.cache_dir / "vix.csv").exists(),
            "prices": {},
            "options_dates": {},
            "pcr_dates": {},
        }
        for sym in tickers:
            status["prices"][sym] = (self.cache_dir / "prices" / f"{sym}.csv").exists()
            opts_dir = self.cache_dir / "options" / sym
            status["options_dates"][sym] = len(list(opts_dir.glob("*.json"))) if opts_dir.exists() else 0
            pcr_dir = self.cache_dir / "pcr" / sym
            status["pcr_dates"][sym] = len(list(pcr_dir.glob("*.json"))) if pcr_dir.exists() else 0
        return status
