#!/usr/bin/env python3
"""
Populate the earnings date cache using Nasdaq's public earnings calendar API.

Strategy: query the calendar date-by-date (weekdays only) over the full
trade history period plus the next 90 days, then build a per-ticker index
for every ticker that appears in your trade data.

Run once (or weekly):
  python fetch_earnings.py

Estimated time: ~3-5 minutes for ~700 weekdays.
Progress saves after every day — safe to stop and resume.
"""
import json
import sys
import time
import random
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

CACHE_FILE  = Path(".earnings_cache.json")
TRADES_FILE = Path("output/all_trades.csv")

NASDAQ_URL  = "https://api.nasdaq.com/api/calendar/earnings"
HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nasdaq.com/",
}

# Query from the start of the trade history through 90 days ahead
HISTORY_START = date(2023, 12, 1)
LOOKAHEAD_DAYS = 90


# ── Cache helpers ────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ── Nasdaq fetch ──────────────────────────────────────────────────────────────

def fetch_date(query_date: date, retries: int = 3) -> list[dict]:
    """Return all earnings rows for a given date from Nasdaq."""
    for attempt in range(retries):
        try:
            r = requests.get(
                NASDAQ_URL,
                params={"date": str(query_date)},
                headers=HEADERS,
                timeout=15,
            )
            if r.status_code == 200:
                payload = r.json()
                if payload is None:
                    continue  # empty body — retry
                return (payload.get("data") or {}).get("rows", []) or []
            if r.status_code == 429 and attempt < retries - 1:
                time.sleep(15 + random.uniform(0, 5))
                continue
        except (requests.RequestException, ValueError):
            if attempt < retries - 1:
                time.sleep(5)
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TRADES_FILE.exists():
        print("Run  python main.py  first to generate output/all_trades.csv")
        sys.exit(1)

    # Load traded tickers
    df = pd.read_csv(TRADES_FILE)
    traded = set(df["ticker"].dropna().unique().tolist())
    print(f"Tracking {len(traded)} unique tickers from trade history.\n")

    # Build the date range to query (weekdays only)
    end_date  = date.today() + timedelta(days=LOOKAHEAD_DAYS)
    all_dates = [
        HISTORY_START + timedelta(days=i)
        for i in range((end_date - HISTORY_START).days + 1)
        if (HISTORY_START + timedelta(days=i)).weekday() < 5  # Mon–Fri
    ]

    # Load existing cache and figure out which dates already have a record
    cache = load_cache()
    meta_key = "__dates_fetched__"
    fetched_dates = set(cache.get(meta_key, {}).get("dates", []))

    to_fetch = [d for d in all_dates if str(d) not in fetched_dates]
    print(f"  {len(all_dates)} weekdays in range  ({HISTORY_START} → {end_date})")
    print(f"  {len(fetched_dates)} already cached — skipping")
    print(f"  {len(to_fetch)} dates to fetch")
    print(f"  Estimated time: ~{len(to_fetch) * 0.4 / 60:.0f}–{len(to_fetch) * 0.6 / 60:.0f} min\n")

    # Build per-ticker → dates mapping from existing cache (excluding meta key)
    earnings_map: dict[str, set[str]] = {}
    for sym, entry in cache.items():
        if sym == meta_key:
            continue
        for d in entry.get("dates", []):
            earnings_map.setdefault(sym, set()).add(d)

    hits = 0
    for i, query_date in enumerate(to_fetch, 1):
        date_str = str(query_date)
        pct      = i / len(to_fetch) * 100
        eta_min  = (len(to_fetch) - i) * 0.5 / 60

        rows = fetch_date(query_date)

        # Cross-reference with traded tickers
        matched = []
        for row in rows:
            sym = row.get("symbol", "").strip().upper()
            if sym in traded:
                earnings_map.setdefault(sym, set()).add(date_str)
                matched.append(sym)
                hits += 1

        label = f"  {', '.join(matched)}" if matched else ""
        print(f"  [{i:>4}/{len(to_fetch)}  {pct:4.0f}%  ETA ~{eta_min:.0f}m]"
              f"  {date_str}  ({len(rows):>3} companies){label}")

        # Mark date as fetched and persist after every day
        fetched_dates.add(date_str)
        cache[meta_key] = {"dates": sorted(fetched_dates)}
        today_str = str(date.today())
        for sym, dates in earnings_map.items():
            cache[sym] = {"dates": sorted(dates), "fetched": today_str}
        save_cache(cache)

        time.sleep(0.4 + random.uniform(0, 0.3))

    # Final summary
    tickers_with_data = [s for s in earnings_map if earnings_map[s]]
    print(f"\nDone.")
    print(f"  {len(tickers_with_data)} traded tickers have earnings dates in cache")
    print(f"  {hits} total ticker-date pairs found")
    print(f"  Cache saved to {CACHE_FILE}")


if __name__ == "__main__":
    main()
