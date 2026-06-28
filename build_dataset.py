#!/usr/bin/env python3
"""
Phase 1: Build the enriched feature dataset.

Pipeline:
  1. Load all E*Trade CSVs → raw transactions
  2. Parse + match → round-trip trades
  3. Filter to 0-7 DTE short puts on watchlist
  4. yfinance features (existing pipeline from features.py)
  5. AlphaVantage enrichment (VIX, spot, options IV/Greeks, PCR)
  6. Save → output/dataset_phase1.csv

Usage:
  python build_dataset.py                   # free tier: VIX + prices only
  python build_dataset.py --premium         # + IV, delta, theta, PCR (requires AV premium)
  python build_dataset.py --skip-yf         # skip yfinance step (AV-only, faster)
  python build_dataset.py --cache-status    # show what's already cached, then exit

Requires:
  ALPHAVANTAGE_API_KEY=<key> in .env
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

from src.loader import load_all
from src.parser import enrich_options
from src.matcher import match_trades
from src.features import build_features, FEATURE_COLS
from src.strategy import STRATEGY
from src.av_fetcher import AlphaVantageFetcher, RateLimitError
from src.enricher import enrich_with_av


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase 1 enriched dataset")
    ap.add_argument("--premium", action="store_true",
                    help="Fetch HISTORICAL_OPTIONS + PCR (requires AV premium plan)")
    ap.add_argument("--skip-yf", action="store_true",
                    help="Skip yfinance feature build (quicker, loses HV/return cols)")
    ap.add_argument("--call-gap", type=float, default=13.0,
                    help="Seconds between AV API calls (default 13 ≈ 4/min)")
    ap.add_argument("--cache-status", action="store_true",
                    help="Print cache coverage and exit without fetching")
    args = ap.parse_args()

    # ── AlphaVantage fetcher (optional) ──────────────────────────────────────
    fetcher = None
    try:
        fetcher = AlphaVantageFetcher(call_gap_sec=args.call_gap)
    except ValueError:
        log.info("No ALPHAVANTAGE_API_KEY in .env — skipping AV enrichment")
        log.info("Add key to .env and re-run to get VIX/price data from AV")

    if args.cache_status:
        if fetcher is None:
            log.error("--cache-status requires ALPHAVANTAGE_API_KEY in .env")
            sys.exit(1)
        status = fetcher.cache_status(STRATEGY["watchlist"])
        print("\nCache status:")
        print(f"  VIX cached      : {status['vix']}")
        print(f"  Prices cached   :")
        for sym, cached in status["prices"].items():
            mark = "✓" if cached else "✗"
            opts = status["options_dates"][sym]
            pcr = status["pcr_dates"][sym]
            print(f"    {sym:<6} {mark}   options_dates={opts}  pcr_dates={pcr}")
        sys.exit(0)

    # ── Steps 1-3: Load and match trades ─────────────────────────────────────
    log.info("Loading E*Trade transaction data…")
    raw = load_all()
    log.info("Parsing %d raw rows…", len(raw))
    parsed = enrich_options(raw)
    log.info("Matching round-trip trades…")
    trades = match_trades(parsed)

    closed = trades[trades["is_closed"]]
    strat = closed[
        (closed["strategy"] == "PUT_short") &
        (closed["dte_at_open"].fillna(99) <= 7)
    ]
    log.info("Closed trades: %d  |  0-7 DTE short puts: %d", len(closed), len(strat))

    # ── Step 4: yfinance features ─────────────────────────────────────────────
    if not args.skip_yf:
        log.info("Building yfinance features…")
        features = build_features(trades)
        if features.empty:
            log.error("Feature build returned no rows.")
            sys.exit(1)
        log.info("yfinance features: %d rows × %d cols", len(features), len(features.columns))
    else:
        features = strat.copy()
        features["is_win"] = (features["net_pnl"] > 0).astype(int)
        log.info("Skipping yfinance — %d raw strategy trades", len(features))

    # ── Step 5: AlphaVantage enrichment ───────────────────────────────────────
    if fetcher is None:
        log.info("Skipping AV enrichment (no API key)")
        enriched = features
    else:
        log.info("Starting AlphaVantage enrichment (premium=%s)…", args.premium)
        log.info("  Free-tier: VIX history + underlying prices for all tickers")
        if args.premium:
            log.info("  Premium:   HISTORICAL_OPTIONS + PCR per (ticker, date)")
        else:
            log.info("  Run with --premium once you have an AV premium plan to add IV/delta/PCR")
        try:
            enriched = enrich_with_av(
                features,
                fetcher,
                fetch_options=args.premium,
                fetch_pcr=args.premium,
            )
        except RateLimitError as e:
            log.warning("Hit AV daily rate limit: %s", e)
            log.warning("Free tier = 25 calls/day. Re-run tomorrow or upgrade plan.")
            enriched = features

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    out = Path("output/dataset_phase1.csv")
    out.parent.mkdir(exist_ok=True)
    enriched.to_csv(out, index=False)
    log.info("Saved %d rows × %d cols → %s", len(enriched), len(enriched.columns), out)

    # Missing-value report
    all_feat = FEATURE_COLS + ["av_vix", "av_vix_rank", "av_spot",
                               "iv_actual", "delta_actual", "theta_actual",
                               "iv_rank_52w", "pcr_at_open"]
    present = [c for c in all_feat if c in enriched.columns]
    missing = enriched[present].isna().mean().sort_values(ascending=False)
    high_miss = missing[missing > 0.05]
    if not high_miss.empty:
        log.warning("Features with >5%% missing values:")
        for col, pct in high_miss.items():
            log.warning("  %-22s  %.0f%% missing", col, pct * 100)
    else:
        log.info("All features have <5%% missing — dataset looks clean.")

    log.info("Done. Next: python train_model.py")


if __name__ == "__main__":
    main()
