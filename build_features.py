#!/usr/bin/env python3
"""
Build the feature dataset for ML model training.

Loads all E*Trade trade history, matches round-trip trades, filters to the
0-7 DTE short put strategy, and attaches market-context features from yfinance.
Saves the enriched dataset to output/features.csv.

Usage:
  python build_features.py

Output:
  output/features.csv   — one row per strategy trade, all features + is_win label
"""
from __future__ import annotations

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
from src.parser import parse_transactions
from src.matcher import match_trades
from src.features import build_features, FEATURE_COLS


def main() -> None:
    log.info("Loading E*Trade transaction data…")
    raw = load_all()

    log.info("Parsing %d raw rows…", len(raw))
    parsed = parse_transactions(raw)

    log.info("Matching round-trip trades…")
    trades = match_trades(parsed)

    closed = trades[trades["is_closed"]]
    log.info(
        "Total closed trades: %d  |  short puts 0-7 DTE: %d",
        len(closed),
        len(closed[
            (closed["strategy"] == "PUT_short") &
            (closed["dte_at_open"].fillna(99) <= 7)
        ]),
    )

    log.info("Building features (fetching historical price data via yfinance)…")
    features = build_features(trades)

    if features.empty:
        log.error("Feature build returned no rows. Exiting.")
        sys.exit(1)

    out_path = Path("output/features.csv")
    out_path.parent.mkdir(exist_ok=True)
    features.to_csv(out_path, index=False)
    log.info("Saved %d rows × %d cols → %s", len(features), len(features.columns), out_path)

    # Quick sanity check
    missing = features[FEATURE_COLS].isna().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.05]
    if not high_missing.empty:
        log.warning("Features with >5%% missing values:")
        for col, pct in high_missing.items():
            log.warning("  %-22s  %.1f%% missing", col, pct * 100)
    else:
        log.info("All features have <5%% missing — dataset looks clean.")

    log.info("Done. Run 'python train_model.py' next to train the ML model.")


if __name__ == "__main__":
    main()
