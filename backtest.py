#!/usr/bin/env python3
"""
Run the data-derived strategy backtest on historical trades.

Usage:
  python backtest.py
"""
import time

from src.backtester import print_backtest_report, run_backtest
from src.earnings import earnings_impact_report
from src.loader import load_all
from src.matcher import match_trades
from src.parser import enrich_options


def main() -> None:
    t0 = time.time()

    print("Loading historical trades...")
    transactions = load_all()
    transactions = enrich_options(transactions)
    trades = match_trades(transactions)
    closed = trades[trades["is_closed"]]
    print(f"  {len(closed):,} closed trades loaded")

    print("Running backtest...")
    result = run_backtest(trades)
    print_backtest_report(result)

    earnings_impact_report(result["strategy_trades"])

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
