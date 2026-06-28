#!/usr/bin/env python3
"""
Options Trader — Phase 1: Data Engine & Trade Analytics
"""

import time

from src.loader import load_all
from src.matcher import match_trades
from src.parser import enrich_options
from src.reporter import run_report


def main() -> None:
    t0 = time.time()

    print("Loading all E*TRADE transaction files...")
    transactions = load_all()
    print(f"  {len(transactions):,} rows loaded ({transactions['is_option'].sum():,} options, {transactions['is_equity'].sum():,} equity)")

    print("\nParsing option contract fields...")
    transactions = enrich_options(transactions)
    parsed_count = transactions["contract_key"].notna().sum()
    print(f"  {parsed_count:,} option rows successfully parsed")

    print("\nRunning FIFO trade matching...")
    trades = match_trades(transactions)
    closed = trades[trades["is_closed"]]
    print(f"  {len(trades):,} total trades found ({len(closed):,} closed, {(trades['closed_by'] == 'open').sum()} open)")

    print("\nGenerating report...")
    run_report(transactions, trades)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
