#!/usr/bin/env python3
"""
Honest chain-data backtest — replays the live strategy (scanner rules +
risk engine + exits) against real historical EOD options chains.

Unlike backtest.py (which filters YOUR historical fills and inherits your
discretion), this simulates what the algo itself would have traded on every
trading day. Requires historical chain data on disk or an AlphaVantage
premium key (HISTORICAL_OPTIONS endpoint) to download it.

Usage:
  python chain_backtest.py --start 2024-01-02 --end 2026-06-01
  python chain_backtest.py --start ... --end ... --variant derived
  python chain_backtest.py --prefetch        # download chains to cache only

Exit variants compared (--variant all, default):
  derived         2x stop, hold to expiry otherwise   (current STRATEGY)
  hold_only       no stop, always hold to expiry      (the raw derived rule)
  profit_take_50  2x stop + close at 50% of max profit
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from dotenv import load_dotenv

load_dotenv()

from src.av_fetcher import AlphaVantageFetcher, PremiumRequired, RateLimitError
from src.chain_backtester import av_rows_to_puts, simulate, summarize
from src.earnings import straddles_earnings
from src.strategy import STRATEGY

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

VARIANTS = {
    "derived":        {"stop_loss_multiplier": 2.0, "profit_target_pct": None},
    "hold_only":      {"stop_loss_multiplier": 1e9, "profit_target_pct": None},
    "profit_take_50": {"stop_loss_multiplier": 2.0, "profit_target_pct": 0.5},
}


class AVChainSource:
    """ChainSource backed by the AlphaVantage disk cache (src/av_fetcher.py)."""

    def __init__(self, fetcher: AlphaVantageFetcher, watchlist: list[str]):
        self.fetcher = fetcher
        self._prices: dict[str, dict] = {}
        for sym in watchlist:
            try:
                df = fetcher.get_daily_prices(sym)
                self._prices[sym] = {d.date(): float(c) for d, c in df["close"].items()}
            except Exception as e:
                log.warning("%s: no daily prices — %s", sym, e)
                self._prices[sym] = {}

    def get_puts(self, symbol: str, day: date) -> list[dict]:
        try:
            return av_rows_to_puts(self.fetcher.get_historical_options(symbol, str(day)))
        except (PremiumRequired, RateLimitError):
            raise
        except Exception as e:
            log.warning("%s %s: chain fetch failed — %s", symbol, day, e)
            return []

    def get_underlying(self, symbol: str, day: date) -> float | None:
        return self._prices.get(symbol, {}).get(day)

    def trading_days(self, start: date, end: date) -> list[date]:
        days: set[date] = set()
        for price_map in self._prices.values():
            days.update(d for d in price_map if start <= d <= end)
        return sorted(days)


def _print_report(name: str, trades: list[dict]) -> None:
    m = summarize(trades)
    stops = sum(1 for t in trades if t["exit_reason"] == "stop_loss")
    expired = sum(1 for t in trades if t["exit_reason"] == "expired")
    print(f"\n  ── {name} {'─' * (50 - len(name))}")
    print(f"  Trades        : {m['trades']:>10,}   (expired {expired}, stopped {stops})")
    if m["trades"]:
        print(f"  Win rate      : {m['win_rate']*100:>9.1f}%")
        print(f"  Total P&L     : ${m['total_pnl']:>+12,.2f}")
        print(f"  Avg P&L/trade : ${m['avg_pnl']:>+12,.2f}")
        pf = m["profit_factor"]
        print(f"  Profit factor : {pf if pf is not None else 'inf':>10}")
        print(f"  Max drawdown  : ${m['max_drawdown']:>+12,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chain-data strategy backtest")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--variant", default="all", choices=["all", *VARIANTS])
    ap.add_argument("--prefetch", action="store_true",
                    help="Download/cache chains for the window, no simulation")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    cfg = STRATEGY
    watchlist = [t for t in cfg["watchlist"] if t not in cfg.get("tickers_exclude", [])]
    fetcher = AlphaVantageFetcher()
    source = AVChainSource(fetcher, watchlist)

    try:
        vix_series = fetcher.get_vix_history()
        vix = {d.date(): float(v) for d, v in vix_series.items()}
    except Exception as e:
        log.warning("VIX history unavailable (%s) — gate disabled for backtest", e)
        vix = None

    if args.prefetch:
        days = source.trading_days(start, end)
        total = len(days) * len(watchlist)
        log.info("Prefetching %d (symbol, day) chains...", total)
        done = 0
        for day in days:
            for sym in watchlist:
                try:
                    source.get_puts(sym, day)
                except (PremiumRequired, RateLimitError) as e:
                    print(f"\n  ABORTED: {e}\n  Cached {done}/{total} before stopping.")
                    sys.exit(1)
                done += 1
                if done % 100 == 0:
                    log.info("  %d / %d", done, total)
        print(f"\n  Prefetch complete: {done} chain snapshots cached.")
        return

    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    print(f"\n  CHAIN BACKTEST  {start} → {end}   capital ${args.capital:,.0f}")
    print(f"  Watchlist: {len(watchlist)} tickers | VIX gate: "
          f"{'on' if vix else 'off (no data)'} | earnings filter: on")

    try:
        for name in names:
            variant_cfg = {**cfg, "exit": VARIANTS[name]}
            trades = simulate(
                source, variant_cfg, start, end,
                vix=vix,
                earnings_blocked=straddles_earnings,
                capital=args.capital,
            )
            _print_report(name, trades)
    except PremiumRequired:
        print(
            "\n  Historical chains are not cached and your AlphaVantage key is\n"
            "  free-tier (HISTORICAL_OPTIONS is premium). Either upgrade the key\n"
            "  and run with --prefetch first, or point the backtest at another\n"
            "  chain data source (see AVChainSource for the 3-method interface).\n"
        )
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
