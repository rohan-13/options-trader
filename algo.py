#!/usr/bin/env python3
"""
Live options algo — scans for short put opportunities and places orders via E*Trade.

Usage:
  python algo.py --scan          # find opportunities, print them, no orders placed
  python algo.py --trade         # scan and place real limit orders
  python algo.py --status        # show open positions and open orders
  python algo.py --monitor       # watch open short puts, enforce the 2x stop-loss
  python algo.py --monitor --once     # single stop-loss check (for cron)
  python algo.py --sandbox       # use E*Trade sandbox (add to any flag)

Kill switch:
  touch HALT   → blocks all NEW entries (--trade). The --monitor stop-loss
  keeps running regardless: protective closes are never halted.

Credentials:
  Copy .env.example → .env and fill in ETRADE_CONSUMER_KEY / ETRADE_CONSUMER_SECRET.
  Access tokens expire at midnight ET daily — re-run to re-authenticate.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from src import risk
from src.etrade_api import ETrade
from src.scanner import scan_watchlist
from src.strategy import STRATEGY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_signals(signals: list[dict]) -> None:
    if not signals:
        print("  No qualifying opportunities found.")
        return
    hdr = f"  {'Symbol':<8} {'Expiry':<12} {'DTE':>4} {'Strike':>8} {'Bid':>6} {'Ask':>6} {'Premium':>9} {'OTM%':>6} {'OI':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for s in signals:
        print(
            f"  {s['symbol']:<8} {str(s['expiry']):<12} {s['dte']:>4} "
            f"{s['strike']:>8.1f} {s['bid']:>6.2f} {s['ask']:>6.2f} "
            f"${s['premium']:>8.2f} {s['otm_pct']:>5.1f}% {s['open_interest']:>7,}"
        )


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_scan(api: ETrade) -> None:
    cfg = STRATEGY
    watchlist_size = len([t for t in cfg["watchlist"] if t not in cfg.get("tickers_exclude", [])])
    print(f"\n  Scanning {watchlist_size} tickers for {cfg['name']} opportunities...")
    signals = scan_watchlist(api)
    print(f"\n  {len(signals)} qualifying opportunities:\n")
    _print_signals(signals)
    print(f"\n  Run with --trade to place limit orders for all {len(signals)} opportunities.")


def fetch_vix() -> float | None:
    """Spot VIX via yfinance; None on any failure (risk gate fails closed)."""
    try:
        import yfinance as yf
        return float(yf.Ticker("^VIX").fast_info["lastPrice"])
    except Exception as e:
        log.warning("VIX fetch failed: %s", e)
        return None


def cmd_trade(api: ETrade, account_key: str) -> None:
    cfg = STRATEGY
    risk_cfg = cfg["risk"]

    # Gate 1: kill switch — a HALT file blocks all new entries.
    if risk.halt_active(risk_cfg["halt_file"]):
        print(f"\n  HALTED: '{risk_cfg['halt_file']}' file exists — no new orders. "
              f"Delete it to resume trading.")
        return

    # Gate 2: VIX circuit breaker (fails closed if VIX can't be fetched).
    allowed, reason = risk.vix_allows_entry(fetch_vix(), risk_cfg["vix_max"])
    print(f"\n  {reason}")
    if not allowed:
        print("  No orders placed.")
        return

    watchlist_size = len([t for t in cfg["watchlist"] if t not in cfg.get("tickers_exclude", [])])
    print(f"  Scanning {watchlist_size} tickers...")
    signals = scan_watchlist(api)

    if not signals:
        print("  No qualifying opportunities — no orders placed.")
        return

    # Gate 3: portfolio limits against live positions and buying power.
    open_positions = risk.parse_short_puts(api.get_portfolio(account_key))
    buying_power = risk.extract_buying_power(api.get_balance(account_key))
    for s in signals:
        s["qty"] = cfg["qty"]
    approved, rejected = risk.filter_signals(signals, open_positions, risk_cfg, buying_power)

    print(f"\n  {len(signals)} signals → {len(approved)} approved, {len(rejected)} rejected by risk limits")
    print(f"  Open short puts: {len(open_positions)} | Cash buying power: ${buying_power:,.0f}")
    for sig, why in rejected:
        log.info("REJECTED %s %s $%.0fP: %s", sig["symbol"], sig["expiry"], sig["strike"], why)

    if not approved:
        print("  Nothing passed the risk gates — no orders placed.")
        return

    print(f"\n  Placing {len(approved)} orders...\n")
    placed = 0
    for s in approved:
        order = ETrade.build_sell_put_order(
            symbol=s["symbol"],
            expiry=s["expiry"],
            strike=s["strike"],
            qty=cfg["qty"],
            limit_price=s["mid"],
        )
        try:
            preview = api.preview_order(account_key, order)
            preview_id = preview["PreviewOrderResponse"]["PreviewIds"]["previewId"]
            result = api.place_order(account_key, order, preview_id)
            order_id = result["PlaceOrderResponse"]["OrderIds"]["orderId"]
            log.info(
                "ORDER PLACED: %s %s $%.0fP @ $%.2f  (credit $%.0f)  ID=%s",
                s["symbol"], s["expiry"], s["strike"], s["mid"], s["premium"], order_id,
            )
            placed += 1
        except Exception as e:
            log.error("Failed to place %s %s $%sP: %s", s["symbol"], s["expiry"], s["strike"], e)

    print(f"\n  {placed} of {len(approved)} orders placed successfully.")


def cmd_status(api: ETrade, account_key: str) -> None:
    print("\n  ── Open Positions ──────────────────────────────────────────────")
    try:
        portfolio = api.get_portfolio(account_key)
        for acct in portfolio:
            for pos in acct.get("Position", []):
                prod = pos.get("Product", {})
                sym  = prod.get("symbol", "?")
                mv   = pos.get("marketValue", 0)
                cost = pos.get("totalCost", 0)
                pnl  = mv - cost
                print(f"    {sym:<15} MV=${float(mv):>10,.2f}   P&L=${pnl:>+10,.2f}")
    except Exception as e:
        log.error("Could not fetch portfolio: %s", e)

    print("\n  ── Open Orders ─────────────────────────────────────────────────")
    try:
        orders = api.get_orders(account_key)
        if not orders:
            print("    None")
        for o in orders:
            details = o.get("OrderDetail", [{}])[0]
            instruments = details.get("Instrument", [{}])
            prod = instruments[0].get("Product", {}) if instruments else {}
            sym = prod.get("symbol", "?")
            action = instruments[0].get("orderAction", "?") if instruments else "?"
            price = details.get("limitPrice", "?")
            status = o.get("status", "?")
            print(f"    #{o.get('orderId')}  {sym:<10} {action:<15} @ ${price}  [{status}]")
    except Exception as e:
        log.error("Could not fetch orders: %s", e)


def monitor_pass(api: ETrade, account_key: str) -> int:
    """One stop-loss sweep over all open short puts. Returns orders placed."""
    cfg = STRATEGY
    multiplier = cfg["exit"]["stop_loss_multiplier"]
    buffer_pct = cfg["risk"].get("stop_buffer_pct", 0.05)

    positions = risk.parse_short_puts(api.get_portfolio(account_key))
    if not positions:
        log.info("No open short puts to monitor.")
        return 0

    no_quote = [p for p in positions if p.current_price is None]
    for p in no_quote:
        log.warning("%s %s $%.0fP: no live quote — cannot evaluate stop", p.symbol, p.expiry, p.strike)

    breaches = risk.positions_to_close(positions, multiplier, buffer_pct)
    log.info(
        "Monitoring %d short put(s): %d breach(es) of the %.1fx stop.",
        len(positions), len(breaches), multiplier,
    )
    if not breaches:
        return 0

    open_orders = []
    try:
        open_orders = api.get_orders(account_key)
    except Exception as e:
        log.warning("Could not fetch open orders for dedup: %s", e)

    placed = 0
    for pos, limit in breaches:
        if risk.has_pending_close(open_orders, pos):
            log.info("%s %s $%.0fP: close order already pending — skipping", pos.symbol, pos.expiry, pos.strike)
            continue
        log.warning(
            "STOP-LOSS: %s %s $%.0fP entry=$%.2f now=$%.2f → buy-to-close %d @ $%.2f limit",
            pos.symbol, pos.expiry, pos.strike, pos.entry_price, pos.current_price, pos.qty, limit,
        )
        order = ETrade.build_buy_to_close_order(
            symbol=pos.symbol, expiry=pos.expiry, strike=pos.strike,
            qty=pos.qty, limit_price=limit,
        )
        try:
            preview = api.preview_order(account_key, order)
            preview_id = preview["PreviewOrderResponse"]["PreviewIds"]["previewId"]
            result = api.place_order(account_key, order, preview_id)
            order_id = result["PlaceOrderResponse"]["OrderIds"]["orderId"]
            log.warning("STOP ORDER PLACED: %s ID=%s", pos.symbol, order_id)
            placed += 1
        except Exception as e:
            log.error("Failed to place stop order for %s: %s", pos.symbol, e)
    return placed


def cmd_monitor(api: ETrade, account_key: str, interval: int, once: bool) -> None:
    print(f"\n  Stop-loss monitor: {STRATEGY['exit']['stop_loss_multiplier']:.1f}x premium, "
          f"checking every {interval}s (Ctrl-C to stop)\n")
    while True:
        try:
            monitor_pass(api, account_key)
        except Exception as e:
            log.error("Monitor pass failed: %s", e)
        if once:
            return
        time.sleep(interval)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Options Algo — E*Trade short put scanner")
    parser.add_argument("--scan",    action="store_true", help="Scan only, no orders")
    parser.add_argument("--trade",   action="store_true", help="Scan and place limit orders")
    parser.add_argument("--status",  action="store_true", help="Show positions and open orders")
    parser.add_argument("--monitor", action="store_true", help="Enforce stop-loss on open short puts")
    parser.add_argument("--once",    action="store_true", help="With --monitor: single pass, then exit")
    parser.add_argument("--interval", type=int, default=120, help="Monitor poll seconds (default 120)")
    parser.add_argument("--sandbox", action="store_true", help="Use E*Trade sandbox environment")
    args = parser.parse_args()

    if not any([args.scan, args.trade, args.status, args.monitor]):
        parser.print_help()
        sys.exit(0)

    api = ETrade(sandbox=args.sandbox)
    api.authenticate()

    accounts = api.get_accounts()
    if not accounts:
        log.error("No accounts found.")
        sys.exit(1)

    # Use ETRADE_ACCOUNT_KEY env var or first account
    account_key = os.environ.get("ETRADE_ACCOUNT_KEY") or accounts[0]["accountIdKey"]
    log.info("Account: %s", account_key)

    if args.status:
        cmd_status(api, account_key)
    elif args.monitor:
        cmd_monitor(api, account_key, interval=args.interval, once=args.once)
    elif args.trade:
        cmd_trade(api, account_key)
    else:
        cmd_scan(api)


if __name__ == "__main__":
    main()
