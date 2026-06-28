"""
Option chain scanner: finds short put opportunities matching the strategy.
Earnings filter: skips any expiry that falls on or after an upcoming earnings date
so the position is never held through an earnings announcement.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from src.earnings import straddles_earnings
from src.etrade_api import ETrade
from src.strategy import STRATEGY

log = logging.getLogger(__name__)


def _target_expiries(api: ETrade, symbol: str, dte_max: int) -> list[date]:
    today = date.today()
    cutoff = today + timedelta(days=dte_max)
    try:
        return [d for d in api.get_expiry_dates(symbol) if today <= d <= cutoff]
    except Exception as e:
        log.warning("%s: could not fetch expiry dates — %s", symbol, e)
        return []


def scan_ticker(api: ETrade, symbol: str, cfg: dict | None = None) -> list[dict]:
    """Return qualifying short put opportunities for one ticker."""
    cfg = cfg or STRATEGY
    signals: list[dict] = []

    try:
        quotes = api.get_quote([symbol])
        q = quotes.get(symbol, {})
        last = q.get("All", {}).get("lastTrade") or q.get("All", {}).get("ask")
        if not last:
            log.warning("%s: no price data", symbol)
            return []
        last = float(last)

        expiries = _target_expiries(api, symbol, cfg["dte_max"])
        if not expiries:
            log.debug("%s: no expiries within %d DTE", symbol, cfg["dte_max"])
            return []

        target_strike = last * (1 - cfg.get("otm_pct_fallback", 0.05))

        for expiry in expiries:
            dte = (expiry - date.today()).days
            # Skip any expiry that lands on or after an upcoming earnings date —
            # a single earnings surprise can wipe out multiple wins.
            if straddles_earnings(symbol, date.today(), expiry):
                log.info("%s: skipping %s expiry — earnings window detected", symbol, expiry)
                continue
            try:
                options = api.get_option_chain(
                    symbol=symbol,
                    expiry=expiry,
                    chain_type="PUT",
                    strikes_near=target_strike,
                    n_strikes=8,
                )
            except Exception as e:
                log.warning("%s %s: chain fetch failed — %s", symbol, expiry, e)
                continue

            for opt in options:
                bid    = float(opt.get("bid", 0))
                ask    = float(opt.get("ask", 0))
                oi     = int(opt.get("openInterest", 0))
                vol    = int(opt.get("volume", 0))
                strike = float(opt.get("strikePrice", 0))
                mid    = round((bid + ask) / 2, 2)
                premium = round(mid * 100, 2)  # per-contract dollar value

                if oi < cfg.get("min_open_interest", 50):
                    continue
                if vol < cfg.get("min_volume", 5):
                    continue
                if not (cfg["premium_min"] <= premium <= cfg["premium_max"]):
                    continue
                if strike >= last:  # must be OTM
                    continue
                if bid <= 0:
                    continue

                signals.append({
                    "symbol": symbol,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "premium": premium,
                    "open_interest": oi,
                    "volume": vol,
                    "underlying_price": last,
                    "otm_pct": round((last - strike) / last * 100, 1),
                })

    except Exception as e:
        log.error("%s: scan error — %s", symbol, e)

    return signals


def scan_watchlist(api: ETrade, cfg: dict | None = None) -> list[dict]:
    """Scan all watchlist tickers; return opportunities sorted by premium (desc)."""
    cfg = cfg or STRATEGY
    exclude = set(cfg.get("tickers_exclude", []))
    watchlist = [t for t in cfg.get("watchlist", []) if t not in exclude]

    all_signals: list[dict] = []
    for symbol in watchlist:
        log.info("Scanning %s ...", symbol)
        all_signals.extend(scan_ticker(api, symbol, cfg))

    all_signals.sort(key=lambda x: x["premium"], reverse=True)
    return all_signals
