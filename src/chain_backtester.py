"""
Chain-snapshot backtest core — data-source agnostic.

Simulates the live strategy (scanner entry rules + risk engine + exit rules)
against historical EOD options chain snapshots. The data source is anything
that yields put dicts: {expiration: date, strike, bid, ask, delta|None}.

Honesty constraints (EOD granularity):
  - Entry requires DTE >= 1: a 0-DTE entry would open and expire inside the
    same EOD snapshot, which can't be simulated without intraday data.
  - Entry credit = mid; exit checks use the ask (cost-to-close), the
    conservative side of the spread.
  - Stops are evaluated once per day at the close. Real intraday breaches
    can be worse — EOD results are an upper bound on stop quality.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Callable, Protocol

from src import risk
from src.risk import stop_loss_triggered

log = logging.getLogger(__name__)


class ChainSource(Protocol):
    """Anything that can serve EOD chain snapshots and underlying closes."""

    def get_puts(self, symbol: str, day: date) -> list[dict]: ...
    def get_underlying(self, symbol: str, day: date) -> float | None: ...
    def trading_days(self, start: date, end: date) -> list[date]: ...


def select_entry(
    puts: list[dict],
    underlying: float,
    cfg: dict,
    today: date,
) -> dict | None:
    """
    Pick the single best short-put candidate from one ticker's EOD chain.

    Mirrors src/scanner.py rules: OTM only, positive bid, premium within
    bounds, DTE within [1, dte_max]. Selection prefers the delta closest to
    target_delta; falls back to the strike nearest underlying × (1 - otm_pct)
    when deltas are unavailable.
    """
    candidates = []
    for p in puts:
        dte = (p["expiration"] - today).days
        if not (1 <= dte <= cfg["dte_max"]):
            continue
        bid, ask = float(p.get("bid") or 0), float(p.get("ask") or 0)
        strike = float(p["strike"])
        if bid <= 0 or strike >= underlying:
            continue
        mid = round((bid + ask) / 2, 2)
        premium = round(mid * 100, 2)
        if not (cfg["premium_min"] <= premium <= cfg["premium_max"]):
            continue
        candidates.append({
            "expiry": p["expiration"],
            "dte": dte,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "premium": premium,
            "delta": p.get("delta"),
            "underlying_price": underlying,
        })

    if not candidates:
        return None

    if all(c["delta"] is not None for c in candidates):
        target = cfg["target_delta"]
        return min(candidates, key=lambda c: abs(float(c["delta"]) - target))

    target_strike = underlying * (1 - cfg["otm_pct_fallback"])
    return min(candidates, key=lambda c: abs(c["strike"] - target_strike))


def check_exit(credit: float, ask: float | None, exit_cfg: dict) -> str | None:
    """
    Exit decision for an open short put marked at today's ask.

    Returns "stop_loss", "profit_target", or None. Stop is checked first.
    A missing quote never exits — the position settles at expiry instead.
    """
    if ask is None:
        return None
    if stop_loss_triggered(credit, ask, exit_cfg["stop_loss_multiplier"]):
        return "stop_loss"
    pt = exit_cfg.get("profit_target_pct")
    if pt is not None and ask <= credit * (1 - pt):
        return "profit_target"
    return None


def settle_at_expiry(
    strike: float,
    underlying_close: float,
    credit: float,
    qty: int,
) -> float:
    """P&L in dollars of holding a short put to expiration."""
    intrinsic = max(0.0, strike - underlying_close)
    return round((credit - intrinsic) * 100.0 * qty, 2)


# ── Reporting ─────────────────────────────────────────────────────────────────

def summarize(trades: list[dict]) -> dict:
    """Win rate, P&L, profit factor, and max drawdown for a closed-trade list."""
    if not trades:
        return {"trades": 0, "win_rate": None, "total_pnl": 0.0,
                "avg_pnl": None, "profit_factor": None, "max_drawdown": 0.0}
    pnls = [t["pnl"] for t in sorted(trades, key=lambda t: t["close_date"])]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    cumulative = peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        peak = max(peak, cumulative)
        max_dd = min(max_dd, cumulative - peak)
    return {
        "trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls), 4),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(sum(pnls) / len(pnls), 2),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else None,
        "max_drawdown": round(max_dd, 2),
    }


def av_rows_to_puts(df) -> list[dict]:
    """Convert an AlphaVantage HISTORICAL_OPTIONS DataFrame to put dicts."""
    if df is None or df.empty or "type" not in df.columns:
        return []
    from datetime import datetime
    puts = []
    for _, row in df[df["type"] == "put"].iterrows():
        try:
            expiration = datetime.strptime(str(row["expiration"]), "%Y-%m-%d").date()
        except ValueError:
            continue
        puts.append({
            "expiration": expiration,
            "strike": float(row["strike"]),
            "bid": float(row.get("bid") or 0),
            "ask": float(row.get("ask") or 0),
            "delta": float(row["delta"]) if row.get("delta") is not None else None,
            "volume": int(row.get("volume") or 0),
            "open_interest": int(row.get("open_interest") or 0),
        })
    return puts


# ── Simulation ────────────────────────────────────────────────────────────────

def _find_quote(puts: list[dict], expiry: date, strike: float) -> float | None:
    """Today's ask for a specific held contract, if quoted."""
    for p in puts:
        if p["expiration"] == expiry and float(p["strike"]) == strike:
            ask = p.get("ask")
            return float(ask) if ask else None
    return None


def simulate(
    source: ChainSource,
    cfg: dict,
    start: date,
    end: date,
    *,
    vix: dict[date, float] | None = None,
    earnings_blocked: Callable[[str, date, date], bool] | None = None,
    capital: float = 100_000.0,
) -> list[dict]:
    """
    Replay the strategy day by day over historical chain snapshots.

    Exits are processed before entries each day. One position per symbol at
    a time. `vix=None` disables the gate (no data); when a vix dict is given,
    missing days fail closed, matching live behavior. Returns closed trades:
    {symbol, open_date, close_date, expiry, strike, qty, credit, premium,
     pnl, exit_reason, dte}.
    """
    exit_cfg = cfg["exit"]
    risk_cfg = cfg["risk"]
    exclude = set(cfg.get("tickers_exclude", []))
    watchlist = [t for t in cfg["watchlist"] if t not in exclude]

    open_positions: list[dict] = []
    trades: list[dict] = []
    realized = 0.0

    def _close(pos: dict, day: date, reason: str, pnl: float) -> None:
        nonlocal realized
        realized += pnl
        trades.append({
            "symbol": pos["symbol"],
            "open_date": pos["open_date"],
            "close_date": day,
            "expiry": pos["expiry"],
            "strike": pos["strike"],
            "qty": pos["qty"],
            "credit": pos["credit"],
            "premium": round(pos["credit"] * 100 * pos["qty"], 2),
            "pnl": pnl,
            "exit_reason": reason,
            "dte": (pos["expiry"] - pos["open_date"]).days,
        })

    for day in source.trading_days(start, end):
        # ── Exits first ──
        still_open: list[dict] = []
        for pos in open_positions:
            if day >= pos["expiry"]:
                close_px = source.get_underlying(pos["symbol"], pos["expiry"])
                if close_px is None:
                    close_px = source.get_underlying(pos["symbol"], day)
                if close_px is None:
                    log.warning("%s: no close for expiry %s — using entry underlying",
                                pos["symbol"], pos["expiry"])
                    close_px = pos["underlying_at_open"]
                pnl = settle_at_expiry(pos["strike"], close_px, pos["credit"], pos["qty"])
                _close(pos, pos["expiry"], "expired", pnl)
                continue
            ask = _find_quote(source.get_puts(pos["symbol"], day), pos["expiry"], pos["strike"])
            if ask is not None:
                pos["last_ask"] = ask
            reason = check_exit(pos["credit"], ask, exit_cfg)
            if reason:
                pnl = round((pos["credit"] - ask) * 100 * pos["qty"], 2)
                _close(pos, day, reason, pnl)
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── Entries ──
        if vix is not None:
            allowed, _ = risk.vix_allows_entry(vix.get(day), risk_cfg["vix_max"])
            if not allowed:
                continue

        held = {p["symbol"] for p in open_positions}
        signals = []
        for symbol in watchlist:
            if symbol in held:
                continue
            underlying = source.get_underlying(symbol, day)
            if underlying is None:
                continue
            pick = select_entry(source.get_puts(symbol, day), underlying, cfg, day)
            if pick is None:
                continue
            if earnings_blocked and earnings_blocked(symbol, day, pick["expiry"]):
                continue
            signals.append({**pick, "symbol": symbol, "qty": cfg["qty"]})

        if not signals:
            continue
        signals.sort(key=lambda s: s["premium"], reverse=True)

        open_as_risk = [
            risk.Position(p["symbol"], p["expiry"], p["strike"], p["qty"], p["credit"])
            for p in open_positions
        ]
        collateral_open = sum(
            p.notional - p.entry_price * 100 * p.qty for p in open_as_risk
        )
        buying_power = max(0.0, capital + realized - collateral_open)
        approved, _rejected = risk.filter_signals(signals, open_as_risk, risk_cfg, buying_power)

        for sig in approved:
            open_positions.append({
                "symbol": sig["symbol"],
                "expiry": sig["expiry"],
                "strike": sig["strike"],
                "qty": sig["qty"],
                "credit": sig["mid"],
                "open_date": day,
                "underlying_at_open": sig["underlying_price"],
                "last_ask": sig["ask"],
            })

    # Window ended with positions still open: mark at last seen ask.
    for pos in open_positions:
        pnl = round((pos["credit"] - pos["last_ask"]) * 100 * pos["qty"], 2)
        _close(pos, end, "open_at_end", pnl)

    return trades
