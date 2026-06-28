"""
Risk engine — trade-time gates and position-time stop-loss rules.

Conventions:
  - Option prices are per-share (entry credit 1.00 = $100/contract premium).
  - Stop-loss triggers when cost-to-close >= multiplier × entry credit.
  - Notional = strike × 100 × qty: the assignment exposure of a short put.
  - All functions here are pure (no network, no broker calls) so they can be
    tested exhaustively and reused by both the live algo and backtests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class Position:
    """An open short put as the risk engine sees it."""
    symbol: str
    expiry: date
    strike: float
    qty: int                       # number of short contracts (positive)
    entry_price: float             # credit received per share
    current_price: float | None = None  # latest cost-to-close per share

    @property
    def notional(self) -> float:
        return notional(self.strike, self.qty)


# ── Portfolio parsing ─────────────────────────────────────────────────────────

def parse_short_puts(portfolio: list[dict]) -> list[Position]:
    """
    Extract short put positions from ETrade.get_portfolio() output.

    Only short puts are returned — they are the positions the stop-loss
    monitor manages. Long options, equities, and short calls are ignored.
    """
    positions: list[Position] = []
    for account in portfolio:
        for pos in account.get("Position", []):
            product = pos.get("Product", {})
            qty = float(pos.get("quantity", 0))
            if (
                product.get("securityType") != "OPTN"
                or product.get("callPut") != "PUT"
                or qty >= 0
            ):
                continue
            try:
                expiry = date(
                    int(product["expiryYear"]),
                    int(product["expiryMonth"]),
                    int(product["expiryDay"]),
                )
            except (KeyError, ValueError):
                continue
            quick = pos.get("Quick") or {}
            last_trade = quick.get("lastTrade")
            positions.append(Position(
                symbol=product.get("symbol", "?"),
                expiry=expiry,
                strike=float(product.get("strikePrice", 0)),
                qty=int(-qty),
                entry_price=float(pos.get("pricePaid", 0)),
                current_price=float(last_trade) if last_trade is not None else None,
            ))
    return positions


# ── Stop-loss ─────────────────────────────────────────────────────────────────

def stop_loss_triggered(
    entry_price: float | None,
    current_price: float | None,
    multiplier: float,
) -> bool:
    """True when the cost to close has reached multiplier × the credit received."""
    if not entry_price or entry_price <= 0:
        return False
    if current_price is None:
        return False
    return current_price >= entry_price * multiplier


def stop_loss_limit_price(current_price: float, buffer_pct: float = 0.05) -> float:
    """Limit price for the closing order: current price plus a slippage buffer."""
    return round(current_price * (1 + buffer_pct), 2)


# ── Monitor decisions ─────────────────────────────────────────────────────────

def positions_to_close(
    positions: list[Position],
    multiplier: float,
    buffer_pct: float = 0.05,
) -> list[tuple[Position, float]]:
    """Return (position, limit_price) for every position breaching the stop."""
    out: list[tuple[Position, float]] = []
    for p in positions:
        if stop_loss_triggered(p.entry_price, p.current_price, multiplier):
            out.append((p, stop_loss_limit_price(p.current_price, buffer_pct)))
    return out


def has_pending_close(orders: list[dict], position: Position) -> bool:
    """True if an open BUY_TO_CLOSE order already exists for this contract."""
    for order in orders:
        for detail in order.get("OrderDetail", []):
            for inst in detail.get("Instrument", []):
                if inst.get("orderAction") != "BUY_TO_CLOSE":
                    continue
                prod = inst.get("Product", {})
                if (
                    prod.get("symbol") == position.symbol
                    and prod.get("callPut") == "PUT"
                    and float(prod.get("strikePrice", -1)) == position.strike
                    and (
                        int(prod.get("expiryYear", 0)),
                        int(prod.get("expiryMonth", 0)),
                        int(prod.get("expiryDay", 0)),
                    ) == (position.expiry.year, position.expiry.month, position.expiry.day)
                ):
                    return True
    return False


# ── Notional ──────────────────────────────────────────────────────────────────

def notional(strike: float, qty: int) -> float:
    return strike * 100.0 * qty


# ── Trade-time signal filtering ───────────────────────────────────────────────

def filter_signals(
    signals: list[dict],
    open_positions: list[Position],
    risk_cfg: dict,
    buying_power: float,
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """
    Approve signals (in priority order) against portfolio-level limits.

    Returns (approved, rejected) where rejected pairs each signal with the
    name of the first limit it breached. Approved signals consume budget,
    so later signals are evaluated against the post-approval portfolio.
    """
    approved: list[dict] = []
    rejected: list[tuple[dict, str]] = []

    open_count = len(open_positions)
    total_notional = sum(p.notional for p in open_positions)
    ticker_notional: dict[str, float] = {}
    for p in open_positions:
        ticker_notional[p.symbol] = ticker_notional.get(p.symbol, 0.0) + p.notional

    bp_budget = buying_power * risk_cfg.get("max_bp_utilization", 0.5)
    collateral_used = 0.0

    for sig in signals:
        qty = int(sig.get("qty", 1))
        sig_notional = notional(sig["strike"], qty)
        sig_collateral = sig_notional - sig.get("premium", 0.0) * qty

        if open_count + len(approved) >= risk_cfg["max_open_positions"]:
            rejected.append((sig, "max_open_positions"))
            continue
        if len(approved) >= risk_cfg["max_new_positions_per_day"]:
            rejected.append((sig, "max_new_positions_per_day"))
            continue
        if (
            ticker_notional.get(sig["symbol"], 0.0) + sig_notional
            > risk_cfg["max_notional_per_ticker"]
        ):
            rejected.append((sig, "max_notional_per_ticker"))
            continue
        if total_notional + sig_notional > risk_cfg["max_total_notional"]:
            rejected.append((sig, "max_total_notional"))
            continue
        if collateral_used + sig_collateral > bp_budget:
            rejected.append((sig, "insufficient_buying_power"))
            continue

        approved.append(sig)
        total_notional += sig_notional
        ticker_notional[sig["symbol"]] = (
            ticker_notional.get(sig["symbol"], 0.0) + sig_notional
        )
        collateral_used += sig_collateral

    return approved, rejected


# ── Buying power ──────────────────────────────────────────────────────────────

def extract_buying_power(balance: dict) -> float:
    """
    Cash buying power from ETrade.get_balance() output.

    Uses cashBuyingPower (not margin) because the strategy treats short puts
    as cash-secured. Fail-closed: anything unparseable returns 0.0, which
    blocks all new entries rather than guessing at available collateral.
    """
    computed = balance.get("BalanceResponse", {}).get("Computed", {})
    try:
        return float(computed["cashBuyingPower"])
    except (KeyError, TypeError, ValueError):
        return 0.0


# ── VIX circuit breaker ───────────────────────────────────────────────────────

def vix_allows_entry(vix: float | None, vix_max: float) -> tuple[bool, str]:
    """Fail-closed: if VIX can't be fetched, no new entries."""
    if vix is None:
        return False, "VIX unavailable — failing closed, no new entries"
    if vix >= vix_max:
        return False, f"VIX {vix:.1f} >= limit {vix_max:.1f} — no new entries"
    return True, f"VIX {vix:.1f} below limit {vix_max:.1f}"


# ── Kill switch ───────────────────────────────────────────────────────────────

def halt_active(halt_file: str | Path) -> bool:
    """True when the kill-switch file exists; create it to stop all new orders."""
    return Path(halt_file).exists()
