"""
Tests for src/risk.py — the trade-time and position-time risk engine.

Conventions under test:
  - Option prices are per-share (entry credit 1.00 = $100/contract premium).
  - Stop-loss triggers when cost-to-close >= multiplier × entry credit
    (multiplier 2.0, credit $1.00 → stop when option trades at $2.00+).
  - Notional = strike × 100 × qty (assignment exposure of a short put).
"""
from __future__ import annotations

from datetime import date

import pytest

from src.risk import (
    Position,
    filter_signals,
    halt_active,
    notional,
    stop_loss_limit_price,
    stop_loss_triggered,
    vix_allows_entry,
)


# ── Stop-loss ─────────────────────────────────────────────────────────────────

def test_stop_loss_triggers_at_multiplier():
    assert stop_loss_triggered(entry_price=1.00, current_price=2.00, multiplier=2.0)


def test_stop_loss_triggers_above_multiplier():
    assert stop_loss_triggered(entry_price=1.00, current_price=3.50, multiplier=2.0)


def test_stop_loss_not_triggered_below_multiplier():
    assert not stop_loss_triggered(entry_price=1.00, current_price=1.99, multiplier=2.0)


def test_stop_loss_never_triggers_without_entry_price():
    # Defensive: a missing/zero entry credit must not fire a panic close.
    assert not stop_loss_triggered(entry_price=0.0, current_price=5.00, multiplier=2.0)


def test_stop_loss_never_triggers_without_current_price():
    assert not stop_loss_triggered(entry_price=1.00, current_price=None, multiplier=2.0)


def test_stop_limit_price_adds_slippage_buffer():
    # Limit must sit above the current price so the closing order actually fills.
    assert stop_loss_limit_price(2.00, buffer_pct=0.05) == 2.10


def test_stop_limit_price_rounds_to_cents():
    assert stop_loss_limit_price(2.33, buffer_pct=0.05) == 2.45


# ── Notional ──────────────────────────────────────────────────────────────────

def test_notional_is_assignment_exposure():
    assert notional(strike=150.0, qty=2) == 30_000.0


# ── Trade-time signal filtering ───────────────────────────────────────────────

RISK_CFG = {
    "max_open_positions": 4,
    "max_new_positions_per_day": 3,
    "max_notional_per_ticker": 30_000.0,
    "max_total_notional": 60_000.0,
    "max_bp_utilization": 0.5,
}


def _sig(symbol="AAPL", strike=100.0, premium=100.0, qty=1):
    return {
        "symbol": symbol,
        "expiry": date(2026, 6, 12),
        "strike": strike,
        "premium": premium,
        "mid": premium / 100.0,
        "qty": qty,
    }


def _pos(symbol="TSLA", strike=100.0, qty=1, entry=1.0):
    return Position(
        symbol=symbol,
        expiry=date(2026, 6, 12),
        strike=strike,
        qty=qty,
        entry_price=entry,
    )


def test_filter_approves_signal_within_all_limits():
    approved, rejected = filter_signals(
        [_sig()], open_positions=[], risk_cfg=RISK_CFG, buying_power=1_000_000
    )
    assert len(approved) == 1
    assert rejected == []


def test_filter_rejects_when_max_open_positions_reached():
    open_pos = [_pos(symbol=s, strike=10.0) for s in ("A", "B", "C", "D")]
    approved, rejected = filter_signals(
        [_sig()], open_positions=open_pos, risk_cfg=RISK_CFG, buying_power=1_000_000
    )
    assert approved == []
    assert rejected[0][1] == "max_open_positions"


def test_filter_counts_approvals_toward_position_cap():
    # 3 already open + 2 candidates: only 1 more fits under max_open_positions=4.
    open_pos = [_pos(symbol=s, strike=10.0) for s in ("A", "B", "C")]
    approved, rejected = filter_signals(
        [_sig("AAPL"), _sig("META")],
        open_positions=open_pos,
        risk_cfg=RISK_CFG,
        buying_power=1_000_000,
    )
    assert len(approved) == 1
    assert rejected[0][1] == "max_open_positions"


def test_filter_caps_new_positions_per_day():
    sigs = [_sig(s, strike=10.0) for s in ("A", "B", "C", "D")]
    approved, rejected = filter_signals(
        sigs, open_positions=[], risk_cfg=RISK_CFG, buying_power=1_000_000
    )
    assert len(approved) == 3
    assert rejected[0][1] == "max_new_positions_per_day"


def test_filter_enforces_per_ticker_notional_including_open_positions():
    # $20K already open on AAPL; a 150-strike put adds $15K → breaches $30K cap.
    open_pos = [_pos(symbol="AAPL", strike=200.0)]  # $20K notional
    approved, rejected = filter_signals(
        [_sig("AAPL", strike=150.0)],
        open_positions=open_pos,
        risk_cfg=RISK_CFG,
        buying_power=1_000_000,
    )
    assert approved == []
    assert rejected[0][1] == "max_notional_per_ticker"


def test_filter_enforces_total_notional():
    # Two $25K signals fit under $60K total; the third breaches it.
    sigs = [_sig(s, strike=250.0) for s in ("A", "B", "C")]
    approved, rejected = filter_signals(
        sigs, open_positions=[], risk_cfg=RISK_CFG, buying_power=1_000_000
    )
    assert [s["symbol"] for s in approved] == ["A", "B"]
    assert rejected[0][1] == "max_total_notional"


def test_filter_enforces_buying_power():
    # Collateral for a cash-secured 100-strike put ≈ $10K - premium.
    # buying_power=10_000 × 0.5 utilization → only $5K available → reject.
    approved, rejected = filter_signals(
        [_sig("AAPL", strike=100.0)],
        open_positions=[],
        risk_cfg=RISK_CFG,
        buying_power=10_000,
    )
    assert approved == []
    assert rejected[0][1] == "insufficient_buying_power"


# ── VIX circuit breaker ───────────────────────────────────────────────────────

def test_vix_allows_entry_below_threshold():
    allowed, _ = vix_allows_entry(18.5, vix_max=30.0)
    assert allowed


def test_vix_blocks_entry_at_or_above_threshold():
    allowed, reason = vix_allows_entry(30.0, vix_max=30.0)
    assert not allowed
    assert "30" in reason


def test_vix_fails_closed_when_unavailable():
    allowed, reason = vix_allows_entry(None, vix_max=30.0)
    assert not allowed
    assert "unavailable" in reason.lower()


# ── Kill switch ───────────────────────────────────────────────────────────────

def test_halt_active_when_file_exists(tmp_path):
    halt = tmp_path / "HALT"
    halt.touch()
    assert halt_active(halt)


def test_halt_inactive_when_file_missing(tmp_path):
    assert not halt_active(tmp_path / "HALT")


# ── Monitor decisions ─────────────────────────────────────────────────────────

from src.risk import has_pending_close, positions_to_close


def test_positions_to_close_returns_triggered_with_limit_price():
    triggered = _pos(symbol="AAPL", entry=1.0)
    triggered.current_price = 2.10
    safe = _pos(symbol="META", entry=1.0)
    safe.current_price = 1.20
    to_close = positions_to_close([triggered, safe], multiplier=2.0, buffer_pct=0.05)
    assert len(to_close) == 1
    pos, limit = to_close[0]
    assert pos.symbol == "AAPL"
    assert limit == 2.21  # 2.10 × 1.05, rounded to cents


def test_positions_to_close_skips_positions_without_quotes():
    p = _pos(entry=1.0)
    p.current_price = None
    assert positions_to_close([p], multiplier=2.0, buffer_pct=0.05) == []


def _open_order(symbol="AAPL", action="BUY_TO_CLOSE", strike=100.0,
                call_put="PUT", status="OPEN"):
    return {
        "orderId": 777,
        "OrderDetail": [{
            "status": status,
            "Instrument": [{
                "orderAction": action,
                "Product": {
                    "symbol": symbol,
                    "callPut": call_put,
                    "securityType": "OPTN",
                    "expiryYear": 2026,
                    "expiryMonth": 6,
                    "expiryDay": 12,
                    "strikePrice": strike,
                },
            }],
        }],
    }


def test_has_pending_close_matches_same_contract():
    pos = _pos(symbol="AAPL", strike=100.0)
    assert has_pending_close([_open_order()], pos)


def test_has_pending_close_ignores_other_contracts_and_actions():
    pos = _pos(symbol="AAPL", strike=100.0)
    assert not has_pending_close([_open_order(symbol="META")], pos)
    assert not has_pending_close([_open_order(strike=95.0)], pos)
    assert not has_pending_close([_open_order(action="SELL_OPEN")], pos)


# ── Buying power extraction ───────────────────────────────────────────────────

from src.risk import extract_buying_power


def test_extract_buying_power_prefers_cash_buying_power():
    balance = {"BalanceResponse": {"Computed": {
        "cashBuyingPower": 25_000.0, "marginBuyingPower": 50_000.0,
    }}}
    assert extract_buying_power(balance) == 25_000.0


def test_extract_buying_power_zero_when_missing():
    # Fail-closed: unparseable balance means no collateral budget.
    assert extract_buying_power({}) == 0.0
    assert extract_buying_power({"BalanceResponse": {}}) == 0.0
