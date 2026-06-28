"""
Tests for src/chain_backtester.py — chain-snapshot backtest core.

The simulator consumes EOD chain snapshots (list of put dicts with
expiration/strike/bid/ask/delta) from any data source. Conventions:
  - Entry requires DTE >= 1: EOD data cannot honestly simulate a 0-DTE
    entry (the position would open and expire inside the same snapshot).
  - Entry credit is the mid; exits are checked against the ask
    (cost-to-close), the conservative side.
  - select_entry returns the single best candidate per ticker-day.
"""
from __future__ import annotations

from datetime import date

from src.chain_backtester import check_exit, select_entry, settle_at_expiry

CFG = {
    "dte_max": 7,
    "premium_min": 30.0,
    "premium_max": 2000.0,
    "target_delta": -0.20,
    "otm_pct_fallback": 0.05,
}

TODAY = date(2026, 6, 8)          # Monday
FRIDAY = date(2026, 6, 12)


def _put(strike, bid, ask, delta=None, expiration=FRIDAY):
    return {"expiration": expiration, "strike": strike, "bid": bid, "ask": ask,
            "delta": delta}


# ── select_entry ──────────────────────────────────────────────────────────────

def test_picks_delta_closest_to_target():
    puts = [
        _put(90.0, 0.50, 0.60, delta=-0.12),
        _put(95.0, 1.00, 1.10, delta=-0.22),   # closest to -0.20
        _put(98.0, 1.80, 1.90, delta=-0.35),
    ]
    pick = select_entry(puts, underlying=100.0, cfg=CFG, today=TODAY)
    assert pick["strike"] == 95.0
    assert pick["premium"] == 105.0            # mid 1.05 × 100
    assert pick["dte"] == 4


def test_falls_back_to_otm_pct_when_deltas_missing():
    puts = [
        _put(90.0, 0.50, 0.60),
        _put(95.0, 1.00, 1.10),                # closest to 100 × 0.95
        _put(98.0, 1.80, 1.90),
    ]
    pick = select_entry(puts, underlying=100.0, cfg=CFG, today=TODAY)
    assert pick["strike"] == 95.0


def test_rejects_itm_zero_bid_premium_bounds_and_far_dte():
    puts = [
        _put(105.0, 2.00, 2.10, delta=-0.60),               # ITM
        _put(95.0, 0.0, 0.10, delta=-0.20),                 # no bid
        _put(94.0, 0.10, 0.15, delta=-0.19),                # premium $12.50 < $30
        _put(96.0, 1.00, 1.10, delta=-0.21,
             expiration=date(2026, 7, 17)),                 # 39 DTE > 7
    ]
    assert select_entry(puts, underlying=100.0, cfg=CFG, today=TODAY) is None


def test_rejects_same_day_expiry():
    puts = [_put(95.0, 1.00, 1.10, delta=-0.20, expiration=TODAY)]
    assert select_entry(puts, underlying=100.0, cfg=CFG, today=TODAY) is None


# ── check_exit ────────────────────────────────────────────────────────────────

EXIT_CFG = {"stop_loss_multiplier": 2.0, "profit_target_pct": None}


def test_stop_fires_when_ask_reaches_multiple_of_credit():
    assert check_exit(credit=1.00, ask=2.00, exit_cfg=EXIT_CFG) == "stop_loss"


def test_no_exit_between_thresholds():
    assert check_exit(credit=1.00, ask=1.50, exit_cfg=EXIT_CFG) is None


def test_profit_target_fires_when_enabled():
    cfg = {"stop_loss_multiplier": 2.0, "profit_target_pct": 0.5}
    assert check_exit(credit=1.00, ask=0.50, exit_cfg=cfg) == "profit_target"


def test_profit_target_disabled_by_default():
    assert check_exit(credit=1.00, ask=0.05, exit_cfg=EXIT_CFG) is None


def test_stop_takes_precedence_over_missing_quote():
    assert check_exit(credit=1.00, ask=None, exit_cfg=EXIT_CFG) is None


# ── settle_at_expiry ──────────────────────────────────────────────────────────

def test_otm_expiry_keeps_full_credit():
    assert settle_at_expiry(strike=95.0, underlying_close=100.0,
                            credit=1.00, qty=1) == 100.0


def test_itm_expiry_pays_intrinsic():
    # Credit $1.00, finishes $3 ITM → P&L = (1.00 − 3.00) × 100 = -$200
    assert settle_at_expiry(strike=95.0, underlying_close=92.0,
                            credit=1.00, qty=1) == -200.0


# ── simulate() ────────────────────────────────────────────────────────────────

from src.chain_backtester import simulate


class FakeSource:
    """In-memory ChainSource: chains[(symbol, day)] = [put dicts],
    prices[(symbol, day)] = underlying close."""

    def __init__(self, chains, prices, days):
        self.chains, self.prices, self.days = chains, prices, days

    def get_puts(self, symbol, day):
        return self.chains.get((symbol, day), [])

    def get_underlying(self, symbol, day):
        return self.prices.get((symbol, day))

    def trading_days(self, start, end):
        return [d for d in self.days if start <= d <= end]


MON, TUE, WED, THU, FRI = (date(2026, 6, d) for d in (8, 9, 10, 11, 12))

SIM_CFG = {
    "watchlist": ["AAPL"],
    "tickers_exclude": [],
    "qty": 1,
    "dte_max": 7,
    "premium_min": 30.0,
    "premium_max": 2000.0,
    "target_delta": -0.20,
    "otm_pct_fallback": 0.05,
    "exit": {"stop_loss_multiplier": 2.0, "profit_target_pct": None},
    "risk": {
        "max_open_positions": 10,
        "max_new_positions_per_day": 5,
        "max_notional_per_ticker": 50_000.0,
        "max_total_notional": 200_000.0,
        "max_bp_utilization": 1.0,
        "vix_max": 30.0,
    },
}


def _chain(bid, ask, delta=-0.20, strike=95.0, expiration=FRI):
    return [{"expiration": expiration, "strike": strike, "bid": bid, "ask": ask,
             "delta": delta}]


def _week_source(tue_ask=1.00, fri_close=100.0):
    """AAPL chain Mon–Thu; entry candidate appears Monday at mid 1.05."""
    chains = {
        ("AAPL", MON): _chain(1.00, 1.10),
        ("AAPL", TUE): _chain(tue_ask - 0.10, tue_ask),
        ("AAPL", WED): _chain(0.40, 0.50),
        ("AAPL", THU): _chain(0.20, 0.30),
    }
    prices = {("AAPL", d): 100.0 for d in (MON, TUE, WED, THU)}
    prices[("AAPL", FRI)] = fri_close
    return FakeSource(chains, prices, [MON, TUE, WED, THU, FRI])


def test_opens_and_expires_otm_for_full_credit():
    trades = simulate(_week_source(), SIM_CFG, MON, FRI)
    assert len(trades) == 1
    t = trades[0]
    assert t["symbol"] == "AAPL"
    assert t["open_date"] == MON
    assert t["exit_reason"] == "expired"
    assert t["pnl"] == 105.0          # mid 1.05 × 100, OTM at Friday close 100


def test_itm_expiry_settles_at_intrinsic():
    trades = simulate(_week_source(fri_close=92.0), SIM_CFG, MON, FRI)
    assert trades[0]["exit_reason"] == "expired"
    assert trades[0]["pnl"] == (1.05 - 3.0) * 100   # $3 ITM


def test_stop_loss_closes_position_midweek():
    trades = simulate(_week_source(tue_ask=2.50), SIM_CFG, MON, FRI)
    t = trades[0]
    assert t["exit_reason"] == "stop_loss"
    assert t["close_date"] == TUE
    assert t["pnl"] == round((1.05 - 2.50) * 100, 2)


def test_no_reentry_while_position_open():
    # Monday's entry stays open all week; Tue-Thu chains also qualify but
    # the same contract must not be doubled up.
    trades = simulate(_week_source(), SIM_CFG, MON, FRI)
    assert len(trades) == 1


def test_vix_gate_blocks_entry_days():
    vix = {d: 35.0 for d in (MON, TUE, WED, THU, FRI)}
    trades = simulate(_week_source(), SIM_CFG, MON, FRI, vix=vix)
    assert trades == []


def test_earnings_filter_blocks_symbol():
    trades = simulate(
        _week_source(), SIM_CFG, MON, FRI,
        earnings_blocked=lambda symbol, start, end: True,
    )
    assert trades == []


def test_max_open_positions_cap():
    chains = {
        ("AAPL", MON): _chain(1.00, 1.10),
        ("META", MON): _chain(1.00, 1.10),
    }
    prices = {("AAPL", d): 100.0 for d in (MON, FRI)}
    prices.update({("META", d): 100.0 for d in (MON, FRI)})
    source = FakeSource(chains, prices, [MON, FRI])
    cfg = {**SIM_CFG, "watchlist": ["AAPL", "META"],
           "risk": {**SIM_CFG["risk"], "max_open_positions": 1}}
    trades = simulate(source, cfg, MON, FRI)
    assert len(trades) == 1


# ── summarize() and AV row conversion ─────────────────────────────────────────

import pandas as pd

from src.chain_backtester import av_rows_to_puts, summarize


def test_summarize_core_metrics():
    trades = [
        {"pnl": 100.0, "close_date": date(2026, 1, 5)},
        {"pnl": 100.0, "close_date": date(2026, 1, 12)},
        {"pnl": -150.0, "close_date": date(2026, 1, 20)},
        {"pnl": 50.0, "close_date": date(2026, 2, 2)},
    ]
    m = summarize(trades)
    assert m["trades"] == 4
    assert m["win_rate"] == 0.75
    assert m["total_pnl"] == 100.0
    assert m["profit_factor"] == round(250.0 / 150.0, 2)
    assert m["max_drawdown"] == -150.0   # peak 200 → trough 50


def test_summarize_empty():
    assert summarize([])["trades"] == 0


def test_av_rows_to_puts_filters_and_converts():
    df = pd.DataFrame([
        {"type": "put", "expiration": "2026-06-12", "strike": 95.0,
         "bid": 1.0, "ask": 1.1, "delta": -0.20, "volume": 10, "open_interest": 500},
        {"type": "call", "expiration": "2026-06-12", "strike": 95.0,
         "bid": 2.0, "ask": 2.1, "delta": 0.55, "volume": 5, "open_interest": 100},
    ])
    puts = av_rows_to_puts(df)
    assert len(puts) == 1
    assert puts[0]["expiration"] == date(2026, 6, 12)
    assert puts[0]["delta"] == -0.20


def test_av_rows_to_puts_empty_frame():
    assert av_rows_to_puts(pd.DataFrame()) == []
