"""
Strategy configuration derived from historical trade analysis.

Key findings from 19,366 closed trades (Dec 2023 – Dec 2025):
  PUT_short   76.5% win rate  +$1.01M   ← primary strategy
  CALL_short  71.9% win rate  +$313K    ← secondary
  PUT_long    30.2% win rate  -$110K    avoid
  CALL_long   32.9% win rate  -$333K    avoid

  0-7 DTE     71.5% win rate  +$1.13M   ← the only bucket that makes money
  8-30 DTE    48.6% win rate  -$256K    avoid

  Let expire  91.7% win rate  +$2.1M    ← hold to expiry
  Buy to close 48.7% win rate -$1.28M   only for stop-losses
"""
from __future__ import annotations

STRATEGY: dict = {
    "name": "0-7 DTE Short Put",
    # ── Entry ─────────────────────────────────────────────────────────────────
    "opt_type": "PUT",
    "direction": "short",
    "dte_min": 0,
    "dte_max": 7,
    "premium_min": 30.0,    # minimum credit per contract (× 100 multiplier)
    "premium_max": 2000.0,  # cap to limit outsized risk
    "qty": 1,
    # ── Ticker universe ───────────────────────────────────────────────────────
    "watchlist": [
        # Top performers by total P&L (≥$10K, win rate ≥ 67%)
        "ENPH", "ADBE", "AAPL", "FSLR", "ALB", "TSLA", "ANET",
        "ABNB", "META", "INTU", "CRM", "CIEN", "EAT", "HD",
        "BHP", "BA", "AA", "NVDA", "CVX", "BMY", "DAL", "AMSC",
    ],
    "tickers_exclude": [
        # Biggest losers by total P&L
        "CONL", "ARM", "AMAT", "BTI", "CRUS", "CVNA", "ALK", "B",
    ],
    # ── Live trading: strike / contract selection ──────────────────────────────
    "target_delta": -0.20,       # ~20 delta OTM put (used if greeks available)
    "otm_pct_fallback": 0.05,    # 5% below spot if greeks unavailable
    "min_open_interest": 50,
    "min_volume": 5,
    # ── Exit ─────────────────────────────────────────────────────────────────
    "exit": {
        "let_expire": True,           # primary: hold to expiry
        "profit_target_pct": None,    # no early profit take
        "stop_loss_multiplier": 2.0,  # buy-to-close when cost-to-close ≥ 2× credit
    },
    # ── Portfolio-level risk limits (enforced by src/risk.py) ─────────────────
    "risk": {
        "max_open_positions": 10,         # short puts open at once, account-wide
        "max_new_positions_per_day": 5,   # per --trade run
        "max_notional_per_ticker": 30_000.0,   # strike × 100 × qty per ticker
        "max_total_notional": 100_000.0,       # total assignment exposure
        "max_bp_utilization": 0.5,        # fraction of buying power usable as collateral
        "vix_max": 30.0,                  # no new entries at/above this VIX (fail-closed)
        "stop_buffer_pct": 0.05,          # stop order limit = current price × 1.05
        "halt_file": "HALT",              # touch this file to block all new orders
    },
}
