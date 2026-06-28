"""
Historical backtest engine.

Applies the derived strategy rules to the closed-trades DataFrame and reports
how strict adherence would have performed vs the full trade history.
"""
from __future__ import annotations

import pandas as pd

from src.patterns import backtest_filter, backtest_metrics
from src.strategy import STRATEGY


def apply_strategy(trades: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Return the subset of closed trades that match the strategy rules."""
    cfg = cfg or STRATEGY
    return backtest_filter(
        trades,
        strategies=[f"{cfg['opt_type']}_{cfg['direction']}"],
        dte_min=cfg["dte_min"],
        dte_max=cfg["dte_max"],
        tickers_include=cfg.get("watchlist") or None,
        tickers_exclude=cfg.get("tickers_exclude"),
        premium_min=cfg["premium_min"],
        premium_max=cfg["premium_max"],
    )


def run_backtest(trades: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Run backtest and return metrics for both all-trades and strategy."""
    cfg = cfg or STRATEGY
    closed = trades[trades["is_closed"]].copy()
    strategy_trades = apply_strategy(trades, cfg)
    return {
        "config": cfg,
        "all_metrics": backtest_metrics(closed),
        "strategy_metrics": backtest_metrics(strategy_trades),
        "all_trades": closed,
        "strategy_trades": strategy_trades,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _sign(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "N/A"


def _val(v, fmt: str = ".2f") -> str:
    return f"{v:{fmt}}" if v is not None else "N/A"


def print_backtest_report(result: dict) -> None:
    cfg = result["config"]
    a = result["all_metrics"]
    s = result["strategy_metrics"]
    st = result["strategy_trades"]

    print(f"\n{'=' * 65}")
    print(f"  STRATEGY BACKTEST: {cfg['name']}")
    print(f"{'=' * 65}")
    print(f"  Rules:")
    print(f"    Type       : {cfg['opt_type']}_{cfg['direction']}")
    print(f"    DTE        : {cfg['dte_min']}–{cfg['dte_max']} days")
    print(f"    Premium    : ${cfg['premium_min']:.0f} – ${cfg['premium_max']:.0f}")
    print(f"    Watchlist  : {len(cfg.get('watchlist', []))} tickers")
    print(f"    Exclude    : {', '.join(cfg.get('tickers_exclude', []))}")

    print(f"\n{'─' * 65}")
    print(f"  {'Metric':<30} {'All Trades':>15} {'Strategy':>15}")
    print(f"{'─' * 65}")

    rows = [
        ("Trades",        f"{a.get('trades', 0):,}",              f"{s.get('trades', 0):,}"),
        ("Win rate",      _pct(a.get("win_rate")),                 _pct(s.get("win_rate"))),
        ("Total P&L",     _sign(a.get("total_pnl", 0)),            _sign(s.get("total_pnl", 0))),
        ("Avg P&L/trade", _sign(a.get("avg_pnl", 0)),              _sign(s.get("avg_pnl", 0))),
        ("Avg win",       _sign(a.get("avg_win", 0)),              _sign(s.get("avg_win", 0))),
        ("Avg loss",      _sign(a.get("avg_loss", 0)),             _sign(s.get("avg_loss", 0))),
        ("Profit factor", _val(a.get("profit_factor"), ".2f"),     _val(s.get("profit_factor"), ".2f")),
        ("Max drawdown",  _sign(a.get("max_drawdown", 0)),         _sign(s.get("max_drawdown", 0))),
        ("Sharpe ratio",  _val(a.get("sharpe", 0), ".2f"),         _val(s.get("sharpe", 0), ".2f")),
        ("Best trade",    _sign(a.get("best_trade", 0)),           _sign(s.get("best_trade", 0))),
        ("Worst trade",   _sign(a.get("worst_trade", 0)),          _sign(s.get("worst_trade", 0))),
    ]

    for label, all_val, strat_val in rows:
        print(f"  {label:<30} {all_val:>15} {strat_val:>15}")

    print(f"{'─' * 65}")

    # Monthly P&L breakdown for strategy trades
    if not st.empty:
        print(f"\n  Monthly P&L (strategy trades only):")
        monthly = (
            st.assign(month=pd.to_datetime(st["open_date"]).dt.to_period("M"))
            .groupby("month")
            .agg(trades=("net_pnl", "count"), pnl=("net_pnl", "sum"), wins=("is_win", "sum"))
            .reset_index()
        )
        monthly["win_rate"] = monthly["wins"] / monthly["trades"]
        monthly["cumulative"] = monthly["pnl"].cumsum()

        print(f"\n  {'Month':<10} {'Trades':>7} {'Win%':>7} {'P&L':>12} {'Cumulative':>12}")
        print(f"  {'─' * 52}")
        for _, row in monthly.iterrows():
            pnl_str = f"+${row['pnl']:,.0f}" if row["pnl"] >= 0 else f"-${abs(row['pnl']):,.0f}"
            cum_str = f"+${row['cumulative']:,.0f}" if row["cumulative"] >= 0 else f"-${abs(row['cumulative']):,.0f}"
            print(f"  {str(row['month']):<10} {int(row['trades']):>7} "
                  f"{row['win_rate']*100:>6.0f}% {pnl_str:>12} {cum_str:>12}")

    n_all = a.get("trades", 1)
    n_strat = s.get("trades", 0)
    print(f"\n  Strategy selected {n_strat:,} of {n_all:,} total trades "
          f"({n_strat / n_all * 100:.1f}%)")
    print(f"{'=' * 65}\n")
