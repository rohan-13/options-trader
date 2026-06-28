"""
Pattern mining on closed trades: surfaces what actually drove winners vs losers.
All functions accept a closed-trades DataFrame and return tidy DataFrames
ready for Plotly charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _stats(df: pd.DataFrame) -> dict:
    wins = df[df["is_win"]]
    losses = df[~df["is_win"]]
    return {
        "trades": len(df),
        "win_rate": len(wins) / len(df) if len(df) else 0,
        "total_pnl": df["net_pnl"].sum(),
        "avg_pnl": df["net_pnl"].mean() if len(df) else 0,
        "avg_win": wins["net_pnl"].mean() if len(wins) else 0,
        "avg_loss": losses["net_pnl"].mean() if len(losses) else 0,
    }


def by_day_of_week(closed: pd.DataFrame) -> pd.DataFrame:
    closed = closed.copy()
    closed["day"] = pd.to_datetime(closed["open_date"]).dt.day_name()
    rows = []
    for day in _DAY_ORDER:
        grp = closed[closed["day"] == day]
        if len(grp) == 0:
            continue
        r = _stats(grp)
        r["day"] = day
        rows.append(r)
    return pd.DataFrame(rows)


def by_month_of_year(closed: pd.DataFrame) -> pd.DataFrame:
    closed = closed.copy()
    closed["month_num"] = pd.to_datetime(closed["open_date"]).dt.month
    rows = []
    for m in range(1, 13):
        grp = closed[closed["month_num"] == m]
        if len(grp) == 0:
            continue
        r = _stats(grp)
        r["month_num"] = m
        r["month"] = _MONTH_NAMES[m]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("month_num")


def by_premium_bucket(closed: pd.DataFrame, strategy_filter: str | None = None) -> pd.DataFrame:
    """Win rate vs size of premium collected (short) or paid (long)."""
    df = closed.copy()
    if strategy_filter:
        df = df[df["strategy"] == strategy_filter]
    df = df[df["premium"] > 0]

    bins = [0, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000, np.inf]
    labels = ["<$50", "$50-100", "$100-200", "$200-300", "$300-500",
              "$500-750", "$750-1k", "$1k-2k", "$2k-5k", "$5k+"]
    df["premium_bucket"] = pd.cut(df["premium"], bins=bins, labels=labels)

    rows = []
    for bucket in labels:
        grp = df[df["premium_bucket"] == bucket]
        if len(grp) < 3:
            continue
        r = _stats(grp)
        r["bucket"] = bucket
        rows.append(r)
    return pd.DataFrame(rows)


def by_holding_period(closed: pd.DataFrame) -> pd.DataFrame:
    """P&L and win rate bucketed by how many days the trade was held."""
    df = closed[closed["holding_days"].notna()].copy()
    bins = [-1, 0, 1, 3, 7, 14, 30, 60, 365]
    labels = ["0d (same-day)", "1d", "2-3d", "4-7d", "8-14d", "15-30d", "31-60d", "60d+"]
    df["hold_bucket"] = pd.cut(df["holding_days"], bins=bins, labels=labels)

    rows = []
    for b in labels:
        grp = df[df["hold_bucket"] == b]
        if len(grp) == 0:
            continue
        r = _stats(grp)
        r["hold_bucket"] = b
        rows.append(r)
    return pd.DataFrame(rows)


def ticker_heatmap(closed: pd.DataFrame, min_trades: int = 10) -> pd.DataFrame:
    """Per-ticker stats, filtered to tickers with at least min_trades."""
    rows = []
    for ticker, grp in closed.groupby("ticker"):
        if len(grp) < min_trades:
            continue
        r = _stats(grp)
        r["ticker"] = ticker
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    return df


def pnl_distribution(closed: pd.DataFrame) -> pd.DataFrame:
    """Raw P&L values for histogram, with strategy label."""
    return closed[["net_pnl", "strategy", "dte_bucket", "is_win"]].copy()


def streak_analysis(closed: pd.DataFrame) -> dict:
    """Longest win and loss streaks."""
    results = closed.sort_values("open_date")["is_win"].tolist()
    max_win = max_loss = cur_win = cur_loss = 0
    for w in results:
        if w:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return {"max_win_streak": max_win, "max_loss_streak": max_loss}


def cumulative_pnl_series(closed: pd.DataFrame) -> pd.DataFrame:
    """Daily cumulative P&L for equity curve."""
    df = closed.sort_values("open_date").copy()
    df["date"] = pd.to_datetime(df["open_date"]).dt.date
    daily = df.groupby("date")["net_pnl"].sum().reset_index()
    daily["cumulative_pnl"] = daily["net_pnl"].cumsum()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def backtest_filter(
    trades: pd.DataFrame,
    strategies: list[str] | None = None,
    dte_min: int = 0,
    dte_max: int = 365,
    tickers_include: list[str] | None = None,
    tickers_exclude: list[str] | None = None,
    premium_min: float = 0,
    premium_max: float = 999_999,
    date_start: str | None = None,
    date_end: str | None = None,
    accounts: list[str] | None = None,
) -> pd.DataFrame:
    """Filter trades for backtesting a custom strategy rule set."""
    df = trades[trades["is_closed"]].copy()

    if strategies:
        df = df[df["strategy"].isin(strategies)]
    if accounts:
        df = df[df["account"].isin(accounts)]

    df = df[
        (df["dte_at_open"].fillna(0) >= dte_min) &
        (df["dte_at_open"].fillna(0) <= dte_max)
    ]

    if tickers_include:
        df = df[df["ticker"].isin(tickers_include)]
    if tickers_exclude:
        df = df[~df["ticker"].isin(tickers_exclude)]

    df = df[
        (df["premium"].fillna(0) >= premium_min) &
        (df["premium"].fillna(0) <= premium_max)
    ]

    if date_start:
        df = df[pd.to_datetime(df["open_date"]) >= pd.to_datetime(date_start)]
    if date_end:
        df = df[pd.to_datetime(df["open_date"]) <= pd.to_datetime(date_end)]

    return df.sort_values("open_date").reset_index(drop=True)


def backtest_metrics(filtered: pd.DataFrame) -> dict:
    """Compute backtest summary metrics from a filtered trades DataFrame."""
    if filtered.empty:
        return {}

    wins = filtered[filtered["is_win"]]
    losses = filtered[~filtered["is_win"]]

    # Equity curve for drawdown
    eq = filtered.sort_values("open_date")["net_pnl"].cumsum()
    rolling_max = eq.cummax()
    drawdown = eq - rolling_max
    max_dd = drawdown.min()

    # Sharpe (annualized, assuming ~252 trading days)
    daily = filtered.groupby(pd.to_datetime(filtered["open_date"]).dt.date)["net_pnl"].sum()
    sharpe = (daily.mean() / daily.std() * (252 ** 0.5)) if daily.std() > 0 else 0

    pf = (
        abs(wins["net_pnl"].sum() / losses["net_pnl"].sum())
        if len(losses) and losses["net_pnl"].sum() != 0
        else None
    )

    return {
        "trades": len(filtered),
        "win_rate": len(wins) / len(filtered) if filtered.shape[0] else 0,
        "total_pnl": filtered["net_pnl"].sum(),
        "avg_pnl": filtered["net_pnl"].mean(),
        "avg_win": wins["net_pnl"].mean() if len(wins) else 0,
        "avg_loss": losses["net_pnl"].mean() if len(losses) else 0,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "best_trade": filtered["net_pnl"].max(),
        "worst_trade": filtered["net_pnl"].min(),
    }
