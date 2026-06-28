"""
Generate summary analytics from the matched trades DataFrame.
Saves CSVs to output/ and prints a console report.
"""

from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _pct(n: float) -> str:
    return f"{n * 100:.1f}%"


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _fmt_pnl(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"


# ---------------------------------------------------------------------------
# Core aggregations
# ---------------------------------------------------------------------------

def _summary_row(df: pd.DataFrame, label: str = "ALL") -> dict:
    closed = df[df["is_closed"]]
    wins = closed[closed["is_win"]]
    losses = closed[~closed["is_win"]]
    return {
        "label": label,
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) if len(closed) else None,
        "total_pnl": closed["net_pnl"].sum(),
        "avg_pnl": closed["net_pnl"].mean() if len(closed) else None,
        "avg_win": wins["net_pnl"].mean() if len(wins) else None,
        "avg_loss": losses["net_pnl"].mean() if len(losses) else None,
        "best_trade": closed["net_pnl"].max() if len(closed) else None,
        "worst_trade": closed["net_pnl"].min() if len(closed) else None,
        "profit_factor": abs(wins["net_pnl"].sum() / losses["net_pnl"].sum())
        if len(losses) and losses["net_pnl"].sum() != 0
        else None,
    }


def _breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for val, grp in df.groupby(col):
        rows.append(_summary_row(grp, str(val)))
    result = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    return result


# ---------------------------------------------------------------------------
# Monthly P&L timeline
# ---------------------------------------------------------------------------

def monthly_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades[trades["is_closed"]].copy()
    closed["month"] = closed["open_date"].dt.to_period("M")
    monthly = (
        closed.groupby("month")["net_pnl"]
        .agg(["sum", "count", lambda x: (x > 0).sum()])
        .rename(columns={"sum": "pnl", "count": "trades", "<lambda_0>": "wins"})
        .reset_index()
    )
    monthly["win_rate"] = monthly["wins"] / monthly["trades"]
    monthly["cumulative_pnl"] = monthly["pnl"].cumsum()
    return monthly


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def run_report(transactions: pd.DataFrame, trades: pd.DataFrame) -> None:
    # ---- save raw data ----
    transactions.to_csv(OUTPUT_DIR / "all_transactions.csv", index=False)
    trades.to_csv(OUTPUT_DIR / "all_trades.csv", index=False)

    closed = trades[trades["is_closed"]]

    # ---- overall summary ----
    _section("OVERALL SUMMARY")
    summary = _summary_row(closed)
    print(f"  Total options transactions : {len(transactions[transactions['is_option']]):,}")
    print(f"  Closed option trades       : {summary['trades']:,}")
    print(f"  Open / unclosed positions  : {(trades['closed_by'] == 'open').sum():,}")
    print(f"  Total net P&L              : {_fmt_pnl(summary['total_pnl'])}")
    print(f"  Win rate                   : {_pct(summary['win_rate'] or 0)}")
    print(f"  Avg P&L per trade          : {_fmt_pnl(summary['avg_pnl'] or 0)}")
    print(f"  Avg winning trade          : {_fmt_pnl(summary['avg_win'] or 0)}")
    print(f"  Avg losing trade           : {_fmt_pnl(summary['avg_loss'] or 0)}")
    print(f"  Profit factor              : {summary['profit_factor']:.2f}" if summary["profit_factor"] else "  Profit factor              : N/A")
    print(f"  Best single trade          : {_fmt_pnl(summary['best_trade'] or 0)}")
    print(f"  Worst single trade         : {_fmt_pnl(summary['worst_trade'] or 0)}")

    # ---- by account ----
    _section("BY ACCOUNT")
    acct_df = _breakdown(closed, "account")
    print(acct_df[["label", "trades", "win_rate", "total_pnl", "avg_pnl", "profit_factor"]].to_string(index=False))
    acct_df.to_csv(OUTPUT_DIR / "by_account.csv", index=False)

    # ---- by strategy ----
    _section("BY STRATEGY  (CALL/PUT × long/short)")
    strat_df = _breakdown(closed, "strategy")
    print(strat_df[["label", "trades", "win_rate", "total_pnl", "avg_pnl", "avg_win", "avg_loss"]].to_string(index=False))
    strat_df.to_csv(OUTPUT_DIR / "by_strategy.csv", index=False)

    # ---- by DTE bucket ----
    _section("BY DTE BUCKET")
    dte_order = {"0-7 DTE": 0, "8-30 DTE": 1, "31-90 DTE": 2, "90+ DTE": 3, "unknown": 4}
    dte_df = _breakdown(closed, "dte_bucket")
    dte_df["_order"] = dte_df["label"].map(dte_order)
    dte_df = dte_df.sort_values("_order").drop(columns="_order")
    print(dte_df[["label", "trades", "win_rate", "total_pnl", "avg_pnl"]].to_string(index=False))
    dte_df.to_csv(OUTPUT_DIR / "by_dte_bucket.csv", index=False)

    # ---- by closed_by ----
    _section("BY CLOSE METHOD")
    close_df = _breakdown(closed, "closed_by")
    print(close_df[["label", "trades", "win_rate", "total_pnl", "avg_pnl"]].to_string(index=False))
    close_df.to_csv(OUTPUT_DIR / "by_close_method.csv", index=False)

    # ---- top tickers ----
    _section("TOP 20 TICKERS  (by trade count)")
    ticker_df = _breakdown(closed, "ticker")
    top_tickers = ticker_df.nlargest(20, "trades")
    print(top_tickers[["label", "trades", "win_rate", "total_pnl", "avg_pnl"]].to_string(index=False))
    ticker_df.to_csv(OUTPUT_DIR / "by_ticker.csv", index=False)

    _section("TOP 20 TICKERS  (by total P&L)")
    top_pnl = ticker_df.nlargest(20, "total_pnl")
    print(top_pnl[["label", "trades", "win_rate", "total_pnl", "avg_pnl"]].to_string(index=False))

    _section("BOTTOM 10 TICKERS  (by total P&L — biggest losers)")
    bot_pnl = ticker_df.nsmallest(10, "total_pnl")
    print(bot_pnl[["label", "trades", "win_rate", "total_pnl", "avg_pnl"]].to_string(index=False))

    # ---- monthly P&L ----
    _section("MONTHLY P&L TIMELINE")
    monthly = monthly_pnl(trades)
    monthly.to_csv(OUTPUT_DIR / "monthly_pnl.csv", index=False)
    print(monthly[["month", "trades", "win_rate", "pnl", "cumulative_pnl"]].to_string(index=False))

    # ---- sizing analysis ----
    _section("POSITION SIZING DISTRIBUTION")
    size_counts = closed["qty"].value_counts().sort_index()
    print("  Contracts per trade:")
    for qty, cnt in size_counts.items():
        bar = "█" * min(int(cnt / size_counts.max() * 30), 30)
        print(f"    {int(qty):>3} contracts: {cnt:>5}  {bar}")

    print(f"\n  Average contracts per trade : {closed['qty'].mean():.2f}")
    print(f"  Median  contracts per trade : {closed['qty'].median():.0f}")

    print(f"\n[Phase 1 complete — outputs saved to {OUTPUT_DIR}/]")
