"""
Options Trader Dashboard — Phase 2
Run: streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import time as _time
from datetime import datetime as _dt

from src.quotes import fetch_quotes
from src.patterns import (
    backtest_filter,
    backtest_metrics,
    by_day_of_week,
    by_holding_period,
    by_month_of_year,
    by_premium_bucket,
    cumulative_pnl_series,
    pnl_distribution,
    streak_analysis,
    ticker_heatmap,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Options Trader Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTPUT_DIR = Path("output")

_COLORS = {
    "green": "#00C896",
    "red": "#FF4B6E",
    "blue": "#4B9BFF",
    "orange": "#FF9444",
    "purple": "#A855F7",
    "bg": "#0E1117",
    "card": "#1A1F2E",
    "text": "#E0E0E0",
}

STRATEGY_COLORS = {
    "PUT_short": _COLORS["green"],
    "CALL_short": _COLORS["blue"],
    "PUT_long": _COLORS["red"],
    "CALL_long": _COLORS["orange"],
}

st.markdown(
    """
    <style>
    .metric-card {
        background: #1A1F2E;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2A2F3E;
    }
    .metric-label { color: #888; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #E0E0E0; font-size: 28px; font-weight: 700; margin-top: 4px; }
    .metric-pos  { color: #00C896; }
    .metric-neg  { color: #FF4B6E; }
    div[data-testid="stTabs"] button { font-size: 15px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_trades() -> pd.DataFrame:
    path = OUTPUT_DIR / "all_trades.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["open_date", "close_date", "expiry"])
    df["is_closed"] = df["closed_by"] != "open"
    df["is_win"] = df["net_pnl"] > 0
    df["month"] = pd.to_datetime(df["open_date"]).dt.to_period("M")
    return df


@st.cache_data(show_spinner=False)
def load_monthly() -> pd.DataFrame:
    path = OUTPUT_DIR / "monthly_pnl.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _refresh():
    """Re-run the Phase 1 pipeline and reload data."""
    import subprocess, sys
    with st.spinner("Re-running data pipeline…"):
        subprocess.run([sys.executable, "main.py"], check=True)
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric_card(col, label: str, value: str, delta: str = "", positive: bool | None = None):
    color_cls = ""
    if positive is True:
        color_cls = "metric-pos"
    elif positive is False:
        color_cls = "metric-neg"
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_cls}">{value}</div>
            {"<div style='color:#888;font-size:13px;margin-top:4px;'>" + delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt(v: float, prefix: str = "$") -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{prefix}{v:,.2f}" if prefix else f"{sign}{v:.1f}"


def _win_color(rate: float) -> str:
    return _COLORS["green"] if rate >= 0.6 else _COLORS["orange"] if rate >= 0.5 else _COLORS["red"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

trades_all = load_trades()
monthly_df = load_monthly()

if trades_all.empty:
    st.error("No data found in output/. Click below to run the data pipeline first.")
    if st.button("Run Data Pipeline"):
        _refresh()
    st.stop()

closed_all = trades_all[trades_all["is_closed"]].copy()

# ---------------------------------------------------------------------------
# Global sidebar filters (affect all tabs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Global Filters")

    accounts = sorted(closed_all["account"].dropna().unique().tolist())
    sel_accounts = st.multiselect("Account", accounts, default=accounts)

    date_min = pd.to_datetime(closed_all["open_date"]).min().date()
    date_max = pd.to_datetime(closed_all["open_date"]).max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

    strategies = sorted(closed_all["strategy"].dropna().unique().tolist())
    sel_strategies = st.multiselect("Strategy", strategies, default=strategies)

    st.divider()
    if st.button("🔄 Refresh Data Pipeline"):
        _refresh()

# Apply global filters
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["account"].isin(sel_accounts)]
    df = df[df["strategy"].isin(sel_strategies)]
    if len(date_range) == 2:
        df = df[
            (pd.to_datetime(df["open_date"]).dt.date >= date_range[0]) &
            (pd.to_datetime(df["open_date"]).dt.date <= date_range[1])
        ]
    return df


closed = apply_filters(closed_all)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
def _render_period_tab(df: pd.DataFrame, period_label: str) -> None:
    """Render a full analytics page for a date-filtered slice of closed trades."""
    if df.empty:
        st.info(f"No closed trades found for {period_label}.")
        return

    wins   = df[df["is_win"]]
    losses = df[~df["is_win"]]
    total_pnl = df["net_pnl"].sum()
    win_rate  = len(wins) / len(df) if len(df) else 0
    pf = (
        abs(wins["net_pnl"].sum() / losses["net_pnl"].sum())
        if len(losses) and losses["net_pnl"].sum() != 0 else 0
    )

    st.title(f"Analytics — {period_label}")

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _metric_card(c1, "Total Net P&L",    f"${total_pnl:,.0f}",          positive=total_pnl >= 0)
    _metric_card(c2, "Win Rate",         f"{win_rate*100:.1f}%",        positive=win_rate >= 0.55)
    _metric_card(c3, "Profit Factor",    f"{pf:.2f}",                   positive=pf >= 1)
    _metric_card(c4, "Closed Trades",    f"{len(df):,}")
    _metric_card(c5, "Avg P&L / Trade",  _fmt(df["net_pnl"].mean()),    positive=df["net_pnl"].mean() >= 0)
    _metric_card(c6, "Max Drawdown",
        _fmt(
            (df.sort_values("open_date")["net_pnl"].cumsum()
             - df.sort_values("open_date")["net_pnl"].cumsum().cummax()).min()
        ),
        positive=False,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cumulative P&L + Strategy Mix ────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Cumulative P&L")
        cum = cumulative_pnl_series(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum["date"], y=cum["cumulative_pnl"],
            mode="lines", fill="tozeroy",
            line=dict(color=_COLORS["green"], width=2),
            fillcolor="rgba(0,200,150,0.12)",
        ))
        fig.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
            xaxis=dict(gridcolor="#1E2433"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Strategy Mix")
        strat_df = (
            df.groupby("strategy")["net_pnl"]
            .agg(["sum", "count"]).reset_index()
            .rename(columns={"sum": "total_pnl", "count": "trades"})
        )
        fig_pie = px.pie(
            strat_df, values="trades", names="strategy",
            color="strategy", color_discrete_map=STRATEGY_COLORS, hole=0.5,
        )
        fig_pie.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Monthly P&L ──────────────────────────────────────────────────────────
    st.subheader("Monthly P&L")
    df_m = df.copy()
    df_m["month_str"] = pd.to_datetime(df_m["open_date"]).dt.strftime("%Y-%m")
    monthly = df_m.groupby("month_str")["net_pnl"].sum().reset_index()
    monthly["color"] = monthly["net_pnl"].apply(
        lambda x: _COLORS["green"] if x >= 0 else _COLORS["red"]
    )
    fig_m = go.Figure(go.Bar(
        x=monthly["month_str"], y=monthly["net_pnl"],
        marker_color=monthly["color"],
        text=monthly["net_pnl"].apply(lambda v: f"${v:+,.0f}"),
        textposition="outside", textfont=dict(size=9),
    ))
    fig_m.update_layout(
        plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
        font_color=_COLORS["text"], height=280,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
        xaxis=dict(gridcolor="#1E2433"),
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # ── DTE breakdown + Strategy P&L ─────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("By DTE Bucket")
        dte_order = {"0-7 DTE": 0, "8-30 DTE": 1, "31-90 DTE": 2, "90+ DTE": 3, "unknown": 4}
        dte_rows = []
        for bucket, grp in df.groupby("dte_bucket"):
            w = grp[grp["is_win"]]
            dte_rows.append({
                "DTE Bucket": bucket,
                "Trades": len(grp),
                "Win %": f"{len(w)/len(grp)*100:.0f}%",
                "Total P&L": f"${grp['net_pnl'].sum():+,.0f}",
                "Avg P&L": f"${grp['net_pnl'].mean():+,.0f}",
            })
        dte_tbl = pd.DataFrame(dte_rows)
        if not dte_tbl.empty:
            dte_tbl["_ord"] = dte_tbl["DTE Bucket"].map(dte_order).fillna(99)
            dte_tbl = dte_tbl.sort_values("_ord").drop(columns="_ord")
        st.dataframe(dte_tbl, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("By Strategy")
        strat_rows = []
        for strat, grp in df.groupby("strategy"):
            w = grp[grp["is_win"]]
            strat_rows.append({
                "Strategy": strat,
                "Trades": len(grp),
                "Win %": f"{len(w)/len(grp)*100:.0f}%",
                "Total P&L": f"${grp['net_pnl'].sum():+,.0f}",
                "Avg P&L": f"${grp['net_pnl'].mean():+,.0f}",
            })
        st.dataframe(
            pd.DataFrame(strat_rows).sort_values("Total P&L", ascending=False),
            use_container_width=True, hide_index=True,
        )

    # ── Top / bottom tickers ─────────────────────────────────────────────────
    st.subheader("Top & Bottom Tickers (min 5 trades)")
    tk_rows = []
    for ticker, grp in df.groupby("ticker"):
        if len(grp) < 5:
            continue
        w = grp[grp["is_win"]]
        tk_rows.append({
            "Ticker": ticker,
            "Trades": len(grp),
            "Win %": round(len(w) / len(grp) * 100, 1),
            "Total P&L": grp["net_pnl"].sum(),
            "Avg P&L": grp["net_pnl"].mean(),
        })
    tk_df = pd.DataFrame(tk_rows).sort_values("Total P&L", ascending=False)

    top_col, bot_col = st.columns(2)
    with top_col:
        st.caption("Top 15 by P&L")
        top15 = tk_df.head(15).copy()
        top15["Total P&L"] = top15["Total P&L"].apply(lambda v: f"${v:+,.0f}")
        top15["Avg P&L"]   = top15["Avg P&L"].apply(lambda v: f"${v:+,.0f}")
        top15["Win %"]     = top15["Win %"].apply(lambda v: f"{v:.0f}%")
        st.dataframe(top15, use_container_width=True, hide_index=True)
    with bot_col:
        st.caption("Bottom 15 by P&L")
        bot15 = tk_df.tail(15).sort_values("Total P&L").copy()
        bot15["Total P&L"] = bot15["Total P&L"].apply(lambda v: f"${v:+,.0f}")
        bot15["Avg P&L"]   = bot15["Avg P&L"].apply(lambda v: f"${v:+,.0f}")
        bot15["Win %"]     = bot15["Win %"].apply(lambda v: f"{v:.0f}%")
        st.dataframe(bot15, use_container_width=True, hide_index=True)


# Period-filtered DataFrames (account/strategy filters from sidebar apply; date is fixed)
def _period_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["account"].isin(sel_accounts)]
    return df[df["strategy"].isin(sel_strategies)]

closed_hist = _period_filter(
    closed_all[pd.to_datetime(closed_all["open_date"]).dt.year < 2026]
)
closed_2026 = _period_filter(
    closed_all[pd.to_datetime(closed_all["open_date"]).dt.year >= 2026]
)


tab_live, tab_overview, tab_hist, tab_2026, tab_strategy, tab_patterns, tab_backtest, tab_explorer, tab_earnings = st.tabs([
    "🔴 Live Positions",
    "📊 Overview (All Time)",
    "📅 2023–2025",
    "📅 2026",
    "🎯 Strategy Analysis",
    "🔍 Pattern Mining",
    "⚙️ Backtester",
    "📋 Trade Explorer",
    "📆 Earnings Calendar",
])

# ===========================================================================
# TAB 0: LIVE POSITIONS
# ===========================================================================
with tab_live:
    st.title("Live Positions")
    st.caption("Option quotes via yfinance (~15-min delayed). Unrealized P&L and theta are estimates based on mid-price.")

    open_pos = trades_all[trades_all["closed_by"] == "open"].copy()

    if open_pos.empty:
        st.info("No open positions found. Run the data pipeline to refresh.")
    else:
        # Controls
        ctrl_l, ctrl_r = st.columns([4, 1])
        with ctrl_r:
            fetch_btn = st.button("🔄 Fetch Live Quotes", type="primary", use_container_width=True)

        _CACHE_KEY = "live_quotes_data"
        _CACHE_TS = "live_quotes_ts"

        if fetch_btn or _CACHE_KEY not in st.session_state:
            with st.spinner("Fetching live quotes…"):
                _enriched = fetch_quotes(open_pos)
            st.session_state[_CACHE_KEY] = _enriched
            st.session_state[_CACHE_TS] = _time.time()
        else:
            _enriched = st.session_state[_CACHE_KEY]

        with ctrl_l:
            _ts = st.session_state.get(_CACHE_TS)
            if _ts:
                st.caption(f"Last updated: {_dt.fromtimestamp(_ts).strftime('%I:%M:%S %p')}")

        _today_ts = pd.Timestamp.now().normalize()
        _active = _enriched[pd.to_datetime(_enriched["expiry"]) >= _today_ts].copy()
        _expired = _enriched[pd.to_datetime(_enriched["expiry"]) < _today_ts]

        if not _expired.empty:
            st.warning(
                f"{len(_expired)} position(s) show an expiry date in the past — "
                "run the data pipeline to close them out."
            )

        if _active.empty:
            st.info("No active open positions.")
        else:
            # ── Summary metric cards ──────────────────────────────────────────
            _upnl_vals = _active["unrealized_pnl"].dropna()
            _total_upnl = _upnl_vals.sum() if not _upnl_vals.empty else None
            _near_exp = int((_active["dte_now"] <= 7).sum())

            # Portfolio dollar theta/day (positive = net theta income for short book)
            _theta_total = None
            if "theta_dollar_day" in _active.columns:
                _td = _active["theta_dollar_day"].dropna()
                if not _td.empty:
                    _theta_total = round(_td.sum(), 2)

            mc1, mc2, mc3, mc4 = st.columns(4)
            _metric_card(mc1, "Open Positions", f"{len(_active)}")
            if _total_upnl is not None:
                _metric_card(mc2, "Unrealized P&L", _fmt(_total_upnl), positive=_total_upnl >= 0)
            else:
                _metric_card(mc2, "Unrealized P&L", "—")
            _metric_card(
                mc3, "Near Expiry (≤7 DTE)", f"{_near_exp}",
                positive=(_near_exp == 0),
            )
            if _theta_total is not None:
                _metric_card(mc4, "Portfolio θ/Day", _fmt(_theta_total), positive=_theta_total >= 0)
            else:
                _metric_card(mc4, "Portfolio θ/Day", "—")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Unrealized P&L bar chart ──────────────────────────────────────
            _chart_df = _active[_active["unrealized_pnl"].notna()].copy()
            if not _chart_df.empty:
                _chart_df["_label"] = (
                    _chart_df["ticker"].astype(str)
                    + " "
                    + _chart_df["opt_type"].astype(str)
                    + " $"
                    + _chart_df["strike"].astype(str)
                    + " "
                    + pd.to_datetime(_chart_df["expiry"]).dt.strftime("%m/%d")
                )
                _fig_pos = px.bar(
                    _chart_df, x="_label", y="unrealized_pnl",
                    color=_chart_df["unrealized_pnl"].apply(
                        lambda v: "profit" if v >= 0 else "loss"
                    ),
                    color_discrete_map={"profit": _COLORS["green"], "loss": _COLORS["red"]},
                    text=_chart_df["unrealized_pnl"].apply(lambda v: f"${v:+,.2f}"),
                )
                _fig_pos.update_layout(
                    plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                    font_color=_COLORS["text"], height=280, showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
                    xaxis_title="", yaxis_title="Unrealized P&L",
                )
                st.plotly_chart(_fig_pos, use_container_width=True)

            # ── Positions table ───────────────────────────────────────────────
            st.subheader("Position Details")

            def _dte_badge(v):
                d = int(v) if pd.notna(v) else 999
                if d <= 3:
                    return f"🔴 {d}d"
                if d <= 7:
                    return f"🟠 {d}d"
                if d <= 14:
                    return f"🟡 {d}d"
                return f"🟢 {d}d"

            def _upnl_fmt(v):
                if pd.isna(v):
                    return "—"
                return f"▲ ${v:,.2f}" if v >= 0 else f"▼ -${abs(v):,.2f}"

            _tbl = _active[[
                "ticker", "strategy", "strike", "expiry", "dte_now", "qty",
                "open_date", "stock_price", "current_price", "bid", "ask",
                "unrealized_pnl", "pct_max_profit", "iv", "theta_per_day",
                "quote_status",
            ]].copy().reset_index(drop=True)

            _tbl["expiry"] = pd.to_datetime(_tbl["expiry"]).dt.strftime("%Y-%m-%d")
            _tbl["open_date"] = pd.to_datetime(_tbl["open_date"]).dt.strftime("%Y-%m-%d")
            _tbl["strike"] = _tbl["strike"].apply(lambda v: f"${v:.1f}" if pd.notna(v) else "")
            _tbl["qty"] = _tbl["qty"].apply(lambda v: str(int(v)) if pd.notna(v) else "")
            _tbl["stock_price"] = _tbl["stock_price"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
            _tbl["current_price"] = _tbl["current_price"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
            _tbl["bid"] = _tbl["bid"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
            _tbl["ask"] = _tbl["ask"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
            _tbl["unrealized_pnl"] = _tbl["unrealized_pnl"].apply(_upnl_fmt)
            _tbl["pct_max_profit"] = _tbl["pct_max_profit"].apply(
                lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
            )
            _tbl["iv"] = _tbl["iv"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
            _tbl["theta_per_day"] = _tbl["theta_per_day"].apply(
                lambda v: f"${v:.4f}" if pd.notna(v) else "—"
            )
            _tbl["dte_now"] = _tbl["dte_now"].apply(_dte_badge)

            _tbl = _tbl.rename(columns={
                "ticker": "Ticker", "strategy": "Strategy", "strike": "Strike",
                "expiry": "Expiry", "dte_now": "DTE", "qty": "Qty",
                "open_date": "Opened", "stock_price": "Stock",
                "current_price": "Option Mid", "bid": "Bid", "ask": "Ask",
                "unrealized_pnl": "Unrealized P&L", "pct_max_profit": "% Max Profit",
                "iv": "IV", "theta_per_day": "θ/Day ($/share)",
                "quote_status": "Status",
            })

            st.dataframe(_tbl, use_container_width=True, hide_index=True)

            if not _CACHE_KEY in st.session_state or fetch_btn:
                st.info("Click **Fetch Live Quotes** to load market data.")

# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab_overview:
    st.title("P&L Overview")

    wins = closed[closed["is_win"]]
    losses = closed[~closed["is_win"]]
    total_pnl = closed["net_pnl"].sum()
    win_rate = len(wins) / len(closed) if len(closed) else 0
    pf = abs(wins["net_pnl"].sum() / losses["net_pnl"].sum()) if len(losses) and losses["net_pnl"].sum() != 0 else 0

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    _metric_card(c1, "Total Net P&L", f"${total_pnl:,.0f}", positive=total_pnl >= 0)
    _metric_card(c2, "Win Rate", f"{win_rate*100:.1f}%", positive=win_rate >= 0.55)
    _metric_card(c3, "Profit Factor", f"{pf:.2f}", positive=pf >= 1)
    _metric_card(c4, "Closed Trades", f"{len(closed):,}")
    _metric_card(c5, "Avg P&L / Trade", _fmt(closed["net_pnl"].mean()), positive=closed["net_pnl"].mean() >= 0)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cumulative P&L + Monthly P&L
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Cumulative P&L")
        cum = cumulative_pnl_series(closed)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum["date"], y=cum["cumulative_pnl"],
            mode="lines", fill="tozeroy",
            line=dict(color=_COLORS["green"], width=2),
            fillcolor="rgba(0,200,150,0.12)",
            name="Cumulative P&L",
        ))
        fig.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
            xaxis=dict(gridcolor="#1E2433"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Strategy Mix")
        strat_summary = (
            closed.groupby("strategy")["net_pnl"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "total_pnl", "count": "trades"})
        )
        fig_pie = px.pie(
            strat_summary, values="trades", names="strategy",
            color="strategy",
            color_discrete_map=STRATEGY_COLORS,
            hole=0.5,
        )
        fig_pie.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Monthly P&L bar
    st.subheader("Monthly P&L")
    closed_m = closed.copy()
    closed_m["month_str"] = pd.to_datetime(closed_m["open_date"]).dt.strftime("%Y-%m")
    monthly = closed_m.groupby("month_str")["net_pnl"].sum().reset_index()
    monthly["color"] = monthly["net_pnl"].apply(lambda x: _COLORS["green"] if x >= 0 else _COLORS["red"])

    fig_monthly = go.Figure(go.Bar(
        x=monthly["month_str"], y=monthly["net_pnl"],
        marker_color=monthly["color"],
        text=monthly["net_pnl"].apply(lambda v: f"${v:+,.0f}"),
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_monthly.update_layout(
        plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
        font_color=_COLORS["text"], height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
        xaxis=dict(gridcolor="#1E2433"),
        showlegend=False,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Streaks + open positions
    col_a, col_b = st.columns(2)
    with col_a:
        streaks = streak_analysis(closed)
        st.metric("Max Win Streak", streaks["max_win_streak"])
        st.metric("Max Loss Streak", streaks["max_loss_streak"])
    with col_b:
        open_pos = trades_all[trades_all["closed_by"] == "open"]
        st.metric("Currently Open Positions", len(open_pos))
        if not open_pos.empty:
            st.dataframe(
                open_pos[["ticker", "opt_type", "direction", "strike", "expiry", "open_date", "open_amount"]]
                .sort_values("expiry"),
                use_container_width=True, hide_index=True,
            )

# ===========================================================================
# TAB 2: STRATEGY ANALYSIS
# ===========================================================================
with tab_strategy:
    st.title("Strategy Analysis")

    # Strategy cards
    strat_cols = st.columns(4)
    for i, strat in enumerate(["PUT_short", "CALL_short", "PUT_long", "CALL_long"]):
        g = closed[closed["strategy"] == strat]
        if g.empty:
            continue
        w = g[g["is_win"]]
        pnl = g["net_pnl"].sum()
        wr = len(w) / len(g)
        _metric_card(
            strat_cols[i], strat.replace("_", " "),
            _fmt(pnl), f"{wr*100:.1f}% win rate • {len(g):,} trades",
            positive=pnl >= 0,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Win Rate by Strategy")
        sr = closed.groupby("strategy").apply(
            lambda g: pd.Series({
                "win_rate": (g["is_win"].sum() / len(g)) * 100,
                "trades": len(g),
                "total_pnl": g["net_pnl"].sum(),
            })
        ).reset_index()
        fig = px.bar(
            sr.sort_values("win_rate", ascending=True),
            x="win_rate", y="strategy", orientation="h",
            color="strategy", color_discrete_map=STRATEGY_COLORS,
            text=sr.sort_values("win_rate")["win_rate"].apply(lambda v: f"{v:.1f}%"),
        )
        fig.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=280,
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            xaxis=dict(title="Win Rate (%)", gridcolor="#1E2433"),
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Total P&L by Strategy")
        fig2 = px.bar(
            sr.sort_values("total_pnl", ascending=True),
            x="total_pnl", y="strategy", orientation="h",
            color="strategy", color_discrete_map=STRATEGY_COLORS,
            text=sr.sort_values("total_pnl")["total_pnl"].apply(lambda v: f"${v:+,.0f}"),
        )
        fig2.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=280,
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            xaxis=dict(title="Total P&L ($)", tickprefix="$", gridcolor="#1E2433"),
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # DTE bucket analysis
    st.subheader("DTE Bucket Breakdown")
    dte_order = ["0-7 DTE", "8-30 DTE", "31-90 DTE", "90+ DTE", "unknown"]
    dte_df = closed.groupby("dte_bucket").apply(
        lambda g: pd.Series({
            "trades": len(g),
            "win_rate": g["is_win"].mean() * 100,
            "total_pnl": g["net_pnl"].sum(),
            "avg_pnl": g["net_pnl"].mean(),
        })
    ).reset_index()
    dte_df["dte_bucket"] = pd.Categorical(dte_df["dte_bucket"], categories=dte_order, ordered=True)
    dte_df = dte_df.sort_values("dte_bucket")

    d1, d2 = st.columns(2)
    with d1:
        fig_dte_wr = px.bar(
            dte_df, x="dte_bucket", y="win_rate",
            text=dte_df["win_rate"].apply(lambda v: f"{v:.1f}%"),
            color_discrete_sequence=[_COLORS["blue"]],
            title="Win Rate by DTE",
        )
        fig_dte_wr.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=280, showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title="Win Rate (%)", gridcolor="#1E2433"),
            xaxis_title="",
        )
        st.plotly_chart(fig_dte_wr, use_container_width=True)

    with d2:
        fig_dte_pnl = px.bar(
            dte_df, x="dte_bucket", y="total_pnl",
            text=dte_df["total_pnl"].apply(lambda v: f"${v:+,.0f}"),
            color=dte_df["total_pnl"].apply(lambda v: "pos" if v >= 0 else "neg"),
            color_discrete_map={"pos": _COLORS["green"], "neg": _COLORS["red"]},
            title="Total P&L by DTE",
        )
        fig_dte_pnl.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=280, showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title="P&L ($)", tickprefix="$", gridcolor="#1E2433"),
            xaxis_title="",
        )
        st.plotly_chart(fig_dte_pnl, use_container_width=True)

    # Close method + P&L distribution
    st.subheader("Close Method & P&L Distribution")
    cm1, cm2 = st.columns(2)

    with cm1:
        close_df = closed.groupby("closed_by").apply(
            lambda g: pd.Series({
                "trades": len(g),
                "win_rate": g["is_win"].mean() * 100,
                "total_pnl": g["net_pnl"].sum(),
            })
        ).reset_index()
        fig_cm = px.bar(
            close_df, x="closed_by", y=["total_pnl"],
            color_discrete_sequence=[_COLORS["blue"]],
            title="P&L by Close Method",
            text_auto=True,
        )
        fig_cm.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300, showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
            xaxis_title="",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm2:
        dist_df = pnl_distribution(closed)
        clip_val = dist_df["net_pnl"].quantile(0.99)
        dist_clipped = dist_df["net_pnl"].clip(-clip_val, clip_val)
        fig_hist = px.histogram(
            x=dist_clipped, nbins=80,
            color=dist_df["is_win"].map({True: "Win", False: "Loss"}),
            color_discrete_map={"Win": _COLORS["green"], "Loss": _COLORS["red"]},
            title="P&L Distribution (99th pct clip)",
        )
        fig_hist.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title="Net P&L ($)", tickprefix="$", gridcolor="#1E2433"),
            yaxis=dict(gridcolor="#1E2433"),
            barmode="overlay", bargap=0.1,
            legend=dict(title=""),
        )
        fig_hist.update_traces(opacity=0.7)
        st.plotly_chart(fig_hist, use_container_width=True)

# ===========================================================================
# TAB 3: PATTERN MINING
# ===========================================================================
with tab_patterns:
    st.title("Pattern Mining")
    st.caption("Discover what drove winners vs losers in your trading history.")

    # Day of week
    p1, p2 = st.columns(2)

    with p1:
        st.subheader("Win Rate by Day of Week")
        dow = by_day_of_week(closed)
        fig_dow = px.bar(
            dow, x="day", y="win_rate",
            text=dow["win_rate"].apply(lambda v: f"{v*100:.1f}%"),
            color="win_rate",
            color_continuous_scale=["#FF4B6E", "#FF9444", "#00C896"],
            range_color=[0.4, 0.8],
        )
        fig_dow.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300, showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickformat=".0%", gridcolor="#1E2433"),
            xaxis_title="",
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with p2:
        st.subheader("P&L by Day of Week")
        fig_dow_pnl = px.bar(
            dow, x="day", y="total_pnl",
            text=dow["total_pnl"].apply(lambda v: f"${v:+,.0f}"),
            color=dow["total_pnl"].apply(lambda v: "pos" if v >= 0 else "neg"),
            color_discrete_map={"pos": _COLORS["green"], "neg": _COLORS["red"]},
        )
        fig_dow_pnl.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
            xaxis_title="",
        )
        st.plotly_chart(fig_dow_pnl, use_container_width=True)

    # Month of year
    st.subheader("Seasonality — Month of Year")
    moy = by_month_of_year(closed)
    s1, s2 = st.columns(2)
    with s1:
        fig_moy = px.bar(
            moy, x="month", y="win_rate",
            text=moy["win_rate"].apply(lambda v: f"{v*100:.1f}%"),
            color="win_rate",
            color_continuous_scale=["#FF4B6E", "#FF9444", "#00C896"],
            range_color=[0.4, 0.8],
            title="Win Rate by Month",
        )
        fig_moy.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300, showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(tickformat=".0%", gridcolor="#1E2433"),
            xaxis_title="",
        )
        st.plotly_chart(fig_moy, use_container_width=True)
    with s2:
        fig_moy_pnl = px.bar(
            moy, x="month", y="total_pnl",
            text=moy["total_pnl"].apply(lambda v: f"${v:+,.0f}"),
            color=moy["total_pnl"].apply(lambda v: "pos" if v >= 0 else "neg"),
            color_discrete_map={"pos": _COLORS["green"], "neg": _COLORS["red"]},
            title="Total P&L by Month",
        )
        fig_moy_pnl.update_layout(
            plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
            font_color=_COLORS["text"], height=300, showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
            xaxis_title="",
        )
        st.plotly_chart(fig_moy_pnl, use_container_width=True)

    # Premium size analysis
    st.subheader("Premium Size vs Outcome")
    prem_strat = st.selectbox("Strategy filter for premium analysis", ["All short trades"] + strategies)
    strat_for_prem = None if prem_strat == "All short trades" else prem_strat
    prem_df = by_premium_bucket(closed, strat_for_prem)

    if not prem_df.empty:
        pr1, pr2 = st.columns(2)
        with pr1:
            fig_prem_wr = px.bar(
                prem_df, x="bucket", y="win_rate",
                text=prem_df["win_rate"].apply(lambda v: f"{v*100:.1f}%"),
                color="win_rate",
                color_continuous_scale=["#FF4B6E", "#FF9444", "#00C896"],
                range_color=[0.4, 0.85],
                title="Win Rate by Premium Size",
            )
            fig_prem_wr.update_layout(
                plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                font_color=_COLORS["text"], height=300, showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(tickformat=".0%", gridcolor="#1E2433"),
                xaxis_title="Premium Collected/Paid",
            )
            st.plotly_chart(fig_prem_wr, use_container_width=True)
        with pr2:
            fig_prem_cnt = px.bar(
                prem_df, x="bucket", y="trades",
                text="trades",
                color_discrete_sequence=[_COLORS["blue"]],
                title="Trade Count by Premium Size",
            )
            fig_prem_cnt.update_layout(
                plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                font_color=_COLORS["text"], height=300, showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(gridcolor="#1E2433"),
                xaxis_title="Premium Collected/Paid",
            )
            st.plotly_chart(fig_prem_cnt, use_container_width=True)

    # Holding period
    st.subheader("Holding Period Analysis")
    hold_df = by_holding_period(closed)
    if not hold_df.empty:
        h1, h2 = st.columns(2)
        with h1:
            fig_hold_wr = px.bar(
                hold_df, x="hold_bucket", y="win_rate",
                text=hold_df["win_rate"].apply(lambda v: f"{v*100:.1f}%"),
                color="win_rate",
                color_continuous_scale=["#FF4B6E", "#FF9444", "#00C896"],
                range_color=[0.4, 0.9],
                title="Win Rate by Holding Period",
            )
            fig_hold_wr.update_layout(
                plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                font_color=_COLORS["text"], height=300, showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(tickformat=".0%", gridcolor="#1E2433"),
                xaxis_title="Hold Duration",
            )
            st.plotly_chart(fig_hold_wr, use_container_width=True)
        with h2:
            fig_hold_pnl = px.bar(
                hold_df, x="hold_bucket", y="total_pnl",
                text=hold_df["total_pnl"].apply(lambda v: f"${v:+,.0f}"),
                color=hold_df["total_pnl"].apply(lambda v: "pos" if v >= 0 else "neg"),
                color_discrete_map={"pos": _COLORS["green"], "neg": _COLORS["red"]},
                title="P&L by Holding Period",
            )
            fig_hold_pnl.update_layout(
                plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                font_color=_COLORS["text"], height=300, showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
                xaxis_title="Hold Duration",
            )
            st.plotly_chart(fig_hold_pnl, use_container_width=True)

    # Ticker heatmap
    st.subheader("Ticker Performance")
    min_trades_filter = st.slider("Minimum trades per ticker", 5, 50, 15)
    tk_df = ticker_heatmap(closed, min_trades=min_trades_filter)

    if not tk_df.empty:
        tk1, tk2 = st.columns(2)
        with tk1:
            top20 = tk_df.nlargest(20, "total_pnl")
            fig_tk_top = px.bar(
                top20.sort_values("total_pnl"), x="total_pnl", y="ticker",
                orientation="h",
                color="win_rate", color_continuous_scale=["#FF4B6E", "#FF9444", "#00C896"],
                range_color=[0.4, 0.9],
                text=top20.sort_values("total_pnl")["total_pnl"].apply(lambda v: f"${v:+,.0f}"),
                title="Top 20 Tickers by P&L",
            )
            fig_tk_top.update_layout(
                plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                font_color=_COLORS["text"], height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(tickprefix="$", gridcolor="#1E2433"),
                yaxis_title="", coloraxis_colorbar=dict(title="Win Rate"),
            )
            st.plotly_chart(fig_tk_top, use_container_width=True)
        with tk2:
            bot20 = tk_df.nsmallest(20, "total_pnl")
            fig_tk_bot = px.bar(
                bot20.sort_values("total_pnl", ascending=False), x="total_pnl", y="ticker",
                orientation="h",
                color="win_rate", color_continuous_scale=["#FF4B6E", "#FF9444", "#00C896"],
                range_color=[0.4, 0.9],
                text=bot20.sort_values("total_pnl", ascending=False)["total_pnl"].apply(lambda v: f"${v:+,.0f}"),
                title="Bottom 20 Tickers by P&L",
            )
            fig_tk_bot.update_layout(
                plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                font_color=_COLORS["text"], height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(tickprefix="$", gridcolor="#1E2433"),
                yaxis_title="", coloraxis_showscale=False,
            )
            st.plotly_chart(fig_tk_bot, use_container_width=True)

        # Full ticker table
        with st.expander("Full Ticker Table"):
            display_cols = ["ticker", "trades", "win_rate", "total_pnl", "avg_pnl", "avg_win", "avg_loss"]
            tk_display = tk_df[display_cols].copy()
            tk_display["win_rate"] = (tk_display["win_rate"] * 100).round(1).astype(str) + "%"
            tk_display["total_pnl"] = tk_display["total_pnl"].apply(lambda v: f"${v:+,.2f}")
            tk_display["avg_pnl"] = tk_display["avg_pnl"].apply(lambda v: f"${v:+,.2f}")
            tk_display["avg_win"] = tk_display["avg_win"].apply(lambda v: f"${v:+,.2f}")
            tk_display["avg_loss"] = tk_display["avg_loss"].apply(lambda v: f"${v:+,.2f}")
            st.dataframe(tk_display, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 4: BACKTESTER
# ===========================================================================
with tab_backtest:
    st.title("Strategy Backtester")
    st.caption("Filter your historical trades by rules and see how the strategy would have performed.")

    bt_col, result_col = st.columns([1, 2])

    with bt_col:
        st.subheader("Strategy Rules")

        bt_strategies = st.multiselect(
            "Strategy types", strategies, default=["PUT_short", "CALL_short"],
            key="bt_strat",
        )
        bt_dte_min, bt_dte_max = st.slider("DTE at open range", 0, 120, (0, 7), key="bt_dte")
        bt_prem_min, bt_prem_max = st.slider("Premium range ($)", 0, 2000, (50, 1000), key="bt_prem")

        all_tickers = sorted(closed_all["ticker"].dropna().unique().tolist())
        bt_tickers_exclude = st.multiselect("Exclude tickers", all_tickers, key="bt_excl")

        bt_date_start = st.date_input("Backtest start", value=date_min, key="bt_ds")
        bt_date_end = st.date_input("Backtest end", value=date_max, key="bt_de")

        run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

    with result_col:
        if run_bt or True:  # show results immediately
            filtered = backtest_filter(
                trades_all,
                strategies=bt_strategies if bt_strategies else None,
                dte_min=bt_dte_min,
                dte_max=bt_dte_max,
                tickers_exclude=bt_tickers_exclude if bt_tickers_exclude else None,
                premium_min=bt_prem_min,
                premium_max=bt_prem_max,
                date_start=str(bt_date_start),
                date_end=str(bt_date_end),
            )

            if filtered.empty:
                st.warning("No trades match the current filters.")
            else:
                metrics = backtest_metrics(filtered)

                m1, m2, m3, m4 = st.columns(4)
                _metric_card(m1, "Total P&L", f"${metrics['total_pnl']:+,.0f}", positive=metrics["total_pnl"] >= 0)
                _metric_card(m2, "Win Rate", f"{metrics['win_rate']*100:.1f}%", positive=metrics["win_rate"] >= 0.55)
                _metric_card(m3, "Profit Factor", f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] else "N/A", positive=(metrics["profit_factor"] or 0) >= 1)
                _metric_card(m4, "Trades", f"{metrics['trades']:,}")

                m5, m6, m7, m8 = st.columns(4)
                _metric_card(m5, "Avg P&L", f"${metrics['avg_pnl']:+,.2f}", positive=metrics["avg_pnl"] >= 0)
                _metric_card(m6, "Max Drawdown", f"${metrics['max_drawdown']:,.0f}", positive=False)
                _metric_card(m7, "Sharpe Ratio", f"{metrics['sharpe']:.2f}", positive=metrics["sharpe"] >= 1)
                _metric_card(m8, "Avg Win / Loss", f"${metrics['avg_win']:,.0f} / ${metrics['avg_loss']:,.0f}")

                st.markdown("<br>", unsafe_allow_html=True)

                # Equity curve
                eq = cumulative_pnl_series(filtered)
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=eq["date"], y=eq["cumulative_pnl"],
                    mode="lines", fill="tozeroy",
                    line=dict(color=_COLORS["green"] if metrics["total_pnl"] >= 0 else _COLORS["red"], width=2),
                    fillcolor="rgba(0,200,150,0.1)" if metrics["total_pnl"] >= 0 else "rgba(255,75,110,0.1)",
                ))
                fig_eq.update_layout(
                    plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                    font_color=_COLORS["text"], height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
                    xaxis=dict(gridcolor="#1E2433"),
                    showlegend=False,
                    title="Backtest Equity Curve",
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                # Monthly breakdown for this backtest
                filtered_m = filtered.copy()
                filtered_m["month_str"] = pd.to_datetime(filtered_m["open_date"]).dt.strftime("%Y-%m")
                bt_monthly = filtered_m.groupby("month_str")["net_pnl"].sum().reset_index()
                fig_btm = go.Figure(go.Bar(
                    x=bt_monthly["month_str"], y=bt_monthly["net_pnl"],
                    marker_color=bt_monthly["net_pnl"].apply(lambda v: _COLORS["green"] if v >= 0 else _COLORS["red"]),
                ))
                fig_btm.update_layout(
                    plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                    font_color=_COLORS["text"], height=220,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(tickprefix="$", gridcolor="#1E2433"),
                    xaxis=dict(gridcolor="#1E2433"),
                    showlegend=False,
                    title="Monthly P&L (Backtest)",
                )
                st.plotly_chart(fig_btm, use_container_width=True)

# ===========================================================================
# TAB 5: TRADE EXPLORER
# ===========================================================================
with tab_explorer:
    st.title("Trade Explorer")

    # Filters
    ex1, ex2, ex3, ex4 = st.columns(4)
    ex_ticker = ex1.text_input("Ticker (leave blank for all)").upper().strip()
    ex_strategy = ex2.selectbox("Strategy", ["All"] + strategies, key="ex_strat")
    ex_result = ex3.selectbox("Result", ["All", "Wins only", "Losses only"], key="ex_res")
    ex_closed_by = ex4.selectbox("Close method", ["All", "expiration", "sell_to_close", "buy_to_close"], key="ex_close")

    view = closed_all.copy()
    if ex_ticker:
        view = view[view["ticker"] == ex_ticker]
    if ex_strategy != "All":
        view = view[view["strategy"] == ex_strategy]
    if ex_result == "Wins only":
        view = view[view["is_win"]]
    elif ex_result == "Losses only":
        view = view[~view["is_win"]]
    if ex_closed_by != "All":
        view = view[view["closed_by"] == ex_closed_by]

    view = view.sort_values("open_date", ascending=False)

    st.caption(f"{len(view):,} trades shown")

    display = view[[
        "open_date", "close_date", "account", "ticker", "opt_type", "direction",
        "strike", "expiry", "dte_at_open", "dte_bucket", "qty",
        "premium", "net_pnl", "closed_by", "n_legs",
    ]].copy()
    display["net_pnl"] = display["net_pnl"].apply(lambda v: f"${v:+,.2f}")
    display["premium"] = display["premium"].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
    display["open_date"] = pd.to_datetime(display["open_date"]).dt.strftime("%Y-%m-%d")
    display["close_date"] = pd.to_datetime(display["close_date"]).dt.strftime("%Y-%m-%d").replace("NaT", "—")
    display["expiry"] = pd.to_datetime(display["expiry"]).dt.strftime("%Y-%m-%d")

    st.dataframe(display, use_container_width=True, hide_index=True, height=700)

    csv = view.to_csv(index=False)
    st.download_button("Download filtered trades CSV", csv, "filtered_trades.csv", "text/csv")

# ===========================================================================
# TAB: EARNINGS CALENDAR
# ===========================================================================
with tab_earnings:
    from src.earnings import all_cached_earnings

    st.title("Earnings Calendar")
    st.caption(
        "Earnings dates for every ticker in your trade history. "
        "Run `python fetch_earnings.py` once to populate (or refresh weekly)."
    )

    earn_df = all_cached_earnings()

    if earn_df.empty:
        st.warning(
            "No earnings data in cache yet.  \n"
            "Run the fetcher from your terminal:  \n"
            "```\npython fetch_earnings.py\n```\n"
            "It takes ~20–30 minutes for all 786 tickers and only needs to run once a week."
        )
    else:
        today_d = pd.Timestamp.today().normalize()

        # ── Summary metrics ───────────────────────────────────────────────────
        n_with_next = earn_df["next_earnings"].notna().sum()
        n_week      = (earn_df["days_away"].fillna(999) <= 7).sum()
        n_month     = (earn_df["days_away"].fillna(999) <= 30).sum()
        n_total     = len(earn_df)

        mc1, mc2, mc3, mc4 = st.columns(4)
        _metric_card(mc1, "Tickers with Earnings Data", f"{n_total:,}")
        _metric_card(mc2, "Have Upcoming Date",         f"{n_with_next:,}")
        _metric_card(mc3, "Earnings This Week (≤7d)",   f"{n_week}",  positive=(n_week == 0))
        _metric_card(mc4, "Earnings This Month (≤30d)", f"{n_month}", positive=(n_month == 0))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Attach trade counts ───────────────────────────────────────────────
        trade_counts = (
            closed_all.groupby("ticker")
            .size()
            .reset_index(name="trade_count")
        )
        earn_df = earn_df.merge(trade_counts, on="ticker", how="left")
        earn_df["trade_count"] = earn_df["trade_count"].fillna(0).astype(int)

        # ── Filter controls ───────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        window_opt = fc1.selectbox(
            "Show earnings within",
            ["All tickers", "Next 7 days", "Next 30 days", "Next 90 days", "Upcoming only"],
        )
        traded_tickers = sorted(closed_all["ticker"].dropna().unique().tolist())
        show_traded = fc2.checkbox("Traded tickers only", value=False)
        top_n_opt = fc3.selectbox("Quick filter", ["All", "Top 10 traded", "Top 30 traded", "Top 50 traded"])
        search_ticker = fc4.text_input("Search ticker", "").upper().strip()

        sort_opt = st.radio(
            "Sort by",
            ["Days to earnings", "Most traded", "Ticker (A–Z)"],
            horizontal=True,
        )

        view = earn_df.copy()

        if show_traded:
            view = view[view["ticker"].isin(traded_tickers)]

        if search_ticker:
            view = view[view["ticker"].str.contains(search_ticker, na=False)]

        if window_opt == "Next 7 days":
            view = view[view["days_away"].fillna(999) <= 7]
        elif window_opt == "Next 30 days":
            view = view[view["days_away"].fillna(999) <= 30]
        elif window_opt == "Next 90 days":
            view = view[view["days_away"].fillna(999) <= 90]
        elif window_opt == "Upcoming only":
            view = view[view["next_earnings"].notna()]

        if top_n_opt != "All":
            n = int(top_n_opt.split()[1])
            top_tickers = (
                trade_counts.nlargest(n, "trade_count")["ticker"].tolist()
            )
            view = view[view["ticker"].isin(top_tickers)]

        if sort_opt == "Days to earnings":
            view = view.sort_values("days_away", na_position="last")
        elif sort_opt == "Most traded":
            view = view.sort_values("trade_count", ascending=False)
        else:
            view = view.sort_values("ticker")

        st.caption(f"{len(view):,} tickers shown")

        # ── Upcoming earnings chart (next 30 days) ────────────────────────────
        upcoming_chart = earn_df[earn_df["days_away"].fillna(999) <= 30].copy()
        if not upcoming_chart.empty:
            st.subheader("Earnings in the Next 30 Days")
            upcoming_chart["date_str"] = upcoming_chart["next_earnings"].astype(str)
            day_counts = (
                upcoming_chart.groupby("date_str")["ticker"]
                .apply(lambda x: ", ".join(sorted(x)))
                .reset_index()
                .rename(columns={"ticker": "tickers"})
            )
            day_counts["count"] = day_counts["tickers"].str.count(",") + 1

            fig_earn = go.Figure(go.Bar(
                x=day_counts["date_str"],
                y=day_counts["count"],
                text=day_counts["tickers"],
                textposition="outside",
                textfont=dict(size=9),
                marker_color=_COLORS["orange"],
            ))
            fig_earn.update_layout(
                plot_bgcolor=_COLORS["bg"], paper_bgcolor=_COLORS["bg"],
                font_color=_COLORS["text"], height=260,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(title="# Companies", gridcolor="#1E2433", dtick=1),
                xaxis=dict(gridcolor="#1E2433"),
                showlegend=False,
            )
            st.plotly_chart(fig_earn, use_container_width=True)

        # ── Table ─────────────────────────────────────────────────────────────
        st.subheader("Earnings Date Table")

        def _days_label(v) -> str:
            if pd.isna(v):
                return "—"
            d = int(v)
            if d == 0:
                return "🔴 Today"
            if d <= 3:
                return f"🔴 {d}d"
            if d <= 7:
                return f"🟠 {d}d"
            if d <= 30:
                return f"🟡 {d}d"
            return f"🟢 {d}d"

        tbl = view[["ticker", "trade_count", "next_earnings", "days_away", "last_earnings", "total_dates"]].copy()
        tbl["next_earnings"] = tbl["next_earnings"].apply(
            lambda v: str(v) if pd.notna(v) else "—"
        )
        tbl["last_earnings"] = tbl["last_earnings"].apply(
            lambda v: str(v) if pd.notna(v) else "—"
        )
        tbl["days_away"] = tbl["days_away"].apply(_days_label)
        tbl = tbl.rename(columns={
            "ticker": "Ticker",
            "trade_count": "Trades",
            "next_earnings": "Next Earnings",
            "days_away": "Days Away",
            "last_earnings": "Last Earnings",
            "total_dates": "Quarters Tracked",
        })

        st.dataframe(tbl, use_container_width=True, hide_index=True, height=600)

        # ── Cache info ────────────────────────────────────────────────────────
        cache_path = Path(".earnings_cache.json")
        if cache_path.exists():
            mtime = pd.Timestamp(cache_path.stat().st_mtime, unit="s")
            st.caption(f"Cache last updated: {mtime.strftime('%Y-%m-%d %H:%M')}")

# ===========================================================================
# TAB: 2023–2025 (Historical)
# ===========================================================================
with tab_hist:
    _render_period_tab(closed_hist, "2023–2025 (Historical)")

# ===========================================================================
# TAB: 2026
# ===========================================================================
with tab_2026:
    _render_period_tab(closed_2026, "2026 (Year to Date)")
