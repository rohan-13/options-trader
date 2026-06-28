"""
Live option quote fetcher using yfinance.
Provides per-position unrealized P&L, Greeks (theta via Black-Scholes), and IV.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

_YF_OK = False
try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    pass

_SCIPY_OK = False
try:
    from scipy.stats import norm as _norm
    _SCIPY_OK = True
except ImportError:
    pass


def build_occ_symbol(ticker: str, expiry: pd.Timestamp, opt_type: str, strike: float) -> str:
    cp = "C" if opt_type.upper() == "CALL" else "P"
    return f"{ticker}{expiry.strftime('%y%m%d')}{cp}{int(round(strike * 1000)):08d}"


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _bs_theta(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> Optional[float]:
    """Black-Scholes theta per calendar day, per share. Negative = option loses value."""
    if not _SCIPY_OK or T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    sqT = math.sqrt(T)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
    except (ValueError, ZeroDivisionError):
        return None
    d2 = d1 - sigma * sqT
    n_pdf = _norm.pdf(d1)
    if opt_type.upper() == "CALL":
        theta_annual = (-S * n_pdf * sigma / (2 * sqT)
                        - r * K * math.exp(-r * T) * _norm.cdf(d2))
    else:
        theta_annual = (-S * n_pdf * sigma / (2 * sqT)
                        + r * K * math.exp(-r * T) * _norm.cdf(-d2))
    return round(theta_annual / 365, 4)


def fetch_quotes(open_df: pd.DataFrame, rfr: float = 0.045) -> pd.DataFrame:
    """
    Enrich an open-positions DataFrame with live yfinance market data.

    Adds columns:
      dte_now          — calendar days until expiry (from today)
      stock_price      — current underlying price
      current_price    — option mid-price (bid+ask)/2, fallback to lastPrice
      bid / ask        — live bid / ask
      iv               — implied volatility (decimal)
      unrealized_pnl   — estimated P&L if closed at current mid
      pct_max_profit   — % of max premium captured (short positions only)
      theta_per_day    — BS theta per share per calendar day (negative)
      theta_dollar_day — portfolio dollar theta/day for this position
      quote_status     — "ok" | "expired" | "fetch_failed" | "strike_not_found" | ...
    """
    _BLANKS: dict = dict(
        dte_now=None, stock_price=None, current_price=None,
        bid=None, ask=None, iv=None, unrealized_pnl=None,
        pct_max_profit=None, theta_per_day=None, theta_dollar_day=None,
        quote_status="no_data",
    )

    if open_df.empty:
        return open_df.assign(**_BLANKS)
    if not _YF_OK:
        return open_df.assign(**{**_BLANKS, "quote_status": "install_yfinance"})

    today = pd.Timestamp.now().normalize()
    df = open_df.copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["_expiry_str"] = df["expiry"].dt.strftime("%Y-%m-%d")
    df["dte_now"] = (df["expiry"] - today).dt.days.clip(lower=0).astype(int)

    # Identify which rows are active — skip all network calls for expired positions
    active_mask = df["expiry"] >= today
    active_df = df[active_mask]

    # ---- stock prices: only for tickers with at least one active position ----
    stock_prices: dict[str, Optional[float]] = {}
    for ticker in active_df["ticker"].dropna().unique():
        try:
            stock_prices[ticker] = float(yf.Ticker(str(ticker)).fast_info["lastPrice"])
        except Exception:
            stock_prices[ticker] = None

    # ---- option chains: only for future (ticker, expiry) pairs ----
    chains: dict[tuple[str, str], Optional[dict]] = {}
    for (ticker, expiry_str), _ in active_df.groupby(["ticker", "_expiry_str"]):
        key = (str(ticker), str(expiry_str))
        try:
            ch = yf.Ticker(str(ticker)).option_chain(str(expiry_str))
            chains[key] = {
                "calls": ch.calls.set_index("strike"),
                "puts": ch.puts.set_index("strike"),
            }
        except Exception:
            chains[key] = None

    # ---- per-row enrichment ----
    rows = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        expiry_str = str(row["_expiry_str"])
        strike = float(row["strike"])
        opt_type = str(row.get("opt_type") or "").upper()
        direction = str(row.get("direction") or "short").lower()
        open_amount = float(row.get("open_amount") or 0)
        qty = float(row.get("qty") or 1)
        dte_now = int(row["dte_now"])
        S = stock_prices.get(ticker)

        bid = ask = current_price = iv = None
        status = "ok"

        if pd.to_datetime(expiry_str) < today:
            status = "expired"
        else:
            chain_data = chains.get((ticker, expiry_str))
            if chain_data is None:
                status = "fetch_failed"
            else:
                side = chain_data["calls"] if opt_type == "CALL" else chain_data["puts"]
                idx = side.index.astype(float)
                close = idx[abs(idx - strike) < 0.011]
                if len(close) == 0:
                    status = "strike_not_found"
                else:
                    r = side.loc[close[0]]
                    bid = _safe_float(r.get("bid"))
                    ask = _safe_float(r.get("ask"))
                    lp = _safe_float(r.get("lastPrice"))
                    iv = _safe_float(r.get("impliedVolatility"))
                    if bid is not None and ask is not None and ask > 0:
                        current_price = (bid + ask) / 2.0
                    elif lp is not None:
                        current_price = lp
                    status = "ok" if current_price is not None else "no_quote"

        # unrealized P&L
        upnl = pct_mp = None
        if current_price is not None:
            val_today = current_price * 100.0 * qty
            if direction == "short":
                upnl = round(open_amount - val_today, 2)
                if open_amount > 0:
                    pct_mp = round(upnl / open_amount * 100.0, 1)
            else:
                upnl = round(val_today - abs(open_amount), 2)

        # theta
        theta_share = None
        theta_dollar = None
        if S and iv and dte_now > 0:
            theta_share = _bs_theta(S, strike, dte_now / 365.0, rfr, iv, opt_type)
            if theta_share is not None:
                # Dollar theta benefit/day: negative theta_share is good for shorts
                dir_mult = -1.0 if direction == "short" else 1.0
                theta_dollar = round(dir_mult * theta_share * 100.0 * qty, 2)

        rows.append({
            **row.to_dict(),
            "dte_now": dte_now,
            "stock_price": S,
            "current_price": current_price,
            "bid": bid,
            "ask": ask,
            "iv": iv,
            "unrealized_pnl": upnl,
            "pct_max_profit": pct_mp,
            "theta_per_day": theta_share,
            "theta_dollar_day": theta_dollar,
            "quote_status": status,
        })

    return pd.DataFrame(rows).drop(columns=["_expiry_str"], errors="ignore")
