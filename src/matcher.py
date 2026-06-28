"""
FIFO trade matching: pairs open and close legs for each option contract,
calculates net P&L, and categorizes each completed round-trip trade.
"""

from __future__ import annotations

import pandas as pd


def _dte_bucket(dte: float | None) -> str:
    if dte is None or pd.isna(dte):
        return "unknown"
    d = int(dte)
    if d <= 7:
        return "0-7 DTE"
    if d <= 30:
        return "8-30 DTE"
    if d <= 90:
        return "31-90 DTE"
    return "90+ DTE"


def match_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FIFO matching on all option rows.

    Returns a DataFrame where each row is a completed (or still-open) round-trip
    option trade with net P&L.
    """
    opt = df[df["is_option"] & df["contract_key"].notna()].copy()
    opt = opt.sort_values(["contract_key", "date"]).reset_index(drop=True)

    completed: list[dict] = []

    for key, group in opt.groupby("contract_key", sort=False):
        account, ticker, expiry_str, strike_str, opt_type = key.split("|")
        strike = float(strike_str)

        # --- running state for this contract ---
        running_qty = 0.0
        open_amount = 0.0    # cash from opening legs (negative for long, positive for short)
        close_amount = 0.0   # cash from closing legs
        open_date: pd.Timestamp | None = None
        open_qty = 0.0
        first_direction: str | None = None
        leg_count = 0

        for _, row in group.iterrows():
            qty = float(row["qty"])
            amount = float(row["amount"])
            date = row["date"]
            is_exp = bool(row.get("is_expiration", False))
            dte = row.get("dte")

            # --- expiration closes whatever is open ---
            if is_exp and abs(running_qty) > 1e-6:
                completed.append(
                    _make_trade(
                        account, ticker, row["expiry"], strike, opt_type,
                        open_date, date, open_amount, close_amount,
                        open_qty, first_direction, leg_count + 1, "expiration", dte,
                    )
                )
                running_qty = 0.0
                open_amount = 0.0
                close_amount = 0.0
                open_date = None
                open_qty = 0.0
                first_direction = None
                leg_count = 0
                continue

            if abs(qty) < 1e-6:
                continue

            if abs(running_qty) < 1e-6:
                # Starting a new position
                open_date = date
                first_direction = "long" if qty > 0 else "short"
                open_qty = abs(qty)

            prev_qty = running_qty
            running_qty += qty
            leg_count += 1

            # Is this leg opening or closing?
            is_closing = (prev_qty != 0) and ((prev_qty > 0) != (qty > 0))
            if is_closing:
                close_amount += amount
            else:
                open_amount += amount

            # Position fully closed
            if abs(running_qty) < 1e-6:
                closed_by = "sell_to_close" if first_direction == "long" else "buy_to_close"
                completed.append(
                    _make_trade(
                        account, ticker, row["expiry"], strike, opt_type,
                        open_date, date, open_amount, close_amount,
                        open_qty, first_direction, leg_count, closed_by, dte,
                    )
                )
                running_qty = 0.0
                open_amount = 0.0
                close_amount = 0.0
                open_date = None
                open_qty = 0.0
                first_direction = None
                leg_count = 0

            elif prev_qty != 0 and (prev_qty * running_qty < 0):
                # Direction reversal — record prior position, start fresh
                close_frac = abs(prev_qty) / (abs(prev_qty) + abs(running_qty))
                split_close = close_amount * close_frac
                rem_close = close_amount - split_close

                closed_by = "sell_to_close" if first_direction == "long" else "buy_to_close"
                completed.append(
                    _make_trade(
                        account, ticker, row["expiry"], strike, opt_type,
                        open_date, date, open_amount, split_close,
                        open_qty, first_direction, leg_count, closed_by, dte,
                    )
                )
                open_amount = rem_close
                close_amount = 0.0
                open_date = date
                open_qty = abs(running_qty)
                first_direction = "long" if running_qty > 0 else "short"
                leg_count = 1

        # Still-open at end of data
        if abs(running_qty) > 1e-6:
            expiry_val = group["expiry"].iloc[-1]
            dte_val = group["dte"].iloc[-1]
            completed.append(
                _make_trade(
                    account, ticker, expiry_val, strike, opt_type,
                    open_date, None, open_amount, close_amount,
                    open_qty, first_direction, leg_count, "open", dte_val,
                )
            )

    trades = pd.DataFrame(completed)
    if trades.empty:
        return trades

    trades["is_closed"] = trades["closed_by"] != "open"
    trades["is_win"] = trades["net_pnl"] > 0
    trades["month"] = trades["open_date"].dt.to_period("M")
    trades["year"] = trades["open_date"].dt.year

    return trades.sort_values("open_date").reset_index(drop=True)


def _make_trade(
    account: str,
    ticker: str,
    expiry,
    strike: float,
    opt_type: str,
    open_date,
    close_date,
    open_amount: float,
    close_amount: float,
    qty: float,
    direction: str | None,
    n_legs: int,
    closed_by: str,
    dte_at_open,
) -> dict:
    net_pnl = open_amount + close_amount
    strategy = f"{opt_type}_{direction}" if direction else opt_type
    # For short trades: premium_collected = open_amount (positive)
    # For long trades:  premium_paid = abs(open_amount) (negative open_amount)
    premium = open_amount if direction == "short" else abs(open_amount)
    holding_days = None
    if open_date is not None and close_date is not None:
        holding_days = (close_date - open_date).days

    return {
        "account": account,
        "ticker": ticker,
        "expiry": expiry,
        "strike": strike,
        "opt_type": opt_type,
        "direction": direction,
        "strategy": strategy,
        "open_date": open_date,
        "close_date": close_date,
        "open_amount": round(open_amount, 2),
        "close_amount": round(close_amount, 2),
        "net_pnl": round(net_pnl, 2),
        "premium": round(premium, 2),
        "qty": qty,
        "n_legs": n_legs,
        "closed_by": closed_by,
        "dte_at_open": dte_at_open,
        "dte_bucket": _dte_bucket(dte_at_open),
        "holding_days": holding_days,
    }
