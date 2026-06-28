"""
Parse option metadata from E*TRADE description strings and OCC symbols,
then enrich the raw transactions DataFrame with structured option fields.
"""

import re

import pandas as pd

# Description format: "CALL PLTR   01/03/25    78.000"
_DESC_RE = re.compile(
    r"^(CALL|PUT)\s+(\S+)\s+(\d{2}/\d{2}/\d{2})\s+([\d.]+)",
    re.IGNORECASE,
)

# OCC symbol: "NFLX--251226C00097000"
_OCC_RE = re.compile(r"^([A-Z]{1,6})-*(\d{6})([CP])(\d{8})$")


def _parse_description(desc: str) -> dict | None:
    m = _DESC_RE.match(desc.strip())
    if not m:
        return None
    try:
        expiry = pd.to_datetime(m.group(3), format="%m/%d/%y")
    except Exception:
        return None
    return {
        "opt_type": m.group(1).upper(),
        "ticker": m.group(2).upper(),
        "expiry": expiry,
        "strike": float(m.group(4)),
    }


def _parse_occ(occ: str) -> dict | None:
    m = _OCC_RE.match(occ.strip())
    if not m:
        return None
    try:
        expiry = pd.to_datetime("20" + m.group(2), format="%Y%m%d")
    except Exception:
        return None
    return {
        "opt_type": "CALL" if m.group(3) == "C" else "PUT",
        "ticker": m.group(1).rstrip("-"),
        "expiry": expiry,
        "strike": int(m.group(4)) / 1000.0,
    }


def enrich_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns: opt_type, ticker, expiry, strike, dte, is_expiration,
    contract_key for every option row.
    """
    df = df.copy()

    for col in ["opt_type", "ticker", "expiry", "strike", "dte", "is_expiration", "contract_key"]:
        df[col] = None

    option_mask = df["is_option"].fillna(False)
    if option_mask.sum() == 0:
        return df

    # Parse from description (works for both format variants)
    parsed_series = df.loc[option_mask, "description"].apply(_parse_description)
    valid = parsed_series.notna()
    valid_idx = parsed_series[valid].index
    valid_parsed = parsed_series[valid].tolist()

    df.loc[valid_idx, "opt_type"] = [p["opt_type"] for p in valid_parsed]
    df.loc[valid_idx, "ticker"] = [p["ticker"] for p in valid_parsed]
    df.loc[valid_idx, "expiry"] = pd.to_datetime([p["expiry"] for p in valid_parsed])
    df.loc[valid_idx, "strike"] = [p["strike"] for p in valid_parsed]

    # DTE at time of transaction (clipped to 0 for same-day expirations)
    df.loc[valid_idx, "dte"] = (
        pd.to_datetime(df.loc[valid_idx, "expiry"]) - df.loc[valid_idx, "date"]
    ).dt.days.clip(lower=0)

    # Expiration: option row with amount=0 and price=0 (expired worthless or assigned)
    df.loc[valid_idx, "is_expiration"] = (
        (df.loc[valid_idx, "amount"] == 0.0) & (df.loc[valid_idx, "price"] == 0.0)
    )

    # Canonical contract key for FIFO matching: account|ticker|YYYYMMDD|strike|CALL/PUT
    expiry_strs = pd.to_datetime(df.loc[valid_idx, "expiry"]).dt.strftime("%Y%m%d")
    df.loc[valid_idx, "contract_key"] = (
        df.loc[valid_idx, "account"].astype(str)
        + "|"
        + df.loc[valid_idx, "ticker"].astype(str)
        + "|"
        + expiry_strs.astype(str)
        + "|"
        + df.loc[valid_idx, "strike"].astype(str)
        + "|"
        + df.loc[valid_idx, "opt_type"].astype(str)
    )

    return df
