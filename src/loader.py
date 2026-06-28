import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

_OCC_RE = re.compile(r"^[A-Z]{1,6}-*\d{6}[CP]\d{8}$")


def _extract_account(header_block: str) -> str:
    # Old format: "#####2563"
    m = re.search(r"#####(\d+)", header_block)
    if m:
        return m.group(1)
    # New format: "Account Activity for Joint JTWROS -2563 from ..."
    m = re.search(r"JTWROS\s+-(\d+)", header_block)
    if m:
        return m.group(1)
    return "unknown"


def _to_float(s: str) -> float:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return 0.0


# ── Old format: header starts with "TransactionDate" ──────────────────────────
# Columns (9, comma-split 8): date | txtype | sectype | symbol | qty | amount | price | comm | desc

def _parse_old_format(lines: list[str], account: str, header_idx: int, path: Path) -> list[dict]:
    rows = []
    for raw in lines[header_idx + 1:]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue

        date_s, txtype, sectype, symbol, qty_s, amount_s, price_s, comm_s, desc = parts
        txtype = txtype.strip()
        sectype = sectype.strip()
        symbol = symbol.strip()
        desc = desc.strip()

        try:
            date = pd.to_datetime(date_s.strip(), format="%m/%d/%y")
        except Exception:
            continue

        qty    = _to_float(qty_s)
        amount = _to_float(amount_s)
        price  = _to_float(price_s)
        commission = _to_float(comm_s)

        is_new_fmt = bool(_OCC_RE.match(txtype))
        is_option  = desc.upper().startswith("CALL") or desc.upper().startswith("PUT")

        if is_new_fmt:
            sec_type_clean = symbol
            occ_symbol     = txtype
        else:
            sec_type_clean = sectype
            occ_symbol     = None

        is_equity = (sec_type_clean.strip() == "EQ") and not is_option

        rows.append({
            "date":         date,
            "account":      account,
            "source_file":  path.name,
            "txtype_raw":   txtype,
            "security_type": sec_type_clean,
            "occ_symbol":   occ_symbol,
            "symbol_raw":   symbol if not is_new_fmt else "",
            "qty":          qty,
            "amount":       amount,
            "price":        price,
            "commission":   commission,
            "description":  desc,
            "is_option":    is_option,
            "is_equity":    is_equity,
            "is_new_format": is_new_fmt,
        })
    return rows


# ── New format: header starts with "Activity/Trade Date" ──────────────────────
# Columns (13): trade_date | txn_date | settle_date | activity_type | description
#               | symbol | cusip | qty | price | amount | commission | category | note
#
# Activity Type values for options:
#   "Sold Short"      → open a short position (qty < 0, amount > 0 = credit)
#   "Bought To Cover" → close a short position (qty > 0, amount < 0 = debit)
#   "Bought"          → open a long position   (qty > 0, amount < 0 = debit)
#   "Sold"            → close a long position  (qty < 0, amount > 0 = credit)
#   "Option Expired"  → expiry (amount = 0, price = 0)

def _parse_new_format(lines: list[str], account: str, header_idx: int, path: Path) -> list[dict]:
    rows = []
    for raw in lines[header_idx + 1:]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",", 12)
        if len(parts) < 10:
            continue

        date_s        = parts[0].strip()
        activity_type = parts[3].strip()
        desc          = parts[4].strip()
        symbol        = parts[5].strip()
        qty_s         = parts[7].strip()
        price_s       = parts[8].strip()
        amount_s      = parts[9].strip()
        comm_s        = parts[10].strip() if len(parts) > 10 else "0"

        try:
            date = pd.to_datetime(date_s, format="%m/%d/%y")
        except Exception:
            continue

        qty        = _to_float(qty_s)
        amount     = _to_float(amount_s)
        price      = _to_float(price_s)
        commission = _to_float(comm_s)

        is_option = desc.upper().startswith("CALL") or desc.upper().startswith("PUT")

        # Equity: a real symbol, a buy/sell activity, and not an option description
        is_equity = (
            not is_option
            and symbol not in ("--", "", "OPTN")
            and activity_type in ("Bought", "Sold", "Sold Short", "Bought To Cover")
        )

        rows.append({
            "date":          date,
            "account":       account,
            "source_file":   path.name,
            "txtype_raw":    activity_type,
            "security_type": "OPTN" if is_option else ("EQ" if is_equity else ""),
            "occ_symbol":    None,
            "symbol_raw":    symbol if symbol != "--" else "",
            "qty":           qty,
            "amount":        amount,
            "price":         price,
            "commission":    commission,
            "description":   desc,
            "is_option":     is_option,
            "is_equity":     is_equity,
            "is_new_format": False,
        })
    return rows


# ── Router ────────────────────────────────────────────────────────────────────

def _parse_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return []

    # Search first 10 lines for account info (new format has it on line 3)
    header_block = "".join(lines[:10])
    account = _extract_account(header_block)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("TransactionDate"):
            return _parse_old_format(lines, account, i, path)
        if stripped.startswith("Activity/Trade Date"):
            return _parse_new_format(lines, account, i, path)

    return []


def load_all() -> pd.DataFrame:
    """Load and unify all E*TRADE CSV exports into a single DataFrame."""
    all_rows: list[dict] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        rows = _parse_file(path)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(f"No data loaded from {DATA_DIR}")

    df = pd.DataFrame(all_rows)
    df = df.sort_values("date").reset_index(drop=True)

    df = df.drop_duplicates(
        subset=["date", "account", "qty", "amount", "description"]
    ).reset_index(drop=True)

    return df
