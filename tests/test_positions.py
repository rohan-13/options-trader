"""
Tests for parsing short-put Positions out of E*Trade's portfolio payload.

Fixture mirrors the PortfolioResponse → AccountPortfolio → Position shape
returned by ETrade.get_portfolio(). The parser must pick out ONLY short puts:
those are the positions the stop-loss monitor manages.
"""
from __future__ import annotations

from datetime import date

from src.risk import parse_short_puts


def _option_position(
    symbol="AAPL",
    call_put="PUT",
    qty=-1,
    price_paid=1.25,
    last_trade=1.10,
    strike=190.0,
):
    return {
        "positionId": 12345,
        "quantity": qty,
        "pricePaid": price_paid,
        "totalCost": price_paid * 100 * qty,
        "Product": {
            "symbol": symbol,
            "securityType": "OPTN",
            "callPut": call_put,
            "expiryYear": 2026,
            "expiryMonth": 6,
            "expiryDay": 12,
            "strikePrice": strike,
        },
        "Quick": {"lastTrade": last_trade},
    }


def _stock_position(symbol="NVDA"):
    return {
        "positionId": 99,
        "quantity": 100,
        "pricePaid": 120.0,
        "Product": {"symbol": symbol, "securityType": "EQ"},
        "Quick": {"lastTrade": 130.0},
    }


def _portfolio(positions):
    return [{"accountId": "1", "Position": positions}]


def test_parses_short_put_into_position():
    portfolio = _portfolio([_option_position()])
    positions = parse_short_puts(portfolio)
    assert len(positions) == 1
    p = positions[0]
    assert p.symbol == "AAPL"
    assert p.expiry == date(2026, 6, 12)
    assert p.strike == 190.0
    assert p.qty == 1                # short -1 reported as 1 contract short
    assert p.entry_price == 1.25
    assert p.current_price == 1.10


def test_skips_long_puts_stock_and_short_calls():
    portfolio = _portfolio([
        _option_position(qty=2),                  # long put — not ours to manage
        _stock_position(),                        # equity
        _option_position(call_put="CALL", qty=-1),  # short call — out of scope
        _option_position(symbol="META", qty=-3),  # the one real short put
    ])
    positions = parse_short_puts(portfolio)
    assert [p.symbol for p in positions] == ["META"]
    assert positions[0].qty == 3


def test_missing_quote_yields_none_current_price():
    pos = _option_position()
    del pos["Quick"]
    positions = parse_short_puts(_portfolio([pos]))
    assert positions[0].current_price is None


def test_empty_portfolio():
    assert parse_short_puts([]) == []
    assert parse_short_puts([{"accountId": "1"}]) == []
