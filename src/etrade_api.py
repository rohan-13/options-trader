"""
E*Trade API client — OAuth 1.0a authentication + market data + order execution.

Setup:
  1. Register at developer.etrade.com → get consumer key & secret
  2. Copy .env.example → .env and fill in your credentials
  3. Call api.authenticate() once per day (access tokens expire at midnight ET)
"""
from __future__ import annotations

import os
import uuid
import webbrowser
from datetime import date
from typing import Any

from requests_oauthlib import OAuth1Session

PROD_BASE = "https://api.etrade.com"
SB_BASE   = "https://apisb.etrade.com"
AUTH_URL  = "https://us.etrade.com/e/t/etws/authorize"


class ETrade:
    def __init__(self, sandbox: bool = False) -> None:
        self.consumer_key    = os.environ["ETRADE_CONSUMER_KEY"]
        self.consumer_secret = os.environ["ETRADE_CONSUMER_SECRET"]
        self.base_url        = SB_BASE if sandbox else PROD_BASE
        self._session: OAuth1Session | None = None

    # ── Authentication ────────────────────────────────────────────────────────

    def authenticate(self) -> None:
        """Full OAuth 1.0a flow: opens browser for user authorization."""
        # Step 1: get a request token
        oauth = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            callback_uri="oob",
        )
        r = oauth.fetch_request_token(f"{self.base_url}/v1/oauth/request_token")
        request_token  = r["oauth_token"]
        request_secret = r["oauth_token_secret"]

        # Step 2: user authorizes in browser
        auth_url = f"{AUTH_URL}?key={self.consumer_key}&token={request_token}"
        print(f"\nAuthorize E*Trade access at:\n  {auth_url}\n")
        webbrowser.open(auth_url)
        verifier = input("Enter the PIN/verifier from E*Trade: ").strip()

        # Step 3: exchange for access token
        oauth = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=request_token,
            resource_owner_secret=request_secret,
            verifier=verifier,
        )
        tokens = oauth.fetch_access_token(f"{self.base_url}/v1/oauth/access_token")
        self._build_session(tokens["oauth_token"], tokens["oauth_token_secret"])
        print("Authentication successful.\n")

    def load_tokens(self, access_token: str, access_secret: str) -> None:
        """Load previously saved tokens to skip the browser flow."""
        self._build_session(access_token, access_secret)

    def _build_session(self, token: str, secret: str) -> None:
        self._session = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=token,
            resource_owner_secret=secret,
        )

    def _require_session(self) -> OAuth1Session:
        if self._session is None:
            raise RuntimeError("Call authenticate() or load_tokens() first.")
        return self._session

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> Any:
        r = self._require_session().get(
            f"{self.base_url}{path}",
            params=params,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        r = self._require_session().post(
            f"{self.base_url}{path}",
            json=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.json()

    # ── Account info ──────────────────────────────────────────────────────────

    def get_accounts(self) -> list[dict]:
        data = self._get("/v1/accounts/list")
        return data["AccountListResponse"]["Accounts"]["Account"]

    def get_portfolio(self, account_key: str) -> list[dict]:
        data = self._get(f"/v1/accounts/{account_key}/portfolio")
        return data.get("PortfolioResponse", {}).get("AccountPortfolio", [])

    def get_balance(self, account_key: str) -> dict:
        return self._get(f"/v1/accounts/{account_key}/balance",
                         params={"instType": "BROKERAGE", "realTimeNAV": "true"})

    # ── Market data ───────────────────────────────────────────────────────────

    def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        """Return a {symbol: quote_dict} mapping."""
        joined = ",".join(symbols)
        data = self._get(f"/v1/market/quote/{joined}")
        quotes = data["QuoteResponse"]["QuoteData"]
        if isinstance(quotes, dict):
            quotes = [quotes]
        return {q["Product"]["symbol"]: q for q in quotes}

    def get_expiry_dates(self, symbol: str) -> list[date]:
        """Return available expiry dates for a symbol (weekly + monthly)."""
        data = self._get("/v1/market/optionexpiredate", params={"symbol": symbol})
        dates = []
        for item in data.get("OptionExpireDateResponse", {}).get("ExpirationDate", []):
            try:
                dates.append(date(int(item["year"]), int(item["month"]), int(item["day"])))
            except (KeyError, ValueError):
                pass
        return sorted(dates)

    def get_option_chain(
        self,
        symbol: str,
        expiry: date,
        chain_type: str = "PUT",
        strikes_near: float | None = None,
        n_strikes: int = 10,
    ) -> list[dict]:
        """Return a list of option contracts for the given expiry."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "expiryYear": expiry.year,
            "expiryMonth": expiry.month,
            "expiryDay": expiry.day,
            "chainType": chain_type,
            "noOfStrikes": n_strikes,
            "includeWeekly": "y",
            "skipAdjusted": "y",
            "optionCategory": "STANDARD",
        }
        if strikes_near is not None:
            params["strikePriceNear"] = strikes_near

        data = self._get("/v1/market/optionchains", params=params)
        pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])
        options = []
        for pair in pairs:
            opt = pair.get("Put") or pair.get("Call")
            if opt:
                options.append(opt)
        return options

    # ── Orders ────────────────────────────────────────────────────────────────

    def preview_order(self, account_key: str, order: dict) -> dict:
        return self._post(
            f"/v1/accounts/{account_key}/orders/preview",
            {"PreviewOrderRequest": order},
        )

    def place_order(self, account_key: str, order: dict, preview_id: int) -> dict:
        payload = dict(order)
        payload["previewId"] = preview_id
        return self._post(
            f"/v1/accounts/{account_key}/orders/place",
            {"PlaceOrderRequest": payload},
        )

    def cancel_order(self, account_key: str, order_id: int) -> dict:
        return self._post(
            f"/v1/accounts/{account_key}/orders/cancel",
            {"CancelOrderRequest": {"orderId": order_id}},
        )

    def get_orders(self, account_key: str, status: str = "OPEN") -> list[dict]:
        data = self._get(
            f"/v1/accounts/{account_key}/orders",
            params={"status": status},
        )
        orders = data.get("OrdersResponse", {}).get("Order", [])
        return orders if isinstance(orders, list) else [orders]

    # ── Order builder ─────────────────────────────────────────────────────────

    @staticmethod
    def build_sell_put_order(
        symbol: str,
        expiry: date,
        strike: float,
        qty: int,
        limit_price: float,
    ) -> dict:
        """Build a SELL_OPEN PUT order (net credit limit)."""
        return {
            "orderType": "OPTN",
            "clientOrderId": str(uuid.uuid4())[:20],
            "Order": [{
                "priceType": "NET_CREDIT",
                "limitPrice": round(limit_price, 2),
                "term": "GOOD_FOR_DAY",
                "marketSession": "REGULAR",
                "Instrument": [{
                    "Product": {
                        "securityType": "OPTN",
                        "symbol": symbol,
                        "callPut": "PUT",
                        "expiryYear": expiry.year,
                        "expiryMonth": expiry.month,
                        "expiryDay": expiry.day,
                        "strikePrice": strike,
                    },
                    "orderAction": "SELL_OPEN",
                    "quantityType": "QUANTITY",
                    "quantity": float(qty),
                }],
            }],
        }

    @staticmethod
    def build_buy_to_close_order(
        symbol: str,
        expiry: date,
        strike: float,
        qty: int,
        limit_price: float,
    ) -> dict:
        """Build a BUY_TO_CLOSE PUT order (stop-loss)."""
        return {
            "orderType": "OPTN",
            "clientOrderId": str(uuid.uuid4())[:20],
            "Order": [{
                "priceType": "NET_DEBIT",
                "limitPrice": round(limit_price, 2),
                "term": "GOOD_FOR_DAY",
                "marketSession": "REGULAR",
                "Instrument": [{
                    "Product": {
                        "securityType": "OPTN",
                        "symbol": symbol,
                        "callPut": "PUT",
                        "expiryYear": expiry.year,
                        "expiryMonth": expiry.month,
                        "expiryDay": expiry.day,
                        "strikePrice": strike,
                    },
                    "orderAction": "BUY_TO_CLOSE",
                    "quantityType": "QUANTITY",
                    "quantity": float(qty),
                }],
            }],
        }
