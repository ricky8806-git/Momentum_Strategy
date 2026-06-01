"""
Thin wrapper around Alpaca paper-trading API.
Loads credentials from environment variables via python-dotenv.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


def _get_client() -> "TradingClient":
    if not _ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py not installed. Run: pip install alpaca-py")
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    base_url   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    paper = "paper-api" in base_url
    return TradingClient(api_key, secret_key, paper=paper)


def submit_market_order(
    ticker: str,
    side: str,           # "buy" or "sell"
    qty: float,
    dry_run: bool = False,
) -> dict:
    """
    Submit a market day order.  Returns dict with order_id and status.
    If dry_run=True, logs intent and returns a fake order dict.
    """
    if dry_run:
        print(f"[DRY-RUN] {side.upper()} {qty} {ticker} @ market")
        return {"order_id": "DRY-RUN", "status": "pending", "ticker": ticker,
                "side": side, "qty": qty}

    client = _get_client()
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=side_enum,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(req)
    return {
        "order_id": str(order.id),
        "status":   str(order.status),
        "ticker":   ticker,
        "side":     side,
        "qty":      qty,
    }


def get_account_nav(dry_run: bool = False) -> float:
    """Return current portfolio equity from Alpaca account."""
    if dry_run:
        return 100_000.0
    client = _get_client()
    account = client.get_account()
    return float(account.equity)


def get_account_cash(dry_run: bool = False) -> float:
    """Return current cash balance from Alpaca account."""
    if dry_run:
        return 0.0
    client = _get_client()
    account = client.get_account()
    return float(account.cash)


def get_position_qty(ticker: str, dry_run: bool = False) -> float:
    """Return actual shares held for ticker; 0.0 if no position exists."""
    if dry_run:
        return 0.0
    client = _get_client()
    try:
        pos = client.get_open_position(ticker)
        return float(pos.qty)
    except Exception:
        return 0.0
