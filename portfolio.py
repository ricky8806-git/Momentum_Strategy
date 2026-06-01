"""Portfolio state management, position sizing, entry/exit logic, SPY sleeve."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import config


# ── State I/O ─────────────────────────────────────────────────────────────

def initial_state() -> dict:
    return {
        "holdings": {},
        "spy_shares": 0.0,
        "last_rebalance": None,
        "nav": 0.0,
    }


def load_state(path: str = config.STATE_FILE) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return initial_state()


def save_state(state: dict, path: str = config.STATE_FILE) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Position sizing ───────────────────────────────────────────────────────

def compute_entry_size(nav: float, n_total: int, price: float) -> float:
    """
    Return number of shares to buy.
    Allocation = min(MAX_POSITION_WEIGHT, 1/n_total) * NAV.
    Floor to whole shares.
    """
    if n_total <= 0 or price <= 0:
        return 0.0
    alloc_pct = min(config.MAX_POSITION_WEIGHT, 1.0 / n_total)
    dollar_alloc = alloc_pct * nav
    return math.floor(dollar_alloc / price)


# ── Exit logic ─────────────────────────────────────────────────────────────

def check_stop_losses(
    holdings: dict[str, Any],
    current_prices: dict[str, float],
) -> list[dict]:
    """
    Return list of {ticker, reason, price} for any position whose
    current price is < entry_price * (1 - STOP_LOSS_PCT).
    """
    exits = []
    for ticker, pos in holdings.items():
        price = current_prices.get(ticker)
        if price is None:
            continue
        threshold = pos["entry_price"] * (1 - config.STOP_LOSS_PCT)
        if price < threshold:
            exits.append({"ticker": ticker, "reason": "stop_loss", "price": price})
    return exits


def apply_exits(
    state: dict,
    exit_orders: list[dict],
) -> float:
    """
    Remove exited tickers from state["holdings"].
    Returns total cash freed by the exits.
    """
    cash = 0.0
    for order in exit_orders:
        ticker = order["ticker"]
        price  = order["price"]
        if ticker in state["holdings"]:
            shares = state["holdings"][ticker]["shares"]
            cash  += shares * price
            del state["holdings"][ticker]
    return cash


def apply_entries(
    state: dict,
    entries: list[dict],
    cash_available: float,
    n_total: int,
    entry_date: str,
) -> None:
    """
    Buy entries from available cash.  Each entry dict must have {ticker, price}.
    Cash is divided evenly across all new entries so later ones aren't starved.
    Modifies state["holdings"] in-place.
    """
    new_entries = [e for e in entries if e["ticker"] not in state["holdings"]]
    if not new_entries or cash_available <= 0:
        return
    per_entry_cash = cash_available / len(new_entries)
    for entry in new_entries:
        ticker = entry["ticker"]
        price  = entry["price"]
        shares = compute_entry_size(state["nav"], n_total, price)
        cost   = shares * price
        if cost > per_entry_cash:
            shares = math.floor(per_entry_cash / price)
            cost   = shares * price
        if shares <= 0:
            continue
        state["holdings"][ticker] = {
            "shares":      float(shares),
            "entry_price": float(price),
            "entry_date":  entry_date,
        }


# ── SPY sleeve ────────────────────────────────────────────────────────────

def adjust_spy_sleeve(state: dict, idle_cash: float, spy_price: float) -> None:
    """Set spy_shares so that idle_cash is invested in SPY (floored at 0)."""
    if spy_price <= 0:
        return
    state["spy_shares"] = max(0.0, idle_cash / spy_price)


def compute_idle_cash(state: dict, current_prices: dict[str, float]) -> float:
    """NAV minus market value of all equity holdings (including SPY sleeve)."""
    equity = sum(
        pos["shares"] * current_prices.get(ticker, pos["entry_price"])
        for ticker, pos in state["holdings"].items()
    )
    spy_value = state["spy_shares"] * current_prices.get("SPY", 0.0)
    return state["nav"] - equity - spy_value
