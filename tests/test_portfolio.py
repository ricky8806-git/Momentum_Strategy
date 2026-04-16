import json
import os
import tempfile
import pandas as pd
import pytest

import config
from portfolio import (
    load_state,
    save_state,
    initial_state,
    compute_entry_size,
    check_stop_losses,
    apply_exits,
    apply_entries,
    adjust_spy_sleeve,
)


EMPTY_STATE = {
    "holdings": {},
    "spy_shares": 0.0,
    "last_rebalance": None,
    "nav": 100_000.0,
}


def test_load_state_returns_fresh_when_file_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        state = load_state(path)
        assert state["holdings"] == {}
        assert state["nav"] == 0.0


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        state = dict(EMPTY_STATE)
        state["holdings"]["AAPL"] = {"shares": 10.0, "entry_price": 150.0, "entry_date": "2024-01-05"}
        save_state(state, path)
        loaded = load_state(path)
        assert loaded["holdings"]["AAPL"]["shares"] == 10.0


def test_compute_entry_size_uses_max_weight():
    """Entry size should not exceed MAX_POSITION_WEIGHT * NAV."""
    nav = 100_000.0
    n_total = 10
    price = 200.0
    shares = compute_entry_size(nav, n_total, price)
    cost = shares * price
    assert cost <= config.MAX_POSITION_WEIGHT * nav + price  # allow rounding up by 1 share


def test_compute_entry_size_uses_equal_weight():
    """When 100%/n_total < MAX_POSITION_WEIGHT, use 100%/n_total."""
    nav = 100_000.0
    n_total = 20   # 1/20 = 5% < 7%
    price = 100.0
    shares = compute_entry_size(nav, n_total, price)
    expected_weight = 1.0 / n_total
    cost = shares * price
    assert cost <= expected_weight * nav + price


def test_check_stop_losses_flags_below_threshold():
    """Position down 11% should be flagged."""
    holdings = {
        "AAPL": {"shares": 10.0, "entry_price": 100.0, "entry_date": "2024-01-02"},
        "MSFT": {"shares": 5.0,  "entry_price": 100.0, "entry_date": "2024-01-02"},
    }
    current_prices = {"AAPL": 88.0, "MSFT": 95.0}  # AAPL down 12%, MSFT down 5%
    exits = check_stop_losses(holdings, current_prices)
    tickers = [e["ticker"] for e in exits]
    assert "AAPL" in tickers
    assert "MSFT" not in tickers


def test_apply_exits_removes_holdings_and_returns_cash():
    state = {
        "holdings": {
            "AAPL": {"shares": 10.0, "entry_price": 100.0, "entry_date": "2024-01-02"},
            "MSFT": {"shares": 5.0,  "entry_price": 100.0, "entry_date": "2024-01-02"},
        },
        "spy_shares": 0.0,
        "nav": 100_000.0,
        "last_rebalance": None,
    }
    exit_orders = [{"ticker": "AAPL", "reason": "stop_loss", "price": 88.0}]
    cash_freed = apply_exits(state, exit_orders)
    assert "AAPL" not in state["holdings"]
    assert "MSFT" in state["holdings"]
    assert cash_freed == pytest.approx(10.0 * 88.0)


def test_apply_entries_deploys_cash():
    state = {
        "holdings": {},
        "spy_shares": 0.0,
        "nav": 100_000.0,
        "last_rebalance": None,
    }
    cash_available = 50_000.0
    entries = [
        {"ticker": "NVDA", "price": 500.0},
        {"ticker": "META", "price": 400.0},
    ]
    apply_entries(state, entries, cash_available, n_total=2, entry_date="2024-01-08")
    assert "NVDA" in state["holdings"]
    assert "META" in state["holdings"]
    assert state["holdings"]["NVDA"]["shares"] > 0


def test_adjust_spy_sleeve_sets_shares():
    state = {
        "holdings": {"AAPL": {"shares": 10.0, "entry_price": 150.0, "entry_date": "2024-01-02"}},
        "spy_shares": 0.0,
        "nav": 100_000.0,
        "last_rebalance": None,
    }
    equity_value = 10.0 * 155.0  # AAPL at 155
    idle_cash = state["nav"] - equity_value
    spy_price = 480.0
    adjust_spy_sleeve(state, idle_cash, spy_price)
    expected_shares = idle_cash / spy_price
    assert state["spy_shares"] == pytest.approx(expected_shares, rel=0.01)
