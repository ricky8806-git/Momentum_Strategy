"""
Smoke test: backtest runs on a short window without crashing.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from backtest import run_backtest


def _make_mock_prices(tickers, n_days=400):
    """Create synthetic price data for testing."""
    rng = pd.bdate_range("2022-01-03", periods=n_days)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    data = {}
    for field in fields:
        for t in tickers:
            if field == "Volume":
                data[(field, t)] = np.full(n_days, 1_000_000.0)
            elif field == "High":
                base = 150 + np.arange(n_days) * 0.3
                data[(field, t)] = base * 1.01
            elif field == "Low":
                base = 150 + np.arange(n_days) * 0.3
                data[(field, t)] = base * 0.99
            else:
                data[(field, t)] = 150 + np.arange(n_days) * 0.3
    return pd.DataFrame(data, index=rng, columns=cols)


def test_backtest_runs_without_error():
    """run_backtest should complete and return a DataFrame with nav column."""
    tickers = [f"STK{i:02d}" for i in range(20)] + ["SPY"]
    mock_prices = _make_mock_prices(tickers)

    with patch("backtest.data_loader.get_sp500_symbols", return_value=tickers[:-1]), \
         patch("backtest.data_loader.fetch_prices",      return_value=mock_prices):
        results = run_backtest(
            start="2022-09-01",
            end="2023-01-31",
            initial_nav=100_000.0,
        )

    assert isinstance(results, pd.DataFrame)
    assert "nav" in results.columns
    assert "n_holdings" in results.columns
    assert len(results) > 0


def test_backtest_nav_never_negative():
    tickers = [f"STK{i:02d}" for i in range(20)] + ["SPY"]
    mock_prices = _make_mock_prices(tickers)

    with patch("backtest.data_loader.get_sp500_symbols", return_value=tickers[:-1]), \
         patch("backtest.data_loader.fetch_prices",      return_value=mock_prices):
        results = run_backtest(
            start="2022-09-01",
            end="2023-01-31",
            initial_nav=100_000.0,
        )
    assert (results["nav"] >= 0).all()
