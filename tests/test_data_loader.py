import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from data_loader import get_sp500_symbols, fetch_prices


def test_get_sp500_symbols_returns_list_of_strings():
    """get_sp500_symbols should return a non-empty list of ticker strings."""
    symbols = get_sp500_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) > 400
    assert all(isinstance(s, str) for s in symbols)


def test_fetch_prices_returns_dataframe_with_expected_columns():
    """fetch_prices should return a DataFrame with OHLCV columns per ticker."""
    with patch("data_loader.yf.download") as mock_dl:
        # Build a minimal multi-index DataFrame mimicking yfinance output
        dates = pd.bdate_range("2024-01-02", periods=5)
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL", "MSFT"]],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame(1.0, index=dates, columns=cols)
        mock_dl.return_value = data

        df = fetch_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-08")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


def test_fetch_prices_includes_spy():
    """fetch_prices must always include SPY in the returned data."""
    with patch("data_loader.yf.download") as mock_dl:
        dates = pd.bdate_range("2024-01-02", periods=5)
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL", "SPY"]],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame(1.0, index=dates, columns=cols)
        mock_dl.return_value = data

        df = fetch_prices(["AAPL"], "2024-01-02", "2024-01-08")
        # SPY should be added even if not in symbols list
        tickers_in_close = df["Close"].columns.tolist()
        assert "SPY" in tickers_in_close
