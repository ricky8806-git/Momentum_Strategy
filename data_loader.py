"""Fetch S&P 500 universe and OHLC prices via yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

import config


def get_sp500_symbols() -> list[str]:
    """Return list of current S&P 500 tickers from GitHub CSV."""
    df = pd.read_csv(config.SP500_URL)
    # Column is 'Symbol' in the constituents CSV
    symbols = df["Symbol"].dropna().str.strip().str.upper().tolist()
    # yfinance uses '-' not '.' for some tickers (e.g. BRK.B -> BRK-B)
    symbols = [s.replace(".", "-") for s in symbols]
    return symbols


def fetch_prices(
    symbols: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download daily OHLCV for *symbols* + SPY between *start* and *end*.

    Returns a DataFrame with a MultiIndex on columns: (field, ticker).
    Fields: Open, High, Low, Close, Volume.
    """
    tickers = list(dict.fromkeys(symbols + ["SPY"]))  # deduplicate, keep SPY
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # yfinance returns a MultiIndex DataFrame when multiple tickers requested
    if not isinstance(raw.columns, pd.MultiIndex):
        # Single-ticker edge case — wrap it
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])
    return raw
