"""Compute indicators, apply hard filters, and rank universe."""

from __future__ import annotations

import pandas as pd
import numpy as np

import config


# ── Per-ticker indicators ─────────────────────────────────────────────────

def compute_indicators(close: pd.Series) -> pd.DataFrame:
    """
    Given a close-price Series (DatetimeIndex, business days), return a
    DataFrame with one row per date containing all indicator columns.
    """
    df = pd.DataFrame({"close": close})

    df["ma50"]  = close.rolling(config.MA_SHORT).mean()
    df["ma100"] = close.rolling(config.MA_MID).mean()
    df["ma200"] = close.rolling(config.MA_LONG).mean()

    hi20 = close.rolling(config.RANGE_WINDOW).max()
    lo20 = close.rolling(config.RANGE_WINDOW).min()
    rng  = hi20 - lo20
    df["range_pos"] = np.where(rng > 0, (close - lo20) / rng, np.nan)
    df["high20"]    = hi20

    return df


def apply_hard_filters(row: pd.Series) -> bool:
    """
    Return True if all 5 hard filters pass for the given indicator row.

    row must contain: close, ma100, ma200, ma50, range_pos, high20
    """
    if pd.isna([row["ma50"], row["ma100"], row["ma200"],
                row["range_pos"], row["high20"]]).any():
        return False

    checks = [
        row["close"] > row["ma100"],                              # filter 1
        row["close"] > row["ma200"],                              # filter 2
        row["ma50"]  > row["ma200"],                              # filter 3
        row["range_pos"] >= config.RANGE_POS_MIN,                 # filter 4
        row["close"] >= config.HIGH20_MIN_PCT * row["high20"],    # filter 5
    ]
    return all(checks)


def compute_momentum_score(close: pd.Series) -> float:
    """
    Return 0.60 × 126d_return + 0.40 × 63d_return.
    Returns NaN if insufficient history.
    """
    if len(close) < config.MOM_WINDOW_LONG + 1:
        return float("nan")
    ret_long  = close.iloc[-1] / close.iloc[-(config.MOM_WINDOW_LONG + 1)] - 1
    ret_short = close.iloc[-1] / close.iloc[-(config.MOM_WINDOW_SHORT + 1)] - 1
    return config.MOM_WEIGHT_LONG * ret_long + config.MOM_WEIGHT_SHORT * ret_short


# ── Cross-sectional ranking ───────────────────────────────────────────────

def get_eligible_tickers(
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    For each ticker in prices_df["Close"] (or prices_df directly if it's a plain DataFrame),
    evaluate hard filters on *as_of_date*.
    Return DataFrame with columns [ticker, score, passes_filter].
    """
    # Handle both MultiIndex (field, ticker) and plain (ticker) DataFrames
    if isinstance(prices_df.columns, pd.MultiIndex):
        close_all = prices_df["Close"]
    else:
        close_all = prices_df

    results = []
    for ticker in close_all.columns:
        series = close_all[ticker].dropna()
        if as_of_date not in series.index:
            continue
        series = series.loc[:as_of_date]
        if len(series) < config.MA_LONG + 5:
            continue
        ind  = compute_indicators(series)
        last = ind.loc[as_of_date]
        passes = apply_hard_filters(last)
        score  = compute_momentum_score(series) if passes else float("nan")
        results.append({"ticker": ticker, "score": score,
                         "passes_filter": passes})
    return pd.DataFrame(results)


def rank_universe(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank eligible tickers as of the last available date in prices_df.
    Returns top TOP_N tickers sorted by score descending.
    Accepts either a MultiIndex (field, ticker) DataFrame or a plain (ticker) DataFrame.
    """
    if isinstance(prices_df.columns, pd.MultiIndex):
        as_of = prices_df["Close"].index[-1]
    else:
        as_of = prices_df.index[-1]
    eligible = get_eligible_tickers(prices_df, as_of)
    eligible = eligible[eligible["passes_filter"]].copy()
    eligible.sort_values("score", ascending=False, inplace=True)
    eligible["rank"] = range(1, len(eligible) + 1)
    return eligible.head(config.TOP_N).reset_index(drop=True)
