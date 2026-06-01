import pandas as pd
import numpy as np
import pytest
from signals import (
    compute_indicators,
    apply_hard_filters,
    compute_momentum_score,
    rank_universe,
    get_eligible_tickers,
)


def _make_close(n: int = 300, base: float = 150.0) -> pd.Series:
    """Create a steadily-rising close series of length n."""
    rng = pd.bdate_range("2023-01-02", periods=n)
    prices = base + np.arange(n) * 0.5
    return pd.Series(prices, index=rng, name="AAPL")


def test_compute_indicators_ma100_length():
    close = _make_close(300)
    ind = compute_indicators(close)
    assert "ma100" in ind.columns
    assert len(ind) == len(close)


def test_compute_indicators_range_position_bounds():
    close = _make_close(300)
    ind = compute_indicators(close)
    valid = ind["range_pos"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_apply_hard_filters_passes_strong_uptrend():
    """A stock in a clear uptrend should pass all 5 filters."""
    close = _make_close(300, base=150.0)
    ind = compute_indicators(close)
    last = ind.iloc[-1]
    result = apply_hard_filters(last)
    assert result is True


def test_apply_hard_filters_fails_below_ma200():
    """Stock well below its 200d MA should fail."""
    rng = pd.bdate_range("2023-01-02", periods=300)
    prices = np.concatenate([
        150 + np.arange(250) * 0.2,   # gentle rise for warm-up
        np.linspace(200, 80, 50),      # sharp drop
    ])
    close = pd.Series(prices, index=rng, name="TEST")
    ind = compute_indicators(close)
    last = ind.iloc[-1]
    result = apply_hard_filters(last)
    assert result is False


def test_compute_momentum_score_uses_weights():
    close = _make_close(300)
    score = compute_momentum_score(close)
    ret_126 = close.iloc[-1] / close.iloc[-127] - 1
    ret_63  = close.iloc[-1] / close.iloc[-64] - 1
    expected = 0.60 * ret_126 + 0.40 * ret_63
    assert abs(score - expected) < 1e-10


def test_eligible_tickers_exposes_return_components():
    """get_eligible_tickers() must include ret_long and ret_short columns."""
    idx    = pd.bdate_range("2022-01-03", periods=260)
    prices = pd.DataFrame({"AAPL": np.linspace(100, 160, 260)}, index=idx)
    result = get_eligible_tickers(prices, idx[-1])
    assert "ret_long"  in result.columns, "ret_long column missing"
    assert "ret_short" in result.columns, "ret_short column missing"
    row = result[result["ticker"] == "AAPL"]
    assert len(row) == 1
    assert isinstance(float(row.iloc[0]["ret_long"]),  float)
    assert isinstance(float(row.iloc[0]["ret_short"]), float)


def test_rank_universe_returns_top_n():
    """rank_universe should return exactly TOP_N tickers in descending order."""
    import config
    n = 30
    rng = pd.bdate_range("2023-01-02", periods=300)
    close_dict = {}
    for i in range(n):
        prices = 150 + np.arange(300) * (0.1 + i * 0.05)
        close_dict[f"TICK{i:02d}"] = pd.Series(prices, index=rng)

    prices_df = pd.DataFrame(close_dict)
    ranked = rank_universe(prices_df)
    assert len(ranked) == config.TOP_N
    assert all(ranked["score"].iloc[i] >= ranked["score"].iloc[i+1]
               for i in range(len(ranked) - 1))
