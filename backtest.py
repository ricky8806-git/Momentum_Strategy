"""
Standalone backtest engine.

Usage:
    python backtest.py --start 2022-01-01 --end 2024-12-31 --nav 100000
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import config
import data_loader
from signals import get_eligible_tickers
from portfolio import check_stop_losses


def _fridays_in_range(start: str, end: str) -> list[pd.Timestamp]:
    dates = pd.bdate_range(start, end)
    return [d for d in dates if d.weekday() == 4]


def _next_open_price(
    prices: pd.DataFrame, ticker: str, after: pd.Timestamp
) -> tuple[float, pd.Timestamp] | None:
    """Return (open_price, date) on the first business day after *after*."""
    opens = prices["Open"][ticker].dropna()
    future = opens[opens.index > after]
    if future.empty:
        return None
    return float(future.iloc[0]), future.index[0]


def run_backtest(
    start: str,
    end: str,
    initial_nav: float = 100_000.0,
) -> pd.DataFrame:
    """
    Simulate the momentum strategy from *start* to *end*.
    Returns daily NAV DataFrame with columns [nav, n_holdings].
    """
    # ── Fetch data ────────────────────────────────────────────────────────
    symbols = data_loader.get_sp500_symbols()
    fetch_start = (
        pd.Timestamp(start) - pd.offsets.BDay(config.WARMUP_DAYS)
    ).strftime("%Y-%m-%d")
    prices = data_loader.fetch_prices(symbols, fetch_start, end)
    close  = prices["Close"]

    # ── State ─────────────────────────────────────────────────────────────
    holdings: dict[str, dict] = {}
    spy_shares = 0.0
    cash = initial_nav  # all cash starts undeployed; invested in SPY below

    nav_series = []
    bdays   = pd.bdate_range(start, end)
    fridays = set(_fridays_in_range(start, end))

    for day in bdays:
        if day not in close.index:
            continue

        day_prices = close.loc[day].to_dict()

        # ── 1. Check stop-losses on today's close ─────────────────────────
        stop_exits = check_stop_losses(holdings, day_prices)
        for ex in stop_exits:
            t = ex["ticker"]
            if t not in holdings:
                continue
            info = _next_open_price(prices, t, day)
            if info:
                sell_price, _ = info
                cash += holdings[t]["shares"] * sell_price
                del holdings[t]

        # ── 2. Weekly rebalance signal on Friday ──────────────────────────
        if day in fridays:
            data_to_date = prices.loc[:day]
            eligible = get_eligible_tickers(data_to_date, day)
            if eligible.empty or "passes_filter" not in eligible.columns:
                eligible = pd.DataFrame({"ticker": pd.Series([], dtype=str),
                                         "score": pd.Series([], dtype=float),
                                         "passes_filter": pd.Series([], dtype=bool)})
            eligible_pass = eligible[eligible["passes_filter"]].copy()
            eligible_pass.sort_values("score", ascending=False, inplace=True)
            eligible_pass["rank"] = range(1, len(eligible_pass) + 1)
            top_n_set = set(eligible_pass.head(config.TOP_N)["ticker"].tolist())

            # ── Exits (filter fail or rank > EXIT_RANK_THRESHOLD) ─────────
            to_exit = []
            for t in list(holdings.keys()):
                row = eligible[eligible["ticker"] == t]
                if row.empty or not bool(row.iloc[0]["passes_filter"]):
                    to_exit.append(t)
                    continue
                rank_row = eligible_pass[eligible_pass["ticker"] == t]
                if rank_row.empty or int(rank_row.iloc[0]["rank"]) > config.EXIT_RANK_THRESHOLD:
                    to_exit.append(t)

            for t in to_exit:
                info = _next_open_price(prices, t, day)
                if info and t in holdings:
                    sell_price, _ = info
                    cash += holdings[t]["shares"] * sell_price
                    del holdings[t]

            # ── Entries (top-N not already held) ──────────────────────────
            entries_to_buy = [t for t in top_n_set if t not in holdings]
            n_total = len(top_n_set)
            for t in entries_to_buy:
                info = _next_open_price(prices, t, day)
                if info is None:
                    continue
                buy_price, _ = info
                alloc_pct = min(config.MAX_POSITION_WEIGHT, 1.0 / max(n_total, 1))
                # Use full portfolio equity as NAV basis for sizing
                current_equity = sum(
                    holdings[h]["shares"] * day_prices.get(h, holdings[h]["entry_price"])
                    for h in holdings
                )
                nav_basis = cash + current_equity
                shares = math.floor(alloc_pct * nav_basis / buy_price)
                cost   = shares * buy_price
                if cost > cash:
                    shares = math.floor(cash / buy_price)
                    cost   = shares * buy_price
                if shares <= 0:
                    continue
                holdings[t] = {
                    "shares":      float(shares),
                    "entry_price": float(buy_price),
                    "entry_date":  str(day.date()),
                }
                cash -= cost

            # ── SPY sleeve: invest remaining cash ─────────────────────────
            info = _next_open_price(prices, "SPY", day)
            if info:
                spy_open, _ = info
                if spy_open > 0:
                    spy_shares = cash / spy_open
                    cash = 0.0

        # ── 3. Daily NAV ──────────────────────────────────────────────────
        equity = sum(
            pos["shares"] * day_prices.get(t, pos["entry_price"])
            for t, pos in holdings.items()
        )
        spy_price_today = day_prices.get("SPY", 0.0)
        spy_val = spy_shares * spy_price_today
        nav = equity + spy_val + cash
        nav_series.append({"date": day, "nav": nav, "n_holdings": len(holdings)})

    return pd.DataFrame(nav_series).set_index("date")


def compute_metrics(nav_df: pd.DataFrame) -> dict:
    """Compute annualized return, Sharpe ratio, and max drawdown."""
    nav = nav_df["nav"]
    daily_ret = nav.pct_change().dropna()
    years = len(nav) / 252

    ann_return = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    ann_vol    = daily_ret.std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else float("nan")

    rolling_max = nav.cummax()
    drawdown    = (nav - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    return {
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "avg_holdings": nav_df["n_holdings"].mean(),
        "final_nav":    nav.iloc[-1],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--nav",   type=float, default=100_000.0)
    args = parser.parse_args()

    print(f"Running backtest {args.start} → {args.end}  initial NAV=${args.nav:,.0f}")
    results = run_backtest(args.start, args.end, args.nav)
    metrics = compute_metrics(results)

    print("\n── Performance Metrics ──────────────────────────────")
    print(f"  Annualized Return : {metrics['ann_return']:+.1%}")
    print(f"  Annualized Vol    : {metrics['ann_vol']:.1%}")
    print(f"  Sharpe Ratio      : {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown']:.1%}")
    print(f"  Avg Holdings      : {metrics['avg_holdings']:.1f}")
    print(f"  Final NAV         : ${metrics['final_nav']:,.0f}")
