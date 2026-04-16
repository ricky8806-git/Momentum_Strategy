"""
Weekly rebalance scheduler using APScheduler.
Runs every Monday at 09:35 ET (5 minutes after open).

Usage:
    python scheduler.py             # start live scheduler
    python scheduler.py --dry-run   # print intended trades only
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import pandas as pd
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()

import config
import data_loader
from signals import get_eligible_tickers
from portfolio import (
    load_state,
    save_state,
    apply_exits,
    apply_entries,
    check_stop_losses,
    adjust_spy_sleeve,
    compute_idle_cash,
)
import broker

ET = pytz.timezone("America/New_York")


# ── Logging helpers ───────────────────────────────────────────────────────

def _log_trade(
    date: str,
    ticker: str,
    action: str,
    shares: float,
    price: float,
    reason: str,
    nav_after: float,
    order_id: str = "",
) -> None:
    file_exists = os.path.exists(config.TRADES_LOG)
    with open(config.TRADES_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "ticker", "action", "shares", "price",
                              "reason", "nav_after", "order_id"])
        writer.writerow([date, ticker, action, shares, price, reason,
                         nav_after, order_id])


def _log_rebalance(
    date: str,
    n_holdings: int,
    nav: float,
    spy_pct: float,
    entries: list[str],
    exits: list[str],
    stop_losses: list[str],
) -> None:
    file_exists = os.path.exists(config.REBALANCE_LOG)
    with open(config.REBALANCE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "n_holdings", "nav", "spy_sleeve_pct",
                              "entries", "exits", "stop_losses"])
        writer.writerow([date, n_holdings, nav, spy_pct,
                         "|".join(entries), "|".join(exits), "|".join(stop_losses)])


# ── Core rebalance logic ──────────────────────────────────────────────────

def run_rebalance(dry_run: bool = False) -> None:
    """
    1. Fetch latest prices (through today for signals).
    2. Compute most recent Friday's signals.
    3. Determine exits, stop-losses, entries.
    4. Execute trades via Alpaca (or print in dry_run mode).
    5. Persist state and logs.
    """
    today = pd.Timestamp.now(tz=ET).normalize().tz_localize(None)
    today_str = today.strftime("%Y-%m-%d")
    print(f"\n[{today_str}] Starting rebalance (dry_run={dry_run})")

    # ── Load state ────────────────────────────────────────────────────────
    state = load_state()
    if state["nav"] == 0.0:
        state["nav"] = broker.get_account_nav(dry_run=dry_run)
        print(f"  Bootstrapped NAV: ${state['nav']:,.2f}")

    # ── Fetch prices ──────────────────────────────────────────────────────
    symbols = data_loader.get_sp500_symbols()
    fetch_start = (today - pd.offsets.BDay(config.WARMUP_DAYS)).strftime("%Y-%m-%d")
    prices = data_loader.fetch_prices(symbols, fetch_start, today_str)

    # Signal day = most recent Friday on or before today
    bdays = pd.bdate_range(fetch_start, today_str)
    fridays = [d for d in bdays if d.weekday() == 4]
    signal_day = fridays[-1] if fridays else today

    # ── Friday signals ────────────────────────────────────────────────────
    eligible = get_eligible_tickers(prices.loc[:signal_day], signal_day)
    eligible_pass = eligible[eligible["passes_filter"]].copy()
    eligible_pass.sort_values("score", ascending=False, inplace=True)
    eligible_pass["rank"] = range(1, len(eligible_pass) + 1)
    top_n_set = set(eligible_pass.head(config.TOP_N)["ticker"].tolist())

    # Latest available prices for order sizing
    close_today = prices["Close"].loc[:today].iloc[-1].to_dict()
    open_today  = prices["Open"].loc[:today].iloc[-1].to_dict()

    # ── Stop-loss check ───────────────────────────────────────────────────
    stop_exits = check_stop_losses(state["holdings"], close_today)
    stop_tickers = [e["ticker"] for e in stop_exits]

    # ── Filter + rank exits ───────────────────────────────────────────────
    filter_exits = []
    for t in list(state["holdings"].keys()):
        if t in stop_tickers:
            continue
        row = eligible[eligible["ticker"] == t]
        if row.empty or not bool(row.iloc[0]["passes_filter"]):
            filter_exits.append({"ticker": t, "reason": "filter_exit",
                                  "price": open_today.get(t, 0.0)})
            continue
        rank_row = eligible_pass[eligible_pass["ticker"] == t]
        if rank_row.empty or int(rank_row.iloc[0]["rank"]) > config.EXIT_RANK_THRESHOLD:
            filter_exits.append({"ticker": t, "reason": "rank_exit",
                                  "price": open_today.get(t, 0.0)})

    # ── Execute exits (stop-loss first, then filter/rank) ─────────────────
    all_exits = stop_exits + filter_exits
    for ex in all_exits:
        t = ex["ticker"]
        shares = state["holdings"].get(t, {}).get("shares", 0)
        price  = float(ex.get("price") or open_today.get(t, 0.0))
        result = broker.submit_market_order(t, "sell", shares, dry_run=dry_run)
        _log_trade(today_str, t, "sell", shares, price, ex["reason"],
                   state["nav"], result.get("order_id", ""))

    cash_freed = apply_exits(state, all_exits)

    # ── Entries ───────────────────────────────────────────────────────────
    entries_to_buy = [t for t in top_n_set if t not in state["holdings"]]
    n_total = len(top_n_set)
    entry_orders = [{"ticker": t, "price": float(open_today.get(t, 0.0))}
                    for t in entries_to_buy]

    idle = compute_idle_cash(state, close_today) + cash_freed
    apply_entries(state, entry_orders, idle, n_total, today_str)

    for t in entries_to_buy:
        if t not in state["holdings"]:
            continue
        shares = state["holdings"][t]["shares"]
        price  = float(open_today.get(t, 0.0))
        result = broker.submit_market_order(t, "buy", shares, dry_run=dry_run)
        _log_trade(today_str, t, "buy", shares, price, "entry",
                   state["nav"], result.get("order_id", ""))

    # ── SPY sleeve ────────────────────────────────────────────────────────
    idle_now = compute_idle_cash(state, close_today)
    spy_price = float(open_today.get("SPY") or close_today.get("SPY") or 0.0)
    old_spy = state["spy_shares"]
    adjust_spy_sleeve(state, idle_now, spy_price)
    spy_delta = state["spy_shares"] - old_spy
    if abs(spy_delta) >= 0.5:
        side   = "buy" if spy_delta > 0 else "sell"
        result = broker.submit_market_order("SPY", side, abs(spy_delta), dry_run=dry_run)
        _log_trade(today_str, "SPY", side, abs(spy_delta), spy_price, "spy_sleeve",
                   state["nav"], result.get("order_id", ""))

    # ── Update state ──────────────────────────────────────────────────────
    state["last_rebalance"] = today_str
    state["nav"] = broker.get_account_nav(dry_run=dry_run)
    save_state(state)

    # ── Log rebalance summary ─────────────────────────────────────────────
    spy_val = state["spy_shares"] * spy_price
    spy_pct = spy_val / state["nav"] if state["nav"] > 0 else 0.0
    _log_rebalance(
        today_str,
        len(state["holdings"]),
        state["nav"],
        spy_pct,
        entries_to_buy,
        [e["ticker"] for e in filter_exits],
        stop_tickers,
    )
    print(f"  Done. Holdings: {len(state['holdings'])}  NAV: ${state['nav']:,.2f}")


# ── Scheduler entry point ─────────────────────────────────────────────────

def start_scheduler(dry_run: bool = False) -> None:
    scheduler = BlockingScheduler(timezone=ET)
    scheduler.add_job(
        run_rebalance,
        "cron",
        day_of_week="mon",
        hour=config.REBALANCE_HOUR,
        minute=config.REBALANCE_MINUTE,
        kwargs={"dry_run": dry_run},
        id="weekly_rebalance",
    )
    print(f"Scheduler started. Next run: Monday {config.REBALANCE_HOUR}:{config.REBALANCE_MINUTE:02d} ET")
    scheduler.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    start_scheduler(dry_run=args.dry_run)
