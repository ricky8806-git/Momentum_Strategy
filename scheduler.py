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


# ── Per-run markdown report ───────────────────────────────────────────────

def generate_report(
    report_path: str,
    signal_day: pd.Timestamp,
    nav: float,
    n_universe: int,
    n_scanned: int,
    eligible_pass: pd.DataFrame,
    entries: list[dict],
    exits: list[dict],
    held_tickers: list[str],
) -> None:
    """
    Write a markdown rebalance summary to report_path.

    eligible_pass : DataFrame with [ticker, score, passes_filter, ret_long, ret_short, rank]
                    sorted by score descending.
    entries       : list of {ticker, shares, price}
    exits         : list of {ticker, shares, price, reason}
    held_tickers  : tickers carried over with no action this run
    """
    run_str   = pd.Timestamp.now().strftime("%Y-%m-%d")
    sig_str   = signal_day.strftime("%Y-%m-%d")
    n_pass    = len(eligible_pass)
    top_n     = eligible_pass.head(config.TOP_N)
    entry_set = {e["ticker"] for e in entries}

    lines: list[str] = [
        f"# Momentum Strategy Rebalance — {run_str}",
        "",
        "## Summary",
        f"- **Signal Date (Friday):** {sig_str}",
        f"- **Post-Rebalance NAV:** ${nav:,.2f}",
        f"- **Holdings:** {len(held_tickers) + len(entries)}",
        "",
        "## Universe Scan",
        "| Step | Count |",
        "|------|-------|",
        f"| S&P 500 symbols attempted | {n_universe} |",
        f"| Sufficient price history  | {n_scanned} |",
        f"| Passed all 5 hard filters | {n_pass} |",
        f"| Selected (Top {config.TOP_N}) | {min(n_pass, config.TOP_N)} |",
        "",
        "### Hard Filter Criteria",
        "1. Close > 100-day MA",
        "2. Close > 200-day MA",
        "3. 50-day MA > 200-day MA",
        f"4. Range position ≥ {config.RANGE_POS_MIN:.0%} of 20-day high-low range",
        f"5. Close ≥ {config.HIGH20_MIN_PCT:.0%} of 20-day high",
        "",
        f"## Selected Portfolio (Top {config.TOP_N} by momentum score)",
        "| Rank | Ticker | Score | 6M Return | 3M Return | Action |",
        "|------|--------|-------|-----------|-----------|--------|",
    ]

    for _, row in top_n.iterrows():
        action = "**NEW ENTRY**" if row["ticker"] in entry_set else "held"
        ret_l  = f"{row['ret_long']  * 100:+.1f}%" if pd.notna(row["ret_long"])  else "—"
        ret_s  = f"{row['ret_short'] * 100:+.1f}%" if pd.notna(row["ret_short"]) else "—"
        lines.append(
            f"| {int(row['rank'])} | {row['ticker']} "
            f"| {row['score']:.3f} | {ret_l} | {ret_s} | {action} |"
        )

    lines.append("")

    if exits:
        lines += [
            "## Exits",
            "| Ticker | Shares | Price | Reason |",
            "|--------|--------|-------|--------|",
        ]
        for ex in exits:
            lines.append(
                f"| {ex['ticker']} | {int(ex.get('shares', 0))} "
                f"| ${float(ex.get('price', 0)):.2f} | {ex.get('reason', '')} |"
            )
    else:
        lines += ["## Exits", "_No exits this week._"]
    lines.append("")

    if entries:
        lines += [
            "## Entries",
            "| Ticker | Shares | Price | Est. Cost |",
            "|--------|--------|-------|-----------|",
        ]
        for en in entries:
            cost = en.get("shares", 0) * en.get("price", 0)
            lines.append(
                f"| {en['ticker']} | {int(en.get('shares', 0))} "
                f"| ${float(en.get('price', 0)):.2f} | ${cost:,.0f} |"
            )
    else:
        lines += ["## Entries", "_No new entries this week._"]
    lines.append("")

    if held_tickers:
        lines += [
            "## Continued Holdings (no action)",
            ", ".join(sorted(held_tickers)),
            "",
        ]

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Report saved → {report_path}")


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

    # ── Load state & sync account values from Alpaca ──────────────────────
    state = load_state()
    account_nav  = broker.get_account_nav(dry_run=dry_run)
    account_cash = broker.get_account_cash(dry_run=dry_run)
    state["nav"] = account_nav
    print(f"  NAV: ${account_nav:,.2f}  Cash: ${account_cash:,.2f}")

    # ── Fetch prices ──────────────────────────────────────────────────────
    symbols = data_loader.get_sp500_symbols()
    fetch_start = (today - pd.offsets.BDay(config.WARMUP_DAYS)).strftime("%Y-%m-%d")
    prices = data_loader.fetch_prices(symbols, fetch_start, today_str)

    # Signal day = most recent Friday that has price data
    price_index = prices.index
    available_fridays = [d for d in price_index if d.weekday() == 4]
    signal_day = available_fridays[-1] if available_fridays else price_index[-1]

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
        ex["shares"] = shares                                      # persist for report
        price  = float(ex.get("price") or open_today.get(t, 0.0))
        result = broker.submit_market_order(t, "sell", shares, dry_run=dry_run)
        if not dry_run:
            _log_trade(today_str, t, "sell", shares, price, ex["reason"],
                       state["nav"], result.get("order_id", ""))

    cash_freed = apply_exits(state, all_exits)

    # ── Entries ───────────────────────────────────────────────────────────
    entries_to_buy = [t for t in top_n_set if t not in state["holdings"]]
    n_total = len(top_n_set)
    entry_orders = [{"ticker": t, "price": float(open_today.get(t, 0.0))}
                    for t in entries_to_buy]

    # Use real Alpaca cash (+ proceeds from any exits just submitted) for sizing
    idle = account_cash + cash_freed
    apply_entries(state, entry_orders, idle, n_total, today_str)

    for t in entries_to_buy:
        if t not in state["holdings"]:
            continue
        shares = state["holdings"][t]["shares"]
        price  = float(open_today.get(t, 0.0))
        result = broker.submit_market_order(t, "buy", shares, dry_run=dry_run)
        if not dry_run:
            _log_trade(today_str, t, "buy", shares, price, "entry",
                       state["nav"], result.get("order_id", ""))

    # ── SPY sleeve ────────────────────────────────────────────────────────
    # Compute idle cash after entries, then size SPY sleeve against real position
    entry_costs = sum(
        state["holdings"][t]["shares"] * open_today.get(t, 0.0)
        for t in entries_to_buy
        if t in state["holdings"]
    )
    idle_now = account_cash + cash_freed - entry_costs
    spy_price = float(open_today.get("SPY") or close_today.get("SPY") or 0.0)
    # Sync spy_shares to actual Alpaca position before computing delta
    actual_spy = broker.get_position_qty("SPY", dry_run=dry_run)
    state["spy_shares"] = actual_spy if not dry_run else state["spy_shares"]
    old_spy = state["spy_shares"]
    adjust_spy_sleeve(state, idle_now, spy_price)
    spy_delta = state["spy_shares"] - old_spy
    spy_order_qty = int(abs(spy_delta))  # Alpaca requires whole shares for market orders
    if spy_order_qty >= 1:
        side   = "buy" if spy_delta > 0 else "sell"
        result = broker.submit_market_order("SPY", side, spy_order_qty, dry_run=dry_run)
        if not dry_run:
            _log_trade(today_str, "SPY", side, spy_order_qty, spy_price, "spy_sleeve",
                       state["nav"], result.get("order_id", ""))
        # Sync state to actual integer shares purchased/sold
        state["spy_shares"] = old_spy + spy_order_qty if side == "buy" else max(0.0, old_spy - spy_order_qty)

    # ── Update state (skipped in dry-run — no real orders were placed) ──────
    state["last_rebalance"] = today_str
    state["nav"] = broker.get_account_nav(dry_run=dry_run)
    if not dry_run:
        save_state(state)
        # ── Log rebalance summary ─────────────────────────────────────────
        spy_val = state["spy_shares"] * spy_price
        spy_pct = spy_val / state["nav"] if state["nav"] > 0 else 0.0
        entries_executed = [t for t in entries_to_buy if t in state["holdings"]]
        _log_rebalance(
            today_str,
            len(state["holdings"]),
            state["nav"],
            spy_pct,
            entries_executed,
            [e["ticker"] for e in filter_exits],
            stop_tickers,
        )
        # ── Per-run markdown report ───────────────────────────────────────
        entries_detail = [
            {
                "ticker": t,
                "shares": state["holdings"][t]["shares"],
                "price":  float(open_today.get(t, 0.0)),
            }
            for t in entries_executed
        ]
        held_tickers_list = [t for t in state["holdings"] if t not in set(entries_executed)]
        generate_report(
            report_path=os.path.join("reports", f"{today_str}_rebalance_report.md"),
            signal_day=signal_day,
            nav=state["nav"],
            n_universe=len(symbols),
            n_scanned=len(eligible),
            eligible_pass=eligible_pass,
            entries=entries_detail,
            exits=all_exits,
            held_tickers=held_tickers_list,
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
