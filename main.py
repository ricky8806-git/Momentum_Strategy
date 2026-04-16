"""
Entry point for the Momentum Strategy.

Usage:
    python main.py --backtest --start 2022-01-01 --end 2024-12-31
    python main.py --run
    python main.py --dry-run
"""

from __future__ import annotations

import argparse
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="US Equity Momentum Strategy"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backtest", action="store_true",
                       help="Run historical backtest")
    group.add_argument("--run",      action="store_true",
                       help="Start live weekly scheduler")
    group.add_argument("--dry-run",  action="store_true",
                       help="Run one rebalance in dry-run mode (no orders placed)")

    parser.add_argument("--start",   default="2022-01-03",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end",     default="2024-12-31",
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--nav",     type=float, default=100_000.0,
                        help="Initial NAV for backtest")
    args = parser.parse_args()

    if args.backtest:
        from backtest import run_backtest, compute_metrics
        print(f"Backtesting {args.start} → {args.end}  NAV=${args.nav:,.0f}")
        results = run_backtest(args.start, args.end, args.nav)
        metrics = compute_metrics(results)
        print("\n── Performance Metrics ──────────────────────────────")
        print(f"  Annualized Return : {metrics['ann_return']:+.1%}")
        print(f"  Sharpe Ratio      : {metrics['sharpe']:.2f}")
        print(f"  Max Drawdown      : {metrics['max_drawdown']:.1%}")
        print(f"  Avg Holdings      : {metrics['avg_holdings']:.1f}")
        print(f"  Final NAV         : ${metrics['final_nav']:,.0f}")

    elif args.run:
        from scheduler import start_scheduler
        start_scheduler(dry_run=False)

    elif args.dry_run:
        from scheduler import run_rebalance
        run_rebalance(dry_run=True)


if __name__ == "__main__":
    main()
