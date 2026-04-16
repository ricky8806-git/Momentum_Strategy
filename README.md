# US Equity Momentum Strategy

Weekly-rebalancing equity momentum strategy on S&P 500 constituents with Alpaca paper-trading integration.

## Strategy Overview

- **Universe:** Current S&P 500 constituents (via GitHub CSV)
- **Signal day:** Friday close
- **Execution day:** Monday open (9:35 ET)
- **Hard filters (all 5 must pass):**
  - Close > 100-day MA
  - Close > 200-day MA
  - 50-day MA > 200-day MA
  - 20-day range position ≥ 65%
  - Close ≥ 90% of 20-day high
- **Ranking:** 60% × 126-day return + 40% × 63-day return; top 15 names held
- **Exits:** Stop-loss (−10% from entry), filter failure, rank > 23
- **Position sizing:** min(7%, 1/n_holdings) × NAV per new entry
- **Idle cash:** Invested in SPY at all times

## Setup

```bash
git clone https://github.com/ricky8806-git/Momentum_Strategy.git
cd Momentum_Strategy
pip install -r requirements.txt
```

Create a `.env` file with your Alpaca paper-trading credentials:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

## Usage

```bash
# Run 3-year backtest
python main.py --backtest --start 2022-01-03 --end 2024-12-31 --nav 100000

# Start live weekly scheduler (paper trading, runs every Monday at 9:35 ET)
python main.py --run

# Single dry-run rebalance — prints intended trades without placing orders
python main.py --dry-run
```

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | All tunable strategy parameters |
| `data_loader.py` | S&P 500 universe fetch + yfinance OHLC download |
| `signals.py` | Indicators, hard filters, momentum ranking |
| `portfolio.py` | State management, position sizing, entry/exit, SPY sleeve |
| `broker.py` | Alpaca API wrapper (market orders, account NAV) |
| `backtest.py` | Historical simulation engine |
| `scheduler.py` | APScheduler weekly job + CSV trade/rebalance logging |
| `main.py` | CLI entry point |

## Running Tests

```bash
pytest tests/ -v
```

## State Files (gitignored)

- `portfolio_state.json` — current holdings, SPY shares, NAV, last rebalance date
- `trades_log.csv` — all trade executions with order IDs
- `rebalance_log.csv` — weekly rebalance summaries
