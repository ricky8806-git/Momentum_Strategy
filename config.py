"""All tunable strategy parameters in one place."""

# ── Universe ──────────────────────────────────────────────────────────────
SP500_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
    "/main/data/constituents.csv"
)

# ── Data fetch ────────────────────────────────────────────────────────────
WARMUP_DAYS = 260          # extra business days fetched before start date

# ── Moving-average windows (business days) ───────────────────────────────
MA_SHORT  = 50
MA_MID    = 100
MA_LONG   = 200

# ── Hard-filter thresholds ───────────────────────────────────────────────
RANGE_POS_MIN   = 0.65     # (close - 20d_low) / (20d_high - 20d_low) >= this
HIGH20_MIN_PCT  = 0.90     # close >= 90% of 20-day high

# ── Momentum score ────────────────────────────────────────────────────────
MOM_WINDOW_LONG  = 126     # business days (~6 months)
MOM_WINDOW_SHORT = 63      # business days (~3 months)
MOM_WEIGHT_LONG  = 0.60
MOM_WEIGHT_SHORT = 0.40

# ── Portfolio construction ────────────────────────────────────────────────
TOP_N               = 15   # target holdings
EXIT_RANK_THRESHOLD = 23   # sell if rank > this
MAX_POSITION_WEIGHT = 0.07 # 7% of NAV per new entry
STOP_LOSS_PCT       = 0.10 # 10% below entry price

# ── Rebalance schedule ────────────────────────────────────────────────────
REBALANCE_DAY    = "monday"  # execution day
SIGNAL_DAY       = "friday"  # signal computation day
REBALANCE_HOUR   = 9         # ET
REBALANCE_MINUTE = 35        # 5 min after open

# ── File paths ────────────────────────────────────────────────────────────
STATE_FILE        = "portfolio_state.json"
TRADES_LOG        = "trades_log.csv"
REBALANCE_LOG     = "rebalance_log.csv"
