"""Global settings & credentials (edit as needed)."""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Data paths ──────────────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")          # 1‑min bars
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")  # pre‑aggregated bars

# Loader directory: set to RAW_DATA_DIR to resample minute data, or PROCESSED_DATA_DIR for ready daily bars
DATA_DIR = RAW_DATA_DIR

# Minutes per bar when resampling minute data.  None → aggregate to daily.
RESAMPLE_MINUTES = 30

# ─── Strategy / risk parameters ─────────────────────────────────────
INITIAL_CASH = 10_000.0
MAX_POSITION_PCT = 0.20
CASH_BUFFER_PCT = 0.05
MAX_SECTOR_POSITIONS = 3
TRAIL_PERCENT = 0.05
MIN_EDGE = 0.0005        # minimum probability edge (50.05%) to cover slippage

# ─── Model output path ─────────────────────────────────────────────
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")

# ─── Alpaca credentials ───────────────────────────────────────────── ─────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_PAPER = True  # False → live trading