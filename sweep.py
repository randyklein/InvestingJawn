# sweep.py
import csv
from itertools import product
from pathlib import Path
from joblib import Parallel, delayed
from logger_setup import get_logger

from backtesting import run_once

log = get_logger(__name__)

# --- parameter grid -------------------------------------------------
param_grid = {
    "p_long":          [0.56, 0.58, 0.60],
    "p_short":         [0.44, 0.42, 0.40],
    "max_long_short":  [6, 8, 10],
    "trail_percent":   [0.04, 0.05],
}
WIN_START = "2025-01-02"
WIN_END   = "2025-03-31"

# --- flatten grid
keys = list(param_grid.keys())
grid = [dict(zip(keys, values)) for values in product(*(param_grid[k] for k in keys))]

# --- wrapper for parallel call
def run_cfg(cfg):
    try:
        res = run_once(**cfg, start_date=WIN_START, end_date=WIN_END)
        log.info("✓ Finished %s", cfg)
        return {**cfg, **res}
    except Exception as e:
        log.error("✗ Failed %s: %s", cfg, str(e))
        return {**cfg, "final": None, "sharpe": None, "mdd": None, "trades": None}

# --- parallel run
results = Parallel(n_jobs=-1, backend="loky")(delayed(run_cfg)(cfg) for cfg in grid)

# --- write to CSV
Path("logs").mkdir(exist_ok=True)
out_path = Path("logs/experiment_results_parallel.csv")
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

log.info("✅ Saved parallel results to %s", out_path)
