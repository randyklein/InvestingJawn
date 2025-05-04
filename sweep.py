"""
Parameter sweep – slippage-aware.
Results (including net-after-tax) are written to logs/experiment_results.csv
"""

from itertools import product
from pathlib import Path
import csv
from logger_setup import get_logger
from backtesting import run_once

log = get_logger(__name__)

# ─── parameter grid ───────────────────────────────────────────────
param_grid = {
    "p_long":          [0.54, 0.56, 0.58, 0.60],
    "p_short":         [0.42],              # keep best so far
    "max_long_short":  [6, 8, 10],
    "trail_percent":   [0.03, 0.04],
    "min_edge":        [0.0005, 0.0010],    # slippage buffer test
}

# date window you want to evaluate
WIN_START = "2023-01-03"
WIN_END   = "2024-12-31"

# ─── run grid sequentially (simple & RAM-friendly) ───────────────
keys = list(param_grid.keys())
results = []

for vals in product(*(param_grid[k] for k in keys)):
    cfg = dict(zip(keys, vals))
    log.info("Running %s", cfg)
    res = run_once(**cfg, start_date=WIN_START, end_date=WIN_END)
    results.append({**cfg, **res})
    log.info("Result %s", res)

# ─── save CSV ─────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
out_path = Path("logs/experiment_results.csv")
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

log.info("Saved sweep results → %s", out_path)
