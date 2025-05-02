"""Experiment runner: grid-search over strategy parameters."""

import csv
from itertools import product
from pathlib import Path
from logger_setup import get_logger

from backtesting import run_once

log = get_logger(__name__)

# --- parameter grid -------------------------------------------------
param_grid = {
    "p_long":          [0.54, 0.55, 0.57],
    "p_short":         [0.42],
    "max_long_short":  [10],
    "trail_percent":   [0.04],
}

# short evaluation window for speed
WIN_START = "2023-01-01"
WIN_END   = "2024-12-31"

# --------------------------------------------------------------------
keys = list(param_grid.keys())
results = []

for vals in product(*(param_grid[k] for k in keys)):
    cfg = dict(zip(keys, vals))
    log.info("Running %s", cfg)
    res = run_once(**cfg, start_date=WIN_START, end_date=WIN_END)
    results.append({**cfg, **res})
    log.info("Result %s", res)

# save to CSV
Path("logs").mkdir(exist_ok=True)
out_path = Path("logs/experiment_results.csv")
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
log.info("Saved grid results → %s", out_path)
