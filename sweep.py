import csv
from itertools import product
from pathlib import Path
import pandas as pd
from logger_setup import get_logger
from backtesting import run_once

log = get_logger(__name__)

# ─── prepare ticker‐set mapping ───────────────────────────────────
ticker_files = {
    "top200": "universe/top200.csv",
    #"top100": "universe/top100.csv",
    #"top50":  "universe/top50.csv",
}
ticker_sets = {
    name: pd.read_csv(path)["symbol"].tolist()
    for name, path in ticker_files.items()
}

# ─── param grid (universe included) ──────────────────────────────
param_grid = {
    "universe":        list(ticker_sets.keys()),  # <-- NEW dimension
    "p_long":          [0.60, 0.62, 0.64],
    "p_short":         [0.42],
    "max_long_short":  [4, 6],
    "trail_percent":   [0.04],
    "min_edge":        [0.001, 0.0015],
    "trade_shorts":    [False],
}

# ─── evaluation window ────────────────────────────────────────────
WIN_START = "2023-01-03"
WIN_END   = "2024-12-31"

# ─── generate all combos ──────────────────────────────────────────
keys = list(param_grid.keys())
all_cfgs = [
    dict(zip(keys, vals))
    for vals in product(*(param_grid[k] for k in keys))
]

# ─── run sweep ────────────────────────────────────────────────────
results = []
for cfg in all_cfgs:
    universe_name = cfg.pop("universe")
    tickers = ticker_sets[universe_name]
    log.info("Running %s on %s", cfg, universe_name)
    try:
        out = run_once(
            **cfg,
            start_date=WIN_START,
            end_date=WIN_END,
            tickers=tickers,
        )
    except Exception as e:
        log.error("✗ Failed %s on %s: %s", cfg, universe_name, e)
        out = dict(final=None, sharpe=None, mdd=None,
                   trades=None, cagr=None,
                   gross_pnl=None, tax_paid=None,
                   net_after_tax=None)
    row = {"universe": universe_name, **cfg, **out}
    results.append(row)
    log.info("Result %s", row)

# ─── save CSV ─────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
out_path = Path("logs/experiment_results_full.csv")
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

log.info("✅ Saved combined sweep results → %s", out_path)
