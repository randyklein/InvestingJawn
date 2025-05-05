"""
Memory-safe multiprocess parameter sweep
---------------------------------------
Example:
    python sweep.py --workers 4 --universes top200 top100 top50
"""

import csv, argparse, time
from itertools import product
from pathlib import Path
from multiprocessing import Process, Queue
import pandas as pd
from logger_setup import get_logger
from utils.sweep_worker import worker    # <== new helper

log = get_logger(__name__)

# ---- ticker CSV mapping -------------------------------------------------
ticker_files = {
    "top200": "universe/top200.csv",
    "top100": "universe/top100.csv",
    "top50":  "universe/top50.csv",
}

# ---- grid to test -------------------------------------------------------
param_grid = {
    "p_long":          [0.61, 0.62, 0.63],
    "p_short":         [0.41, 0.42, 0.43],
    "max_long_short":  [3, 4, 5],
    "trail_percent":   [0.03, 0.04, 0.05],
    "min_edge":        [0.001],                # fixed
    "trade_shorts":    [False],                # long-only
}

WIN_START = "2023-01-03"
WIN_END   = "2024-12-31"
CSV_PATH  = Path("logs/experiment_results_full.csv")

# ------------------------------------------------------------------------
def build_tasks():
    keys = list(param_grid.keys())
    for vals in product(*(param_grid[k] for k in keys)):
        cfg = dict(zip(keys, vals))
        cfg["_start"] = WIN_START      # embed date window in cfg
        cfg["_end"]   = WIN_END
        yield cfg

def main(universes, n_workers):
    task_q, result_q = Queue(), Queue()

    # enqueue ALL configs for each universe
    for u in universes:
        for cfg in build_tasks():
            task_q.put(cfg)
        task_q.put(None)     # poison pill for that worker

    # start one worker per universe (bounded by --workers)
    procs = []
    for u in universes[:n_workers]:
        p = Process(target=worker,
                    args=(u, ticker_files[u], task_q, result_q),
                    daemon=True)
        p.start()
        procs.append(p)

    # incremental CSV write
    CSV_PATH.parent.mkdir(exist_ok=True)
    wrote_header = False
    finished = 0
    total_tasks = len(universes)*len(list(build_tasks()))
    with CSV_PATH.open("w", newline="") as f:
        writer = None
        while finished < total_tasks:
            res = result_q.get()
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=res.keys())
                writer.writeheader()
                wrote_header = True
            writer.writerow(res); f.flush()
            finished += 1
            if finished % 10 == 0:
                log.info("%d / %d done (%0.1f%%)",
                         finished, total_tasks,
                         100*finished/total_tasks)

    for p in procs:
        p.join()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="Max parallel universes")
    ap.add_argument("--universes", nargs="+",
                    default=list(ticker_files.keys()))
    args = ap.parse_args()
    main(args.universes, args.workers)
