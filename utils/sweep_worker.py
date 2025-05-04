"""
A single worker process: loads one universe into memory ONCE,
then executes many run_once() calls sent via multiprocessing Queue.
"""
import sys, pickle, queue
from multiprocessing import Queue
from backtesting import run_once
from logger_setup import get_logger
import pandas as pd

log = get_logger(__name__)

def worker(universe_name: str, tickers_csv: str,
           task_q: Queue, result_q: Queue):
    tickers = pd.read_csv(tickers_csv)["symbol"].tolist()
    log.info("Worker %s loaded %d tickers", universe_name, len(tickers))

    while True:
        try:
            cfg = task_q.get(timeout=5)
        except queue.Empty:
            continue
        if cfg is None:        # poison pill → shut down
            break
        try:
            # REMOVE the special keys before expanding cfg
            start = cfg.pop("_start")
            end   = cfg.pop("_end")

            # Now call run_once with clean kwargs
            res = run_once(
                **cfg,
                start_date=start,
                end_date=end,
                tickers=tickers,
            )

            # Build your result row
            result_q.put({"universe": universe_name, **cfg, **res})

        except Exception as e:
            log.error("Fail %s : %s", cfg, e)
            result_q.put({"universe": universe_name, **cfg,
                          "final": None})
