"""Reusable logging helper.

Usage:
    from logger_setup import get_logger
    log = get_logger(__name__)          # in any script
"""

import logging
import os
import sys
from datetime import date
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _new_file_handler(script_name: str) -> logging.Handler:
    """Return a FileHandler logs/<script>_YYYY-MM-DD.log (append mode)."""
    logfile = LOG_DIR / f"{script_name}_{date.today()}.log"
    handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    return handler


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a logger that writes to both stdout *and* a dated file.
    Calling this multiple times in the same process reuses handlers,
    so you won’t get duplicate lines.
    """
    logger = logging.getLogger(name or "root")

    if logger.handlers:        # already configured – just return it
        return logger

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))

    # File handler – derive filename from the caller’s module name
    script = (name or "log").split(".")[-1]
    fh = _new_file_handler(script)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False   # stop double logging through root
    return logger
