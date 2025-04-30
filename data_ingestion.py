"""Load parquet price data and return a dict[ticker → OHLCV DataFrame].

• If DATA_DIR contains **1‑minute bars** (duplicate timestamps), they’re resampled to
  *RESAMPLE_MINUTES* (default 15) using OHLC + sum(volume).
• If RESAMPLE_MINUTES is None, data are aggregated to *daily* bars.
• If DATA_DIR already holds aggregated bars, they’re loaded as‑is.

All column names are normalised to Title‑case so downstream code can assume
`Open/High/Low/Close/Volume`.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
from config import DATA_DIR, RESAMPLE_MINUTES

# ─────────────────────────────────────────────────────────────────────

REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def _list_tickers() -> List[str]:
    """Return ticker symbols from *.parquet filenames in DATA_DIR."""
    return sorted(os.path.splitext(f)[0] for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.title() for c in df.columns]
    for alt in ("Timestamp", "Datetime"):
        if alt in df.columns and "Date" not in df.columns:
            df.rename(columns={alt: "Date"}, inplace=True)
    # Drop single‑value Symbol column if present
    if "Symbol" in df.columns and df["Symbol"].nunique() == 1:
        df.drop(columns=["Symbol"], inplace=True)
    return df


def _aggregate(df: pd.DataFrame, minutes: int | None) -> pd.DataFrame:
    """Aggregate minute bars to *minutes*‑bars or daily if minutes is None."""
    df = df.set_index(pd.to_datetime(df["Date"]))
    rule = f"{minutes}min" if minutes else "1D"
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna()
    out.index.name = "Date"
    return out.sort_index()


def load_price_data(tickers: List[str] | None = None) -> Dict[str, pd.DataFrame]:
    if tickers is None:
        tickers = _list_tickers()

    data: Dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        fp = os.path.join(DATA_DIR, f"{tkr}.parquet")
        raw = pd.read_parquet(fp)
        raw = _norm_cols(raw)

        # Heuristic: duplicates in Date ⇒ minute bars
        if raw["Date"].duplicated().any():
            df = _aggregate(raw, RESAMPLE_MINUTES)
        else:
            raw["Date"] = pd.to_datetime(raw["Date"])
            df = raw.set_index("Date").sort_index()
            # If still minute‑level but unique timestamps and we WANT to resample:
            if RESAMPLE_MINUTES and (df.index.freq is None or df.index.freq < pd.Timedelta(f"{RESAMPLE_MINUTES}min")):
                df = _aggregate(df.reset_index(), RESAMPLE_MINUTES)

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{tkr}: missing {missing} after processing")
        data[tkr] = df
    return data