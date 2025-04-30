"""Compute technical‑indicator features for ML model."""
import pandas as pd
import numpy as np


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["SMA_5"] = out["Close"].rolling(5).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["RSI_14"] = _rsi(out["Close"], 14)
    out["BB_MID"] = out["Close"].rolling(20).mean()
    out["BB_STD"] = out["Close"].rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + 2 * out["BB_STD"]
    out["BB_LOWER"] = out["BB_MID"] - 2 * out["BB_STD"]
    # lagged returns
    out["Return_1"] = out["Return"].shift(1)
    out["Return_2"] = out["Return"].shift(2)
    out["Return_5"] = out["Return"].shift(5)
    
    # 7. ATR (Average True Range) – 14 periods
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs()
    ], axis=1).max(axis=1)
    out["ATR_14"] = tr.rolling(14).mean()

    # 8. 1-bar momentum (redundant but explicit)
    out["Mom_1"] = out["Return"]

    # 9-10. Time-of-day cyclical encoding
    if isinstance(df.index, pd.DatetimeIndex):
        minutes = df.index.hour * 60 + df.index.minute
        day_minutes = 1440  # 24×60
        out["TOD_sin"] = np.sin(2 * np.pi * minutes / day_minutes)
        out["TOD_cos"] = np.cos(2 * np.pi * minutes / day_minutes)

    # 11. VWAP gap (requires intraday volume/price)
    if "Volume" in df.columns:
        pv = (df["Close"] * df["Volume"]).cumsum()
        cum_vol = df["Volume"].cumsum().replace(0, np.nan)
        vwap = pv / cum_vol
        out["VWAP_gap"] = df["Close"] / vwap - 1

    
    out.dropna(inplace=True)
    return out