"""LightGBM probability-ranked intraday strategy (fixed 5 % stop)."""

from __future__ import annotations
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import backtrader as bt
import joblib

from config import MODEL_PATH, MAX_POSITION_PCT, CASH_BUFFER_PCT

# ───────────────────────── indicators ────────────────────────────
class _Indicators(bt.Indicator):
    lines = (
        "sma5", "sma20", "rsi14",
        "bb_upper", "bb_lower",
        "ret1", "ret2", "ret5",
        "atr14", "mom1",
    )

    def __init__(self):
        self.lines.sma5  = bt.ind.SMA(self.data.close, period=5)
        self.lines.sma20 = bt.ind.SMA(self.data.close, period=20)
        self.lines.rsi14 = bt.ind.RSI(self.data.close, period=14)
        bb = bt.ind.BollingerBands(self.data.close, period=20, devfactor=2)
        self.lines.bb_upper = bb.top
        self.lines.bb_lower = bb.bot
        self.lines.ret1 = (self.data.close / self.data.close(-1)) - 1
        self.lines.ret2 = (self.data.close / self.data.close(-2)) - 1
        self.lines.ret5 = (self.data.close / self.data.close(-5)) - 1
        self.lines.atr14 = bt.ind.ATR(self.data, period=14)
        self.lines.mom1  = self.lines.ret1


class MLProbabilisticStrategy(bt.Strategy):
    params = dict(
        min_bars=30,
        p_long=0.58,
        p_short=0.42,
        max_long_short=10,
        trail_percent=0.05,
    )

    FEATURE_COLS: List[str] = [
        "SMA_5", "SMA_20", "RSI_14", "BB_UPPER", "BB_LOWER",
        "Return_1", "Return_2", "Return_5",
        "ATR_14", "Mom_1", "TOD_sin", "TOD_cos", "VWAP_gap"
    ]

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.ind_map: Dict[bt.DataBase, _Indicators] = {
            d: _Indicators(d) for d in self.datas
        }

    def next(self):
        if len(self) < self.p.min_bars:
            return

        scores = []
        for d in self.datas:
            ind = self.ind_map[d]
            ts = d.datetime.datetime(0)
            minutes = ts.hour * 60 + ts.minute
            tod_sin = math.sin(2 * math.pi * minutes / 1440)
            tod_cos = math.cos(2 * math.pi * minutes / 1440)

            # 20-bar VWAP gap
            if len(d) >= 20:
                pv_sum = sum(d.close[-i] * d.volume[-i] for i in range(20))
                vol_sum = sum(d.volume[-i] for i in range(20))
                vwap_gap = d.close[0] / (pv_sum / vol_sum) - 1 if vol_sum else np.nan
            else:
                vwap_gap = np.nan

            feats = [
                ind.sma5[0], ind.sma20[0], ind.rsi14[0],
                ind.bb_upper[0], ind.bb_lower[0],
                ind.ret1[0], ind.ret2[0], ind.ret5[0],
                ind.atr14[0], ind.mom1[0],
                tod_sin, tod_cos, vwap_gap,
            ]
            if np.isnan(feats).any():
                continue

            proba = self.model.predict_proba(
                pd.DataFrame([feats], columns=self.FEATURE_COLS)
            )[0, 1]
            scores.append((d, proba))

        longs = [d for d, p in sorted(scores, key=lambda x: -x[1])
                 if p >= self.p.p_long][: self.p.max_long_short]
        shorts = [d for d, p in sorted(scores, key=lambda x:  x[1])
                  if p <= self.p.p_short][: self.p.max_long_short]

        if not longs and not shorts:
            return

        target_pct = min(
            MAX_POSITION_PCT,
            (1 - CASH_BUFFER_PCT) / (len(longs) + len(shorts)),
        )

        # Close stale
        for d in self.datas:
            pos = self.getposition(d).size
            if pos > 0 and d not in longs:
                self.close(d)
            elif pos < 0 and d not in shorts:
                self.close(d)

        # Open / rebalance
        for d in longs:
            self.order_target_percent(d, target_pct)
            if self.getposition(d).size == 0:
                self.sell(d, exectype=bt.Order.StopTrail,
                          trailpercent=self.p.trail_percent)

        for d in shorts:
            self.order_target_percent(d, -target_pct)
            if self.getposition(d).size == 0:
                self.buy(d, exectype=bt.Order.StopTrail,
                         trailpercent=self.p.trail_percent)


# alias for older import
MLTradingStrategy = MLProbabilisticStrategy
