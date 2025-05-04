"""LightGBM probability-ranked intraday strategy (slippage-aware).

Key upgrades
------------
* MIN_EDGE filter: trade only if |p − 0.5| ≥ MIN_EDGE
* Single trailing stop attached after entry (notify_order)
* Cleans duplicate target_pct logic
"""

from __future__ import annotations
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import backtrader as bt
import joblib

from config import MODEL_PATH, MAX_POSITION_PCT, CASH_BUFFER_PCT, MIN_EDGE


class _Indicators(bt.Indicator):
    lines = ("sma5", "sma20", "rsi14",
             "bb_upper", "bb_lower",
             "ret1", "ret2", "ret5",
             "atr14", "mom1")

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
        p_long=0.56,
        p_short=0.42,
        max_long_short=10,
        trail_percent=0.04,
        min_edge=MIN_EDGE,
    )

    FEATURE_COLS: List[str] = [
        "SMA_5", "SMA_20", "RSI_14", "BB_UPPER", "BB_LOWER",
        "Return_1", "Return_2", "Return_5",
        "ATR_14", "Mom_1", "TOD_sin", "TOD_cos", "VWAP_gap",
    ]

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.ind: Dict[bt.DataBase, _Indicators] = {d: _Indicators(d) for d in self.datas}
        self.entry_orders: Dict[bt.DataBase, bt.Order] = {}
        self.stop_orders: Dict[bt.DataBase, bt.Order] = {}

    # ---- attach one trailing stop after fill ---------------------
    def notify_order(self, order: bt.Order):
        if order.status != order.Completed:
            return
        d = order.data
        if self.entry_orders.get(d) is order:
            tp = self.p.trail_percent
            stop = (self.sell if order.size > 0 else self.buy)(
                d, exectype=bt.Order.StopTrail,
                trailpercent=tp, parent=order)
            self.stop_orders[d] = stop
            del self.entry_orders[d]

    # ---- main step ----------------------------------------------
    def next(self):
        if len(self) < self.p.min_bars:
            return

        scores = []
        for d in self.datas:
            ind = self.ind[d]
            ts = d.datetime.datetime(0)
            mins = ts.hour * 60 + ts.minute
            tod_sin = math.sin(2 * math.pi * mins / 1440)
            tod_cos = math.cos(2 * math.pi * mins / 1440)

            # 20-bar VWAP gap
            if len(d) >= 20:
                pv = sum(d.close[-i] * d.volume[-i] for i in range(20))
                vol = sum(d.volume[-i] for i in range(20))
                vwap_gap = d.close[0] / (pv / vol) - 1 if vol else np.nan
            else:
                vwap_gap = np.nan

            feats = [ind.sma5[0], ind.sma20[0], ind.rsi14[0],
                     ind.bb_upper[0], ind.bb_lower[0],
                     ind.ret1[0], ind.ret2[0], ind.ret5[0],
                     ind.atr14[0], ind.mom1[0],
                     tod_sin, tod_cos, vwap_gap]

            if np.isnan(feats).any():
                continue

            p_up = self.model.predict_proba(
                pd.DataFrame([feats], columns=self.FEATURE_COLS))[0, 1]
            scores.append((d, p_up))

        longs  = [d for d,p in scores
                  if p >= self.p.p_long  and (p - 0.5) >= self.p.min_edge]
        shorts = [d for d,p in scores
                  if p <= self.p.p_short and (0.5 - p) >= self.p.min_edge]

        longs  = sorted(longs,  key=lambda d: -dict(scores)[d])[: self.p.max_long_short]
        shorts = sorted(shorts, key=lambda d:  dict(scores)[d])[: self.p.max_long_short]

        if not longs and not shorts:
            return

        base_pct  = (1 - CASH_BUFFER_PCT) / (len(longs) + len(shorts))
        target_pct = min(base_pct, MAX_POSITION_PCT)

        # ---- close stale ----------------------------------------
        for d in self.datas:
            pos = self.getposition(d).size
            if pos > 0 and d not in longs:
                self.close(d)
            elif pos < 0 and d not in shorts:
                self.close(d)

        # ---- open / rebalance -----------------------------------
        for d in longs:
            self.entry_orders[d] = self.order_target_percent(d,  target_pct)
        for d in shorts:
            self.entry_orders[d] = self.order_target_percent(d, -target_pct)


MLTradingStrategy = MLProbabilisticStrategy
