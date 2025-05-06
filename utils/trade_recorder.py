# utils/trade_recorder.py
from __future__ import annotations
import backtrader as bt
from datetime import datetime

class TradeRecorder(bt.Analyzer):
    """
    Collect every *closed* trade in a list of dicts.
    The analysis dict looks like:  {"trades": [ {...}, {...}, ... ]}
    """
    def start(self):
        self._records = []

    def notify_trade(self, trade: bt.Trade):
        if not trade.isclosed:
            return
        bar_dt = self.strategy.datas[0].datetime.datetime()
        self._records.append({
            "datetime": bar_dt,               # bar that closed the trade
            "ticker":   trade.data._name,
            "size":     trade.size,
            "price_in": trade.price,
            "price_out": trade.price + trade.pnlcomm / trade.size
                         if trade.size else None,
            "pnl":      trade.pnlcomm,
        })

    def get_analysis(self):
        return {"trades": self._records}
