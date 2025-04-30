# live/broker_alpaca.py
"""Alpaca live broker integration for Backtrader using API v2."""
import backtrader as bt
from alpaca.trading.client import TradingClient

class AlpacaV2Store(bt.broker.BrokerBase):
    pass  # TODO full store implementation