"""Minimal daily‑bar live deployment via Alpaca."""
import backtrader as bt
import alpaca_backtrader_api as alpaca
from strategy import MLTradingStrategy
from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER


def run_live(symbols: list[str]):
    store = alpaca.AlpacaStore(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
        paper=ALPACA_PAPER,
        usePolygon=False,
    )
    broker = store.getbroker()
    cerebro = bt.Cerebro()
    cerebro.setbroker(broker)

    for sym in symbols:
        data = store.getdata(dataname=sym, timeframe=bt.TimeFrame.Days)
        cerebro.adddata(data)

    cerebro.addstrategy(MLTradingStrategy)
    print("🚀 starting live trading … (Ctrl+C to exit)")
    cerebro.run()


if __name__ == "__main__":
    run_live(["AAPL", "MSFT", "TSLA"])  # or any list; unlimited universe supported