"""Run the full backtest and print high‑level stats."""
import backtrader as bt
from data_ingestion import load_price_data
from strategy import MLProbabilisticStrategy as MLTradingStrategy
from config import INITIAL_CASH

from logger_setup import get_logger
log = get_logger(__name__)


def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)

    cerebro.broker.setcommission(leverage=1.0)   # disable margin/over-shorting
    
    for tkr, df in load_price_data().items():
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=tkr))

    cerebro.addstrategy(MLTradingStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    log.info("Starting capital:", cerebro.broker.getvalue())
    res = cerebro.run()[0]
    log.info("Final capital:", cerebro.broker.getvalue())
    log.info("Sharpe:", res.analyzers.sharpe.get_analysis())
    log.info("MaxDD:", res.analyzers.dd.get_analysis()["max"]["drawdown"], "%")
    log.info("Trades:", res.analyzers.trades.get_analysis().total)


if __name__ == "__main__":
    run_backtest()