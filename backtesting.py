"""Run one back-test of the ML strategy (callable).

Use run_once() programmatically or execute the file directly for
a single default run.

Example short-window test:
    python backtesting.py  # default full-period
Or from code:
    from backtesting import run_once
    res = run_once(start_date="2025-01-02", end_date="2025-03-31",
                   p_long=0.58, p_short=0.42, max_long_short=10)
"""

from datetime import datetime as _dt
from typing import Optional, Dict, Any

import backtrader as bt
from logger_setup import get_logger

from config import INITIAL_CASH
from data_ingestion import load_price_data
from strategy import MLTradingStrategy

log = get_logger(__name__)


def run_once(
    *,
    p_long: float = 0.58,
    p_short: float = 0.42,
    max_long_short: int = 10,
    trail_percent: float = 0.05,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single back-test with the given parameters."""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(leverage=1.0)  # cash-only

    # date window
    fd = _dt.fromisoformat(start_date).date() if start_date else None
    td = _dt.fromisoformat(end_date).date() if end_date else None

    # data feeds
    for tkr, df in load_price_data().items():
        # ── make index tz-naive so date slicing works everywhere
        if getattr(df.index, "tz", None) is not None:
            df = df.tz_localize(None)
        if fd or td:
            df = df.loc[fd:td]
        cerebro.adddata(
            bt.feeds.PandasData(
                dataname=df,
                name=tkr,
                fromdate=fd,
                todate=td,
            )
        )

    # strategy
    cerebro.addstrategy(
        MLTradingStrategy,
        p_long=p_long,
        p_short=p_short,
        max_long_short=max_long_short,
        trail_percent=trail_percent,
    )

    # analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run()[0]

    final   = cerebro.broker.getvalue()
    sharpe  = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    mdd     = strat.analyzers.dd.get_analysis()["max"]["drawdown"]
    tradesA = strat.analyzers.trades.get_analysis()
    closed  = tradesA.total.closed if hasattr(tradesA, "total") else tradesA.get("total", 0)

    results = dict(final=final, sharpe=sharpe, mdd=mdd, trades=closed)

    # tolerate None and keep %s placeholders
    log.info(
        "Run p_long=%.2f p_short=%.2f maxLS=%d trail=%.2f%%  →  "
        "Final %.2f  Sharpe %s  MaxDD %.2f%%  Trades %d",
        p_long, p_short, max_long_short, trail_percent * 100,
        final, f"{sharpe:.3f}" if sharpe is not None else "nan",
        mdd, closed
    )
    
    return results


if __name__ == "__main__":
    run_once()   # full-period default
