"""Run one back-test of the ML strategy (callable).

Example quick slice
    from backtesting import run_once
    res = run_once(
        start_date="2025-01-02", end_date="2025-03-31",
        p_long=0.58, p_short=0.42, max_long_short=10,
        trail_percent=0.05, min_edge=0.001, trade_shorts=False,
        tickers=["AAPL","MSFT","GOOG"]
    )
"""

import argparse
import inspect
from datetime import datetime as _dt
from typing import Optional, Dict, Any, List

import backtrader as bt
from logger_setup import get_logger

from config import INITIAL_CASH, MIN_EDGE
from data_ingestion import load_price_data
from strategy import MLTradingStrategy
from tax_analyzer import TaxAnalyzer

log = get_logger(__name__)


def run_once(
    *,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    tickers:    Optional[List[str]] = None,
    # everything else is a hyperparam forwarded to the strategy:
    **params: Any
) -> Dict[str, Any]:
    """
    Run a single back-test.

    - `start_date`/`end_date`: ISO dates for slicing.
    - `tickers`: optional list to restrict universe.
    - All other settings (p_long, p_short, min_edge, trade_shorts, etc.)
      pass via `params` and are auto-logged & forwarded to the strategy.
    """

    # ─── Parse dates ────────────────────────────────────────────────
    fd = _dt.fromisoformat(start_date).date() if start_date else None
    td = _dt.fromisoformat(end_date).date()   if end_date   else None

    # ─── Snapshot hyper-parameters for logging ─────────────────────
    hp = params.copy()
    hp_str = " ".join(f"{k}={v!r}" for k, v in hp.items())

    # ─── Cerebro setup ─────────────────────────────────────────────
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(leverage=1.0)

    # Slippage (override via params["slip_perc"] if desired)
    slip_perc = params.get("slip_perc", 0.0002)
    cerebro.broker.set_slippage_perc(
        perc=slip_perc, slip_open=True, slip_match=True
    )

    # Prevent shorting against negative cash
    cerebro.broker.set_shortcash(False)

    # Tax analyzer (override via params["tax_rate"])
    tax_rate = params.get("tax_rate", 0.24)
    cerebro.addanalyzer(TaxAnalyzer, _name="tax", rate=tax_rate)

    # ─── Load & clean data ──────────────────────────────────────────
    price_data = load_price_data(tickers)
    for tkr, df in price_data.items():
        if getattr(df.index, "tz", None) is not None:
            df = df.tz_localize(None)

        if fd or td:
            df = df.loc[fd:td]

        # Basic hygiene
        df = df[df["Close"] >= 1.00]
        df = df[df["Volume"] > 0]
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        df = df[~df.index.duplicated(keep="last")]

        if df.empty or len(df) < params.get("min_bars", 30):
            continue

        cerebro.adddata(
            bt.feeds.PandasData(
                dataname=df, name=tkr, fromdate=fd, todate=td
            )
        )

    if not cerebro.datas:
        log.warning("No data feeds for %s→%s; skipping.", fd, td)
        return {
            "start": fd, "end": td, "final": None, "sharpe": None,
            "mdd": None, "trades": 0, "cagr": None,
            "gross_pnl": None, "tax_paid": None, "net_after_tax": None
        }

    # ─── Strategy & analyzers ───────────────────────────────────────
    cerebro.addstrategy(MLTradingStrategy, **params)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run()[0]

    # ─── Metrics ────────────────────────────────────────────────────
    final  = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    mdd    = strat.analyzers.dd.get_analysis()["max"]["drawdown"]
    trades = strat.analyzers.trades.get_analysis()["total"]["closed"]

    tax = strat.analyzers.tax.get_analysis()
    window_days = (td - fd).days if fd and td else 0
    cagr = (
        (final / INITIAL_CASH) ** (1 / (window_days / 365.25)) - 1
        if final > 0 and window_days >= 30 else None
    )

    results = {
        "start": fd or "FULL",
        "end":   td or "FULL",
        "final": final,
        "sharpe": sharpe,
        "mdd":    mdd,
        "trades": trades,
        "cagr":   cagr,
        **tax
    }

    log.info(
        "Run [%s→%s] %s → Final %.2f  CAGR %s  Sharpe %s  MaxDD %.2f%%  Trades %d",
        fd or "BEGIN", td or "END",
        hp_str,
        final,
        f"{cagr:.2%}"   if cagr   is not None else "nan",
        f"{sharpe:.3f}" if sharpe is not None else "nan",
        mdd,
        trades,
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one backtest")
    parser.add_argument("--start",   help="YYYY-MM-DD")
    parser.add_argument("--end",     help="YYYY-MM-DD")
    parser.add_argument(
        "--tickers", nargs="+",
        help="Optional list of tickers (default = all)"
    )
    args = parser.parse_args()

    run_once(
        start_date=args.start,
        end_date=args.end,
        tickers=args.tickers
    )
