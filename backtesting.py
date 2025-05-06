"""Run one back-test of the ML strategy (callable)."""

from __future__ import annotations
import argparse
from datetime import datetime as _dt
from typing import Optional, Dict, Any, List

import pandas as pd
import backtrader as bt
from logger_setup import get_logger

from config import INITIAL_CASH
from data_ingestion import load_price_data
from strategy import MLTradingStrategy
from tax_analyzer import TaxAnalyzer

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
def run_once(
    *,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    tickers:    Optional[List[str]] = None,
    **params: Any,          # p_long, p_short, min_edge, trade_shorts, …
) -> Dict[str, Any]:
    """Run a single back-test and return a metrics dict."""

    # ── time window (full-day inclusive) ────────────────────────────
    fd = pd.to_datetime(start_date) if start_date else None
    td = (
        pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        if end_date else None
    )

    hp_str = " ".join(f"{k}={v!r}" for k, v in params.items())

    # ── Cerebro core ────────────────────────────────────────────────
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(leverage=1.0)

    cerebro.broker.set_slippage_perc(
        perc=params.get("slip_perc", 0.0002), slip_open=True, slip_match=True
    )
    cerebro.broker.set_shortcash(False)

    cerebro.addanalyzer(TaxAnalyzer, _name="tax", rate=params.get("tax_rate", 0.24))

    # ── load once, then slice & clean ───────────────────────────────
    price_data = load_price_data(tickers)
    for tkr, df in price_data.items():
        if getattr(df.index, "tz", None):
            df = df.tz_localize(None)

        if fd is not None or td is not None:
            df = df.loc[fd:td]

        df = df[
            (df["Close"] >= 1.0)
            & (df["Volume"] > 0)
        ].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        df = df[~df.index.duplicated(keep="last")]

        if len(df) < params.get("min_bars", 30):
            continue

        cerebro.adddata(
            bt.feeds.PandasData(dataname=df, name=tkr, fromdate=fd, todate=td)
        )

    if not cerebro.datas:
        log.warning("No data feeds for %s → %s; skipped.", fd, td)
        return {
            "start": fd, "end": td, "final": None, "sharpe": None,
            "mdd": None, "trades": 0, "cagr": None,
            "gross_pnl": None, "tax_paid": None, "net_after_tax": None,
        }

    log.info("→ Running on %d tickers", len(cerebro.datas))

    # ── strategy & analyzers ────────────────────────────────────────
    cerebro.addstrategy(MLTradingStrategy, **params)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown,     _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run()[0]

    # ── metrics ─────────────────────────────────────────────────────
    final  = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    mdd    = strat.analyzers.dd.get_analysis()["max"]["drawdown"]
    trades = strat.analyzers.trades.get_analysis().get("total", {}).get("closed", 0)
    tax    = strat.analyzers.tax.get_analysis()

    if fd is not None and td is not None and final > 0 and (td - fd).days >= 30:
        years = (td - fd).days / 365.25
        cagr  = (final / INITIAL_CASH) ** (1 / years) - 1
    else:
        cagr = None

    results = {
        "start": fd or "FULL",
        "end":   td or "FULL",
        "final": final,
        "sharpe": sharpe,
        "mdd":    mdd,
        "trades": trades,
        "cagr":   cagr,
        **tax,
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


# ── CLI helper ───────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run one back-test")
    p.add_argument("--start",   help="YYYY-MM-DD")
    p.add_argument("--end",     help="YYYY-MM-DD")
    p.add_argument("--tickers", nargs="+", help="Restrict to these tickers")
    args = p.parse_args()

    run_once(start_date=args.start, end_date=args.end, tickers=args.tickers)
