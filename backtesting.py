"""Run one back-test of the ML strategy (callable).

Example quick slice
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
from tax_analyzer import TaxAnalyzer

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

    cerebro.addanalyzer(TaxAnalyzer, _name="tax", rate=0.24)

    # ── NEW: 0.02 % slippage each way (≈ 0.04 % round-trip) ──
    cerebro.broker.set_slippage_perc(
        perc=0.0002,        # 0.02 %
        slip_open=True,     # apply on entry
        slip_match=True     # and on exit/stop
)

    # forbid entering new shorts when cash is negative
    cerebro.broker.set_shortcash(False)

    # date window (naive dates)
    fd = _dt.fromisoformat(start_date).date() if start_date else None
    td = _dt.fromisoformat(end_date).date()   if end_date   else None

    # ─────────────────── add data feeds ────────────────────────────
    for tkr, df in load_price_data().items():
        # 1. drop timezone
        if getattr(df.index, "tz", None) is not None:
            df = df.tz_localize(None)

        # 2. slice to window
        if fd or td:
            df = df.loc[fd:td]

        # 3. basic hygiene: remove bad rows & dups
        df = df.loc[df["Close"] >= 1.00]           # price floor every row
        df = df[df["Volume"] > 0]
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        df = df[~df.index.duplicated(keep="last")]


        # 4. skip too-short feeds
        if df.empty or len(df) < 30:
            continue

        cerebro.adddata(
            bt.feeds.PandasData(
                dataname=df,
                name=tkr,
                fromdate=fd,
                todate=td,
            )
        )

    # ─────────────────── strategy & analyzers ──────────────────────
    cerebro.addstrategy(
        MLTradingStrategy,
        p_long=p_long,
        p_short=p_short,
        max_long_short=max_long_short,
        trail_percent=trail_percent,
    )
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run()[0]

    # ─────────────────── results ───────────────────────────────────
    final   = cerebro.broker.getvalue()
    sharpe  = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    mdd     = strat.analyzers.dd.get_analysis()["max"]["drawdown"]
    tradesA = strat.analyzers.trades.get_analysis()
    closed  = tradesA["total"]["closed"] if "total" in tradesA else 0

    # NEW — tax analyzer results
    tax = strat.analyzers.tax.get_analysis()   # {'gross_pnl': …, 'tax_paid': …, 'net_after_tax': …}

    # CAGR (only if window ≥ 30 days and equity positive)
    window_days = (td - fd).days if fd and td else 0
    if final > 0 and window_days >= 30:
        years = window_days / 365.25
        cagr  = (final / INITIAL_CASH) ** (1 / years) - 1
    else:
        cagr = None

    results = dict(final=final, sharpe=sharpe, mdd=mdd,
                   trades=closed, cagr=cagr)
    
    results.update(tax)        # ← merge the three tax keys into results

    log.info(
        "Run p_long=%.2f p_short=%.2f maxLS=%d trail=%.2f%%  →  "
        "Final %.2f  CAGR %s  Sharpe %s  MaxDD %.2f%%  Trades %d",
        p_long, p_short, max_long_short, trail_percent * 100,
        final,
        f"{cagr:.2%}" if cagr is not None else "nan",
        f"{sharpe:.3f}" if sharpe is not None else "nan",
        mdd, closed
    )
    return results


if __name__ == "__main__":
    run_once()      # full-period default
