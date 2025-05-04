import backtrader as bt

class TaxAnalyzer(bt.Analyzer):
    """Flat-rate short-term tax approximation."""
    params = (("rate", 0.24),)        # combined Fed+state default

    def start(self):
        self.pnl = 0.0

    def notify_trade(self, trade):
        if trade.isclosed:
            self.pnl += trade.pnlcomm     # already net of comm/slippage

    def get_analysis(self):
        tax = max(0.0, self.pnl) * self.p.rate
        return {
            "gross_pnl": self.pnl,
            "tax_paid": tax,
            "net_after_tax": self.pnl - tax,
        }