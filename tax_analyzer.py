import backtrader as bt

class TaxAnalyzer(bt.Analyzer):
    params = (("rate", 0.24),)          # federal + state short-term

    def start(self):
        self.gross_pnl = 0.0

    def notify_trade(self, trade):
        if trade.isclosed:
            self.gross_pnl += trade.pnlcomm   # already net of slippage & comm

    def get_analysis(self):
        tax = max(0.0, self.gross_pnl) * self.p.rate
        return {
            "gross_pnl": self.gross_pnl,
            "tax_paid": tax,
            "net_after_tax": self.gross_pnl - tax,
        }
