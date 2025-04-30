# universe/refresh_universe.py
"""Download latest top‑200 by 30‑day dollar volume via Alpaca data API."""

import datetime as dt
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest

# Load .env API keys
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# Alpaca clients
client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

OUT_PATH = Path("universe/top200.csv")

def refresh():
    end = dt.date.today()
    start = end - dt.timedelta(days=30)

    # Step 1: Get all tradable U.S. assets
    assets = trading_client.get_all_assets(GetAssetsRequest(status="active"))
    symbols = [a.symbol for a in assets if a.tradable and a.exchange in ("NYSE", "NASDAQ")]

    print(f"Found {len(symbols)} tradable tickers...")

    # Step 2: Get 30-day bars in batches of 200
    all_records = []

    for i in range(0, len(symbols), 200):
        batch = symbols[i:i + 200]
        try:
            bars = client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end
                )
            )
            df = bars.df
            if df.empty:
                continue

            # Step 3: Compute average dollar volume
            grouped = df.groupby("symbol").agg({"close": "mean", "volume": "mean"})
            grouped["dollar_volume"] = grouped["close"] * grouped["volume"]
            all_records.append(grouped[["dollar_volume"]])

        except Exception as e:
            print(f"Failed on batch {i}–{i+200}: {e}")

    # Step 4: Combine & rank top 200
    if not all_records:
        print("No data returned – exiting.")
        return

    full = pd.concat(all_records)
    top200 = full.sort_values("dollar_volume", ascending=False).head(200)
    top200.reset_index().to_csv(OUT_PATH, index=False)
    print(f"Saved top 200 to {OUT_PATH}")

if __name__ == "__main__":
    refresh()
