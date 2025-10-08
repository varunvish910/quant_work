#!/usr/bin/env python3
"""
Debug why 45% of trades have no quotes
"""

from polygon import RESTClient
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime

# Configuration
API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
DATE = '2025-10-06'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
TRADES_FILE = OUTPUT_DIR / f"{DATE}.csv.gz"

print("="*80)
print("DEBUGGING NO-QUOTE RATE")
print("="*80)

# Load 10 SPY trades for detailed debugging
print(f"\n📊 Loading 10 SPY trades for debugging...")
spy_trades = []
with gzip.open(TRADES_FILE, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)
            if len(pd.concat(spy_trades)) >= 10:
                break

trades = pd.concat(spy_trades, ignore_index=True).head(10)
print(f"   ✅ Loaded {len(trades)} trades\n")

# Initialize client
client = RESTClient(API_KEY)

# Debug each trade
for idx, trade in trades.iterrows():
    print(f"\n{'='*80}")
    print(f"TRADE {idx + 1}")
    print(f"{'='*80}")

    ticker = trade['ticker']
    price = trade['price']
    timestamp_ns = trade['sip_timestamp']
    size = trade['size']

    # Parse timestamp
    timestamp_sec = timestamp_ns / 1_000_000_000
    dt = datetime.fromtimestamp(timestamp_sec)

    print(f"\n📊 Trade Details:")
    print(f"   Ticker: {ticker}")
    print(f"   Price: ${price:.2f}")
    print(f"   Size: {size}")
    print(f"   Timestamp (ns): {timestamp_ns}")
    print(f"   Timestamp (dt): {dt}")
    print(f"   Date/Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # Time window
    window_start = timestamp_ns - 1_000_000_000  # 1 second before
    window_end = timestamp_ns + 1_000_000_000    # 1 second after

    print(f"\n🕐 Time Windows to Try:")

    # Try different window sizes
    windows = [
        ('±1 second', 1_000_000_000),
        ('±5 seconds', 5_000_000_000),
        ('±10 seconds', 10_000_000_000),
        ('±30 seconds', 30_000_000_000),
        ('±1 minute', 60_000_000_000),
    ]

    for window_name, window_size in windows:
        try:
            quotes = []
            for quote in client.list_quotes(
                ticker=ticker,
                timestamp_gte=timestamp_ns - window_size,
                timestamp_lte=timestamp_ns + window_size,
                order="asc",
                limit=10,
                sort="timestamp",
            ):
                quotes.append(quote)

            print(f"   {window_name:15s}: {len(quotes):3d} quotes", end='')

            if len(quotes) > 0:
                # Find closest quote before trade
                pre_trade = [q for q in quotes if q.sip_timestamp <= timestamp_ns]
                if pre_trade:
                    closest = pre_trade[-1]
                    time_diff = (timestamp_ns - closest.sip_timestamp) / 1_000_000_000
                    print(f" (closest: {time_diff:.3f}s before trade)")
                else:
                    print(f" (all quotes AFTER trade)")
            else:
                print("")

        except Exception as e:
            print(f"   {window_name:15s}: ERROR - {e}")

    # Also try getting ANY quotes for this ticker on this day
    print(f"\n🔍 Checking if ticker has ANY quotes on {DATE}:")

    # Get start/end of trading day (9:30 AM - 4:00 PM ET)
    day_start = datetime(2025, 10, 6, 9, 30, 0).timestamp() * 1_000_000_000
    day_end = datetime(2025, 10, 6, 16, 0, 0).timestamp() * 1_000_000_000

    try:
        all_quotes = []
        for quote in client.list_quotes(
            ticker=ticker,
            timestamp_gte=int(day_start),
            timestamp_lte=int(day_end),
            order="asc",
            limit=100,
            sort="timestamp",
        ):
            all_quotes.append(quote)

        print(f"   Found {len(all_quotes)} quotes during trading hours")

        if len(all_quotes) > 0:
            first_quote = all_quotes[0]
            last_quote = all_quotes[-1]

            first_dt = datetime.fromtimestamp(first_quote.sip_timestamp / 1_000_000_000)
            last_dt = datetime.fromtimestamp(last_quote.sip_timestamp / 1_000_000_000)

            print(f"   First quote: {first_dt.strftime('%H:%M:%S')}")
            print(f"   Last quote:  {last_dt.strftime('%H:%M:%S')}")
            print(f"   Trade time:  {dt.strftime('%H:%M:%S')}")

            # Check if trade is outside quote range
            if timestamp_ns < first_quote.sip_timestamp:
                print(f"   ⚠️  Trade occurred BEFORE first quote!")
            elif timestamp_ns > last_quote.sip_timestamp:
                print(f"   ⚠️  Trade occurred AFTER last quote!")
            else:
                print(f"   ✅ Trade within quote range")
        else:
            print(f"   ❌ NO quotes found for this ticker on {DATE}")
            print(f"   This option may be:")
            print(f"      - Very illiquid (no market makers)")
            print(f"      - Newly listed")
            print(f"      - Not covered by quote feed")

    except Exception as e:
        print(f"   ERROR: {e}")

print(f"\n{'='*80}")
print("Debug complete!")
print(f"{'='*80}")
