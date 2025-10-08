#!/usr/bin/env python3
"""
Debug why 4% of trades have no quotes even with ±10 second window
"""

from polygon import RESTClient
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime, timezone
import time

API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
DATE = '2025-10-06'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
TRADES_FILE = OUTPUT_DIR / f"{DATE}.csv.gz"

print("="*80)
print("DEBUGGING THE 4% MISSING QUOTES")
print("="*80)

# Load 100 trades and classify them
print(f"\n📊 Loading and classifying 100 trades...")
spy_trades = []
with gzip.open(TRADES_FILE, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)
            if len(pd.concat(spy_trades)) >= 100:
                break

trades = pd.concat(spy_trades, ignore_index=True).head(100)

# Parse option type
def parse_option_type(ticker):
    try:
        if not ticker.startswith('O:SPY'):
            return None
        parts = ticker[5:]
        if len(parts) >= 15:
            return parts[6].lower()
    except:
        pass
    return None

trades['option_type'] = trades['ticker'].apply(parse_option_type)

client = RESTClient(API_KEY)

# Track which trades have no quotes
no_quote_trades = []

for idx, trade in trades.iterrows():
    ticker = trade['ticker']
    timestamp_ns = trade['sip_timestamp']

    window_start = timestamp_ns - 10_000_000_000
    window_end = timestamp_ns + 10_000_000_000

    try:
        quotes = []
        for quote in client.list_quotes(
            ticker=ticker,
            timestamp_gte=window_start,
            timestamp_lte=window_end,
            limit=50
        ):
            quotes.append(quote)

        if len(quotes) == 0:
            no_quote_trades.append({
                'idx': idx,
                'ticker': ticker,
                'price': trade['price'],
                'size': trade['size'],
                'timestamp_ns': timestamp_ns,
                'option_type': trade['option_type']
            })
    except Exception as e:
        no_quote_trades.append({
            'idx': idx,
            'ticker': ticker,
            'price': trade['price'],
            'size': trade['size'],
            'timestamp_ns': timestamp_ns,
            'option_type': trade['option_type'],
            'error': str(e)
        })

    if idx % 20 == 0:
        print(f"   Processed {idx+1}/100 trades...")

    if idx % 5 == 0:
        time.sleep(1)

print(f"\n✅ Found {len(no_quote_trades)} trades with no quotes")

if len(no_quote_trades) == 0:
    print("\n🎉 All trades have quotes! (This run had 0% missing)")
    print("   The 4% missing rate was from the previous run")
    print("   It may vary run-to-run due to:")
    print("      - API timeouts")
    print("      - Network issues")
    print("      - Temporary API unavailability")
else:
    print(f"\n🔍 Analyzing the {len(no_quote_trades)} trades with no quotes:")

    for i, trade in enumerate(no_quote_trades, 1):
        print(f"\n{'='*80}")
        print(f"MISSING QUOTE TRADE #{i}")
        print(f"{'='*80}")

        ticker = trade['ticker']
        timestamp_ns = trade['timestamp_ns']

        dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)

        print(f"\n   Ticker: {ticker}")
        print(f"   Price: ${trade['price']:.2f}")
        print(f"   Size: {trade['size']}")
        print(f"   Timestamp: {timestamp_ns}")
        print(f"   Time (UTC): {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Option type: {trade['option_type']}")

        if 'error' in trade:
            print(f"   Error: {trade['error']}")

        # Try to get ANY quotes for this ticker on this day
        print(f"\n   🔍 Checking if ticker has ANY quotes on {DATE}:")

        # Market hours: 9:30 AM - 4:00 PM ET (13:30 - 20:00 UTC)
        day_start_utc = datetime(2025, 10, 6, 13, 30, 0, tzinfo=timezone.utc).timestamp() * 1_000_000_000
        day_end_utc = datetime(2025, 10, 6, 20, 0, 0, tzinfo=timezone.utc).timestamp() * 1_000_000_000

        try:
            all_quotes = []
            for quote in client.list_quotes(
                ticker=ticker,
                timestamp_gte=int(day_start_utc),
                timestamp_lte=int(day_end_utc),
                limit=100
            ):
                all_quotes.append(quote)
                if len(all_quotes) >= 100:
                    break

            if len(all_quotes) > 0:
                print(f"      ✅ Found {len(all_quotes)} quotes during market hours")

                first_q = all_quotes[0]
                last_q = all_quotes[-1]

                first_dt = datetime.fromtimestamp(first_q.sip_timestamp / 1_000_000_000, tz=timezone.utc)
                last_dt = datetime.fromtimestamp(last_q.sip_timestamp / 1_000_000_000, tz=timezone.utc)

                print(f"      First quote: {first_dt.strftime('%H:%M:%S')}")
                print(f"      Last quote:  {last_dt.strftime('%H:%M:%S')}")
                print(f"      Trade time:  {dt.strftime('%H:%M:%S')}")

                # Check if trade is outside quote range
                if timestamp_ns < first_q.sip_timestamp:
                    diff_sec = (first_q.sip_timestamp - timestamp_ns) / 1_000_000_000
                    print(f"      ⚠️  Trade {diff_sec:.1f} seconds BEFORE first quote")
                elif timestamp_ns > last_q.sip_timestamp:
                    diff_sec = (timestamp_ns - last_q.sip_timestamp) / 1_000_000_000
                    print(f"      ⚠️  Trade {diff_sec:.1f} seconds AFTER last quote")
                else:
                    print(f"      🤔 Trade within quote range, but no quotes in ±10s window")
                    print(f"         This suggests:")
                    print(f"            - Very illiquid option (sparse quotes)")
                    print(f"            - Quotes more than 10 seconds apart")

                # Try wider window
                print(f"\n      🧪 Trying ±60 second window:")
                wide_window_quotes = []
                for quote in client.list_quotes(
                    ticker=ticker,
                    timestamp_gte=timestamp_ns - 60_000_000_000,
                    timestamp_lte=timestamp_ns + 60_000_000_000,
                    limit=10
                ):
                    wide_window_quotes.append(quote)

                print(f"         Found {len(wide_window_quotes)} quotes in ±60s window")

                if len(wide_window_quotes) > 0:
                    closest = min(wide_window_quotes, key=lambda q: abs(q.sip_timestamp - timestamp_ns))
                    time_diff = abs(closest.sip_timestamp - timestamp_ns) / 1_000_000_000
                    print(f"         Closest quote: {time_diff:.1f} seconds away")
                    print(f"         → Option is ILLIQUID (quotes >10s apart)")
            else:
                print(f"      ❌ NO quotes found during entire market day")
                print(f"         This option is either:")
                print(f"            - Extremely illiquid")
                print(f"            - Newly listed")
                print(f"            - Expired/delisted")
                print(f"            - Not in quote feed")

        except Exception as e:
            print(f"      ❌ Error: {e}")

print(f"\n{'='*80}")
print("Debug complete!")
print(f"{'='*80}")
