#!/usr/bin/env python3
"""
Test the quote API approach with a small sample of trades
"""

import pandas as pd
import gzip
from pathlib import Path
import requests
import time

# Configuration
POLYGON_API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
DATE = '2025-10-06'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
TRADES_FILE = OUTPUT_DIR / f"{DATE}.csv.gz"

print("="*80)
print("TESTING QUOTE API APPROACH")
print("Testing with 10 sample trades")
print("="*80)

# Load trades
print(f"\n📊 Loading trades from {TRADES_FILE}...")
with gzip.open(TRADES_FILE, 'rt') as f:
    trades = pd.read_csv(f, nrows=100000)  # Read first 100k rows

# Filter for SPY
spy_trades = trades[trades['ticker'].str.match(r'^O:SPY\d', na=False)]
print(f"   Found {len(spy_trades):,} SPY trades in first 100k rows")

# Take small sample
sample_trades = spy_trades.head(10)
print(f"   Testing with {len(sample_trades)} trades")

print(f"\n📋 Sample trades:")
print(sample_trades[['ticker', 'price', 'size', 'sip_timestamp']].to_string())

# Test API for each trade
print(f"\n🧪 Testing Quote API...")

for idx, trade in sample_trades.iterrows():
    ticker = trade['ticker']
    price = trade['price']
    timestamp_ns = trade['sip_timestamp']
    timestamp_ms = int(timestamp_ns / 1_000_000)

    print(f"\n   Trade {idx+1}:")
    print(f"      Ticker: {ticker}")
    print(f"      Price: ${price:.2f}")
    print(f"      Timestamp: {timestamp_ns}")

    # Fetch quote
    url = f"https://api.polygon.io/v3/quotes/{ticker}"
    params = {
        'timestamp': timestamp_ms,
        'limit': 1,
        'order': 'desc',
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"      API Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                bid = result.get('bid_price', 0)
                ask = result.get('ask_price', 0)

                print(f"      Quote: Bid=${bid:.2f}, Ask=${ask:.2f}")

                # Classify
                mid = (bid + ask) / 2
                if price >= ask * 0.99:
                    direction = 'BUY'
                elif price <= bid * 1.01:
                    direction = 'SELL'
                elif price > mid:
                    direction = 'BUY'
                else:
                    direction = 'SELL'

                print(f"      Direction: {direction}")
            else:
                print(f"      ⚠️  No quote data in response")
                print(f"      Response: {data}")
        else:
            print(f"      ❌ API Error: {response.status_code}")
            print(f"      Response: {response.text[:200]}")

    except Exception as e:
        print(f"      ❌ Exception: {e}")

    # Rate limit
    time.sleep(0.2)

print(f"\n{'='*80}")
print("Test complete!")
print(f"{'='*80}")
