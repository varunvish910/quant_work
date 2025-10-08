#!/usr/bin/env python3
"""
Deep debug of why quote API isn't returning data for trades
"""

from polygon import RESTClient
import requests
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
print("DEEP DEBUG: QUOTE API FOR TRADES")
print("="*80)

# Load one SPY trade
print(f"\n📊 Loading one SPY trade...")
with gzip.open(TRADES_FILE, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
        if len(spy_chunk) > 0:
            trade = spy_chunk.iloc[0]
            break

ticker = trade['ticker']
price = trade['price']
timestamp_ns = trade['sip_timestamp']
size = trade['size']

dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)

print(f"\n   Ticker: {ticker}")
print(f"   Price: ${price:.2f}")
print(f"   Size: {size}")
print(f"   Timestamp (ns): {timestamp_ns}")
print(f"   Time (UTC): {dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")

# Test 1: Direct REST API call (no SDK)
print(f"\n{'='*80}")
print("TEST 1: Direct REST API Call (no SDK)")
print(f"{'='*80}")

# Build URL
base_url = f"https://api.polygon.io/v3/quotes/{ticker}"

# Try different timestamp formats
test_params = [
    {
        'name': 'Nanoseconds',
        'timestamp.gte': timestamp_ns - 10_000_000_000,
        'timestamp.lte': timestamp_ns + 10_000_000_000,
        'limit': 10,
        'apiKey': API_KEY
    },
    {
        'name': 'Milliseconds',
        'timestamp.gte': int((timestamp_ns - 10_000_000_000) / 1_000_000),
        'timestamp.lte': int((timestamp_ns + 10_000_000_000) / 1_000_000),
        'limit': 10,
        'apiKey': API_KEY
    },
    {
        'name': 'Date string (no time)',
        'timestamp.gte': '2025-10-06',
        'timestamp.lte': '2025-10-06',
        'limit': 10,
        'apiKey': API_KEY
    },
]

for params in test_params:
    test_name = params.pop('name')
    print(f"\n🧪 {test_name}:")
    print(f"   URL: {base_url}")
    print(f"   Params: {params}")

    try:
        response = requests.get(base_url, params=params, timeout=30)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")

            if 'results' in data:
                print(f"   Results count: {len(data.get('results', []))}")
                if len(data.get('results', [])) > 0:
                    quote = data['results'][0]
                    print(f"   First quote: {quote}")

            if 'status' in data:
                print(f"   Status: {data['status']}")
            if 'message' in data:
                print(f"   Message: {data['message']}")
        else:
            print(f"   Error: {response.text[:200]}")

    except Exception as e:
        print(f"   Exception: {e}")

    params['name'] = test_name  # Restore for next iteration

# Test 2: Check if we can get ANY quotes for this ticker
print(f"\n{'='*80}")
print("TEST 2: Get ANY quotes for this ticker (no filters)")
print(f"{'='*80}")

url = f"https://api.polygon.io/v3/quotes/{ticker}"
params = {
    'limit': 5,
    'order': 'desc',
    'apiKey': API_KEY
}

print(f"\n   URL: {url}")
print(f"   Params: {params}")

try:
    response = requests.get(url, params=params, timeout=30)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"   Response: {data}")

        if 'results' in data and len(data['results']) > 0:
            print(f"\n   ✅ Found {len(data['results'])} quotes!")
            for i, q in enumerate(data['results'], 1):
                q_ts = q.get('sip_timestamp', 0)
                q_dt = datetime.fromtimestamp(q_ts / 1_000_000_000, tz=timezone.utc) if q_ts else 'N/A'
                print(f"\n   Quote {i}:")
                print(f"      Timestamp: {q_ts}")
                print(f"      Time: {q_dt}")
                print(f"      Bid: ${q.get('bid_price', 0):.2f}")
                print(f"      Ask: ${q.get('ask_price', 0):.2f}")
        else:
            print(f"\n   ❌ No quotes found")
    else:
        print(f"   Error: {response.text}")

except Exception as e:
    print(f"   Exception: {e}")

# Test 3: Try using Python SDK with different parameters
print(f"\n{'='*80}")
print("TEST 3: Python SDK with verbose output")
print(f"{'='*80}")

client = RESTClient(API_KEY)

print(f"\n   Using list_quotes with timestamp_gte and timestamp_lte...")
print(f"   Window: ±10 seconds around trade")

try:
    quotes = []
    iterator = client.list_quotes(
        ticker=ticker,
        timestamp_gte=timestamp_ns - 10_000_000_000,
        timestamp_lte=timestamp_ns + 10_000_000_000,
        limit=50,
        order='asc'
    )

    print(f"   Iterator created: {iterator}")

    for idx, quote in enumerate(iterator):
        quotes.append(quote)
        if idx < 3:  # Print first 3
            print(f"\n   Quote {idx+1}:")
            print(f"      Timestamp: {quote.sip_timestamp}")
            print(f"      Bid: ${quote.bid_price:.2f}")
            print(f"      Ask: ${quote.ask_price:.2f}")

        if idx >= 49:  # Limit to 50
            break

    print(f"\n   Total quotes found: {len(quotes)}")

except Exception as e:
    print(f"   Exception: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check API limits/rate limiting
print(f"\n{'='*80}")
print("TEST 4: Check API rate limit status")
print(f"{'='*80}")

url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2025-10-06/2025-10-06"
params = {'apiKey': API_KEY}

try:
    response = requests.get(url, params=params, timeout=10)

    print(f"   Request to: {url}")
    print(f"   Status: {response.status_code}")
    print(f"\n   Response Headers:")
    for key in ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-Request-ID']:
        if key in response.headers:
            print(f"      {key}: {response.headers[key]}")

except Exception as e:
    print(f"   Exception: {e}")

print(f"\n{'='*80}")
print("Debug complete!")
print(f"{'='*80}")
