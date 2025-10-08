#!/usr/bin/env python3
"""
Simple test of quote API with real SPY trades
"""

import requests
import time

# Test data (from grep output)
test_trades = [
    {
        'ticker': 'O:SPY251006C00550000',
        'price': 119.96,
        'timestamp_ns': 1759759211412000000
    },
    {
        'ticker': 'O:SPY251006C00556000',
        'price': 114.42,
        'timestamp_ns': 1759761009076000000
    },
    {
        'ticker': 'O:SPY251006C00559000',
        'price': 111.43,
        'timestamp_ns': 1759761009076000000
    }
]

POLYGON_API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

print("="*80)
print("TESTING QUOTE API WITH REAL SPY TRADES")
print("="*80)

for i, trade in enumerate(test_trades, 1):
    print(f"\n🧪 Test {i}:")
    print(f"   Ticker: {trade['ticker']}")
    print(f"   Price: ${trade['price']:.2f}")

    # Convert timestamp
    timestamp_ms = int(trade['timestamp_ns'] / 1_000_000)

    # API request
    url = f"https://api.polygon.io/v3/quotes/{trade['ticker']}"
    params = {
        'timestamp': timestamp_ms,
        'limit': 1,
        'order': 'desc',
        'apiKey': POLYGON_API_KEY
    }

    print(f"   API URL: {url}")
    print(f"   Timestamp: {timestamp_ms}")

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")

            if 'results' in data:
                if len(data['results']) > 0:
                    result = data['results'][0]
                    bid = result.get('bid_price', 0)
                    ask = result.get('ask_price', 0)

                    print(f"   ✅ Quote found:")
                    print(f"      Bid: ${bid:.2f}")
                    print(f"      Ask: ${ask:.2f}")
                    print(f"      Mid: ${(bid + ask) / 2:.2f}")

                    # Classify
                    mid = (bid + ask) / 2
                    if trade['price'] >= ask * 0.99:
                        direction = 'BUY'
                    elif trade['price'] <= bid * 1.01:
                        direction = 'SELL'
                    elif trade['price'] > mid:
                        direction = 'BUY'
                    else:
                        direction = 'SELL'

                    print(f"      Direction: {direction}")
                else:
                    print(f"   ⚠️  No results in response")
            else:
                print(f"   ⚠️  No 'results' key in response")
                print(f"   Full response: {data}")
        else:
            print(f"   ❌ Error {response.status_code}")
            print(f"   Response: {response.text[:300]}")

    except Exception as e:
        print(f"   ❌ Exception: {e}")

    # Rate limit
    time.sleep(1)

print(f"\n{'='*80}")
print("Test complete!")
print(f"{'='*80}")
