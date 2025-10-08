#!/usr/bin/env python3
"""
Test different timestamp formats with Polygon API
"""

from polygon import RESTClient
from datetime import datetime, timezone

API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

# Known trade details
ticker = 'O:SPY251006C00550000'
trade_timestamp_ns = 1759759211412000000  # 2025-10-06 14:00:11 UTC

print("="*80)
print("TESTING DIFFERENT TIMESTAMP FORMATS")
print("="*80)

print(f"\n📊 Target:")
print(f"   Ticker: {ticker}")
print(f"   Trade timestamp: {trade_timestamp_ns}")

dt = datetime.fromtimestamp(trade_timestamp_ns / 1_000_000_000, tz=timezone.utc)
print(f"   Trade time (UTC): {dt.strftime('%Y-%m-%d %H:%M:%S')}")

client = RESTClient(API_KEY)

# Test 1: Try with nanosecond timestamps (what we're using)
print(f"\n🧪 Test 1: Using nanosecond timestamps")
window_start_ns = trade_timestamp_ns - 10_000_000_000  # 10 seconds before
window_end_ns = trade_timestamp_ns + 10_000_000_000    # 10 seconds after

print(f"   Window: {window_start_ns} to {window_end_ns}")

try:
    quotes = []
    for q in client.list_quotes(
        ticker=ticker,
        timestamp_gte=window_start_ns,
        timestamp_lte=window_end_ns,
        limit=10
    ):
        quotes.append(q)
    print(f"   Result: {len(quotes)} quotes found")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Try with millisecond timestamps
print(f"\n🧪 Test 2: Using millisecond timestamps")
window_start_ms = int(trade_timestamp_ns / 1_000_000)  # Convert to milliseconds
window_end_ms = window_start_ms + 10000  # 10 seconds

print(f"   Window: {window_start_ms} to {window_end_ms}")

try:
    quotes = []
    for q in client.list_quotes(
        ticker=ticker,
        timestamp_gte=window_start_ms,
        timestamp_lte=window_end_ms,
        limit=10
    ):
        quotes.append(q)
    print(f"   Result: {len(quotes)} quotes found")
    if quotes:
        for i, q in enumerate(quotes[:3]):
            print(f"      Quote {i+1}: bid=${q.bid_price:.2f}, ask=${q.ask_price:.2f}, ts={q.sip_timestamp}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Try with datetime objects
print(f"\n🧪 Test 3: Using datetime strings")
window_start_dt = datetime.fromtimestamp((trade_timestamp_ns - 10_000_000_000) / 1_000_000_000, tz=timezone.utc)
window_end_dt = datetime.fromtimestamp((trade_timestamp_ns + 10_000_000_000) / 1_000_000_000, tz=timezone.utc)

print(f"   Window: {window_start_dt} to {window_end_dt}")

try:
    quotes = []
    for q in client.list_quotes(
        ticker=ticker,
        timestamp_gte=window_start_dt,
        timestamp_lte=window_end_dt,
        limit=10
    ):
        quotes.append(q)
    print(f"   Result: {len(quotes)} quotes found")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Just try to get ANY quotes for this ticker
print(f"\n🧪 Test 4: Get ANY quotes (no timestamp filter)")

try:
    quotes = []
    for q in client.list_quotes(
        ticker=ticker,
        limit=10
    ):
        quotes.append(q)
    print(f"   Result: {len(quotes)} quotes found")
    if quotes:
        first_q = quotes[0]
        print(f"   First quote timestamp: {first_q.sip_timestamp}")
        first_dt = datetime.fromtimestamp(first_q.sip_timestamp / 1_000_000_000, tz=timezone.utc)
        print(f"   First quote time: {first_dt}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n{'='*80}")
print("Tests complete!")
print(f"{'='*80}")
