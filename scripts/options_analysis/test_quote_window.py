#!/usr/bin/env python3
"""
Test getting quotes within a time window around each trade
Using Polygon RESTClient with list_quotes
"""

from polygon import RESTClient
import time

# Test with real SPY trade from Oct 6
test_trade = {
    'ticker': 'O:SPY251006C00550000',
    'price': 119.96,
    'timestamp_ns': 1759759211412000000,  # Oct 6, 2025
    'size': 5
}

API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

print("="*80)
print("TESTING QUOTE API WITH TIME WINDOW")
print("="*80)

client = RESTClient(API_KEY)

print(f"\n📊 Test Trade:")
print(f"   Ticker: {test_trade['ticker']}")
print(f"   Price: ${test_trade['price']:.2f}")
print(f"   Timestamp: {test_trade['timestamp_ns']}")

# Create time window: ±1 second around trade
timestamp_ns = test_trade['timestamp_ns']
window_start = timestamp_ns - 1_000_000_000  # 1 second before
window_end = timestamp_ns + 1_000_000_000    # 1 second after

print(f"\n🕐 Time Window:")
print(f"   Start: {window_start}")
print(f"   Trade: {timestamp_ns}")
print(f"   End:   {window_end}")

# Fetch quotes
print(f"\n📡 Fetching quotes...")

try:
    quotes = []
    for quote in client.list_quotes(
        ticker=test_trade['ticker'],
        timestamp_gte=window_start,
        timestamp_lte=window_end,
        order="asc",
        limit=50,
        sort="timestamp",
    ):
        quotes.append(quote)

    print(f"   ✅ Found {len(quotes)} quotes in time window")

    if len(quotes) > 0:
        print(f"\n📋 Quotes:")
        for i, q in enumerate(quotes[:10], 1):  # Show first 10
            print(f"\n   Quote {i}:")
            print(f"      Timestamp: {q.sip_timestamp}")
            print(f"      Bid: ${q.bid_price:.2f} (size: {q.bid_size})")
            print(f"      Ask: ${q.ask_price:.2f} (size: {q.ask_size})")
            print(f"      Mid: ${(q.bid_price + q.ask_price) / 2:.2f}")

        # Find quote closest to trade (before trade)
        pre_trade_quotes = [q for q in quotes if q.sip_timestamp <= timestamp_ns]

        if pre_trade_quotes:
            closest_quote = pre_trade_quotes[-1]  # Last quote before trade

            print(f"\n✅ Closest quote before trade:")
            print(f"   Timestamp: {closest_quote.sip_timestamp}")
            print(f"   Bid: ${closest_quote.bid_price:.2f}")
            print(f"   Ask: ${closest_quote.ask_price:.2f}")
            print(f"   Mid: ${(closest_quote.bid_price + closest_quote.ask_price) / 2:.2f}")

            # Classify trade
            bid = closest_quote.bid_price
            ask = closest_quote.ask_price
            mid = (bid + ask) / 2
            price = test_trade['price']

            if price >= ask * 0.99:
                direction = 'BUY'
            elif price <= bid * 1.01:
                direction = 'SELL'
            elif price > mid:
                direction = 'BUY'
            else:
                direction = 'SELL'

            print(f"\n🎯 Trade Classification:")
            print(f"   Trade price: ${price:.2f}")
            print(f"   Quote mid: ${mid:.2f}")
            print(f"   Direction: {direction}")
        else:
            print(f"\n⚠️  No quotes before trade timestamp")
    else:
        print(f"\n⚠️  No quotes found in time window")
        print(f"   This may mean:")
        print(f"   1. No quote data available for this option")
        print(f"   2. Option was illiquid (wide spreads, few quotes)")
        print(f"   3. API doesn't have historical intraday quotes")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("Test complete!")
print(f"{'='*80}")
