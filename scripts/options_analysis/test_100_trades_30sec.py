#!/usr/bin/env python3
"""
Test quote window approach with 100 SPY trades - FINAL OPTIMIZED
Using ±30 second window to catch illiquid options
"""

from polygon import RESTClient
import pandas as pd
import gzip
from pathlib import Path
import time
from collections import defaultdict

# Configuration
API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
DATE = '2025-10-06'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
TRADES_FILE = OUTPUT_DIR / f"{DATE}.csv.gz"

print("="*80)
print("TESTING QUOTE WINDOW WITH 100 SPY TRADES (FINAL OPTIMIZED)")
print("Using ±30 second window to catch illiquid options")
print("="*80)

# Load trades
print(f"\n📊 Step 1: Loading SPY trades...")
spy_trades = []
with gzip.open(TRADES_FILE, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)
            if len(pd.concat(spy_trades)) >= 100:
                break

trades = pd.concat(spy_trades, ignore_index=True).head(100)
print(f"   ✅ Loaded {len(trades)} SPY trades")

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

# Initialize client
client = RESTClient(API_KEY)

# Classify trades
print(f"\n📡 Step 2: Fetching quotes and classifying trades...")
print(f"   Using ±30 second time window (was ±10 seconds)")

results = []
api_calls = 0
no_quotes = 0
quote_counts = []
start_time = time.time()

for idx, trade in trades.iterrows():
    ticker = trade['ticker']
    price = trade['price']
    timestamp_ns = trade['sip_timestamp']
    size = trade['size']
    option_type = trade['option_type']

    # Time window: ±30 seconds (was ±10 seconds)
    window_start = timestamp_ns - 30_000_000_000  # 30 seconds before
    window_end = timestamp_ns + 30_000_000_000    # 30 seconds after

    try:
        # Fetch quotes
        quotes = []
        for quote in client.list_quotes(
            ticker=ticker,
            timestamp_gte=window_start,
            timestamp_lte=window_end,
            order="asc",
            limit=50,
            sort="timestamp",
        ):
            quotes.append(quote)

        api_calls += 1
        quote_counts.append(len(quotes))

        # Find closest quote before trade
        pre_trade_quotes = [q for q in quotes if q.sip_timestamp <= timestamp_ns]

        if pre_trade_quotes:
            closest_quote = pre_trade_quotes[-1]
            bid = closest_quote.bid_price
            ask = closest_quote.ask_price
            mid = (bid + ask) / 2

            # Classify
            if price >= ask * 0.99:
                direction = 'BUY'
            elif price <= bid * 1.01:
                direction = 'SELL'
            elif price > mid:
                direction = 'BUY'
            else:
                direction = 'SELL'

            results.append({
                'ticker': ticker,
                'price': price,
                'size': size,
                'option_type': option_type,
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'direction': direction,
                'quotes_found': len(quotes)
            })
        else:
            no_quotes += 1
            results.append({
                'ticker': ticker,
                'price': price,
                'size': size,
                'option_type': option_type,
                'bid': 0,
                'ask': 0,
                'mid': 0,
                'direction': 'UNKNOWN',
                'quotes_found': len(quotes)
            })

    except Exception as e:
        no_quotes += 1
        results.append({
            'ticker': ticker,
            'price': price,
            'size': size,
            'option_type': option_type,
            'bid': 0,
            'ask': 0,
            'mid': 0,
            'direction': 'UNKNOWN',
            'quotes_found': 0
        })
        if idx < 5:  # Only print first few errors
            print(f"   ⚠️  Error on trade {idx+1}: {e}")

    # Progress
    if (idx + 1) % 20 == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        remaining = (100 - (idx + 1)) / rate if rate > 0 else 0
        print(f"   Progress: {idx+1}/100 trades ({rate:.1f} trades/sec, ~{remaining:.0f}s remaining)")

    # Rate limiting
    if api_calls % 5 == 0:
        time.sleep(1)

elapsed_total = time.time() - start_time

print(f"\n   ✅ Classification complete!")
print(f"      Time: {elapsed_total:.1f} seconds")
print(f"      Rate: {100/elapsed_total:.1f} trades/sec")
print(f"      API calls: {api_calls}")
print(f"      No quotes: {no_quotes} ({no_quotes/100*100:.1f}%)")
print(f"      Avg quotes per trade: {sum(quote_counts)/len(quote_counts):.1f}")

# Analyze results
results_df = pd.DataFrame(results)

print(f"\n📊 Step 3: Analyzing results...")

# Direction distribution
print(f"\n   Trade Direction Distribution:")
direction_counts = results_df['direction'].value_counts()
for direction, count in direction_counts.items():
    pct = count / len(results_df) * 100
    print(f"      {direction:8s}: {count:>3} ({pct:>5.1f}%)")

# By option type
calls = results_df[results_df['option_type'] == 'c']
puts = results_df[results_df['option_type'] == 'p']

print(f"\n   By Option Type:")
print(f"      Calls: {len(calls)}")
print(f"      Puts:  {len(puts)}")

if len(calls) > 0:
    call_buy = calls[calls['direction'] == 'BUY']['size'].sum()
    call_sell = calls[calls['direction'] == 'SELL']['size'].sum()
    call_total = calls['size'].sum()
    print(f"\n   Call Volume:")
    print(f"      BUY:  {call_buy:>10,.0f} ({call_buy/call_total*100:.1f}%)")
    print(f"      SELL: {call_sell:>10,.0f} ({call_sell/call_total*100:.1f}%)")
    print(f"      Net:  {call_buy - call_sell:>10,.0f}")

if len(puts) > 0:
    put_buy = puts[puts['direction'] == 'BUY']['size'].sum()
    put_sell = puts[puts['direction'] == 'SELL']['size'].sum()
    put_total = puts['size'].sum()
    print(f"\n   Put Volume:")
    print(f"      BUY:  {put_buy:>10,.0f} ({put_buy/put_total*100:.1f}%)")
    print(f"      SELL: {put_sell:>10,.0f} ({put_sell/put_total*100:.1f}%)")
    print(f"      Net:  {put_buy - put_sell:>10,.0f}")

# Quote coverage stats
print(f"\n   Quote Coverage Stats:")
print(f"      Trades with quotes: {(results_df['quotes_found'] > 0).sum()} ({(results_df['quotes_found'] > 0).sum()/100*100:.1f}%)")
print(f"      Trades without quotes: {(results_df['quotes_found'] == 0).sum()} ({(results_df['quotes_found'] == 0).sum()/100*100:.1f}%)")
print(f"      Min quotes: {results_df['quotes_found'].min()}")
print(f"      Max quotes: {results_df['quotes_found'].max()}")
print(f"      Avg quotes: {results_df['quotes_found'].mean():.1f}")

# Extrapolate to full dataset
print(f"\n📈 Extrapolation to Full Dataset:")
print(f"   Sample: 100 trades in {elapsed_total:.1f}s")
print(f"   Full dataset: ~769,000 trades")
print(f"   Estimated time: {(769000/100) * elapsed_total / 60:.1f} minutes")
print(f"   Estimated time: {(769000/100) * elapsed_total / 3600:.1f} hours")

# Comparison table
print(f"\n🔄 Performance Comparison:")
print(f"   {'Window':<15} {'No-Quote Rate':<15} {'Coverage':<15}")
print(f"   {'-'*45}")
print(f"   {'±1 second':<15} {'45%':<15} {'55%':<15}")
print(f"   {'±10 seconds':<15} {'4%':<15} {'96%':<15}")
print(f"   {'±30 seconds':<15} {f'{no_quotes}%':<15} {f'{100-no_quotes}%':<15}")

print(f"\n{'='*80}")
print("✅ Test complete!")
print(f"{'='*80}")
