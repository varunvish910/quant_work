#!/usr/bin/env python3
"""
Match SPY Options Trades to Quotes Using Polygon REST API
Smart sampling approach to reduce API calls from 769k to ~15k
"""

import pandas as pd
import gzip
from pathlib import Path
from polygon import RESTClient
import time
from datetime import datetime
import numpy as np
import sys

print("="*80)
print("SPY OPTIONS - TRADE TO QUOTE MATCHING (REST API)")
print("="*80)
sys.stdout.flush()

# Configuration
API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
TARGET_DATE = '2025-10-06'
DATA_DIR = Path('trade_and_quote_data/data_management/flatfiles')
OUTPUT_DIR = Path('analysis_outputs/oct_2025_spy_options')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADES_FILE = DATA_DIR / f"{TARGET_DATE}.csv.gz"

# Rate limiting
REQUESTS_PER_SECOND = 5
DELAY_BETWEEN_REQUESTS = 1.0 / REQUESTS_PER_SECOND

print(f"\n📋 Configuration:")
print(f"   Date: {TARGET_DATE}")
print(f"   API Key: ...{API_KEY[-8:]}")
print(f"   Rate limit: {REQUESTS_PER_SECOND} req/sec")
print(f"   Trades file: {TRADES_FILE.name}")

# Phase 1: Load SPY trades
print(f"\n{'='*80}")
print("PHASE 1: Load SPY Trades")
print(f"{'='*80}")

spy_trades = []
total_chunks = 0

print(f"Loading trades from {TRADES_FILE.name}...")

with gzip.open(TRADES_FILE, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        total_chunks += 1
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)

        if total_chunks % 10 == 0:
            total_collected = sum(len(t) for t in spy_trades)
            print(f"   Processed {total_chunks} chunks, found {total_collected:,} SPY trades", end='\r')

trades = pd.concat(spy_trades, ignore_index=True)
print(f"\n✅ Loaded {len(trades):,} SPY trades from {total_chunks} chunks")

# Phase 2: Analyze volume distribution
print(f"\n{'='*80}")
print("PHASE 2: Analyze Volume Distribution")
print(f"{'='*80}")

# Calculate volume by ticker
volume_by_ticker = trades.groupby('ticker')['size'].sum().sort_values(ascending=False)

print(f"\n📊 Volume Statistics:")
print(f"   Unique tickers: {len(volume_by_ticker):,}")
print(f"   Total volume: {volume_by_ticker.sum():,.0f}")
print(f"   Mean volume per ticker: {volume_by_ticker.mean():,.0f}")
print(f"   Median volume per ticker: {volume_by_ticker.median():,.0f}")

print(f"\n   Top 10 Most Active Tickers:")
for i, (ticker, vol) in enumerate(volume_by_ticker.head(10).items(), 1):
    pct = vol / volume_by_ticker.sum() * 100
    print(f"      {i:2d}. {ticker:30s} {vol:>15,.0f} ({pct:>5.2f}%)")

# Phase 3: Smart sampling strategy
print(f"\n{'='*80}")
print("PHASE 3: Design Smart Sampling Strategy")
print(f"{'='*80}")

# Calculate cumulative volume
volume_by_ticker_sorted = volume_by_ticker.sort_values(ascending=False)
cumulative_pct = (volume_by_ticker_sorted.cumsum() / volume_by_ticker_sorted.sum() * 100)

# Find how many tickers cover 80% of volume
tickers_for_80pct = (cumulative_pct <= 80).sum()
pct_of_tickers = tickers_for_80pct / len(volume_by_ticker) * 100

print(f"\n   Volume Concentration:")
print(f"      Top {tickers_for_80pct} tickers ({pct_of_tickers:.1f}%) = 80% of volume")
print(f"      Remaining {len(volume_by_ticker) - tickers_for_80pct} tickers = 20% of volume")

# Strategy: Full coverage for top tickers, sample the rest
high_volume_tickers = set(volume_by_ticker_sorted.head(tickers_for_80pct).index)
low_volume_tickers = set(volume_by_ticker_sorted.tail(len(volume_by_ticker) - tickers_for_80pct).index)

SAMPLE_RATE_LOW_VOLUME = 0.10  # 10% sample for low volume tickers

print(f"\n   Sampling Strategy:")
print(f"      High-volume ({len(high_volume_tickers)} tickers): 100% coverage")
print(f"      Low-volume ({len(low_volume_tickers)} tickers): {SAMPLE_RATE_LOW_VOLUME*100:.0f}% sample")

# Apply sampling
trades['is_high_volume'] = trades['ticker'].isin(high_volume_tickers)

# For low volume, randomly sample
np.random.seed(42)
trades['random'] = np.random.random(len(trades))
trades['include'] = (trades['is_high_volume']) | ((~trades['is_high_volume']) & (trades['random'] < SAMPLE_RATE_LOW_VOLUME))

sampled_trades = trades[trades['include']].copy()

print(f"\n   Sampling Results:")
print(f"      Original trades: {len(trades):,}")
print(f"      Sampled trades: {len(sampled_trades):,}")
print(f"      Reduction: {(1 - len(sampled_trades)/len(trades))*100:.1f}%")
print(f"      Volume coverage: {sampled_trades['size'].sum() / trades['size'].sum() * 100:.1f}%")

# Phase 4: Aggregate into timestamp buckets
print(f"\n{'='*80}")
print("PHASE 4: Aggregate Into Timestamp Buckets")
print(f"{'='*80}")

# Round timestamps to nearest second
sampled_trades['timestamp_sec'] = (sampled_trades['sip_timestamp'] // 1_000_000_000) * 1_000_000_000

# Group by ticker and timestamp_sec
print(f"\n   Aggregating trades by ticker + 1-second timestamp...")

aggregated = sampled_trades.groupby(['ticker', 'timestamp_sec']).agg({
    'price': 'mean',  # Average price for the second
    'size': 'sum',    # Total size
    'sip_timestamp': 'count'  # Number of trades
}).reset_index()

aggregated.rename(columns={'sip_timestamp': 'trade_count'}, inplace=True)

print(f"\n   Aggregation Results:")
print(f"      Original sampled trades: {len(sampled_trades):,}")
print(f"      Unique ticker-timestamp combos: {len(aggregated):,}")
print(f"      Reduction: {(1 - len(aggregated)/len(sampled_trades))*100:.1f}%")
print(f"      Avg trades per bucket: {aggregated['trade_count'].mean():.1f}")

api_calls_needed = len(aggregated)
estimated_time_min = api_calls_needed * DELAY_BETWEEN_REQUESTS / 60

print(f"\n   API Call Estimates:")
print(f"      API calls needed: {api_calls_needed:,}")
print(f"      Est. time: {estimated_time_min:.0f} minutes ({estimated_time_min/60:.1f} hours)")

# Phase 5: Fetch quotes from Polygon API
print(f"\n{'='*80}")
print("PHASE 5: Fetch Quotes via Polygon REST API")
print(f"{'='*80}")

print(f"\n⚠️  WARNING: About to make {api_calls_needed:,} API calls")
print(f"   Estimated time: {estimated_time_min:.0f} minutes")
print(f"   Rate limit: {REQUESTS_PER_SECOND} req/sec")

# Initialize client
client = RESTClient(API_KEY)

# Storage for results
quote_results = []
errors = []

print(f"\n   Starting API calls...")
start_time = time.time()

for idx, row in aggregated.iterrows():
    ticker = row['ticker']
    timestamp_ns = row['timestamp_sec']

    # Progress indicator
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        remaining = (len(aggregated) - idx - 1) / rate / 60
        print(f"   Progress: {idx+1:,}/{len(aggregated):,} ({(idx+1)/len(aggregated)*100:.1f}%) | {rate:.1f} req/s | ETA: {remaining:.0f} min", end='\r')

    try:
        # Fetch quotes around trade timestamp
        quotes = client.list_quotes(
            ticker=ticker,
            timestamp_gte=timestamp_ns - 5_000_000_000,  # 5 seconds before
            timestamp_lte=timestamp_ns,                   # Up to trade time
            limit=50,
            order="desc"  # Most recent first
        )

        # Convert to list if generator
        quotes_list = list(quotes)

        if quotes_list:
            # Use most recent quote
            quote = quotes_list[0]

            quote_results.append({
                'ticker': ticker,
                'timestamp_sec': timestamp_ns,
                'avg_price': row['price'],
                'total_size': row['size'],
                'trade_count': row['trade_count'],
                'bid': quote.bid_price if hasattr(quote, 'bid_price') else None,
                'ask': quote.ask_price if hasattr(quote, 'ask_price') else None,
                'bid_size': quote.bid_size if hasattr(quote, 'bid_size') else None,
                'ask_size': quote.ask_size if hasattr(quote, 'ask_size') else None,
                'quote_timestamp': quote.sip_timestamp if hasattr(quote, 'sip_timestamp') else None
            })
        else:
            # No quotes found
            quote_results.append({
                'ticker': ticker,
                'timestamp_sec': timestamp_ns,
                'avg_price': row['price'],
                'total_size': row['size'],
                'trade_count': row['trade_count'],
                'bid': None,
                'ask': None,
                'bid_size': None,
                'ask_size': None,
                'quote_timestamp': None
            })

    except Exception as e:
        errors.append({
            'ticker': ticker,
            'timestamp': timestamp_ns,
            'error': str(e)
        })

        # Add placeholder
        quote_results.append({
            'ticker': ticker,
            'timestamp_sec': timestamp_ns,
            'avg_price': row['price'],
            'total_size': row['size'],
            'trade_count': row['trade_count'],
            'bid': None,
            'ask': None,
            'bid_size': None,
            'ask_size': None,
            'quote_timestamp': None
        })

    # Rate limiting
    time.sleep(DELAY_BETWEEN_REQUESTS)

elapsed_time = time.time() - start_time

print(f"\n\n✅ API calls complete!")
print(f"   Total time: {elapsed_time/60:.1f} minutes")
print(f"   Actual rate: {len(aggregated)/elapsed_time:.1f} req/sec")
print(f"   Errors: {len(errors)}")

# Convert to DataFrame
matched = pd.DataFrame(quote_results)

# Calculate match statistics
total_requests = len(matched)
matched_with_quotes = matched['bid'].notna().sum()
match_rate = matched_with_quotes / total_requests * 100

print(f"\n📊 Match Quality:")
print(f"   Total requests: {total_requests:,}")
print(f"   Matched with quotes: {matched_with_quotes:,}")
print(f"   Match rate: {match_rate:.1f}%")

if match_rate < 90:
    print(f"   ⚠️  Match rate below 90% - may need adjustment")
elif match_rate < 95:
    print(f"   ✅ Good match rate")
else:
    print(f"   ✅ Excellent match rate!")

# Phase 6: Classify trade direction
print(f"\n{'='*80}")
print("PHASE 6: Classify Trade Direction")
print(f"{'='*80}")

def classify_direction(row):
    """Classify trade as BUY or SELL based on price vs bid/ask"""
    price = row['avg_price']
    bid = row['bid']
    ask = row['ask']

    # Handle missing quotes
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
        return 'UNKNOWN'

    # Calculate spread
    spread = ask - bid
    mid = (bid + ask) / 2

    # Tolerance based on spread width
    if spread > mid * 0.1:  # Wide spread (>10%)
        tolerance = 0.05
    else:
        tolerance = 0.01

    # Classify
    if price >= ask * (1 - tolerance):
        return 'BUY'
    elif price <= bid * (1 + tolerance):
        return 'SELL'
    elif price > mid:
        return 'BUY'
    else:
        return 'SELL'

matched['direction'] = matched.apply(classify_direction, axis=1)

# Direction statistics
direction_counts = matched['direction'].value_counts()

print(f"\n   Trade Direction Distribution (by unique combinations):")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    if direction in direction_counts:
        count = direction_counts[direction]
        pct = count / len(matched) * 100
        print(f"      {direction:8s}: {count:>10,} ({pct:>5.1f}%)")

# Weight by volume
volume_by_direction = matched.groupby('direction')['total_size'].sum()

print(f"\n   Volume-Weighted Direction Distribution:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    if direction in volume_by_direction:
        vol = volume_by_direction[direction]
        pct = vol / volume_by_direction.sum() * 100
        print(f"      {direction:8s}: {vol:>15,.0f} ({pct:>5.1f}%)")

# Phase 7: Analyze by option type
print(f"\n{'='*80}")
print("PHASE 7: Calculate Dealer Positioning")
print(f"{'='*80}")

def parse_option_type(ticker):
    """Extract C or P from option ticker"""
    try:
        if not ticker.startswith('O:SPY'):
            return None
        parts = ticker[5:]
        if len(parts) >= 15:
            return parts[6].lower()
    except:
        pass
    return None

matched['option_type'] = matched['ticker'].apply(parse_option_type)

calls = matched[matched['option_type'] == 'c']
puts = matched[matched['option_type'] == 'p']

print(f"\n   CALL OPTIONS:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    vol = calls[calls['direction'] == direction]['total_size'].sum()
    pct = vol / calls['total_size'].sum() * 100 if len(calls) > 0 else 0
    print(f"      {direction:8s}: {vol:>15,.0f} ({pct:>5.1f}%)")

call_buy = calls[calls['direction'] == 'BUY']['total_size'].sum()
call_sell = calls[calls['direction'] == 'SELL']['total_size'].sum()
net_call_buy = call_buy - call_sell

print(f"      Net buying:  {net_call_buy:>15,.0f}")

print(f"\n   PUT OPTIONS:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    vol = puts[puts['direction'] == direction]['total_size'].sum()
    pct = vol / puts['total_size'].sum() * 100 if len(puts) > 0 else 0
    print(f"      {direction:8s}: {vol:>15,.0f} ({pct:>5.1f}%)")

put_buy = puts[puts['direction'] == 'BUY']['total_size'].sum()
put_sell = puts[puts['direction'] == 'SELL']['total_size'].sum()
net_put_buy = put_buy - put_sell

print(f"      Net buying:  {net_put_buy:>15,.0f}")

# Dealer positioning
print(f"\n⚡ Dealer Positioning Analysis:")

print(f"\n   Customer Flow (Net Buying = BUY - SELL):")
print(f"      Calls: {net_call_buy:>15,.0f} (positive = net buying)")
print(f"      Puts:  {net_put_buy:>15,.0f} (positive = net buying)")

print(f"\n   Dealer Positioning (opposite of customer):")
print(f"      Calls: {-net_call_buy:>15,.0f} (negative = SHORT calls)")
print(f"      Puts:  {-net_put_buy:>15,.0f} (negative = SHORT puts)")

# Gamma interpretation
dealer_short_calls = net_call_buy
dealer_short_puts = net_put_buy

if dealer_short_calls > 0 and dealer_short_puts > 0:
    gamma_status = "🔴 NET SHORT GAMMA (Volatility amplifier)"
    risk_level = "HIGH"
elif dealer_short_calls < 0 and dealer_short_puts < 0:
    gamma_status = "🟢 NET LONG GAMMA (Volatility dampener)"
    risk_level = "LOW"
else:
    gamma_status = "🟡 MIXED GAMMA (Directional bias)"
    risk_level = "MODERATE"

print(f"\n   Gamma Assessment: {gamma_status}")
print(f"   Risk Implication: {risk_level}")

# Phase 8: Save results
print(f"\n{'='*80}")
print("PHASE 8: Save Results")
print(f"{'='*80}")

# Save matched data
output_file = OUTPUT_DIR / f"{TARGET_DATE}_classified_trades.csv"
matched.to_csv(output_file, index=False)

file_size_mb = output_file.stat().st_size / (1024**2)
print(f"\n   ✅ Saved: {output_file.name} ({file_size_mb:.1f} MB)")

# Save summary statistics
summary = {
    'date': TARGET_DATE,
    'total_trades': len(trades),
    'sampled_trades': len(sampled_trades),
    'api_calls': len(aggregated),
    'match_rate': match_rate,
    'call_buy': call_buy,
    'call_sell': call_sell,
    'net_call_buy': net_call_buy,
    'put_buy': put_buy,
    'put_sell': put_sell,
    'net_put_buy': net_put_buy,
    'gamma_status': gamma_status,
    'risk_level': risk_level
}

summary_df = pd.DataFrame([summary])
summary_file = OUTPUT_DIR / f"{TARGET_DATE}_dealer_positioning_summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"   ✅ Saved: {summary_file.name}")

# Final summary
print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

print(f"\n📊 Key Findings:")
print(f"   Match rate: {match_rate:.1f}%")
print(f"   Net call buying: {net_call_buy:,.0f}")
print(f"   Net put buying: {net_put_buy:,.0f}")
print(f"   Dealer status: {gamma_status}")
print(f"   Risk level: {risk_level}")

print(f"\n💾 Files created:")
print(f"   {output_file}")
print(f"   {summary_file}")

print(f"\n✅ Analysis complete!")
