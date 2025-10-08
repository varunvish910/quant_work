#!/usr/bin/env python3
"""
TEST VERSION: Download single day of quotes and validate approach
Before downloading 145 GB, test on smaller sample to verify methodology
"""

import boto3
from botocore.config import Config
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime
import numpy as np

print("="*80)
print("SPY OPTIONS - QUOTE MATCHING TEST (Single Day)")
print("="*80)

# AWS credentials
AWS_ACCESS_KEY = '86959ae1-29bc-4433-be13-1a41b935d9d1'
AWS_SECRET_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

# Initialize S3
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

s3 = session.client(
    's3',
    endpoint_url='https://files.polygon.io',
    config=Config(signature_version='s3v4'),
)

# Configuration - USING EARLIER DATE TO TEST
TEST_DATE = '2025-09-02'  # First day we have, smaller file
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADES_FILE = OUTPUT_DIR / f"{TEST_DATE}.csv.gz"
QUOTES_FILE = OUTPUT_DIR / f"{TEST_DATE}_quotes.csv.gz"

print(f"\n📋 Test Configuration:")
print(f"   Date: {TEST_DATE}")
print(f"   Trades file: {TRADES_FILE.name}")
print(f"   Quotes file: {QUOTES_FILE.name}")

# Check disk space first
import shutil
disk_usage = shutil.disk_usage('.')
free_gb = disk_usage.free / (1024**3)
print(f"\n💾 Disk Space Check:")
print(f"   Free: {free_gb:.1f} GB")

if free_gb < 50:
    print(f"   ⚠️  WARNING: Less than 50 GB free!")
    print(f"   Recommend at least 50 GB for testing")

# Step 1: Check quote file size before downloading
print(f"\n📏 Step 1: Checking quote file size on S3...")

year = TEST_DATE[:4]
month = TEST_DATE[5:7]
quotes_key = f"us_options_opra/quotes_v1/{year}/{month}/{TEST_DATE}.csv.gz"

try:
    response = s3.head_object(Bucket='flatfiles', Key=quotes_key)
    quote_size_gb = response['ContentLength'] / (1024**3)
    print(f"   Quote file size: {quote_size_gb:.2f} GB")

    if quote_size_gb > free_gb * 0.8:
        print(f"   ❌ INSUFFICIENT SPACE: Need {quote_size_gb:.2f} GB, have {free_gb:.1f} GB")
        exit(1)
    else:
        print(f"   ✅ Sufficient space available")
except Exception as e:
    print(f"   ❌ Error checking file: {e}")
    exit(1)

# Step 2: Download quotes (with confirmation)
if not QUOTES_FILE.exists():
    print(f"\n📥 Step 2: Downloading quotes ({quote_size_gb:.2f} GB)...")
    print(f"   This will take approximately {quote_size_gb*0.5:.0f}-{quote_size_gb*2:.0f} minutes")

    # Auto-proceed for testing
    print(f"\n   Auto-proceeding with download (test mode)...")

    try:
        print(f"   Downloading from S3...")
        s3.download_file('flatfiles', quotes_key, str(QUOTES_FILE))
        actual_size_gb = QUOTES_FILE.stat().st_size / (1024**3)
        print(f"   ✅ Downloaded {actual_size_gb:.2f} GB")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        exit(1)
else:
    actual_size_gb = QUOTES_FILE.stat().st_size / (1024**3)
    print(f"\n✅ Quotes already downloaded ({actual_size_gb:.2f} GB)")

# Step 3: Load SPY trades
print(f"\n📊 Step 3: Loading SPY trades...")

spy_trades = []
with gzip.open(TRADES_FILE, 'rt') as f:
    chunk_num = 0
    for chunk in pd.read_csv(f, chunksize=100000):
        chunk_num += 1
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)

trades = pd.concat(spy_trades, ignore_index=True)
print(f"   ✅ Loaded {len(trades):,} SPY trades")

# Step 4: Sample quotes (don't load ALL quotes yet)
print(f"\n📊 Step 4: Sampling quotes to estimate counts...")

sample_quotes = []
with gzip.open(QUOTES_FILE, 'rt') as f:
    # Read first 1M rows to estimate
    chunk = pd.read_csv(f, nrows=1000000)
    spy_sample = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]
    sample_quotes.append(spy_sample)

    sample_spy_count = len(spy_sample)
    sample_total = len(chunk)
    spy_ratio = sample_spy_count / sample_total

print(f"   Sample: {sample_spy_count:,} SPY quotes in {sample_total:,} total rows")
print(f"   SPY ratio: {spy_ratio:.2%}")

# Estimate total SPY quotes
print(f"\n📊 Estimating total quote file size...")
with gzip.open(QUOTES_FILE, 'rt') as f:
    # Count lines approximately
    total_lines = 0
    for _ in f:
        total_lines += 1
        if total_lines % 1000000 == 0:
            print(f"   Counting... {total_lines/1e6:.0f}M lines", end='\r')

estimated_spy_quotes = int(total_lines * spy_ratio)
print(f"\n   Total lines: {total_lines:,}")
print(f"   Estimated SPY quotes: {estimated_spy_quotes:,}")

memory_estimate_gb = estimated_spy_quotes * 200 / (1024**3)  # ~200 bytes per row
print(f"   Estimated memory needed: {memory_estimate_gb:.2f} GB")

if memory_estimate_gb > 10:
    print(f"   ⚠️  Large memory requirement - will process in chunks")

# Step 5: Load SPY quotes in chunks
print(f"\n📊 Step 5: Loading SPY quotes (chunked processing)...")

spy_quotes = []
quote_chunk_num = 0

with gzip.open(QUOTES_FILE, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        quote_chunk_num += 1
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]

        if len(spy_chunk) > 0:
            # Only keep necessary columns to save memory
            spy_chunk = spy_chunk[['ticker', 'sip_timestamp', 'bid', 'ask', 'bid_size', 'ask_size']]
            spy_quotes.append(spy_chunk)

        if quote_chunk_num % 100 == 0:
            total_collected = sum(len(q) for q in spy_quotes)
            print(f"   Processed {quote_chunk_num} chunks, collected {total_collected:,} SPY quotes", end='\r')

quotes = pd.concat(spy_quotes, ignore_index=True)
print(f"\n   ✅ Loaded {len(quotes):,} SPY quotes")

# Step 6: Prepare for matching
print(f"\n🔗 Step 6: Preparing data for matching...")

# Convert timestamps
trades['timestamp'] = pd.to_datetime(trades['sip_timestamp'], unit='ns')
quotes['timestamp'] = pd.to_datetime(quotes['sip_timestamp'], unit='ns')

# Sort both by ticker and timestamp
print(f"   Sorting trades...")
trades = trades.sort_values(['ticker', 'timestamp'])
print(f"   Sorting quotes...")
quotes = quotes.sort_values(['ticker', 'timestamp'])

print(f"   ✅ Data prepared")

# Step 7: Match trades to quotes
print(f"\n🔗 Step 7: Matching trades to quotes...")
print(f"   Using asof merge (backward direction)")

matched = pd.merge_asof(
    trades,
    quotes,
    on='timestamp',
    by='ticker',
    direction='backward',  # Use most recent quote before trade
    suffixes=('_trade', '_quote')
)

print(f"   ✅ Matching complete")

# Calculate match statistics
total_trades = len(matched)
matched_with_quotes = matched[matched['bid'].notna()].shape[0]
match_rate = matched_with_quotes / total_trades * 100

print(f"\n📊 Match Quality:")
print(f"   Total trades: {total_trades:,}")
print(f"   Matched with quotes: {matched_with_quotes:,}")
print(f"   Match rate: {match_rate:.1f}%")

if match_rate < 90:
    print(f"   ⚠️  Match rate below 90% - may need adjustment")
elif match_rate < 95:
    print(f"   ✅ Good match rate")
else:
    print(f"   ✅ Excellent match rate!")

# Step 8: Classify trade direction
print(f"\n🎯 Step 8: Classifying trade direction...")

def classify_direction(row):
    """Classify trade as BUY or SELL based on price vs bid/ask"""
    price = row['price']
    bid = row['bid']
    ask = row['ask']

    # Handle missing quotes
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
        return 'UNKNOWN'

    # Calculate spread
    spread = ask - bid
    mid = (bid + ask) / 2

    # Tolerance based on spread width
    if spread > mid * 0.1:  # Wide spread
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

print(f"\n   Trade Direction Distribution:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    if direction in direction_counts:
        count = direction_counts[direction]
        pct = count / len(matched) * 100
        print(f"      {direction:8s}: {count:>10,} ({pct:>5.1f}%)")

unknown_pct = direction_counts.get('UNKNOWN', 0) / len(matched) * 100
if unknown_pct > 10:
    print(f"   ⚠️  High UNKNOWN rate - quote coverage may be limited")

# Step 9: Analyze by option type
print(f"\n📊 Step 9: Analyzing by option type...")

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

matched['option_type'] = matched['ticker'].apply(parse_option_type)

calls = matched[matched['option_type'] == 'c']
puts = matched[matched['option_type'] == 'p']

print(f"\n   CALL OPTIONS:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    vol = calls[calls['direction'] == direction]['size'].sum()
    pct = vol / calls['size'].sum() * 100 if len(calls) > 0 else 0
    print(f"      {direction:8s}: {vol:>15,.0f} ({pct:>5.1f}%)")

call_buy = calls[calls['direction'] == 'BUY']['size'].sum()
call_sell = calls[calls['direction'] == 'SELL']['size'].sum()
net_call_buy = call_buy - call_sell

print(f"      Net buying:  {net_call_buy:>15,.0f}")

print(f"\n   PUT OPTIONS:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    vol = puts[puts['direction'] == direction]['size'].sum()
    pct = vol / puts['size'].sum() * 100 if len(puts) > 0 else 0
    print(f"      {direction:8s}: {vol:>15,.0f} ({pct:>5.1f}%)")

put_buy = puts[puts['direction'] == 'BUY']['size'].sum()
put_sell = puts[puts['direction'] == 'SELL']['size'].sum()
net_put_buy = put_buy - put_sell

print(f"      Net buying:  {net_put_buy:>15,.0f}")

# Step 10: Dealer positioning
print(f"\n⚡ Step 10: Estimating dealer positioning...")

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

# Step 11: Save results
print(f"\n💾 Step 11: Saving results...")

output_file = OUTPUT_DIR / f"{TEST_DATE}_classified_trades.csv"
# Save only essential columns to reduce file size
save_cols = ['ticker', 'timestamp', 'price', 'size', 'bid', 'ask', 'direction', 'option_type']
matched[save_cols].to_csv(output_file, index=False)

file_size_mb = output_file.stat().st_size / (1024**2)
print(f"   ✅ Saved: {output_file.name} ({file_size_mb:.1f} MB)")

# Summary report
print(f"\n{'='*80}")
print("TEST SUMMARY")
print(f"{'='*80}")

print(f"\n📊 Data Quality:")
print(f"   Match rate: {match_rate:.1f}%")
print(f"   UNKNOWN rate: {unknown_pct:.1f}%")
print(f"   Classification quality: {'✅ GOOD' if unknown_pct < 5 else '⚠️ ACCEPTABLE' if unknown_pct < 10 else '❌ POOR'}")

print(f"\n⚡ Dealer Positioning:")
print(f"   {gamma_status}")
print(f"   Risk: {risk_level}")

print(f"\n💾 Storage Used:")
print(f"   Quotes file: {actual_size_gb:.2f} GB")
print(f"   Output file: {file_size_mb:.1f} MB")
print(f"   Total: {actual_size_gb:.2f} GB")

print(f"\n📈 Extrapolation to Oct 6:")
print(f"   Oct 6 quote file: ~145 GB (estimated)")
print(f"   Oct 6 trades: ~769k (vs {len(trades):,} today)")
print(f"   Scaling factor: {769000/len(trades):.1f}x")
print(f"   Est. processing time: {actual_size_gb * 5 * 769000/len(trades)/60:.0f} minutes")

print(f"\n✅ Methodology Validation:")
if match_rate > 95 and unknown_pct < 5:
    print(f"   ✅ EXCELLENT - Ready to process Oct 6")
elif match_rate > 90 and unknown_pct < 10:
    print(f"   ✅ GOOD - Can proceed with Oct 6")
else:
    print(f"   ⚠️  NEEDS REVIEW - Check parameters before Oct 6")

print(f"\n{'='*80}")
print("Test complete!")
print(f"{'='*80}")
