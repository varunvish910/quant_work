#!/usr/bin/env python3
"""
Download SPY options quotes and match with trades for proper BTO/STO classification
"""

import boto3
from botocore.config import Config
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime
import numpy as np

print("="*80)
print("SPY OPTIONS - TRADE DIRECTION CLASSIFICATION")
print("Downloading quotes and matching with trades")
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

# Configuration
DATE = '2025-10-06'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADES_FILE = OUTPUT_DIR / f"{DATE}.csv.gz"
QUOTES_FILE = OUTPUT_DIR / f"{DATE}_quotes.csv.gz"

# Download quotes file (WARNING: ~145 GB!)
print(f"\n📥 Step 1: Downloading quotes for {DATE}")
print(f"   ⚠️  WARNING: Quote file is ~145 GB compressed!")
print(f"   This will take significant time and disk space.")

user_confirm = input("\n   Continue with download? (yes/no): ")

if user_confirm.lower() != 'yes':
    print("\n❌ Download cancelled by user")
    exit(0)

# Download quotes
if not QUOTES_FILE.exists():
    print(f"\n   Downloading from S3...")
    year = DATE[:4]
    month = DATE[5:7]
    quotes_key = f"us_options_opra/quotes_v1/{year}/{month}/{DATE}.csv.gz"

    try:
        s3.download_file('flatfiles', quotes_key, str(QUOTES_FILE))
        file_size_gb = QUOTES_FILE.stat().st_size / (1024 ** 3)
        print(f"   ✅ Downloaded {file_size_gb:.1f} GB")
    except Exception as e:
        print(f"   ❌ Error downloading quotes: {e}")
        exit(1)
else:
    file_size_gb = QUOTES_FILE.stat().st_size / (1024 ** 3)
    print(f"   ✅ Quotes already downloaded ({file_size_gb:.1f} GB)")

# Load trades (we already have this)
print(f"\n📊 Step 2: Loading SPY trades...")

spy_trades = []
with gzip.open(TRADES_FILE, 'rt') as f:
    chunk_num = 0
    for chunk in pd.read_csv(f, chunksize=100000):
        chunk_num += 1
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)
        if chunk_num % 10 == 0:
            print(f"   Processed {chunk_num} chunks...")

trades = pd.concat(spy_trades, ignore_index=True)
print(f"   ✅ Loaded {len(trades):,} SPY trades")

# Load quotes (filter for SPY only while reading)
print(f"\n📊 Step 3: Loading SPY quotes (this will take time)...")
print(f"   Note: Processing 145 GB of quote data...")

spy_quotes = []
with gzip.open(QUOTES_FILE, 'rt') as f:
    chunk_num = 0
    for chunk in pd.read_csv(f, chunksize=100000):
        chunk_num += 1
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]
        if len(spy_chunk) > 0:
            spy_quotes.append(spy_chunk)

        if chunk_num % 100 == 0:
            print(f"   Processed {chunk_num} chunks... ({chunk_num * 100000 / 1e6:.0f}M rows)")

quotes = pd.concat(spy_quotes, ignore_index=True)
print(f"   ✅ Loaded {len(quotes):,} SPY quotes")

# Match trades to quotes
print(f"\n🔗 Step 4: Matching trades to quotes by timestamp...")

# Convert timestamps to datetime
trades['timestamp'] = pd.to_datetime(trades['sip_timestamp'], unit='ns')
quotes['timestamp'] = pd.to_datetime(quotes['sip_timestamp'], unit='ns')

# Sort both by ticker and timestamp
trades = trades.sort_values(['ticker', 'timestamp'])
quotes = quotes.sort_values(['ticker', 'timestamp'])

print(f"   Performing asof merge (nearest quote before each trade)...")

# For each trade, find the most recent quote (asof merge)
matched = pd.merge_asof(
    trades,
    quotes[['ticker', 'timestamp', 'bid', 'ask', 'bid_size', 'ask_size']],
    on='timestamp',
    by='ticker',
    direction='backward',
    suffixes=('_trade', '_quote')
)

print(f"   ✅ Matched {len(matched):,} trades to quotes")

# Classify trade direction
print(f"\n🎯 Step 5: Classifying trade direction (BTO/STO/BTC/STC)...")

def classify_trade_direction(row):
    """
    Classify trade based on price vs bid/ask

    Logic:
    - Trade >= ask: Customer buying (BTO if opening, BTC if closing)
    - Trade <= bid: Customer selling (STO if opening, STC if closing)
    - Between bid/ask: Use midpoint
    """
    price = row['price']
    bid = row['bid']
    ask = row['ask']

    # Handle missing quotes
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
        return 'UNKNOWN'

    # Calculate spread
    spread = ask - bid
    mid = (bid + ask) / 2

    # Wide spread threshold (>10% of mid)
    if spread > mid * 0.1:
        # Very wide spread, less reliable
        tolerance = 0.05
    else:
        tolerance = 0.01

    # Classify
    if price >= ask * (1 - tolerance):
        return 'BUY'  # Customer buying (aggressive)
    elif price <= bid * (1 + tolerance):
        return 'SELL'  # Customer selling (aggressive)
    elif price > mid:
        return 'BUY'  # At mid but leaning buy
    else:
        return 'SELL'  # At mid but leaning sell

matched['direction'] = matched.apply(classify_trade_direction, axis=1)

# Count by direction
direction_counts = matched['direction'].value_counts()
print(f"\n   Trade Direction Distribution:")
for direction, count in direction_counts.items():
    pct = count / len(matched) * 100
    print(f"      {direction:8s}: {count:>10,} ({pct:>5.1f}%)")

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

matched['option_type'] = matched['ticker'].apply(parse_option_type)

# Analyze by option type and direction
print(f"\n📊 Step 6: Analyzing by option type and direction...")

calls = matched[matched['option_type'] == 'c']
puts = matched[matched['option_type'] == 'p']

print(f"\n   CALLS:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    vol = calls[calls['direction'] == direction]['size'].sum()
    pct = vol / calls['size'].sum() * 100 if len(calls) > 0 else 0
    print(f"      {direction:8s}: {vol:>15,.0f} ({pct:>5.1f}%)")

print(f"\n   PUTS:")
for direction in ['BUY', 'SELL', 'UNKNOWN']:
    vol = puts[puts['direction'] == direction]['size'].sum()
    pct = vol / puts['size'].sum() * 100 if len(puts) > 0 else 0
    print(f"      {direction:8s}: {vol:>15,.0f} ({pct:>5.1f}%)")

# Calculate dealer positioning
print(f"\n⚡ Step 7: Estimating dealer positioning...")

call_buy_vol = calls[calls['direction'] == 'BUY']['size'].sum()
call_sell_vol = calls[calls['direction'] == 'SELL']['size'].sum()
put_buy_vol = puts[puts['direction'] == 'BUY']['size'].sum()
put_sell_vol = puts[puts['direction'] == 'SELL']['size'].sum()

# Dealer position = opposite of customer flow
dealer_short_calls = call_buy_vol - call_sell_vol  # Net customer buying
dealer_short_puts = put_buy_vol - put_sell_vol     # Net customer buying

print(f"\n   Customer Flow:")
print(f"      Call buying:  {call_buy_vol:>15,.0f}")
print(f"      Call selling: {call_sell_vol:>15,.0f}")
print(f"      Net call buying: {dealer_short_calls:>12,.0f}")
print(f"")
print(f"      Put buying:   {put_buy_vol:>15,.0f}")
print(f"      Put selling:  {put_sell_vol:>15,.0f}")
print(f"      Net put buying:  {dealer_short_puts:>12,.0f}")

print(f"\n   Dealer Positioning (assuming dealers on opposite side):")
print(f"      Dealer SHORT calls: {dealer_short_calls:>15,.0f} (negative gamma)")
print(f"      Dealer SHORT puts:  {dealer_short_puts:>15,.0f} (negative gamma)")

# Simple gamma proxy (ATM has highest gamma)
# We'll estimate gamma as 1.0 for ATM, 0.5 for near OTM, 0.2 for far OTM

if dealer_short_calls > 0 and dealer_short_puts > 0:
    print(f"\n   🔴 Dealers appear NET SHORT gamma (vulnerable to volatility)")
    print(f"      → Market moves will be amplified by dealer hedging")
elif dealer_short_calls < 0 and dealer_short_puts < 0:
    print(f"\n   🟢 Dealers appear NET LONG gamma (stabilizing)")
    print(f"      → Market moves will be dampened by dealer hedging")
else:
    print(f"\n   🟡 Dealers appear MIXED positioning")

# Save results
output_file = OUTPUT_DIR / f"{DATE}_classified_trades.csv"
matched.to_csv(output_file, index=False)
print(f"\n✅ Saved classified trades to: {output_file}")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
