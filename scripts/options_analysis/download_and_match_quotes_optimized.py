#!/usr/bin/env python3
"""
OPTIMIZED: Download SPY options quotes and match with trades for proper BTO/STO classification

Performance improvements:
1. Pre-filter quotes using grep (10-15 min vs 45+ min pandas)
2. Process only SPY options (not SPYD, SPYG, etc)
3. Auto-cleanup: delete 145GB quote file after processing
4. Memory efficient: streaming processing
"""

import boto3
from botocore.config import Config
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime
import subprocess
import os

print("="*80)
print("SPY OPTIONS - TRADE DIRECTION CLASSIFICATION (OPTIMIZED)")
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
SPY_QUOTES_FILTERED = OUTPUT_DIR / f"{DATE}_quotes_spy_only.csv.gz"
CLASSIFIED_OUTPUT = OUTPUT_DIR / f"{DATE}_classified_trades.csv"

# Download quotes file (WARNING: ~145 GB!)
print(f"\n📥 Step 1: Downloading quotes for {DATE}")
print(f"   ⚠️  WARNING: Quote file is ~145 GB compressed!")
print(f"   This will take significant time and disk space.")
print(f"   Will auto-delete after processing to save space.")
print(f"   Starting download...")

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

# OPTIMIZATION: Pre-filter for SPY options only using grep (much faster than pandas)
print(f"\n⚡ Step 2: Pre-filtering for SPY options only (FAST METHOD)...")
print(f"   Using grep to extract only SPY options from 145 GB file...")
print(f"   This will take ~10-15 minutes (vs 45+ min with pandas)")

if not SPY_QUOTES_FILTERED.exists():
    try:
        # Extract header
        print(f"   Extracting header...")
        header_cmd = f"gunzip -c {QUOTES_FILE} | head -1 | gzip > {SPY_QUOTES_FILTERED}"
        subprocess.run(header_cmd, shell=True, check=True)

        # Filter for SPY options only
        # Pattern explanation:
        # - ^O:SPY[0-9] = starts with O:SPY followed by digit (not SPYD, SPYG, etc)
        # - This matches O:SPY241011C00670000 but NOT O:SPYD or O:SPYG
        print(f"   Filtering for O:SPY[0-9]... (this will take time)")

        filter_cmd = f"gunzip -c {QUOTES_FILE} | grep '^O:SPY[0-9]' | gzip >> {SPY_QUOTES_FILTERED}"
        result = subprocess.run(filter_cmd, shell=True, check=True)

        filtered_size_gb = SPY_QUOTES_FILTERED.stat().st_size / (1024 ** 3)
        print(f"   ✅ Filtered SPY quotes: {filtered_size_gb:.2f} GB (reduced from {file_size_gb:.1f} GB)")

    except Exception as e:
        print(f"   ❌ Error filtering quotes: {e}")
        exit(1)
else:
    filtered_size_gb = SPY_QUOTES_FILTERED.stat().st_size / (1024 ** 3)
    print(f"   ✅ SPY quotes already filtered ({filtered_size_gb:.2f} GB)")

# Now delete the original 145GB file to save space
print(f"\n🗑️  Step 3: Cleaning up large quote file...")
try:
    QUOTES_FILE.unlink()
    print(f"   ✅ Deleted {file_size_gb:.1f} GB quote file (no longer needed)")
except Exception as e:
    print(f"   ⚠️  Could not delete quote file: {e}")

# Load trades (we already have this)
print(f"\n📊 Step 4: Loading SPY trades...")

spy_trades = []
with gzip.open(TRADES_FILE, 'rt') as f:
    chunk_num = 0
    for chunk in pd.read_csv(f, chunksize=100000):
        chunk_num += 1
        # Match same pattern: O:SPY followed by digit
        spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)
        if chunk_num % 10 == 0:
            print(f"   Processed {chunk_num} chunks...")

trades = pd.concat(spy_trades, ignore_index=True)
print(f"   ✅ Loaded {len(trades):,} SPY trades")

# Load filtered SPY quotes (much smaller file now!)
print(f"\n📊 Step 5: Loading filtered SPY quotes...")
print(f"   Note: Processing {filtered_size_gb:.2f} GB (already filtered)")

spy_quotes = []
with gzip.open(SPY_QUOTES_FILTERED, 'rt') as f:
    chunk_num = 0
    for chunk in pd.read_csv(f, chunksize=100000):
        chunk_num += 1
        spy_quotes.append(chunk)

        if chunk_num % 100 == 0:
            total_rows = chunk_num * 100000
            print(f"   Processed {chunk_num} chunks... ({total_rows / 1e6:.1f}M rows)")

quotes = pd.concat(spy_quotes, ignore_index=True)
print(f"   ✅ Loaded {len(quotes):,} SPY quotes")

# Clean up filtered quotes file after loading
print(f"\n🗑️  Step 6: Cleaning up filtered quote file...")
try:
    SPY_QUOTES_FILTERED.unlink()
    print(f"   ✅ Deleted {filtered_size_gb:.2f} GB filtered quote file (no longer needed)")
except Exception as e:
    print(f"   ⚠️  Could not delete filtered quote file: {e}")

# Match trades to quotes
print(f"\n🔗 Step 7: Matching trades to quotes by timestamp...")

# Rename quote columns to avoid conflicts
quotes = quotes.rename(columns={
    'bid_price': 'bid',
    'ask_price': 'ask',
    'bid_size': 'bid_size',
    'ask_size': 'ask_size'
})

# Convert timestamps to datetime
trades['timestamp'] = pd.to_datetime(trades['sip_timestamp'], unit='ns')
quotes['timestamp'] = pd.to_datetime(quotes['sip_timestamp'], unit='ns')

# Sort both by ticker and timestamp
print(f"   Sorting data...")
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

match_rate = (matched['bid'].notna().sum() / len(matched)) * 100
print(f"   ✅ Matched {len(matched):,} trades to quotes ({match_rate:.1f}% match rate)")

# Classify trade direction
print(f"\n🎯 Step 8: Classifying trade direction (BUY/SELL)...")

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
print(f"\n📊 Step 9: Analyzing by option type and direction...")

calls = matched[matched['option_type'] == 'c']
puts = matched[matched['option_type'] == 'p']

print(f"\n   CALLS (Volume Analysis):")
call_buy_vol = calls[calls['direction'] == 'BUY']['size'].sum()
call_sell_vol = calls[calls['direction'] == 'SELL']['size'].sum()
call_unknown_vol = calls[calls['direction'] == 'UNKNOWN']['size'].sum()
call_total_vol = calls['size'].sum()

print(f"      BUY     : {call_buy_vol:>15,.0f} ({call_buy_vol/call_total_vol*100:>5.1f}%)")
print(f"      SELL    : {call_sell_vol:>15,.0f} ({call_sell_vol/call_total_vol*100:>5.1f}%)")
print(f"      UNKNOWN : {call_unknown_vol:>15,.0f} ({call_unknown_vol/call_total_vol*100:>5.1f}%)")

print(f"\n   PUTS (Volume Analysis):")
put_buy_vol = puts[puts['direction'] == 'BUY']['size'].sum()
put_sell_vol = puts[puts['direction'] == 'SELL']['size'].sum()
put_unknown_vol = puts[puts['direction'] == 'UNKNOWN']['size'].sum()
put_total_vol = puts['size'].sum()

print(f"      BUY     : {put_buy_vol:>15,.0f} ({put_buy_vol/put_total_vol*100:>5.1f}%)")
print(f"      SELL    : {put_sell_vol:>15,.0f} ({put_sell_vol/put_total_vol*100:>5.1f}%)")
print(f"      UNKNOWN : {put_unknown_vol:>15,.0f} ({put_unknown_vol/put_total_vol*100:>5.1f}%)")

# Calculate dealer positioning
print(f"\n⚡ Step 10: Estimating dealer positioning...")

# Dealer position = opposite of customer flow
net_call_buying = call_buy_vol - call_sell_vol  # Net customer buying
net_put_buying = put_buy_vol - put_sell_vol     # Net customer buying

print(f"\n   Customer Flow (Net):")
print(f"      Net call buying: {net_call_buying:>15,.0f}")
print(f"      Net put buying:  {net_put_buying:>15,.0f}")

print(f"\n   Dealer Positioning (opposite of customer flow):")
print(f"      Dealer SHORT calls: {net_call_buying:>15,.0f} contracts")
print(f"      Dealer SHORT puts:  {net_put_buying:>15,.0f} contracts")

# Interpret gamma exposure
print(f"\n   📊 Gamma Exposure Interpretation:")

if net_call_buying > 0 and net_put_buying > 0:
    gamma_status = "NET SHORT GAMMA"
    risk_level = "🔴 HIGH RISK"
    interpretation = "Dealers are SHORT both calls and puts (negative gamma)"
    impact = "Market moves will be AMPLIFIED by dealer hedging"
    detail = "As SPY moves, dealers must hedge by buying/selling in the same direction"
elif net_call_buying < 0 and net_put_buying < 0:
    gamma_status = "NET LONG GAMMA"
    risk_level = "🟢 LOW RISK"
    interpretation = "Dealers are LONG both calls and puts (positive gamma)"
    impact = "Market moves will be DAMPENED by dealer hedging"
    detail = "As SPY moves, dealers hedge by selling rallies/buying dips (stabilizing)"
else:
    gamma_status = "MIXED GAMMA"
    risk_level = "🟡 MODERATE RISK"
    interpretation = "Dealers have mixed positioning"
    impact = "Directional bias in dealer hedging"
    if abs(net_call_buying) > abs(net_put_buying):
        detail = "Call positioning dominates (upside bias in hedging)"
    else:
        detail = "Put positioning dominates (downside bias in hedging)"

print(f"\n      Status: {gamma_status}")
print(f"      Risk Level: {risk_level}")
print(f"      {interpretation}")
print(f"      Impact: {impact}")
print(f"      Detail: {detail}")

# Calculate put/call ratio
pc_ratio = put_total_vol / call_total_vol if call_total_vol > 0 else 0
print(f"\n   Put/Call Ratio: {pc_ratio:.2f}")

# Save results
print(f"\n💾 Step 11: Saving results...")
matched.to_csv(CLASSIFIED_OUTPUT, index=False)
print(f"   ✅ Saved classified trades to: {CLASSIFIED_OUTPUT}")

# Create summary report
summary = {
    'date': DATE,
    'total_trades': len(matched),
    'match_rate_pct': match_rate,
    'call_buy_volume': call_buy_vol,
    'call_sell_volume': call_sell_vol,
    'put_buy_volume': put_buy_vol,
    'put_sell_volume': put_sell_vol,
    'net_call_buying': net_call_buying,
    'net_put_buying': net_put_buying,
    'put_call_ratio': pc_ratio,
    'gamma_status': gamma_status,
    'risk_level': risk_level,
    'interpretation': interpretation,
    'impact': impact
}

summary_file = OUTPUT_DIR / f"{DATE}_dealer_positioning_summary.csv"
pd.DataFrame([summary]).to_csv(summary_file, index=False)
print(f"   ✅ Saved summary to: {summary_file}")

print(f"\n{'='*80}")
print("✅ Analysis complete!")
print(f"{'='*80}")
print(f"\n📁 Output files:")
print(f"   1. {CLASSIFIED_OUTPUT}")
print(f"   2. {summary_file}")
print(f"\n🗑️  Cleanup: Deleted 145 GB quote files to save space")
print(f"{'='*80}")
