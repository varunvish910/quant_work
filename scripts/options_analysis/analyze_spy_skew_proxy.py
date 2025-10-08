#!/usr/bin/env python3
"""
Analyze SPY Options Skew Proxy from Trade Data
Since we can't calculate true IV without quotes, we'll use price/volume metrics
"""

import pandas as pd
import numpy as np
import gzip
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("SPY OPTIONS SKEW ANALYSIS (Trade-Based Proxy)")
print("="*80)
print("\n⚠️ NOTE: This is NOT true IV smile - we need quotes for that")
print("This shows price and volume patterns as a proxy for skew\n")

# Load October 6 data
data_file = Path('trade_and_quote_data/data_management/flatfiles/2025-10-06.csv.gz')
print(f"📊 Loading SPY options data...")

spy_trades = []
with gzip.open(data_file, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY')]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)

df = pd.concat(spy_trades, ignore_index=True)
print(f"   ✅ Loaded {len(df):,} trades\n")

# Parse options
def parse_ticker(ticker):
    try:
        parts = ticker.replace('O:', '')
        expiry_str = parts[3:9]
        year = int('20' + expiry_str[:2])
        month = int(expiry_str[2:4])
        day = int(expiry_str[4:6])
        expiry = datetime(year, month, day)
        option_type = parts[9].lower()
        strike = int(parts[10:]) / 1000
        return pd.Series({
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'dte': (expiry - datetime(2025, 10, 6)).days
        })
    except:
        return pd.Series({'option_type': None, 'strike': None, 'expiry': None, 'dte': None})

print("⚙️ Parsing option details...")
option_details = df['ticker'].apply(parse_ticker)
df = pd.concat([df, option_details], axis=1)
df = df.dropna(subset=['option_type', 'strike', 'expiry'])
print(f"   ✅ Parsed {len(df):,} valid trades\n")

# Current SPY price (approximate)
spy_price = 669

# =============================================================================
# ANALYZE THIS WEEK'S EXPIRIES
# =============================================================================

print("="*80)
print("THIS WEEK'S EXPIRY ANALYSIS (Oct 7-11)")
print("="*80)

# Filter for this week (0-5 DTE)
this_week = df[df['dte'].between(0, 5)].copy()
this_week['moneyness'] = this_week['strike'] / spy_price

print(f"\n📊 This Week's Statistics:")
print(f"   Total trades: {len(this_week):,}")
print(f"   Unique expiries: {this_week['expiry'].nunique()}")
print(f"   Unique strikes: {this_week['strike'].nunique()}")

# Group by strike and type for volume-weighted average price
strike_analysis = this_week.groupby(['strike', 'option_type']).agg({
    'price': 'mean',
    'size': 'sum'
}).reset_index()

# Separate calls and puts
calls = strike_analysis[strike_analysis['option_type'] == 'c'].set_index('strike')
puts = strike_analysis[strike_analysis['option_type'] == 'p'].set_index('strike')

# =============================================================================
# PRICE SKEW (Proxy for IV Skew)
# =============================================================================

print("\n📈 PRICE-BASED SKEW ANALYSIS:")

# For each strike, calculate put/call price ratio (adjusted for moneyness)
common_strikes = sorted(set(calls.index) & set(puts.index))

skew_data = []
for strike in common_strikes:
    moneyness = strike / spy_price

    # Skip if too far from ATM
    if abs(moneyness - 1) > 0.05:  # Only look at ±5% from ATM
        continue

    call_price = calls.loc[strike, 'price']
    put_price = puts.loc[strike, 'price']

    # Adjust for intrinsic value
    call_intrinsic = max(0, spy_price - strike)
    put_intrinsic = max(0, strike - spy_price)

    call_time_value = call_price - call_intrinsic
    put_time_value = put_price - put_intrinsic

    if call_time_value > 0 and put_time_value > 0:
        skew_data.append({
            'strike': strike,
            'moneyness': moneyness,
            'call_tv': call_time_value,
            'put_tv': put_time_value,
            'skew_ratio': put_time_value / call_time_value if moneyness < 1 else call_time_value / put_time_value
        })

skew_df = pd.DataFrame(skew_data)

if len(skew_df) > 0:
    print(f"\n   Strike | Moneyness | Call TV | Put TV | Skew")
    print("   " + "-"*50)
    for _, row in skew_df.head(10).iterrows():
        print(f"   ${row['strike']:.0f}  |   {row['moneyness']:.3f}  | ${row['call_tv']:.2f}  | ${row['put_tv']:.2f}  | {row['skew_ratio']:.2f}")

# =============================================================================
# VOLUME SKEW
# =============================================================================

print("\n📊 VOLUME SKEW ANALYSIS:")

# Calculate volume distribution by moneyness
this_week['moneyness_bucket'] = pd.cut(
    this_week['moneyness'],
    bins=[0, 0.95, 0.98, 1.00, 1.02, 1.05, 2.0],
    labels=['<95%', '95-98%', '98-100%', '100-102%', '102-105%', '>105%']
)

volume_by_moneyness = this_week.groupby(['moneyness_bucket', 'option_type'])['size'].sum().unstack(fill_value=0)

print("\n   Moneyness   | Call Volume | Put Volume | P/C Ratio")
print("   " + "-"*55)
for bucket in volume_by_moneyness.index:
    call_vol = volume_by_moneyness.loc[bucket, 'c'] if 'c' in volume_by_moneyness.columns else 0
    put_vol = volume_by_moneyness.loc[bucket, 'p'] if 'p' in volume_by_moneyness.columns else 0
    pc_ratio = put_vol / call_vol if call_vol > 0 else 0
    print(f"   {str(bucket):12s} | {call_vol:11,.0f} | {put_vol:10,.0f} | {pc_ratio:6.2f}")

# =============================================================================
# 25-DELTA PROXY (Volume-Weighted)
# =============================================================================

print("\n🎯 25-DELTA PROXY ANALYSIS:")
print("   (Using 95% and 105% moneyness as proxy for 25-delta)\n")

# Approximate 25-delta strikes (roughly 95% for puts, 105% for calls)
put_25d_strikes = this_week[(this_week['moneyness'].between(0.94, 0.96)) &
                            (this_week['option_type'] == 'p')]
call_25d_strikes = this_week[(this_week['moneyness'].between(1.04, 1.06)) &
                             (this_week['option_type'] == 'c')]

if len(put_25d_strikes) > 0 and len(call_25d_strikes) > 0:
    avg_put_25d_price = (put_25d_strikes['price'] * put_25d_strikes['size']).sum() / put_25d_strikes['size'].sum()
    avg_call_25d_price = (call_25d_strikes['price'] * call_25d_strikes['size']).sum() / call_25d_strikes['size'].sum()

    # Normalize by distance from ATM
    put_25d_normalized = avg_put_25d_price / (spy_price * 0.05)  # 5% OTM
    call_25d_normalized = avg_call_25d_price / (spy_price * 0.05)

    skew_proxy = (put_25d_normalized - call_25d_normalized) / call_25d_normalized * 100

    print(f"   25D Put avg price:  ${avg_put_25d_price:.2f}")
    print(f"   25D Call avg price: ${avg_call_25d_price:.2f}")
    print(f"   Normalized skew proxy: {skew_proxy:+.1f}%")

    if skew_proxy > 20:
        print(f"   → High put skew (crash protection demand)")
    elif skew_proxy > 0:
        print(f"   → Normal put skew")
    else:
        print(f"   → Unusual call skew")

# =============================================================================
# HISTORICAL COMPARISON (Load September data if available)
# =============================================================================

print("\n="*80)
print("HISTORICAL SKEW CHANGES (Sept vs Oct)")
print("="*80)

# Check for September files
sept_files = list(Path('trade_and_quote_data/data_management/flatfiles/').glob('2025-09-*.csv.gz'))

if sept_files:
    print(f"\n📊 Found {len(sept_files)} September files")

    # Load last September file for comparison
    last_sept = sorted(sept_files)[-1]
    print(f"   Comparing with {last_sept.name}")

    sept_trades = []
    with gzip.open(last_sept, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=50000):
            spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY')]
            if len(spy_chunk) > 0:
                sept_trades.append(spy_chunk)

    if sept_trades:
        sept_df = pd.concat(sept_trades, ignore_index=True)
        sept_details = sept_df['ticker'].apply(parse_ticker)
        sept_df = pd.concat([sept_df, sept_details], axis=1)
        sept_df = sept_df.dropna(subset=['option_type', 'strike'])

        # Calculate September metrics
        sept_week = sept_df[sept_df['dte'].between(0, 5)]

        if len(sept_week) > 0:
            sept_pc = sept_week[sept_week['option_type'] == 'p']['size'].sum() / \
                     sept_week[sept_week['option_type'] == 'c']['size'].sum()
            oct_pc = this_week[this_week['option_type'] == 'p']['size'].sum() / \
                    this_week[this_week['option_type'] == 'c']['size'].sum()

            print(f"\n   WEEKLY P/C RATIO CHANGE:")
            print(f"   September: {sept_pc:.2f}")
            print(f"   October:   {oct_pc:.2f}")
            print(f"   Change:    {(oct_pc/sept_pc - 1)*100:+.1f}%")

            if oct_pc > sept_pc * 1.2:
                print(f"   → Significant increase in put demand")
            elif oct_pc < sept_pc * 0.8:
                print(f"   → Significant decrease in put demand")
else:
    print("\n   ⚠️ No September data files found for comparison")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n="*80)
print("KEY LIMITATIONS & INTERPRETATIONS")
print("="*80)

print("\n⚠️ IMPORTANT CAVEATS:")
print("   • This is NOT true implied volatility smile")
print("   • We're using trade prices as proxy (not mid-quotes)")
print("   • Cannot calculate actual Greeks without IV")
print("   • Volume patterns may reflect many factors beyond skew")

print("\n📊 WHAT THIS DATA SUGGESTS:")
if len(skew_df) > 0 and skew_proxy > 0:
    print("   • Put options trading at premium to calls (normal)")
    print("   • Downside protection being sought")
elif len(skew_df) > 0:
    print("   • Relatively balanced put/call pricing")

print(f"\n💡 TO GET TRUE IV SMILE, WE NEED:")
print("   1. Download quotes file (~145GB)")
print("   2. Match to trades by timestamp")
print("   3. Calculate mid-price for each strike")
print("   4. Use Black-Scholes to derive IV")
print("   5. Plot IV vs Strike/Delta")

print("\n="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")