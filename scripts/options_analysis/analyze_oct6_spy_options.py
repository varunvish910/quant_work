#!/usr/bin/env python3
"""
Analyze SPY Options for October 6, 2025
Focus on positioning, heavily bid strikes, and expiry concentrations
"""

import pandas as pd
import numpy as np
import gzip
from datetime import datetime, timedelta
from pathlib import Path

print("="*80)
print("SPY OPTIONS ANALYSIS - OCTOBER 6, 2025")
print("="*80)

# Load the October 6 trade data
data_file = Path('trade_and_quote_data/data_management/flatfiles/2025-10-06.csv.gz')

if not data_file.exists():
    print(f"❌ Data file not found: {data_file}")
    exit(1)

print(f"\n📊 Loading SPY options data from {data_file.name}...")

# Read the compressed file
spy_trades = []
with gzip.open(data_file, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        # Filter for SPY options (ticker starts with O:SPY)
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY')]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)

df = pd.concat(spy_trades, ignore_index=True) if spy_trades else pd.DataFrame()

print(f"   ✅ Loaded {len(df):,} SPY options trades")

if len(df) == 0:
    print("   ⚠️ No SPY options trades found")
    exit(1)

# Parse option details from ticker
def parse_ticker(ticker):
    """Parse Polygon option ticker format: O:SPY241011C00670000"""
    try:
        parts = ticker.replace('O:', '')

        # Extract expiry (YYMMDD format)
        expiry_str = parts[3:9]
        year = int('20' + expiry_str[:2])
        month = int(expiry_str[2:4])
        day = int(expiry_str[4:6])
        expiry = datetime(year, month, day)

        # Extract type and strike
        option_type = parts[9].lower()  # 'c' or 'p'
        strike = int(parts[10:]) / 1000  # Convert from thousandths

        return pd.Series({
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'dte': (expiry - datetime(2025, 10, 6)).days
        })
    except:
        return pd.Series({'option_type': None, 'strike': None, 'expiry': None, 'dte': None})

print("\n⚙️ Parsing option details...")
option_details = df['ticker'].apply(parse_ticker)
df = pd.concat([df, option_details], axis=1)

# Remove invalid rows
df = df.dropna(subset=['option_type', 'strike', 'expiry'])

# Convert timestamp to datetime (using sip_timestamp column)
df['sip_timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')
df['hour'] = df['sip_timestamp'].dt.hour

print(f"   ✅ Parsed {len(df):,} valid SPY options trades")

# =============================================================================
# OVERALL STATISTICS
# =============================================================================

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

total_volume = df['size'].sum()
call_volume = df[df['option_type'] == 'c']['size'].sum()
put_volume = df[df['option_type'] == 'p']['size'].sum()
pc_ratio = put_volume / call_volume if call_volume > 0 else 0

print(f"\n📊 VOLUME BREAKDOWN:")
print(f"   Total Volume: {total_volume:,} contracts")
print(f"   Call Volume:  {call_volume:,} ({call_volume/total_volume*100:.1f}%)")
print(f"   Put Volume:   {put_volume:,} ({put_volume/total_volume*100:.1f}%)")
print(f"   Put/Call Ratio: {pc_ratio:.2f}")

if pc_ratio > 1.2:
    print(f"   → ⚠️ Elevated put activity (hedging/bearish)")
elif pc_ratio < 0.7:
    print(f"   → 📈 Call-heavy (bullish/complacent)")
else:
    print(f"   → Balanced sentiment")

# Dollar volume
df['dollar_volume'] = df['price'] * df['size'] * 100  # Options are 100 shares
total_dollar_volume = df['dollar_volume'].sum()

print(f"\n💰 DOLLAR VOLUME:")
print(f"   Total: ${total_dollar_volume:,.0f}")
print(f"   Calls: ${df[df['option_type'] == 'c']['dollar_volume'].sum():,.0f}")
print(f"   Puts:  ${df[df['option_type'] == 'p']['dollar_volume'].sum():,.0f}")

# =============================================================================
# STRIKE ANALYSIS - Find Heavily Bid Strikes
# =============================================================================

print("\n" + "="*80)
print("STRIKE ANALYSIS - HEAVILY BID STRIKES")
print("="*80)

# Assuming SPY is around 669
spy_price = 669

# Categorize strikes
df['moneyness'] = np.where(
    df['option_type'] == 'c',
    df['strike'] / spy_price - 1,  # Calls: positive if OTM
    1 - df['strike'] / spy_price   # Puts: positive if OTM
)

df['strike_category'] = pd.cut(
    df['moneyness'],
    bins=[-np.inf, -0.02, 0, 0.02, 0.05, np.inf],
    labels=['Deep ITM', 'ITM', 'ATM', 'OTM', 'Far OTM']
)

# Top strikes by volume
print("\n🎯 TOP 10 STRIKES BY VOLUME:")
strike_volume = df.groupby(['strike', 'option_type']).agg({
    'size': 'sum',
    'dollar_volume': 'sum'
}).reset_index()

top_strikes = strike_volume.nlargest(10, 'size')
for _, row in top_strikes.iterrows():
    strike_type = 'Call' if row['option_type'] == 'c' else 'Put'
    moneyness = (row['strike'] / spy_price - 1) * 100
    otm_itm = 'OTM' if moneyness > 0 else 'ITM' if moneyness < 0 else 'ATM'
    print(f"   ${row['strike']:.0f} {strike_type} ({otm_itm:3s} {moneyness:+.1f}%): "
          f"{row['size']:8,.0f} contracts, ${row['dollar_volume']/1e6:6.1f}M")

# Analyze strike distribution
print("\n📊 STRIKE DISTRIBUTION:")
strike_dist = df.groupby('strike_category')['size'].sum()
for category, volume in strike_dist.items():
    pct = volume / total_volume * 100
    print(f"   {category:10s}: {volume:10,.0f} ({pct:5.1f}%)")

# =============================================================================
# EXPIRY ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("EXPIRY CONCENTRATION ANALYSIS")
print("="*80)

# Group by DTE buckets
dte_buckets = pd.cut(
    df['dte'],
    bins=[-1, 0, 1, 7, 30, 60, 365],
    labels=['0DTE', '1DTE', '1Week', '1Month', '2Month', '>2Month']
)

df['dte_bucket'] = dte_buckets

print("\n📅 VOLUME BY EXPIRY:")
expiry_volume = df.groupby('dte_bucket')['size'].sum()
for bucket, volume in expiry_volume.items():
    pct = volume / total_volume * 100
    print(f"   {bucket:10s}: {volume:10,.0f} ({pct:5.1f}%)")

# Specific expiry analysis
print("\n🎯 TOP 10 EXPIRIES BY VOLUME:")
expiry_detail = df.groupby(['expiry', 'dte']).agg({
    'size': 'sum',
    'dollar_volume': 'sum'
}).reset_index()

top_expiries = expiry_detail.nlargest(10, 'size')
for _, row in top_expiries.iterrows():
    print(f"   {row['expiry'].strftime('%Y-%m-%d')} ({int(row['dte']):3d} DTE): "
          f"{row['size']:8,.0f} contracts, ${row['dollar_volume']/1e6:6.1f}M")

# =============================================================================
# PUT/CALL RATIO BY EXPIRY
# =============================================================================

print("\n" + "="*80)
print("PUT/CALL RATIO BY EXPIRY BUCKET")
print("="*80)

pc_by_expiry = df.groupby(['dte_bucket', 'option_type'])['size'].sum().unstack(fill_value=0)
pc_by_expiry['pc_ratio'] = pc_by_expiry['p'] / pc_by_expiry['c']

print("\n📊 P/C RATIO TERM STRUCTURE:")
for bucket in pc_by_expiry.index:
    ratio = pc_by_expiry.loc[bucket, 'pc_ratio']
    call_vol = pc_by_expiry.loc[bucket, 'c']
    put_vol = pc_by_expiry.loc[bucket, 'p']

    signal = ""
    if ratio > 1.5:
        signal = " → ⚠️ Heavy hedging"
    elif ratio > 1.2:
        signal = " → Elevated puts"
    elif ratio < 0.7:
        signal = " → Bullish"

    print(f"   {bucket:10s}: {ratio:5.2f} (C:{call_vol:8,.0f} P:{put_vol:8,.0f}){signal}")

# =============================================================================
# INTRADAY PATTERNS
# =============================================================================

print("\n" + "="*80)
print("INTRADAY TRADING PATTERNS")
print("="*80)

hourly_volume = df.groupby('hour')['size'].sum()
print("\n⏰ VOLUME BY HOUR (ET):")
for hour in range(9, 17):
    if hour in hourly_volume.index:
        vol = hourly_volume[hour]
        pct = vol / total_volume * 100
        bar = '█' * int(pct)
        print(f"   {hour:02d}:00  {bar:20s} {vol:8,.0f} ({pct:4.1f}%)")

# =============================================================================
# LARGE TRADES (POTENTIAL INSTITUTIONAL)
# =============================================================================

print("\n" + "="*80)
print("LARGE TRADES ANALYSIS (>100 contracts)")
print("="*80)

large_trades = df[df['size'] > 100].copy()
print(f"\n📊 LARGE TRADE STATISTICS:")
print(f"   Count: {len(large_trades):,} trades")
print(f"   Volume: {large_trades['size'].sum():,} contracts")
print(f"   % of Total Volume: {large_trades['size'].sum() / total_volume * 100:.1f}%")

# Large trade P/C ratio
large_pc = large_trades[large_trades['option_type'] == 'p']['size'].sum() / \
           large_trades[large_trades['option_type'] == 'c']['size'].sum()
print(f"   Large Trade P/C Ratio: {large_pc:.2f}")

if large_pc > pc_ratio * 1.2:
    print(f"   → Institutions more bearish than retail")
elif large_pc < pc_ratio * 0.8:
    print(f"   → Institutions more bullish than retail")
else:
    print(f"   → Aligned with overall market")

# Top large trades
print("\n🐋 TOP 10 LARGEST TRADES:")
top_large = large_trades.nlargest(10, 'size')[['sip_timestamp', 'ticker', 'size', 'price', 'dollar_volume']]
for _, trade in top_large.iterrows():
    ticker_parts = trade['ticker'].replace('O:SPY', '')
    print(f"   {trade['sip_timestamp'].strftime('%H:%M:%S')}: {trade['size']:5,.0f} contracts "
          f"@ ${trade['price']:6.2f} (${trade['dollar_volume']/1e6:5.1f}M)")

# =============================================================================
# POSITIONING SUMMARY
# =============================================================================

print("\n" + "="*80)
print("POSITIONING SUMMARY & INTERPRETATION")
print("="*80)

print("\n🎯 KEY FINDINGS:")

# 1. Overall sentiment
print(f"\n1. OVERALL SENTIMENT:")
if pc_ratio > 1.2:
    print(f"   • P/C ratio of {pc_ratio:.2f} indicates defensive/bearish positioning")
elif pc_ratio < 0.7:
    print(f"   • P/C ratio of {pc_ratio:.2f} indicates bullish/complacent positioning")
else:
    print(f"   • P/C ratio of {pc_ratio:.2f} indicates balanced positioning")

# 2. Term structure
print(f"\n2. TERM STRUCTURE POSITIONING:")
short_term_pc = pc_by_expiry.iloc[:3]['pc_ratio'].mean() if len(pc_by_expiry) > 3 else 0
long_term_pc = pc_by_expiry.iloc[3:]['pc_ratio'].mean() if len(pc_by_expiry) > 3 else 0

if short_term_pc > long_term_pc * 1.2:
    print(f"   • Near-term hedging elevated (ST P/C: {short_term_pc:.2f} vs LT: {long_term_pc:.2f})")
    print(f"   • Traders expecting near-term volatility")
elif long_term_pc > short_term_pc * 1.2:
    print(f"   • Longer-term hedging elevated")
    print(f"   • Structural concerns but near-term complacency")
else:
    print(f"   • Consistent positioning across timeframes")

# 3. Strike concentration
print(f"\n3. STRIKE CONCENTRATION:")
atm_pct = (strike_dist.get('ATM', 0) / total_volume * 100) if 'ATM' in strike_dist else 0
otm_pct = (strike_dist.get('OTM', 0) / total_volume * 100) if 'OTM' in strike_dist else 0
far_otm_pct = (strike_dist.get('Far OTM', 0) / total_volume * 100) if 'Far OTM' in strike_dist else 0

print(f"   • ATM activity: {atm_pct:.1f}%")
print(f"   • OTM activity: {otm_pct:.1f}%")
print(f"   • Far OTM activity: {far_otm_pct:.1f}%")

if far_otm_pct > 15:
    print(f"   • ⚠️ Elevated tail risk hedging detected")

# 4. Institutional vs Retail
print(f"\n4. INSTITUTIONAL POSITIONING:")
if large_pc > pc_ratio * 1.2:
    print(f"   • Large trades more bearish (P/C: {large_pc:.2f} vs Overall: {pc_ratio:.2f})")
    print(f"   • Smart money positioning defensively")
elif large_pc < pc_ratio * 0.8:
    print(f"   • Large trades more bullish")
    print(f"   • Institutions less concerned than retail")
else:
    print(f"   • Institutional positioning aligned with market")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Date analyzed: October 6, 2025")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")