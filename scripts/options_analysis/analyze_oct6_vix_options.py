#!/usr/bin/env python3
"""
Analyze VIX Options for October 6, 2025
Focus on term structure, hedging demand, and positioning
"""

import pandas as pd
import numpy as np
import gzip
from datetime import datetime, timedelta
from pathlib import Path

print("="*80)
print("VIX OPTIONS ANALYSIS - OCTOBER 6, 2025")
print("="*80)

# Load the October 6 trade data
data_file = Path('trade_and_quote_data/data_management/flatfiles/2025-10-06.csv.gz')

if not data_file.exists():
    print(f"❌ Data file not found: {data_file}")
    exit(1)

print(f"\n📊 Loading VIX options data from {data_file.name}...")

# Read the compressed file looking for VIX options
vix_trades = []
with gzip.open(data_file, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        # Filter for VIX options (ticker starts with O:VIX)
        vix_chunk = chunk[chunk['ticker'].str.startswith('O:VIX')]
        if len(vix_chunk) > 0:
            vix_trades.append(vix_chunk)

df = pd.concat(vix_trades, ignore_index=True) if vix_trades else pd.DataFrame()

print(f"   ✅ Loaded {len(df):,} VIX options trades")

if len(df) == 0:
    print("   ⚠️ No VIX options trades found")
    print("\n📊 Checking if VIX options exist in data...")

    # Sample check for any VIX-related tickers
    with gzip.open(data_file, 'rt') as f:
        sample = pd.read_csv(f, nrows=10000)
        vix_related = sample[sample['ticker'].str.contains('VIX', na=False)]
        if len(vix_related) > 0:
            print(f"   Found VIX-related tickers: {vix_related['ticker'].unique()[:5]}")
        else:
            print("   No VIX-related tickers found in sample")

    # Create a note about VIX data
    print("\n" + "="*80)
    print("VIX OPTIONS DATA NOT AVAILABLE")
    print("="*80)
    print("\n📌 NOTE: VIX options may trade under different symbols:")
    print("   • VIX index options: Usually 'O:VIX' prefix")
    print("   • VIXW weekly options: 'O:VIXW' prefix")
    print("   • VXX ETF options: 'O:VXX' prefix")
    print("\n   The October 6 flat file may not include VIX index options.")
    print("   Consider checking VXX ETF options as a proxy.")

    # Check for VXX options instead
    print("\n📊 Checking for VXX ETF options (VIX proxy)...")
    vxx_trades = []
    with gzip.open(data_file, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=100000):
            vxx_chunk = chunk[chunk['ticker'].str.startswith('O:VXX')]
            if len(vxx_chunk) > 0:
                vxx_trades.append(vxx_chunk)

    if vxx_trades:
        df = pd.concat(vxx_trades, ignore_index=True)
        print(f"   ✅ Found {len(df):,} VXX ETF options trades (using as VIX proxy)")
        ticker_prefix = 'VXX'
    else:
        print("   ⚠️ No VXX options found either")
        print("\nAnalysis cannot continue without VIX/VXX options data.")
        exit(0)
else:
    ticker_prefix = 'VIX'

# Parse option details from ticker
def parse_ticker(ticker):
    """Parse Polygon option ticker format"""
    try:
        parts = ticker.replace(f'O:{ticker_prefix}', '')

        # Extract expiry (YYMMDD format)
        expiry_str = parts[:6]
        year = int('20' + expiry_str[:2])
        month = int(expiry_str[2:4])
        day = int(expiry_str[4:6])
        expiry = datetime(year, month, day)

        # Extract type and strike
        option_type = parts[6].lower()  # 'c' or 'p'
        strike = int(parts[7:]) / 1000  # Convert from thousandths

        return pd.Series({
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'dte': (expiry - datetime(2025, 10, 6)).days
        })
    except:
        return pd.Series({'option_type': None, 'strike': None, 'expiry': None, 'dte': None})

print(f"\n⚙️ Parsing {ticker_prefix} option details...")
option_details = df['ticker'].apply(parse_ticker)
df = pd.concat([df, option_details], axis=1)

# Remove invalid rows
df = df.dropna(subset=['option_type', 'strike', 'expiry'])

# Convert timestamp to datetime
df['sip_timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')
df['hour'] = df['sip_timestamp'].dt.hour

print(f"   ✅ Parsed {len(df):,} valid {ticker_prefix} options trades")

# =============================================================================
# OVERALL STATISTICS
# =============================================================================

print("\n" + "="*80)
print(f"{ticker_prefix} OPTIONS OVERALL STATISTICS")
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

# Interpretation for VIX
if ticker_prefix == 'VIX':
    if pc_ratio > 1.5:
        print(f"   → ⚠️ Heavy put demand (hedging against vol decline)")
    elif pc_ratio < 0.7:
        print(f"   → 📈 Call-heavy (expecting volatility spike)")
    else:
        print(f"   → Balanced positioning")
else:  # VXX
    if pc_ratio > 1.2:
        print(f"   → Bearish on volatility (expecting vol decline)")
    elif pc_ratio < 0.8:
        print(f"   → Bullish on volatility (expecting spike)")

# Dollar volume
df['dollar_volume'] = df['price'] * df['size'] * 100
total_dollar_volume = df['dollar_volume'].sum()

print(f"\n💰 DOLLAR VOLUME:")
print(f"   Total: ${total_dollar_volume:,.0f}")

# =============================================================================
# TERM STRUCTURE ANALYSIS (Key for VIX)
# =============================================================================

print("\n" + "="*80)
print(f"{ticker_prefix} TERM STRUCTURE FROM OPTIONS")
print("="*80)

# Group by expiry and calculate average ATM strike
expiry_analysis = []
current_vix = 16.6  # From earlier analysis

for expiry in df['expiry'].unique():
    expiry_data = df[df['expiry'] == expiry]

    # Find ATM strikes (closest to current VIX/VXX level)
    calls = expiry_data[expiry_data['option_type'] == 'c']

    if len(calls) > 0:
        # Weight strikes by volume to find implied forward level
        weighted_strike = (calls['strike'] * calls['size']).sum() / calls['size'].sum()

        expiry_analysis.append({
            'expiry': expiry,
            'dte': (expiry - datetime(2025, 10, 6)).days,
            'implied_forward': weighted_strike,
            'volume': expiry_data['size'].sum()
        })

expiry_df = pd.DataFrame(expiry_analysis).sort_values('dte')

print(f"\n📊 IMPLIED FORWARD {ticker_prefix} LEVELS:")
print(f"   Current {ticker_prefix}: {current_vix:.1f}")
print("\n   DTE  | Implied Level | Change from Spot")
print("   " + "-"*45)

for _, row in expiry_df.head(10).iterrows():
    change = (row['implied_forward'] / current_vix - 1) * 100

    signal = ""
    if change > 5:
        signal = " → Expecting vol increase"
    elif change < -5:
        signal = " → Expecting vol decrease"

    print(f"   {row['dte']:3d}  |    {row['implied_forward']:6.1f}     |    {change:+6.1f}%{signal}")

# Calculate average term structure
if len(expiry_df) > 0:
    short_term = expiry_df[expiry_df['dte'] <= 30]['implied_forward'].mean()
    medium_term = expiry_df[(expiry_df['dte'] > 30) & (expiry_df['dte'] <= 60)]['implied_forward'].mean()

    if not pd.isna(short_term) and not pd.isna(medium_term):
        term_structure = (medium_term - short_term) / short_term * 100

        print(f"\n📈 TERM STRUCTURE SLOPE:")
        print(f"   30-day implied: {short_term:.1f}")
        print(f"   60-day implied: {medium_term:.1f}")
        print(f"   Slope: {term_structure:+.1f}%")

        if term_structure > 5:
            print(f"   → Contango: Market expects higher volatility ahead")
        elif term_structure < -5:
            print(f"   → Backwardation: Near-term stress expected")
        else:
            print(f"   → Flat: No strong directional bias")

# =============================================================================
# STRIKE ANALYSIS - DEMAND PATTERNS
# =============================================================================

print("\n" + "="*80)
print(f"{ticker_prefix} STRIKE ANALYSIS")
print("="*80)

# Top strikes by volume
strike_volume = df.groupby(['strike', 'option_type']).agg({
    'size': 'sum',
    'dollar_volume': 'sum'
}).reset_index()

print(f"\n🎯 TOP 10 STRIKES BY VOLUME:")
top_strikes = strike_volume.nlargest(10, 'size')
for _, row in top_strikes.iterrows():
    strike_type = 'Call' if row['option_type'] == 'c' else 'Put'
    print(f"   ${row['strike']:.0f} {strike_type}: "
          f"{row['size']:8,.0f} contracts, ${row['dollar_volume']/1e6:6.1f}M")

# =============================================================================
# HEDGING DEMAND ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("HEDGING DEMAND ANALYSIS")
print("="*80)

# Analyze OTM calls (tail risk hedging)
otm_calls = df[(df['option_type'] == 'c') & (df['strike'] > current_vix * 1.1)]
otm_call_volume = otm_calls['size'].sum()
otm_call_pct = otm_call_volume / call_volume * 100 if call_volume > 0 else 0

print(f"\n🛡️ TAIL RISK HEDGING:")
print(f"   OTM Call Volume (>10% OTM): {otm_call_volume:,} contracts")
print(f"   % of Total Calls: {otm_call_pct:.1f}%")

if otm_call_pct > 30:
    print(f"   → ⚠️ HIGH tail risk hedging demand")
elif otm_call_pct > 20:
    print(f"   → Elevated hedging activity")
else:
    print(f"   → Normal hedging levels")

# Analyze put spreads (volatility selling)
itm_puts = df[(df['option_type'] == 'p') & (df['strike'] > current_vix)]
otm_puts = df[(df['option_type'] == 'p') & (df['strike'] < current_vix * 0.9)]

if len(itm_puts) > 0 and len(otm_puts) > 0:
    spread_ratio = itm_puts['size'].sum() / otm_puts['size'].sum()
    print(f"\n📊 PUT SPREAD ACTIVITY:")
    print(f"   ITM/OTM Put Ratio: {spread_ratio:.2f}")
    if spread_ratio > 2:
        print(f"   → Potential put spread selling (bearish vol)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print(f"{ticker_prefix} OPTIONS POSITIONING SUMMARY")
print("="*80)

print(f"\n🎯 KEY FINDINGS:")

# Overall positioning
print(f"\n1. DIRECTIONAL BIAS:")
if pc_ratio > 1.2:
    print(f"   • P/C ratio {pc_ratio:.2f} suggests bearish volatility positioning")
    print(f"   • Traders expecting VIX to decline or remain stable")
elif pc_ratio < 0.8:
    print(f"   • P/C ratio {pc_ratio:.2f} suggests bullish volatility positioning")
    print(f"   • Traders positioning for volatility spike")
else:
    print(f"   • Balanced P/C ratio {pc_ratio:.2f}")

# Term structure insights
if len(expiry_df) > 0:
    print(f"\n2. TERM STRUCTURE INSIGHTS:")
    avg_forward = expiry_df['implied_forward'].mean()
    if avg_forward > current_vix * 1.1:
        print(f"   • Options implying higher forward VIX ({avg_forward:.1f} vs {current_vix:.1f})")
        print(f"   • Market pricing in future volatility")
    elif avg_forward < current_vix * 0.9:
        print(f"   • Options implying lower forward VIX")
        print(f"   • Expecting volatility compression")
    else:
        print(f"   • Forward VIX aligned with spot")

# Hedging demand
print(f"\n3. HEDGING CHARACTERISTICS:")
if otm_call_pct > 25:
    print(f"   • High OTM call demand ({otm_call_pct:.1f}%)")
    print(f"   • Strong tail risk hedging")
else:
    print(f"   • Normal hedging levels")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Date analyzed: October 6, 2025")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")