#!/usr/bin/env python3
"""
Calculate VIX Term Structure from VIX Options Trades
Using ATM strikes as proxy for forward VIX expectations
"""

import pandas as pd
import numpy as np
import gzip
from datetime import datetime, timedelta
from pathlib import Path

print("="*80)
print("VIX TERM STRUCTURE FROM OPTIONS TRADES")
print("="*80)

# Load October 6 trade data
data_file = Path('trade_and_quote_data/data_management/flatfiles/2025-10-06.csv.gz')

print(f"\n📊 Loading VIX options data from {data_file.name}...")

# Read VIX options
vix_trades = []
with gzip.open(data_file, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        # VIX options start with O:VIX
        vix_chunk = chunk[chunk['ticker'].str.startswith('O:VIX')]
        if len(vix_chunk) > 0:
            vix_trades.append(vix_chunk)

if not vix_trades:
    print("   ⚠️ No VIX options found. Checking ticker format...")

    # Sample to see what tickers look like
    with gzip.open(data_file, 'rt') as f:
        sample = pd.read_csv(f, nrows=10000)
        # Look for anything VIX related
        vix_related = sample[sample['ticker'].str.contains('VIX', na=False)]
        if len(vix_related) > 0:
            print(f"   Found VIX-related tickers:")
            for ticker in vix_related['ticker'].unique()[:10]:
                print(f"      {ticker}")

df = pd.concat(vix_trades, ignore_index=True) if vix_trades else pd.DataFrame()

if len(df) == 0:
    print("\n❌ No VIX options trades found")
    print("\nAttempting with VX prefix (VIX futures options)...")

    # Try VX prefix
    vx_trades = []
    with gzip.open(data_file, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=100000):
            vx_chunk = chunk[chunk['ticker'].str.startswith('O:VX')]
            if len(vx_chunk) > 0:
                vx_trades.append(vx_chunk)

    if vx_trades:
        df = pd.concat(vx_trades, ignore_index=True)
        print(f"   ✅ Found {len(df):,} VX options trades")
    else:
        print("   ❌ No VX options either")
        exit(1)

print(f"   ✅ Loaded {len(df):,} VIX/VX options trades")

# Parse option details
def parse_vix_ticker(ticker):
    """Parse VIX option ticker format: O:VIX251022C00010000"""
    try:
        # Remove prefix O:VIX
        if not ticker.startswith('O:VIX'):
            return pd.Series({'option_type': None, 'strike': None, 'expiry': None, 'dte': None})

        parts = ticker.replace('O:VIX', '')

        # Format: YYMMDD + C/P + 8 digits for strike
        # Example: 251022C00010000
        #          YY MM DD C/P STRIKE

        if len(parts) < 15:  # Minimum length check
            return pd.Series({'option_type': None, 'strike': None, 'expiry': None, 'dte': None})

        # Extract date (YYMMDD)
        year = int('20' + parts[0:2])
        month = int(parts[2:4])
        day = int(parts[4:6])
        expiry = datetime(year, month, day)

        # Extract option type (C or P)
        option_type = parts[6].lower()

        # Extract strike (last 8 digits, in thousandths)
        strike_str = parts[7:15]
        strike = int(strike_str) / 1000.0

        return pd.Series({
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'dte': (expiry - datetime(2025, 10, 6)).days
        })

    except Exception as e:
        return pd.Series({'option_type': None, 'strike': None, 'expiry': None, 'dte': None})

print("\n⚙️ Parsing option details...")

# First, let's look at a sample ticker to understand format
sample_tickers = df['ticker'].head(5)
print("   Sample tickers:")
for ticker in sample_tickers:
    print(f"      {ticker}")

option_details = df['ticker'].apply(parse_vix_ticker)
df = pd.concat([df, option_details], axis=1)

# Remove invalid rows
df_valid = df.dropna(subset=['option_type', 'strike', 'expiry'])
print(f"   ✅ Parsed {len(df_valid):,} valid VIX options trades")

if len(df_valid) == 0:
    print("\n⚠️ Could not parse VIX option tickers")
    print("   Ticker format may be different than expected")
    exit(1)

# Current VIX level (approximate)
current_vix = 16.6

print(f"\n📊 Current VIX: {current_vix:.1f}")

# =============================================================================
# CALCULATE TERM STRUCTURE FROM ATM OPTIONS
# =============================================================================

print("\n" + "="*80)
print("VIX TERM STRUCTURE FROM ATM OPTIONS")
print("="*80)

# Group by expiry and find ATM strikes
expiry_analysis = []

for expiry in sorted(df_valid['expiry'].unique()):
    expiry_data = df_valid[df_valid['expiry'] == expiry]

    # Get calls only for ATM analysis
    calls = expiry_data[expiry_data['option_type'] == 'c']

    if len(calls) == 0:
        continue

    # Find volume-weighted average price for strikes near ATM
    # ATM = strikes within 10% of current VIX
    atm_range = (current_vix * 0.9, current_vix * 1.1)
    atm_calls = calls[(calls['strike'] >= atm_range[0]) & (calls['strike'] <= atm_range[1])]

    if len(atm_calls) > 0:
        # Volume-weighted average strike (proxy for forward VIX)
        total_volume = atm_calls['size'].sum()
        if total_volume > 0:
            weighted_strike = (atm_calls['strike'] * atm_calls['size']).sum() / total_volume
            weighted_price = (atm_calls['price'] * atm_calls['size']).sum() / total_volume

            dte = (expiry - datetime(2025, 10, 6)).days

            expiry_analysis.append({
                'expiry': expiry,
                'dte': dte,
                'atm_strike': weighted_strike,
                'atm_price': weighted_price,
                'volume': total_volume,
                'num_trades': len(atm_calls)
            })
    else:
        # If no ATM trades, use closest strike with volume
        calls_with_volume = calls.groupby('strike')['size'].sum().sort_values(ascending=False)
        if len(calls_with_volume) > 0:
            # Find strike closest to current VIX with decent volume
            for strike in calls_with_volume.index:
                if calls_with_volume[strike] > 100:  # Minimum volume threshold
                    dte = (expiry - datetime(2025, 10, 6)).days
                    strike_data = calls[calls['strike'] == strike]
                    weighted_price = (strike_data['price'] * strike_data['size']).sum() / strike_data['size'].sum()

                    expiry_analysis.append({
                        'expiry': expiry,
                        'dte': dte,
                        'atm_strike': strike,
                        'atm_price': weighted_price,
                        'volume': calls_with_volume[strike],
                        'num_trades': len(strike_data)
                    })
                    break

if len(expiry_analysis) > 0:
    expiry_df = pd.DataFrame(expiry_analysis).sort_values('dte')

    print(f"\n📊 IMPLIED VIX FORWARD LEVELS (from ATM call strikes):")
    print(f"   Current VIX: {current_vix:.1f}\n")
    print(f"   {'DTE':>4} | {'Date':^12} | {'ATM Strike':>10} | {'vs Spot':>8} | {'Volume':>10}")
    print("   " + "-"*60)

    for _, row in expiry_df.iterrows():
        if row['dte'] >= 0:  # Only future expiries
            change = (row['atm_strike'] / current_vix - 1) * 100
            date_str = row['expiry'].strftime('%Y-%m-%d')

            # Interpret
            if change > 20:
                signal = " ⚠️"
            elif change > 10:
                signal = " 📈"
            elif change < -10:
                signal = " 📉"
            else:
                signal = ""

            print(f"   {row['dte']:4d} | {date_str:^12} | {row['atm_strike']:10.1f} | {change:+7.1f}% | {row['volume']:10,.0f}{signal}")

    # Calculate average term structure slope
    if len(expiry_df) >= 2:
        # Short-term vs medium-term
        short_term = expiry_df[expiry_df['dte'] <= 30]['atm_strike'].mean() if len(expiry_df[expiry_df['dte'] <= 30]) > 0 else current_vix
        medium_term = expiry_df[(expiry_df['dte'] > 30) & (expiry_df['dte'] <= 90)]['atm_strike'].mean() if len(expiry_df[(expiry_df['dte'] > 30) & (expiry_df['dte'] <= 90)]) > 0 else short_term

        if short_term > 0 and medium_term > 0:
            term_slope = (medium_term - short_term) / short_term * 100

            print(f"\n📈 TERM STRUCTURE ANALYSIS:")
            print(f"   30-day forward VIX (implied): {short_term:.1f}")
            print(f"   60-90 day forward VIX (implied): {medium_term:.1f}")
            print(f"   Term structure slope: {term_slope:+.1f}%")

            if term_slope > 10:
                print(f"   → CONTANGO: Market expects higher volatility ahead")
            elif term_slope < -10:
                print(f"   → BACKWARDATION: Near-term stress expected")
            else:
                print(f"   → FLAT: No strong directional bias")

    # Warning about limitations
    print("\n⚠️ IMPORTANT LIMITATIONS:")
    print("   • This uses volume-weighted strikes, NOT true ATM")
    print("   • Without quotes, can't identify exact ATM strikes")
    print("   • Strikes may be skewed by large trades")
    print("   • This is a PROXY, not actual forward VIX")

else:
    print("\n❌ Could not calculate term structure - no valid expiry data")

# =============================================================================
# ADDITIONAL ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("VIX OPTIONS CHARACTERISTICS")
print("="*80)

# Overall P/C ratio
total_calls = df_valid[df_valid['option_type'] == 'c']['size'].sum()
total_puts = df_valid[df_valid['option_type'] == 'p']['size'].sum()

if total_calls > 0:
    pc_ratio = total_puts / total_calls
    print(f"\n📊 VIX Options P/C Ratio: {pc_ratio:.2f}")

    if pc_ratio > 1.5:
        print("   → Heavy put buying (expecting vol to decline)")
    elif pc_ratio < 0.7:
        print("   → Heavy call buying (expecting vol spike)")
    else:
        print("   → Balanced positioning")

# Most active strikes
print(f"\n🎯 Most Active VIX Strikes (all expiries):")
strike_volume = df_valid.groupby('strike')['size'].sum().sort_values(ascending=False)
for strike, volume in strike_volume.head(10).items():
    distance = (strike / current_vix - 1) * 100
    print(f"   Strike {strike:5.0f} ({distance:+6.1f}% from spot): {volume:10,.0f} contracts")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")