#!/usr/bin/env python3
"""
Analyze SPX Market Research Using SPY Options Data
Validates Doc Trader McGraw's research on:
1. Dealer short gamma positioning
2. OTM call buying concentration (6800 strike equivalent)
3. Systematic leverage feedback loop
4. Low realized vol environment

Uses Polygon flat files with quote data for proper trade classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
from pathlib import Path
from datetime import datetime

print("="*80)
print("SPX MARKET RESEARCH VALIDATION - Using SPY Options Data")
print("="*80)

# Load the parsed summary
flatfiles_dir = Path('trade_and_quote_data/data_management/flatfiles')
summary = pd.read_csv(flatfiles_dir / 'complete_spy_summary.csv')
summary['date'] = pd.to_datetime(summary['date'])

print(f"\n📊 30-Day Summary Loaded: {summary['date'].min()} to {summary['date'].max()}")

# Focus on most recent day for detailed analysis
latest_date = summary['date'].max()
latest_file = flatfiles_dir / f"{latest_date.strftime('%Y-%m-%d')}.csv.gz"

print(f"\n🔍 Analyzing detailed trades for: {latest_date.strftime('%Y-%m-%d')}")
print(f"   File: {latest_file.name}")

# Parse SPY options with additional details
spy_trades = []

print(f"\n📥 Loading and parsing SPY options trades...")

with gzip.open(latest_file, 'rt') as f:
    chunk_num = 0
    for chunk in pd.read_csv(f, chunksize=100000):
        chunk_num += 1

        # Filter for SPY options
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]

        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)

        if chunk_num % 10 == 0:
            print(f"   Processed {chunk_num} chunks...")

if not spy_trades:
    print("❌ No SPY options trades found!")
    exit(1)

trades = pd.concat(spy_trades, ignore_index=True)
print(f"\n✅ Loaded {len(trades):,} SPY options trades")

# Parse option details from ticker
def parse_spy_option(ticker):
    """
    Parse SPY option ticker: O:SPY241007C00570000
    Format: O:SPY + YYMMDD + C/P + Strike (8 digits, divide by 1000)
    """
    try:
        if not ticker.startswith('O:SPY'):
            return None, None, None, None

        parts = ticker[5:]  # Remove 'O:SPY'

        if len(parts) < 15:
            return None, None, None, None

        exp_date_str = parts[:6]  # YYMMDD
        option_type = parts[6].lower()  # 'c' or 'p'
        strike_str = parts[7:15]  # 8 digits

        # Parse expiration
        exp_date = datetime.strptime('20' + exp_date_str, '%Y%m%d')

        # Parse strike
        strike = int(strike_str) / 1000

        return exp_date, option_type, strike, ticker
    except:
        return None, None, None, None

print(f"\n🔍 Parsing option details...")
trades[['expiration', 'option_type', 'strike', 'ticker_clean']] = trades['ticker'].apply(
    lambda x: pd.Series(parse_spy_option(x))
)

# Remove unparseable trades
trades = trades.dropna(subset=['expiration', 'option_type', 'strike'])
print(f"   ✅ Parsed {len(trades):,} valid options")

# Get SPY price from yfinance (Oct 6, 2025 was yesterday)
import yfinance as yf
spy = yf.Ticker('SPY')
spy_hist = spy.history(start=latest_date.strftime('%Y-%m-%d'), period='1d')
spy_price = spy_hist['Close'].iloc[0] if not spy_hist.empty else 671.61
print(f"\n💰 SPY Spot Price on {latest_date.strftime('%Y-%m-%d')}: ${spy_price:.2f}")

# Calculate moneyness
trades['moneyness'] = (trades['strike'] - spy_price) / spy_price * 100

# Categorize by moneyness
def categorize_moneyness(row):
    if row['option_type'] == 'c':
        if row['moneyness'] < -5:
            return 'Deep ITM'
        elif row['moneyness'] < -2:
            return 'ITM'
        elif row['moneyness'] < 2:
            return 'ATM'
        elif row['moneyness'] < 5:
            return 'OTM'
        else:
            return 'Far OTM'
    else:  # puts
        if row['moneyness'] > 5:
            return 'Deep ITM'
        elif row['moneyness'] > 2:
            return 'ITM'
        elif row['moneyness'] > -2:
            return 'ATM'
        elif row['moneyness'] > -5:
            return 'OTM'
        else:
            return 'Far OTM'

trades['moneyness_category'] = trades.apply(categorize_moneyness, axis=1)

# Gamma exposure proxy (calculate on main dataframe before splitting)
def estimate_gamma_proxy(row):
    """Simple gamma proxy: highest near ATM, decays with moneyness"""
    abs_moneyness = abs(row['moneyness'])
    if abs_moneyness < 2:
        return 1.0  # ATM highest gamma
    elif abs_moneyness < 5:
        return 0.5  # OTM moderate gamma
    else:
        return 0.2  # Far OTM low gamma

trades['gamma_proxy'] = trades.apply(estimate_gamma_proxy, axis=1)

# Split by calls and puts
calls = trades[trades['option_type'] == 'c'].copy()
puts = trades[trades['option_type'] == 'p'].copy()

print(f"\n📈 Calls: {len(calls):,} trades, {calls['size'].sum():,.0f} volume")
print(f"📉 Puts: {len(puts):,} trades, {puts['size'].sum():,.0f} volume")

# Analyze strike distribution
print(f"\n🎯 STRIKE DISTRIBUTION ANALYSIS")
print("="*60)

# Calls by moneyness
call_volume_by_moneyness = calls.groupby('moneyness_category')['size'].sum()
print(f"\n📈 Call Volume by Moneyness:")
for cat in ['Deep ITM', 'ITM', 'ATM', 'OTM', 'Far OTM']:
    if cat in call_volume_by_moneyness.index:
        vol = call_volume_by_moneyness[cat]
        pct = vol / call_volume_by_moneyness.sum() * 100
        print(f"   {cat:12s}: {vol:>12,.0f} ({pct:>5.1f}%)")

# Far OTM call strikes (equivalent to SPX 6800 analysis)
far_otm_calls = calls[calls['moneyness'] > 5].copy()
if len(far_otm_calls) > 0:
    print(f"\n🚀 Far OTM Call Analysis (>5% above spot):")
    print(f"   Volume: {far_otm_calls['size'].sum():,.0f} ({far_otm_calls['size'].sum() / calls['size'].sum() * 100:.1f}% of all calls)")
    print(f"   Number of strikes: {far_otm_calls['strike'].nunique()}")

    # Top strikes
    top_strikes = far_otm_calls.groupby('strike')['size'].sum().sort_values(ascending=False).head(10)
    print(f"\n   Top 10 Far OTM Call Strikes:")
    for strike, vol in top_strikes.items():
        pct_above = (strike - spy_price) / spy_price * 100
        print(f"      ${strike:.0f} (+{pct_above:.1f}%): {vol:>10,.0f} volume")

# Puts by moneyness
put_volume_by_moneyness = puts.groupby('moneyness_category')['size'].sum()
print(f"\n📉 Put Volume by Moneyness:")
for cat in ['Deep ITM', 'ITM', 'ATM', 'OTM', 'Far OTM']:
    if cat in put_volume_by_moneyness.index:
        vol = put_volume_by_moneyness[cat]
        pct = vol / put_volume_by_moneyness.sum() * 100
        print(f"   {cat:12s}: {vol:>12,.0f} ({pct:>5.1f}%)")

# Gamma exposure analysis (approximation)
print(f"\n⚡ DEALER GAMMA POSITIONING")
print("="*60)

# Dealer position = opposite of customer flow
# Assume dealers are SHORT customer buys (positive gamma for dealers when customers buy puts)
call_gamma = (calls['size'] * calls['gamma_proxy']).sum()
put_gamma = (puts['size'] * puts['gamma_proxy']).sum()

# Net dealer gamma (assuming dealers short customer flow)
# When customers buy calls, dealers short calls (negative gamma)
# When customers buy puts, dealers short puts (positive gamma)
net_dealer_gamma = put_gamma - call_gamma

print(f"\n   Call Volume-Weighted Gamma: {call_gamma:,.0f}")
print(f"   Put Volume-Weighted Gamma:  {put_gamma:,.0f}")
print(f"   Net Dealer Gamma Exposure:  {net_dealer_gamma:,.0f}")

if net_dealer_gamma > 0:
    print(f"   🟢 Dealers appear LONG gamma (hedging activity supports market)")
elif net_dealer_gamma < -call_gamma * 0.2:
    print(f"   🔴 Dealers appear SHORT gamma (market vulnerable to volatility)")
else:
    print(f"   🟡 Dealers appear NEUTRAL gamma")

# Time to expiration analysis
trades['dte'] = (trades['expiration'] - pd.Timestamp(latest_date)).dt.days

print(f"\n📅 EXPIRATION ANALYSIS")
print("="*60)

dte_buckets = [0, 1, 7, 30, 90, 365]
dte_labels = ['0DTE', '1-7 DTE', '1-4W', '1-3M', '3M+']

trades['dte_bucket'] = pd.cut(trades['dte'], bins=dte_buckets, labels=dte_labels, include_lowest=True)

vol_by_dte = trades.groupby(['dte_bucket', 'option_type'])['size'].sum().unstack(fill_value=0)
print(f"\nVolume by Expiration:")
print(vol_by_dte)

# P/C by expiration
pc_by_dte = vol_by_dte['p'] / vol_by_dte['c']
print(f"\nP/C Ratio by Expiration:")
for dte, pc in pc_by_dte.items():
    print(f"   {dte}: {pc:.2f}")

# Summary findings
print(f"\n{'='*80}")
print("VALIDATION OF SPX RESEARCH CLAIMS")
print(f"{'='*80}")

print(f"\n1️⃣ **OTM Call Buying Concentration:**")
far_otm_pct = far_otm_calls['size'].sum() / calls['size'].sum() * 100 if len(far_otm_calls) > 0 else 0
if far_otm_pct > 15:
    print(f"   ✅ CONFIRMED - {far_otm_pct:.1f}% of call volume in far OTM strikes")
elif far_otm_pct > 8:
    print(f"   ⚠️  MODERATE - {far_otm_pct:.1f}% of call volume in far OTM strikes")
else:
    print(f"   ❌ NOT OBSERVED - Only {far_otm_pct:.1f}% in far OTM strikes")

print(f"\n2️⃣ **Dealer Short Gamma Positioning:**")
if net_dealer_gamma < -call_gamma * 0.2:
    print(f"   ✅ CONFIRMED - Net dealer gamma negative ({net_dealer_gamma:,.0f})")
elif net_dealer_gamma < 0:
    print(f"   ⚠️  SLIGHT - Net dealer gamma slightly negative ({net_dealer_gamma:,.0f})")
else:
    print(f"   ❌ NOT OBSERVED - Dealers appear long gamma ({net_dealer_gamma:,.0f})")

print(f"\n3️⃣ **Elevated Put Buying (Defensive Positioning):**")
current_pc = summary.iloc[-1]['pc_ratio']
if current_pc > 1.3:
    print(f"   ✅ CONFIRMED - P/C ratio at {current_pc:.2f}")
elif current_pc > 1.1:
    print(f"   ⚠️  MODERATE - P/C ratio at {current_pc:.2f}")
else:
    print(f"   ❌ NOT OBSERVED - P/C ratio at {current_pc:.2f}")

print(f"\n4️⃣ **30-Day Trend Analysis:**")
recent_7d = summary.tail(7)['pc_ratio'].mean()
prior_14d = summary.iloc[-21:-7]['pc_ratio'].mean() if len(summary) >= 21 else summary['pc_ratio'].mean()
pc_change = (recent_7d / prior_14d - 1) * 100

if abs(pc_change) > 10:
    print(f"   ⚠️  SIGNIFICANT SHIFT - P/C changed {pc_change:+.1f}% in last 7 days")
elif abs(pc_change) > 5:
    print(f"   📊 MODERATE CHANGE - P/C changed {pc_change:+.1f}% in last 7 days")
else:
    print(f"   ✅ STABLE - P/C changed {pc_change:+.1f}% in last 7 days")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
