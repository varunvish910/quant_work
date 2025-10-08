#!/usr/bin/env python3
"""
SPY Options Anomaly Analysis - October 2025
Analyzing put/call behavior, trade direction, and hedging patterns

Looking for behavioral changes in the rally:
- Are traders hedging with puts or still buying calls?
- Any change in trade direction (BTO/STO/BTC/STC)?
- Unusual put buying that wasn't there before?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

print("="*80)
print("SPY OPTIONS ANOMALY ANALYSIS - October 2025")
print("="*80)

# ============================================================================
# Load Trade Data from Flat Files
# ============================================================================

def load_trade_data(file_path: str) -> pd.DataFrame:
    """Load compressed trade data from flat file"""
    print(f"\n📂 Loading trade data from: {file_path}")

    try:
        if file_path.endswith('.gz'):
            df = pd.read_csv(file_path, compression='gzip')
        else:
            df = pd.read_csv(file_path)

        print(f"   ✅ Loaded {len(df):,} trades")
        return df
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        return None

# Find trade data files
data_dir = Path('/Users/varunvish/code/quant_work/trade_and_quote_data/data_management')
trade_files = list(data_dir.glob('*trades*.csv*'))

print(f"\nFound {len(trade_files)} trade data files:")
for f in trade_files:
    print(f"  - {f.name}")

if not trade_files:
    print("\n⚠️  No trade data files found. Need to download first.")
    print("   Run: python trade_and_quote_data/dealer_positioning/spy_trades_downloader.py")
    exit(1)

# Load most recent file
latest_file = sorted(trade_files)[-1]
trades = load_trade_data(str(latest_file))

if trades is None or len(trades) == 0:
    print("❌ No trade data available")
    exit(1)

# ============================================================================
# Parse and Clean Data
# ============================================================================

print("\n🔧 Parsing trade data...")

# Convert timestamps if needed
if 'timestamp' in trades.columns:
    trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='ns', errors='coerce')
elif 'sip_timestamp' in trades.columns:
    trades['timestamp'] = pd.to_datetime(trades['sip_timestamp'], unit='ns', errors='coerce')

# Extract date
if 'timestamp' in trades.columns:
    trades['date'] = trades['timestamp'].dt.date
    trades['hour'] = trades['timestamp'].dt.hour
    trades['minute'] = trades['timestamp'].dt.minute

print(f"   ✅ Trades from {trades['date'].min()} to {trades['date'].max()}")

# ============================================================================
# Parse Option Details from Ticker
# ============================================================================

def parse_option_ticker(ticker: str) -> dict:
    """Parse option ticker like O:SPY241011C00670000"""
    try:
        if not isinstance(ticker, str) or ':' not in ticker:
            return {'option_type': None, 'strike': None, 'expiry': None}

        parts = ticker.split(':')[1]  # Remove O: prefix

        # Extract expiry (YYMMDD)
        expiry_str = parts[3:9]
        year = "20" + expiry_str[:2]
        month = expiry_str[2:4]
        day = expiry_str[4:6]
        expiry = datetime.strptime(f"{year}{month}{day}", '%Y%m%d').date()

        # Extract type (C or P)
        option_type = parts[9].lower()

        # Extract strike
        strike_str = parts[10:]
        strike = int(strike_str) / 1000  # Convert from thousandths

        return {
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry
        }
    except Exception as e:
        return {'option_type': None, 'strike': None, 'expiry': None}

# Apply parsing
print("\n📋 Parsing option details from tickers...")
parsed = trades['ticker'].apply(parse_option_ticker)
trades['option_type'] = [p['option_type'] for p in parsed]
trades['strike'] = [p['strike'] for p in parsed]
trades['expiry'] = [p['expiry'] for p in parsed]

# Filter valid options only
trades = trades[trades['option_type'].notna()].copy()
print(f"   ✅ {len(trades):,} valid option trades")
print(f"      Calls: {(trades['option_type']=='c').sum():,}")
print(f"      Puts:  {(trades['option_type']=='p').sum():,}")

# ============================================================================
# Classify Trade Direction (BTO/STO/BTC/STC)
# ============================================================================

print("\n🎯 Classifying trade direction...")

# Simple classification based on price relative to mid
def classify_trade_direction(row):
    """Classify trade as BTO/STO/BTC/STC"""
    try:
        price = row.get('price', 0)

        # Try to get bid/ask
        bid = row.get('bid', None)
        ask = row.get('ask', None)

        if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
            # Fallback: use heuristics
            if row.get('size', 0) > 10:
                return 'BTO'  # Large trades likely opening
            else:
                return 'UNKNOWN'

        mid = (bid + ask) / 2

        # Classification logic:
        # Trade at/above ask = Buy
        # Trade at/below bid = Sell
        if price >= ask * 0.99:
            return 'BTO'  # Buy to Open (customer buying)
        elif price <= bid * 1.01:
            return 'STO'  # Sell to Open (customer selling)
        elif price > mid:
            return 'BTO'  # Above mid, likely buy
        else:
            return 'STO'  # Below mid, likely sell
    except:
        return 'UNKNOWN'

trades['trade_direction'] = trades.apply(classify_trade_direction, axis=1)

print(f"   Trade Direction Breakdown:")
print(trades['trade_direction'].value_counts())

# ============================================================================
# Calculate Key Metrics by Date
# ============================================================================

print("\n📊 Calculating daily metrics...")

# Group by date
daily = trades.groupby('date').agg({
    'size': 'sum',
    'price': 'mean'
}).rename(columns={'size': 'total_volume', 'price': 'avg_price'})

# Put/Call volumes
put_volume = trades[trades['option_type']=='p'].groupby('date')['size'].sum()
call_volume = trades[trades['option_type']=='c'].groupby('date')['size'].sum()

daily['put_volume'] = put_volume
daily['call_volume'] = call_volume
daily['put_call_ratio'] = daily['put_volume'] / daily['call_volume'].replace(0, 1)

# Trade direction breakdown
for direction in ['BTO', 'STO', 'BTC', 'STC']:
    direction_volume = trades[trades['trade_direction']==direction].groupby('date')['size'].sum()
    daily[f'{direction}_volume'] = direction_volume

# Fill NaN with 0
daily = daily.fillna(0)

# Calculate rolling metrics (to smooth out noise)
daily['put_call_ratio_7d'] = daily['put_call_ratio'].rolling(7, min_periods=1).mean()
daily['BTO_pct'] = daily['BTO_volume'] / daily['total_volume'] * 100
daily['STO_pct'] = daily['STO_volume'] / daily['total_volume'] * 100

print(f"   ✅ Daily metrics calculated for {len(daily)} days")

# ============================================================================
# Detect Anomalies and Behavioral Changes
# ============================================================================

print("\n🔍 Detecting anomalies and behavioral changes...")

# Calculate z-scores to find outliers
daily['put_call_zscore'] = (daily['put_call_ratio'] - daily['put_call_ratio'].mean()) / daily['put_call_ratio'].std()
daily['BTO_zscore'] = (daily['BTO_pct'] - daily['BTO_pct'].mean()) / daily['BTO_pct'].std()

# Find anomalies (z-score > 2)
anomaly_days = daily[
    (daily['put_call_zscore'].abs() > 2) |
    (daily['BTO_zscore'].abs() > 2)
]

print(f"\n🚨 Found {len(anomaly_days)} anomalous trading days:")
for date, row in anomaly_days.iterrows():
    print(f"   {date}:")
    print(f"      Put/Call Ratio: {row['put_call_ratio']:.2f} (z={row['put_call_zscore']:.2f})")
    print(f"      BTO %: {row['BTO_pct']:.1f}% (z={row['BTO_zscore']:.2f})")

# Detect trend changes (last 7 days vs previous 30 days)
if len(daily) >= 30:
    recent_7d = daily.tail(7)
    previous_30d = daily.iloc[-37:-7]  # Days 37-7 ago

    print(f"\n📈 Recent vs Historical Comparison:")
    print(f"   Put/Call Ratio:")
    print(f"      Last 7 days:    {recent_7d['put_call_ratio'].mean():.2f}")
    print(f"      Prior 30 days:  {previous_30d['put_call_ratio'].mean():.2f}")
    print(f"      Change: {((recent_7d['put_call_ratio'].mean() / previous_30d['put_call_ratio'].mean() - 1) * 100):+.1f}%")

    print(f"\n   BTO Activity (% of volume):")
    print(f"      Last 7 days:    {recent_7d['BTO_pct'].mean():.1f}%")
    print(f"      Prior 30 days:  {previous_30d['BTO_pct'].mean():.1f}%")
    print(f"      Change: {(recent_7d['BTO_pct'].mean() - previous_30d['BTO_pct'].mean()):+.1f}pp")

# ============================================================================
# Visualization
# ============================================================================

print("\n🎨 Creating visualizations...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14))
fig.suptitle('SPY OPTIONS TRADING BEHAVIOR ANALYSIS - Anomaly Detection',
             fontsize=16, fontweight='bold')

dates = daily.index

# 1. Put/Call Ratio
ax1 = axes[0]
ax1.plot(dates, daily['put_call_ratio'], 'b-', linewidth=2, marker='o', markersize=4, label='Daily P/C Ratio')
ax1.plot(dates, daily['put_call_ratio_7d'], 'r-', linewidth=3, label='7-Day Average', alpha=0.7)
ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Neutral (1.0)')
ax1.axhline(y=daily['put_call_ratio'].mean(), color='green', linestyle='--', alpha=0.5, label=f'Mean ({daily["put_call_ratio"].mean():.2f})')

# Highlight anomalies
if len(anomaly_days) > 0:
    anomaly_dates = anomaly_days.index
    anomaly_values = daily.loc[anomaly_dates, 'put_call_ratio']
    ax1.scatter(anomaly_dates, anomaly_values, color='red', s=200, zorder=5,
               marker='*', edgecolors='black', linewidths=2, label='Anomaly')

ax1.set_title('Put/Call Volume Ratio - Are Traders Hedging?', fontsize=12, fontweight='bold')
ax1.set_ylabel('Put/Call Ratio', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Trade Direction Breakdown
ax2 = axes[1]
ax2.plot(dates, daily['BTO_pct'], 'g-', linewidth=2, marker='o', markersize=3, label='BTO %')
ax2.plot(dates, daily['STO_pct'], 'r-', linewidth=2, marker='o', markersize=3, label='STO %')

# Add rolling averages
ax2.plot(dates, daily['BTO_pct'].rolling(7, min_periods=1).mean(), 'g--', linewidth=2, alpha=0.6, label='BTO 7d MA')
ax2.plot(dates, daily['STO_pct'].rolling(7, min_periods=1).mean(), 'r--', linewidth=2, alpha=0.6, label='STO 7d MA')

ax2.set_title('Trade Direction - Opening vs Closing Activity', fontsize=12, fontweight='bold')
ax2.set_ylabel('% of Total Volume', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Volume Breakdown by Option Type
ax3 = axes[2]
width = 0.8
ax3.bar(dates, daily['call_volume'], width, label='Call Volume', color='green', alpha=0.7)
ax3.bar(dates, daily['put_volume'], width, bottom=daily['call_volume'],
       label='Put Volume', color='red', alpha=0.7)

ax3.set_title('Volume Distribution - Calls vs Puts', fontsize=12, fontweight='bold')
ax3.set_ylabel('Contract Volume', fontsize=11)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Anomaly Score Timeline
ax4 = axes[3]
ax4.plot(dates, daily['put_call_zscore'], 'purple', linewidth=2, marker='o', markersize=3, label='P/C Anomaly Score')
ax4.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Anomaly Threshold (+2σ)')
ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.fill_between(dates, -2, 2, alpha=0.1, color='green', label='Normal Range')

ax4.set_title('Anomaly Detection Score (Z-Score)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Standard Deviations', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# Format all x-axes
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

plt.tight_layout()
plt.savefig('SPY_Options_Anomaly_Analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: SPY_Options_Anomaly_Analysis.png")
plt.close()

# ============================================================================
# Summary Report
# ============================================================================

print("\n" + "="*80)
print("SUMMARY REPORT - SPY OPTIONS BEHAVIOR")
print("="*80)

print(f"\n📅 Data Period: {daily.index.min()} to {daily.index.max()}")
print(f"📊 Total Days Analyzed: {len(daily)}")

print(f"\n🎯 Overall Metrics:")
print(f"   Put/Call Ratio (mean):     {daily['put_call_ratio'].mean():.2f}")
print(f"   Put/Call Ratio (recent):   {daily['put_call_ratio'].tail(7).mean():.2f}")
print(f"   BTO Activity (mean):       {daily['BTO_pct'].mean():.1f}%")
print(f"   STO Activity (mean):       {daily['STO_pct'].mean():.1f}%")

print(f"\n🔍 Anomalies Detected: {len(anomaly_days)} days")

if len(daily) >= 30:
    # Behavioral change analysis
    recent_pc = recent_7d['put_call_ratio'].mean()
    previous_pc = previous_30d['put_call_ratio'].mean()
    pc_change_pct = (recent_pc / previous_pc - 1) * 100

    recent_bto = recent_7d['BTO_pct'].mean()
    previous_bto = previous_30d['BTO_pct'].mean()
    bto_change_pp = recent_bto - previous_bto

    print(f"\n📈 Behavioral Changes (Last 7 days vs Prior 30 days):")
    print(f"   Put/Call Ratio: {pc_change_pct:+.1f}% change")
    print(f"   BTO Activity: {bto_change_pp:+.1f}pp change")

    print(f"\n💡 INTERPRETATION:")

    # Interpret Put/Call changes
    if recent_pc > previous_pc * 1.2:
        print(f"   🚨 SIGNIFICANT INCREASE in put buying ({pc_change_pct:+.0f}%)")
        print(f"      → Traders are hedging more / becoming more defensive")
    elif recent_pc < previous_pc * 0.8:
        print(f"   ⚠️  SIGNIFICANT DECREASE in put buying ({pc_change_pct:+.0f}%)")
        print(f"      → Less hedging / more complacent")
    else:
        print(f"   ✅ Put/Call ratio relatively stable ({pc_change_pct:+.0f}%)")

    # Interpret BTO changes
    if bto_change_pp > 10:
        print(f"   🟢 INCREASE in Buy-to-Open activity ({bto_change_pp:+.1f}pp)")
        print(f"      → More fresh positioning / new trades opening")
    elif bto_change_pp < -10:
        print(f"   🔴 DECREASE in Buy-to-Open activity ({bto_change_pp:+.1f}pp)")
        print(f"      → Less new positioning / more closing activity")
    else:
        print(f"   ✅ BTO activity relatively stable ({bto_change_pp:+.1f}pp)")

print("\n" + "="*80)
print("Analysis complete. Check SPY_Options_Anomaly_Analysis.png for visualizations.")
print("="*80)
