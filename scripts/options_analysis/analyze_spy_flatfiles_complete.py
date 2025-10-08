#!/usr/bin/env python3
"""
Complete SPY Options Analysis - 30 Days Historical
Parse all flat files and create comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import gzip
# from tqdm import tqdm  # Not needed, using simple loop
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

print("="*80)
print("SPY OPTIONS - COMPLETE 30-DAY ANALYSIS")
print("="*80)

# Find all downloaded flat files
flatfiles_dir = Path('trade_and_quote_data/data_management/flatfiles')
files = sorted(flatfiles_dir.glob('2025-*.csv.gz'))

print(f"\n📂 Found {len(files)} flat files")
print(f"   Date range: {files[0].stem} to {files[-1].stem}")

# Parse all files
all_data = []

for i, file in enumerate(files, 1):
    # Extract date from filename (handles .csv.gz)
    # file.name = "2025-09-02.csv.gz"
    # We need just "2025-09-02"
    if file.name.endswith('.csv.gz'):
        date_str = file.name[:-7]  # Remove .csv.gz
    elif file.name.endswith('.csv'):
        date_str = file.name[:-4]  # Remove .csv
    else:
        date_str = file.stem
    print(f"\rParsing {i}/{len(files)}: {date_str}", end='', flush=True)

    try:
        spy_trades_day = []

        with gzip.open(file, 'rt') as f:
            for chunk in pd.read_csv(f, chunksize=100000):
                # Filter for SPY options
                spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]
                if len(spy_chunk) > 0:
                    spy_trades_day.append(spy_chunk)

        if spy_trades_day:
            day_df = pd.concat(spy_trades_day, ignore_index=True)

            # Parse option type
            def parse_type(ticker):
                try:
                    if not ticker.startswith('O:SPY'):
                        return None
                    parts = ticker[5:]
                    if len(parts) >= 15:
                        return parts[6].lower()
                    return None
                except:
                    return None

            day_df['option_type'] = day_df['ticker'].apply(parse_type)

            calls = day_df[day_df['option_type'] == 'c']
            puts = day_df[day_df['option_type'] == 'p']

            call_volume = calls['size'].sum() if len(calls) > 0 else 0
            put_volume = puts['size'].sum() if len(puts) > 0 else 0

            all_data.append({
                'date': pd.to_datetime(date_str),
                'total_trades': len(day_df),
                'call_trades': len(calls),
                'put_trades': len(puts),
                'call_volume': call_volume,
                'put_volume': put_volume,
                'pc_ratio': put_volume / max(call_volume, 1)
            })

    except Exception as e:
        print(f"\n   ❌ Error parsing {date_str}: {e}")

# Create DataFrame
df = pd.DataFrame(all_data)
df = df.sort_values('date')

print(f"\n✅ Parsed {len(df)} trading days")
print(f"\n📊 Summary Statistics:")
print(f"   Avg P/C Ratio: {df['pc_ratio'].mean():.2f}")
print(f"   Min P/C Ratio: {df['pc_ratio'].min():.2f} on {df.loc[df['pc_ratio'].idxmin(), 'date'].strftime('%Y-%m-%d')}")
print(f"   Max P/C Ratio: {df['pc_ratio'].max():.2f} on {df.loc[df['pc_ratio'].idxmax(), 'date'].strftime('%Y-%m-%d')}")

# Calculate rolling averages
df['pc_ratio_7d'] = df['pc_ratio'].rolling(7, min_periods=1).mean()
df['pc_ratio_std'] = df['pc_ratio'].rolling(7, min_periods=1).std()

# Detect anomalies
mean_pc = df['pc_ratio'].mean()
std_pc = df['pc_ratio'].std()
df['z_score'] = (df['pc_ratio'] - mean_pc) / std_pc
df['anomaly'] = df['z_score'].abs() > 2

print(f"\n🚨 Anomalous Days (z-score > 2): {df['anomaly'].sum()}")
if df['anomaly'].any():
    for idx, row in df[df['anomaly']].iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: P/C = {row['pc_ratio']:.2f} (z={row['z_score']:.2f})")

# Compare recent vs historical
recent_7d = df.tail(7)
prior_14d = df.iloc[-21:-7] if len(df) >= 21 else df.iloc[:-7]

recent_pc = recent_7d['pc_ratio'].mean()
prior_pc = prior_14d['pc_ratio'].mean() if len(prior_14d) > 0 else mean_pc
pc_change_pct = (recent_pc / prior_pc - 1) * 100 if prior_pc > 0 else 0

print(f"\n📈 Behavioral Change Analysis:")
print(f"   Last 7 days P/C:     {recent_pc:.2f}")
print(f"   Prior 14 days P/C:   {prior_pc:.2f}")
print(f"   Change: {pc_change_pct:+.1f}%")

if abs(pc_change_pct) > 10:
    if pc_change_pct > 0:
        print(f"   🔴 SIGNIFICANT INCREASE in put buying")
    else:
        print(f"   🟢 DECREASE in put buying (more calls)")
else:
    print(f"   ✅ Relatively stable")

# Save summary
summary_file = flatfiles_dir / 'complete_spy_summary.csv'
df.to_csv(summary_file, index=False)
print(f"\n✅ Saved summary to: {summary_file}")

# Create visualizations
print(f"\n🎨 Creating visualizations...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14))
fig.suptitle('SPY OPTIONS - 30-DAY BEHAVIORAL ANALYSIS', fontsize=16, fontweight='bold')

dates = df['date']

# 1. Put/Call Ratio Over Time
ax1 = axes[0]
ax1.plot(dates, df['pc_ratio'], 'b-', linewidth=2, marker='o', markersize=4, label='Daily P/C Ratio')
ax1.plot(dates, df['pc_ratio_7d'], 'r-', linewidth=3, label='7-Day Average', alpha=0.7)
ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Neutral (1.0)')
ax1.axhline(y=mean_pc, color='green', linestyle='--', alpha=0.5, label=f'Mean ({mean_pc:.2f})')

# Highlight anomalies
if df['anomaly'].any():
    anomaly_dates = df[df['anomaly']]['date']
    anomaly_values = df[df['anomaly']]['pc_ratio']
    ax1.scatter(anomaly_dates, anomaly_values, color='red', s=200, zorder=5,
               marker='*', edgecolors='black', linewidths=2, label='Anomaly')

# Shade recent period
ax1.axvspan(recent_7d['date'].iloc[0], recent_7d['date'].iloc[-1], alpha=0.1, color='orange', label='Last 7 Days')

ax1.set_title('Put/Call Volume Ratio - 30-Day Trend', fontsize=12, fontweight='bold')
ax1.set_ylabel('P/C Ratio', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Volume Breakdown
ax2 = axes[1]
width = 0.8
ax2.bar(dates, df['call_volume']/1e6, width, label='Call Volume', color='green', alpha=0.7)
ax2.bar(dates, df['put_volume']/1e6, width, bottom=df['call_volume']/1e6,
       label='Put Volume', color='red', alpha=0.7)

ax2.set_title('Daily Options Volume - Calls vs Puts', fontsize=12, fontweight='bold')
ax2.set_ylabel('Volume (Millions)', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Z-Score (Anomaly Detection)
ax3 = axes[2]
colors = ['red' if z > 2 else 'orange' if z > 1 else 'green' for z in df['z_score'].abs()]
ax3.bar(dates, df['z_score'], color=colors, alpha=0.7)
ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Anomaly Threshold (±2σ)')
ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax3.set_title('Anomaly Score (Z-Score)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Standard Deviations from Mean', fontsize=11)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Rolling Statistics
ax4 = axes[3]
ax4.plot(dates, df['pc_ratio_7d'], 'b-', linewidth=2.5, label='7-Day Average')
ax4.fill_between(dates,
                 df['pc_ratio_7d'] - df['pc_ratio_std'],
                 df['pc_ratio_7d'] + df['pc_ratio_std'],
                 alpha=0.2, color='blue', label='±1 Std Dev')
ax4.axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='High Hedging (1.2)')

ax4.set_title('7-Day Rolling Average with Std Dev Band', fontsize=12, fontweight='bold')
ax4.set_ylabel('P/C Ratio', fontsize=11)
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
plt.savefig('SPY_Options_30Day_Complete_Analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: SPY_Options_30Day_Complete_Analysis.png")

# Create summary text report
print(f"\n{'='*80}")
print("FINAL ANALYSIS SUMMARY")
print(f"{'='*80}")

print(f"\n📊 30-Day Put/Call Behavior:")
print(f"   Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"   Trading Days: {len(df)}")
print(f"   Average P/C: {mean_pc:.2f}")

print(f"\n🔍 Key Findings:")
if recent_pc > 1.2:
    print(f"   🔴 CURRENT: High put buying (P/C: {recent_pc:.2f})")
    print(f"      → Traders are defensive/hedging heavily")
elif recent_pc > 1.0:
    print(f"   🟡 CURRENT: Elevated put activity (P/C: {recent_pc:.2f})")
    print(f"      → Some hedging activity present")
else:
    print(f"   🟢 CURRENT: Call dominant (P/C: {recent_pc:.2f})")
    print(f"      → Bullish positioning, limited hedging")

if abs(pc_change_pct) > 15:
    print(f"\n   ⚠️  WARNING: P/C ratio changed {pc_change_pct:+.1f}% in last 7 days")
    print(f"      → Significant behavioral shift detected")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
