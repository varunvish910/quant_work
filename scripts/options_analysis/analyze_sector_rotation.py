#!/usr/bin/env python3
"""
Analyze sector rotation patterns
Compare sector ETF performance to identify risk-on vs risk-off flows
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path('analysis_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("SECTOR ROTATION ANALYSIS")
print("="*80)

# Sector ETFs
SECTORS = {
    # Cyclical / Risk-On
    'XLY': 'Consumer Discretionary',
    'XLF': 'Financials',
    'XLI': 'Industrials',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLB': 'Materials',

    # Defensive / Risk-Off
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLV': 'Healthcare',

    # Benchmark
    'SPY': 'S&P 500'
}

# Download data
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # Last 3 months

print(f"\n📊 Downloading sector data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

sector_data = {}
for ticker, name in SECTORS.items():
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        sector_data[ticker] = data
        print(f"   ✅ {ticker}: {name}")
    except Exception as e:
        print(f"   ❌ Failed to download {ticker}: {e}")

# Calculate returns
print("\n📈 Calculating sector performance...")

performance = {}
for ticker, data in sector_data.items():
    if len(data) > 0:
        # Various timeframe returns
        performance[ticker] = {
            'name': SECTORS[ticker],
            '5d': (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else np.nan,
            '10d': (data['Close'].iloc[-1] / data['Close'].iloc[-10] - 1) * 100 if len(data) >= 10 else np.nan,
            '30d': (data['Close'].iloc[-1] / data['Close'].iloc[-30] - 1) * 100 if len(data) >= 30 else np.nan,
            '60d': (data['Close'].iloc[-1] / data['Close'].iloc[-60] - 1) * 100 if len(data) >= 60 else np.nan,
            'ytd': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100,
            'current_price': data['Close'].iloc[-1],
            'volatility_30d': data['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
        }

perf_df = pd.DataFrame(performance).T

# Convert numeric columns to float
for col in ['5d', '10d', '30d', '60d', 'ytd', 'current_price', 'volatility_30d']:
    perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')

# Classify sectors
risk_on = ['XLY', 'XLF', 'XLI', 'XLK', 'XLE', 'XLB']
risk_off = ['XLP', 'XLU', 'XLV']

perf_df['category'] = perf_df.index.map(lambda x: 'Risk-On' if x in risk_on else 'Risk-Off' if x in risk_off else 'Benchmark')

# =============================================================================
# VISUALIZATION 1: Sector Performance Heatmap
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for heatmap
heatmap_data = perf_df[['5d', '10d', '30d', '60d']].copy()
heatmap_data.index = perf_df['name']

# Convert to numeric (handle any NaN)
heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')

# Create heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Return (%)'},
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('Sector Performance Heatmap - Multiple Timeframes\nOctober 2025',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Timeframe', fontweight='bold', fontsize=12)
ax.set_ylabel('Sector', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sector_performance_heatmap.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {OUTPUT_DIR / 'sector_performance_heatmap.png'}")

# =============================================================================
# VISUALIZATION 2: Risk-On vs Risk-Off Performance
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 30-day performance comparison
risk_on_perf = perf_df[perf_df['category'] == 'Risk-On']['30d'].dropna()
risk_off_perf = perf_df[perf_df['category'] == 'Risk-Off']['30d'].dropna()
spy_perf = perf_df.loc['SPY', '30d']

# Bar chart
categories = ['Risk-On\nAvg', 'Risk-Off\nAvg', 'SPY\nBenchmark']
values = [risk_on_perf.mean(), risk_off_perf.mean(), spy_perf]
colors = ['#E74C3C' if v < 0 else '#2ECC71' for v in values]

bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_ylabel('30-Day Return (%)', fontweight='bold', fontsize=12)
ax1.set_title('Risk-On vs Risk-Off Performance\n30-Day Returns',
              fontsize=14, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center',
            va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=11)

# Individual sector breakdown
all_sectors = pd.concat([
    risk_on_perf.to_frame('Return').assign(Category='Risk-On'),
    risk_off_perf.to_frame('Return').assign(Category='Risk-Off')
])

# Get names for index
all_sectors.index = all_sectors.index.map(lambda x: perf_df.loc[x, 'name'])

# Grouped bar chart
risk_on_sorted = risk_on_perf.sort_values()
risk_off_sorted = risk_off_perf.sort_values()

y_pos = np.arange(len(risk_on_sorted) + len(risk_off_sorted))
names = list(risk_on_sorted.index.map(lambda x: perf_df.loc[x, 'name'])) + \
        list(risk_off_sorted.index.map(lambda x: perf_df.loc[x, 'name']))
values = list(risk_on_sorted.values) + list(risk_off_sorted.values)
colors = ['#3498DB']*len(risk_on_sorted) + ['#F39C12']*len(risk_off_sorted)

bars = ax2.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('30-Day Return (%)', fontweight='bold', fontsize=12)
ax2.set_title('Individual Sector Performance\n30-Day Returns',
              fontsize=14, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax2.text(val, bar.get_y() + bar.get_height()/2,
            f' {val:.2f}%', va='center', fontweight='bold', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498DB', label='Risk-On'),
                  Patch(facecolor='#F39C12', label='Risk-Off')]
ax2.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'risk_on_vs_risk_off.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR / 'risk_on_vs_risk_off.png'}")

# =============================================================================
# VISUALIZATION 3: Relative Performance Chart
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate cumulative returns relative to SPY
spy_returns = sector_data['SPY']['Close'].pct_change()

for ticker, data in sector_data.items():
    if ticker != 'SPY' and len(data) > 0:
        # Calculate returns
        returns = data['Close'].pct_change()

        # Calculate relative return (sector - SPY)
        relative_returns = returns - spy_returns.reindex(returns.index, method='ffill')

        # Cumulative relative return
        cumulative_rel = (1 + relative_returns).cumprod() - 1

        # Plot
        color = '#3498DB' if ticker in risk_on else '#F39C12'
        linestyle = '-' if ticker in risk_on else '--'
        ax.plot(cumulative_rel.index, cumulative_rel * 100,
               label=f"{SECTORS[ticker]} ({ticker})",
               linewidth=2, alpha=0.8, color=color, linestyle=linestyle)

ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5, label='SPY Benchmark')
ax.set_xlabel('Date', fontweight='bold', fontsize=12)
ax.set_ylabel('Relative Return vs SPY (%)', fontweight='bold', fontsize=12)
ax.set_title('Sector Rotation - Relative Performance vs SPY\nLast 90 Days',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sector_relative_performance.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR / 'sector_relative_performance.png'}")

# =============================================================================
# ANALYSIS & INTERPRETATION
# =============================================================================

print("\n" + "="*80)
print("SECTOR ROTATION ANALYSIS")
print("="*80)

# Current rotation status
print("\n📊 CURRENT SECTOR LEADERSHIP (30-day returns):")
top_performers = perf_df.nlargest(5, '30d')[['name', '30d', 'category']]
for idx, row in top_performers.iterrows():
    print(f"   {row['name']:25s}: {row['30d']:+6.2f}%  [{row['category']}]")

print("\n📉 CURRENT SECTOR LAGGARDS (30-day returns):")
bottom_performers = perf_df.nsmallest(5, '30d')[['name', '30d', 'category']]
for idx, row in bottom_performers.iterrows():
    print(f"   {row['name']:25s}: {row['30d']:+6.2f}%  [{row['category']}]")

# Risk-On vs Risk-Off analysis
print("\n📊 RISK-ON vs RISK-OFF COMPARISON:")
print(f"   Risk-On Average (30d):  {risk_on_perf.mean():+.2f}%")
print(f"   Risk-Off Average (30d): {risk_off_perf.mean():+.2f}%")
print(f"   SPY Benchmark (30d):    {spy_perf:+.2f}%")

rotation_signal = risk_on_perf.mean() - risk_off_perf.mean()
print(f"\n   Rotation Signal: {rotation_signal:+.2f}%")

if rotation_signal > 2:
    print("   → RISK-ON rotation (Cyclicals outperforming)")
    risk_appetite = "HIGH"
elif rotation_signal < -2:
    print("   → RISK-OFF rotation (Defensives outperforming)")
    risk_appetite = "LOW"
else:
    print("   → MIXED/NEUTRAL (No clear rotation)")
    risk_appetite = "NEUTRAL"

# Volatility analysis
print("\n📊 SECTOR VOLATILITY (30-day annualized):")
vol_sorted = perf_df.nlargest(5, 'volatility_30d')[['name', 'volatility_30d']]
for idx, row in vol_sorted.iterrows():
    print(f"   {row['name']:25s}: {row['volatility_30d']:.2f}%")

# Interpretation
print("\n" + "="*80)
print("INTERPRETATION & IMPLICATIONS")
print("="*80)

print(f"\n🎯 MARKET RISK APPETITE: {risk_appetite}")

if risk_appetite == "LOW":
    print("\n   Characteristics:")
    print("   • Defensive sectors (Utilities, Staples, Healthcare) outperforming")
    print("   • Flight to quality/safety")
    print("   • Increased market uncertainty")
    print("\n   Implications:")
    print("   • Supports bearish/cautious outlook")
    print("   • Aligns with elevated model risk signals")
    print("   • Consider overweight defensives")

elif risk_appetite == "HIGH":
    print("\n   Characteristics:")
    print("   • Cyclical sectors (Tech, Discretionary, Financials) outperforming")
    print("   • Risk-seeking behavior")
    print("   • Market confidence high")
    print("\n   Implications:")
    print("   • Contradicts bearish signals")
    print("   • Possible complacency")
    print("   • Watch for rotation breakdown")

else:
    print("\n   Characteristics:")
    print("   • No clear sector leadership")
    print("   • Mixed signals across sectors")
    print("   • Possible transition phase")
    print("\n   Implications:")
    print("   • Market direction uncertain")
    print("   • Wait for clearer rotation signal")
    print("   • Maintain balanced positioning")

# Check for divergences
print("\n⚠️ DIVERGENCE CHECK:")
model_bearish = True  # From our 80% risk signal
sectors_bearish = risk_appetite == "LOW"

if model_bearish and not sectors_bearish:
    print("   🔴 DIVERGENCE DETECTED")
    print("   • Models showing high risk (80%)")
    print("   • But sectors showing risk-on behavior")
    print("   • Possible false signal OR complacency before drop")
elif model_bearish and sectors_bearish:
    print("   ✅ ALIGNMENT")
    print("   • Models and sectors both bearish")
    print("   • Confirms defensive posture")
else:
    print("   ⚠️ Mixed signals - proceed with caution")

print("\n" + "="*80)
print("✅ Sector rotation analysis complete")
print(f"📁 Charts saved to: {OUTPUT_DIR}/")
print("="*80)