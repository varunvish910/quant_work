#!/usr/bin/env python3
"""
Analyze 30-day trends in options flows and market data
Focus on:
1. P/C ratio trends (SPY and VIX)
2. SPY/VIX beta evolution
3. Volume patterns over time
4. Positioning changes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import gzip

OUTPUT_DIR = Path('analysis_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("30-DAY TREND ANALYSIS")
print("="*80)

# =============================================================================
# 1. SPY/VIX DATA FOR LAST 30 DAYS
# =============================================================================

print("\n📊 Downloading 30-day SPY and VIX data...")

end_date = datetime(2025, 10, 6)
start_date = end_date - timedelta(days=45)  # Extra buffer for calculations

spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# Calculate returns
spy['Return'] = spy['Close'].pct_change()
vix['Return'] = vix['Close'].pct_change()
vix['VIX_Change'] = vix['Close'].diff()

# Merge data
data = pd.DataFrame({
    'SPY_Return': spy['Return'],
    'VIX_Return': vix['Return'],
    'SPY_Price': spy['Close'],
    'VIX_Price': vix['Close'],
    'VIX_Change': vix['VIX_Change']
}).dropna()

print(f"   ✅ Loaded {len(data)} days of data")

# Calculate rolling metrics
data['Beta_20d'] = data['SPY_Return'].rolling(20).cov(data['VIX_Return']) / \
                   data['SPY_Return'].rolling(20).var()
data['Corr_20d'] = data['SPY_Return'].rolling(20).corr(data['VIX_Return'])

# Calculate rolling volatility
data['SPY_Vol_20d'] = data['SPY_Return'].rolling(20).std() * np.sqrt(252) * 100
data['VIX_Vol_20d'] = data['VIX_Return'].rolling(20).std() * 100

# Get last 30 days
last_30 = data.tail(30).copy()

# =============================================================================
# 2. OPTIONS DATA (We only have Oct 6, but show historical context)
# =============================================================================

print("\n📊 Analyzing October 6 options data in context...")

# Note: We only have Oct 6 flatfile data
# For trend analysis, we'd need multiple days
# For now, we'll show what we CAN calculate from market data

# =============================================================================
# VISUALIZATION 1: SPY/VIX Beta Trend
# =============================================================================

print("\n📊 Creating SPY/VIX beta trend charts...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Beta Evolution
ax1.plot(last_30.index, last_30['Beta_20d'], linewidth=2.5, color='#E74C3C', marker='o')
ax1.axhline(y=-5, color='green', linestyle='--', alpha=0.5, label='Normal Low (-5)')
ax1.axhline(y=-7, color='orange', linestyle='--', alpha=0.5, label='Normal High (-7)')
ax1.axhline(y=-10, color='red', linestyle='--', alpha=0.5, label='Elevated (-10)')
ax1.fill_between(last_30.index, -5, -7, alpha=0.1, color='green')
ax1.fill_between(last_30.index, -7, -10, alpha=0.1, color='orange')
ax1.set_ylabel('20-Day Beta', fontweight='bold', fontsize=11)
ax1.set_title('SPY/VIX Beta Trend - Last 30 Days\n(VIX % change per 1% SPY move)',
              fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Highlight current level
current_beta = last_30['Beta_20d'].iloc[-1]
ax1.scatter(last_30.index[-1], current_beta, s=200, color='red', zorder=5, edgecolor='black', linewidth=2)
ax1.annotate(f'Current:\n{current_beta:.2f}',
            xy=(last_30.index[-1], current_beta),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
            fontsize=10, fontweight='bold')

# Plot 2: Correlation Trend
ax2.plot(last_30.index, last_30['Corr_20d'], linewidth=2.5, color='#3498DB', marker='o')
ax2.axhline(y=-0.7, color='green', linestyle='--', alpha=0.5, label='Strong Negative')
ax2.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate')
ax2.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='Weak (Warning)')
ax2.fill_between(last_30.index, -1, -0.7, alpha=0.1, color='green')
ax2.fill_between(last_30.index, -0.7, -0.3, alpha=0.1, color='orange')
ax2.set_ylabel('20-Day Correlation', fontweight='bold', fontsize=11)
ax2.set_title('SPY/VIX Correlation Trend - Last 30 Days\nMarket Regime Indicator',
              fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Highlight current level
current_corr = last_30['Corr_20d'].iloc[-1]
ax2.scatter(last_30.index[-1], current_corr, s=200, color='red', zorder=5, edgecolor='black', linewidth=2)

# Plot 3: VIX Level Trend
ax3.plot(last_30.index, last_30['VIX_Price'], linewidth=2.5, color='#E74C3C', marker='o')
ax3.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Low Vol (<15)')
ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Elevated (>20)')
ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='High (>30)')
ax3.fill_between(last_30.index, 0, 15, alpha=0.1, color='green')
ax3.fill_between(last_30.index, 15, 20, alpha=0.1, color='orange')
ax3.set_ylabel('VIX Level', fontweight='bold', fontsize=11)
ax3.set_title('VIX Level Trend - Last 30 Days',
              fontsize=13, fontweight='bold', pad=10)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Add trend annotation
vix_start = last_30['VIX_Price'].iloc[0]
vix_end = last_30['VIX_Price'].iloc[-1]
vix_change = ((vix_end / vix_start) - 1) * 100
trend_color = 'red' if vix_change > 0 else 'green'
ax3.annotate(f'30d Change: {vix_change:+.1f}%',
            xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor=trend_color, alpha=0.3),
            fontsize=10, fontweight='bold', va='top')

# Plot 4: SPY Price with VIX overlay
ax4_twin = ax4.twinx()
ax4.plot(last_30.index, last_30['SPY_Price'], linewidth=2.5, color='#2ECC71',
         marker='o', label='SPY Price')
ax4_twin.plot(last_30.index, last_30['VIX_Price'], linewidth=2.5, color='#E74C3C',
              marker='s', label='VIX Level', alpha=0.7)

ax4.set_ylabel('SPY Price ($)', fontweight='bold', fontsize=11, color='#2ECC71')
ax4_twin.set_ylabel('VIX Level', fontweight='bold', fontsize=11, color='#E74C3C')
ax4.set_title('SPY vs VIX - Last 30 Days\nInverse Relationship Check',
              fontsize=13, fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'beta_trends_30day.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR / 'beta_trends_30day.png'}")

# =============================================================================
# VISUALIZATION 2: Market Volatility and Sensitivity Trends
# =============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# SPY Realized Volatility
ax1.plot(last_30.index, last_30['SPY_Vol_20d'], linewidth=2.5, color='#3498DB', marker='o')
ax1.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Low Vol')
ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Normal Vol')
ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='High Vol')
ax1.fill_between(last_30.index, 0, 15, alpha=0.1, color='green')
ax1.fill_between(last_30.index, 15, 20, alpha=0.1, color='orange')
ax1.set_ylabel('Realized Volatility (%)', fontweight='bold', fontsize=11)
ax1.set_title('SPY Realized Volatility Trend - Last 30 Days\n20-Day Annualized',
              fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Current level
current_vol = last_30['SPY_Vol_20d'].iloc[-1]
ax1.scatter(last_30.index[-1], current_vol, s=200, color='red', zorder=5, edgecolor='black', linewidth=2)
ax1.annotate(f'{current_vol:.1f}%',
            xy=(last_30.index[-1], current_vol),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
            fontsize=10, fontweight='bold')

# Beta magnitude (absolute value)
ax2.plot(last_30.index, abs(last_30['Beta_20d']), linewidth=2.5, color='#E74C3C', marker='o')
ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Normal Sensitivity')
ax2.axhline(y=8, color='orange', linestyle='--', alpha=0.5, label='Elevated')
ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='High Sensitivity')
ax2.fill_between(last_30.index, 0, 5, alpha=0.1, color='green')
ax2.fill_between(last_30.index, 5, 8, alpha=0.1, color='orange')
ax2.set_ylabel('Beta Magnitude (Absolute)', fontweight='bold', fontsize=11)
ax2.set_xlabel('Date', fontweight='bold', fontsize=11)
ax2.set_title('Market Sensitivity to VIX - Last 30 Days\nHigher = More Reactive Market',
              fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Current level
current_beta_abs = abs(last_30['Beta_20d'].iloc[-1])
ax2.scatter(last_30.index[-1], current_beta_abs, s=200, color='red', zorder=5, edgecolor='black', linewidth=2)
ax2.annotate(f'{current_beta_abs:.1f}',
            xy=(last_30.index[-1], current_beta_abs),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'volatility_sensitivity_trends.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR / 'volatility_sensitivity_trends.png'}")

# =============================================================================
# ANALYSIS: What is the beta telling us?
# =============================================================================

print("\n" + "="*80)
print("SPY/VIX BETA INTERPRETATION")
print("="*80)

# Calculate trends
beta_trend = last_30['Beta_20d'].iloc[-1] - last_30['Beta_20d'].iloc[0]
corr_trend = last_30['Corr_20d'].iloc[-1] - last_30['Corr_20d'].iloc[0]

print(f"\n📊 30-DAY BETA EVOLUTION:")
print(f"   Starting Beta (30 days ago): {last_30['Beta_20d'].iloc[0]:.2f}")
print(f"   Current Beta: {last_30['Beta_20d'].iloc[-1]:.2f}")
print(f"   Change: {beta_trend:+.2f}")

if abs(beta_trend) > 2:
    if beta_trend < 0:
        print(f"   → Beta becoming MORE NEGATIVE (market more sensitive)")
        print(f"      Small SPY moves causing larger VIX reactions")
    else:
        print(f"   → Beta becoming LESS NEGATIVE (sensitivity decreasing)")
        print(f"      Market calming down")
else:
    print(f"   → Beta relatively stable")

print(f"\n📊 CORRELATION EVOLUTION:")
print(f"   Starting Correlation: {last_30['Corr_20d'].iloc[0]:.3f}")
print(f"   Current Correlation: {last_30['Corr_20d'].iloc[-1]:.3f}")
print(f"   Change: {corr_trend:+.3f}")

if abs(corr_trend) > 0.1:
    if corr_trend > 0:
        print(f"   ⚠️ Correlation WEAKENING (becoming less negative)")
        print(f"      WARNING: Potential regime change")
        print(f"      VIX may not hedge as effectively")
    else:
        print(f"   ✅ Correlation STRENGTHENING (becoming more negative)")
        print(f"      Normal inverse relationship reinforcing")
else:
    print(f"   → Correlation stable")

print(f"\n📊 WHAT THE CURRENT BETA ({last_30['Beta_20d'].iloc[-1]:.2f}) MEANS:")

beta_magnitude = abs(last_30['Beta_20d'].iloc[-1])

if beta_magnitude > 10:
    print(f"\n   🔴 EXTREME SENSITIVITY")
    print(f"   • 1% SPY decline → ~{beta_magnitude:.0f}% VIX increase")
    print(f"   • Market is VERY jumpy and reactive")
    print(f"   • Small price moves trigger large volatility spikes")
    print(f"\n   Trading Implications:")
    print(f"   → Use WIDE stop losses (expect whipsaws)")
    print(f"   → Reduce position size by 30-50%")
    print(f"   → Mean reversion trades favored")
    print(f"   → Avoid momentum chasing")

elif beta_magnitude > 7:
    print(f"\n   🟡 ELEVATED SENSITIVITY")
    print(f"   • 1% SPY decline → ~{beta_magnitude:.0f}% VIX increase")
    print(f"   • Market more reactive than normal")
    print(f"   • Volatility amplification present")
    print(f"\n   Trading Implications:")
    print(f"   → Widen stop losses by 20-30%")
    print(f"   → Reduce size modestly (10-20%)")
    print(f"   → VIX hedges effective")
    print(f"   → Be cautious with leverage")

else:
    print(f"\n   ✅ NORMAL SENSITIVITY")
    print(f"   • 1% SPY decline → ~{beta_magnitude:.0f}% VIX increase")
    print(f"   • Typical market behavior")
    print(f"\n   Trading Implications:")
    print(f"   → Standard risk management applies")
    print(f"   → Normal position sizing")

# Trend analysis
print(f"\n📊 TREND ANALYSIS:")

# Beta trend
if beta_trend < -1:
    print(f"\n   🔴 INCREASING SENSITIVITY TREND")
    print(f"   • Beta becoming more negative over 30 days")
    print(f"   • Market anxiety building")
    print(f"   • Confirms elevated risk environment")

elif beta_trend > 1:
    print(f"\n   🟢 DECREASING SENSITIVITY TREND")
    print(f"   • Beta becoming less negative over 30 days")
    print(f"   • Market calming down")
    print(f"   • Risk may be overstated")

else:
    print(f"\n   ⚠️ STABLE BUT ELEVATED")
    print(f"   • Beta not changing much")
    print(f"   • But remains at elevated levels")
    print(f"   • Market stuck in high-sensitivity regime")

# VIX level context
print(f"\n📊 VIX LEVEL CONTEXT:")
print(f"   Current VIX: {last_30['VIX_Price'].iloc[-1]:.1f}")
print(f"   30-day change: {vix_change:+.1f}%")
print(f"   30-day range: {last_30['VIX_Price'].min():.1f} - {last_30['VIX_Price'].max():.1f}")

if last_30['VIX_Price'].iloc[-1] > 20:
    print(f"   → Elevated volatility expectations")
elif last_30['VIX_Price'].iloc[-1] < 15:
    print(f"   → Complacent market (low vol)")
else:
    print(f"   → Normal volatility range")

print("\n" + "="*80)
print("SUMMARY & ACTIONABLE INSIGHTS")
print("="*80)

print(f"\n🎯 KEY TAKEAWAYS:")

print(f"\n1. MARKET SENSITIVITY:")
print(f"   Current beta: {last_30['Beta_20d'].iloc[-1]:.2f}")
if beta_magnitude > 7:
    print(f"   ⚠️ Market is JUMPY - expect amplified reactions")
    print(f"   Action: Widen stops, reduce size")
else:
    print(f"   ✅ Normal market dynamics")

print(f"\n2. RELATIONSHIP INTEGRITY:")
print(f"   Correlation: {last_30['Corr_20d'].iloc[-1]:.3f}")
if last_30['Corr_20d'].iloc[-1] > -0.5:
    print(f"   🔴 WARNING: Correlation breakdown possible")
    print(f"   Action: VIX hedges may not work as expected")
else:
    print(f"   ✅ Normal inverse relationship intact")
    print(f"   Action: VIX products effective for hedging")

print(f"\n3. TREND DIRECTION:")
if beta_trend < -1:
    print(f"   📈 Sensitivity INCREASING (beta more negative)")
    print(f"   Supports bearish/cautious outlook")
elif beta_trend > 1:
    print(f"   📉 Sensitivity DECREASING")
    print(f"   May contradict bearish signals")
else:
    print(f"   → Stable (but watch current levels)")

print(f"\n4. VOLATILITY REGIME:")
realized_vol = last_30['SPY_Vol_20d'].iloc[-1]
print(f"   Realized Vol: {realized_vol:.1f}%")
print(f"   VIX: {last_30['VIX_Price'].iloc[-1]:.1f}")
vix_premium = last_30['VIX_Price'].iloc[-1] - realized_vol
print(f"   VIX Premium: {vix_premium:+.1f}%")
if vix_premium > 5:
    print(f"   → Market expecting volatility increase")
elif vix_premium < -5:
    print(f"   → Market expects calmer conditions")
else:
    print(f"   → Fair pricing")

print("\n" + "="*80)
print("✅ 30-day trend analysis complete")
print(f"📁 Charts saved to: {OUTPUT_DIR}/")
print("="*80)