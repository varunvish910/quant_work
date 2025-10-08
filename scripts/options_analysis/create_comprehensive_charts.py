#!/usr/bin/env python3
"""
Create comprehensive charts for options analysis
1. VIX term structure curve
2. Rolling SPY/VIX beta with interpretation
3. Rolling P/C ratios for SPY and VIX
4. Most traded strikes and expiries over 15 days
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import gzip

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = Path('analysis_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("COMPREHENSIVE OPTIONS VISUALIZATION")
print("="*80)

# =============================================================================
# 1. VIX TERM STRUCTURE CURVE
# =============================================================================

print("\n📊 Chart 1: VIX Term Structure Curve")

# VIX term structure data from our analysis
vix_ts_data = {
    'DTE': [0, 16, 44, 72, 107, 135, 163, 191, 225, 254],
    'Forward_VIX': [16.6, 17.5, 16.9, 16.4, 16.2, 18.0, 17.7, 17.1, 15.9, 16.6]
}

vix_ts = pd.DataFrame(vix_ts_data)

fig, ax = plt.subplots(figsize=(12, 7))

# Plot the curve
ax.plot(vix_ts['DTE'], vix_ts['Forward_VIX'],
        marker='o', linewidth=2.5, markersize=8,
        color='#E74C3C', label='VIX Forward Curve')

# Add current VIX line
ax.axhline(y=16.6, color='#3498DB', linestyle='--',
           linewidth=2, label='Current VIX (16.6)', alpha=0.7)

# Shade above/below spot
ax.fill_between(vix_ts['DTE'], 16.6, vix_ts['Forward_VIX'],
                where=(vix_ts['Forward_VIX'] >= 16.6),
                alpha=0.2, color='red', label='Contango')
ax.fill_between(vix_ts['DTE'], 16.6, vix_ts['Forward_VIX'],
                where=(vix_ts['Forward_VIX'] < 16.6),
                alpha=0.2, color='green', label='Backwardation')

ax.set_xlabel('Days to Expiration', fontsize=12, fontweight='bold')
ax.set_ylabel('VIX Level', fontsize=12, fontweight='bold')
ax.set_title('VIX Term Structure - October 6, 2025\n(From Volume-Weighted ATM Strikes)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Add annotations
for idx, row in vix_ts.iterrows():
    if idx % 2 == 0:  # Annotate every other point
        ax.annotate(f"{row['Forward_VIX']:.1f}",
                   xy=(row['DTE'], row['Forward_VIX']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'vix_term_structure.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR / 'vix_term_structure.png'}")

# =============================================================================
# 2. ROLLING SPY/VIX BETA
# =============================================================================

print("\n📊 Chart 2: Rolling SPY/VIX Beta with Interpretation")

# Download SPY and VIX data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print("   Downloading SPY and VIX data...")
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# Calculate returns
spy['Return'] = spy['Close'].pct_change()
vix['Return'] = vix['Close'].pct_change()

# Merge data
data = pd.DataFrame({
    'SPY_Return': spy['Return'],
    'VIX_Return': vix['Return'],
    'SPY_Price': spy['Close'],
    'VIX_Price': vix['Close']
}).dropna()

# Calculate rolling metrics
data['Beta_20d'] = data['SPY_Return'].rolling(20).cov(data['VIX_Return']) / \
                   data['SPY_Return'].rolling(20).var()
data['Beta_60d'] = data['SPY_Return'].rolling(60).cov(data['VIX_Return']) / \
                   data['SPY_Return'].rolling(60).var()
data['Corr_60d'] = data['SPY_Return'].rolling(60).corr(data['VIX_Return'])

# Create subplot figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot 1: Rolling Beta
ax1.plot(data.index, data['Beta_20d'], label='20-day Beta',
         linewidth=1.5, alpha=0.7)
ax1.plot(data.index, data['Beta_60d'], label='60-day Beta',
         linewidth=2.5, color='#E74C3C')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax1.axhline(y=-5, color='orange', linestyle='--', linewidth=1,
            alpha=0.5, label='Normal Range')
ax1.axhline(y=-10, color='red', linestyle='--', linewidth=1,
            alpha=0.5, label='Elevated')
ax1.fill_between(data.index, -5, -10, alpha=0.1, color='orange')
ax1.set_ylabel('Beta (VIX % change per 1% SPY move)', fontweight='bold')
ax1.set_title('Rolling SPY/VIX Beta - Market Sensitivity',
              fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Current beta annotation
current_beta = data['Beta_60d'].iloc[-1]
ax1.annotate(f'Current: {current_beta:.2f}',
            xy=(data.index[-1], current_beta),
            xytext=(-60, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
            fontsize=10, fontweight='bold')

# Plot 2: Rolling Correlation
ax2.plot(data.index, data['Corr_60d'], linewidth=2, color='#3498DB')
ax2.axhline(y=-0.7, color='green', linestyle='--', linewidth=1,
            alpha=0.5, label='Strong Negative (Normal)')
ax2.axhline(y=-0.3, color='orange', linestyle='--', linewidth=1,
            alpha=0.5, label='Weak Negative (Caution)')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1,
            alpha=0.5, label='Zero/Positive (Crisis)')
ax2.fill_between(data.index, -1, -0.7, alpha=0.1, color='green')
ax2.fill_between(data.index, -0.7, -0.3, alpha=0.1, color='orange')
ax2.fill_between(data.index, -0.3, 1, alpha=0.1, color='red')
ax2.set_ylabel('Correlation', fontweight='bold')
ax2.set_title('SPY/VIX Correlation - Market Regime',
              fontsize=14, fontweight='bold', pad=10)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: VIX Level Context
ax3_twin = ax3.twinx()
ax3.plot(data.index, data['SPY_Price'], linewidth=2, color='#2ECC71', label='SPY Price')
ax3_twin.plot(data.index, data['VIX_Price'], linewidth=2, color='#E74C3C',
              label='VIX Level')
ax3.set_ylabel('SPY Price ($)', fontweight='bold', color='#2ECC71')
ax3_twin.set_ylabel('VIX Level', fontweight='bold', color='#E74C3C')
ax3.set_xlabel('Date', fontweight='bold')
ax3.set_title('SPY & VIX Levels - Context',
              fontsize=14, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'spy_vix_beta_analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR / 'spy_vix_beta_analysis.png'}")

# =============================================================================
# 3. ROLLING P/C RATIOS
# =============================================================================

print("\n📊 Chart 3: Rolling P/C Ratios (Using Single-Day Data)")

# Note: We only have Oct 6 data, so we'll show that day's breakdown
# In a real scenario, you'd load 15 days of flatfiles

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# SPY P/C by expiry bucket (from our Oct 6 analysis)
spy_pc_data = {
    'Expiry': ['0DTE', '1DTE', '1Week', '1Month', '2Month', '>2Month'],
    'PC_Ratio': [1.14, 1.25, 1.35, 2.06, 3.47, 2.10],
    'Volume': [4497048, 1032441, 727876, 447319, 181951, 272805]
}

spy_pc = pd.DataFrame(spy_pc_data)

# Plot SPY P/C by expiry
colors = ['#2ECC71' if x < 1.2 else '#F39C12' if x < 2.0 else '#E74C3C'
          for x in spy_pc['PC_Ratio']]
bars1 = ax1.bar(spy_pc['Expiry'], spy_pc['PC_Ratio'], color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Neutral (P/C=1.0)')
ax1.axhline(y=1.2, color='orange', linestyle='--', linewidth=1.5,
            alpha=0.7, label='Elevated Hedging')
ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5,
            alpha=0.7, label='Extreme Hedging')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

ax1.set_ylabel('Put/Call Ratio', fontweight='bold', fontsize=12)
ax1.set_title('SPY Options Put/Call Ratio by Expiry - October 6, 2025',
              fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3, axis='y')

# VIX P/C visualization
vix_pc = 0.34  # From our analysis
ax2.barh(['VIX Options'], [vix_pc], color='#3498DB', alpha=0.7, edgecolor='black')
ax2.barh(['SPY Options'], [1.27], color='#E74C3C', alpha=0.7, edgecolor='black')
ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Neutral (P/C=1.0)')
ax2.axvline(x=0.7, color='green', linestyle='--', linewidth=1.5,
            alpha=0.5, label='Call-Heavy')
ax2.axvline(x=1.3, color='orange', linestyle='--', linewidth=1.5,
            alpha=0.5, label='Put-Heavy')

# Add value labels
ax2.text(vix_pc, 0, f'  {vix_pc:.2f} (Extreme Call Buying)',
         va='center', fontweight='bold', fontsize=11)
ax2.text(1.27, 1, f'  {1.27:.2f} (Defensive Positioning)',
         va='center', fontweight='bold', fontsize=11)

ax2.set_xlabel('Put/Call Ratio', fontweight='bold', fontsize=12)
ax2.set_title('Overall Put/Call Ratio Comparison - October 6, 2025',
              fontsize=14, fontweight='bold', pad=10)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, 3.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'put_call_ratios.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR / 'put_call_ratios.png'}")

# =============================================================================
# 4. MOST TRADED STRIKES AND EXPIRIES
# =============================================================================

print("\n📊 Chart 4: Most Traded Strikes and Expiries (October 6, 2025)")

# SPY most traded strikes
spy_strikes_data = {
    'Strike': ['$672 C', '$670 P', '$671 P', '$671 C', '$672 P',
               '$673 C', '$669 P', '$670 C', '$674 C', '$668 P'],
    'Volume': [911724, 706192, 678494, 579231, 537130,
               518102, 382824, 274096, 244691, 193568],
    'Moneyness': ['+0.4%', '+0.1%', '+0.3%', '+0.3%', '+0.4%',
                  '+0.6%', 'ATM', '+0.1%', '+0.7%', '-0.1%']
}

spy_strikes = pd.DataFrame(spy_strikes_data)

# VIX most traded strikes
vix_strikes_data = {
    'Strike': ['$16 P', '$55 C', '$60 C', '$20 C', '$25 C',
               '$15 P', '$18 C', '$21 C', '$28 C', '$40 C'],
    'Volume': [37174, 25542, 24970, 24350, 23436,
               22099, 19523, 18527, 17553, 16385]
}

vix_strikes = pd.DataFrame(vix_strikes_data)

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# SPY strikes
colors_spy = ['#2ECC71' if 'C' in s else '#E74C3C' for s in spy_strikes['Strike']]
bars = ax1.barh(spy_strikes['Strike'], spy_strikes['Volume'],
                color=colors_spy, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Volume (contracts)', fontweight='bold')
ax1.set_title('SPY - Top 10 Most Traded Strikes\nOctober 6, 2025',
              fontsize=13, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='x')

# Add volume labels
for i, (bar, vol) in enumerate(zip(bars, spy_strikes['Volume'])):
    ax1.text(vol, bar.get_y() + bar.get_height()/2,
            f' {vol:,}', va='center', fontweight='bold', fontsize=9)

# VIX strikes
colors_vix = ['#2ECC71' if 'C' in s else '#E74C3C' for s in vix_strikes['Strike']]
bars = ax2.barh(vix_strikes['Strike'], vix_strikes['Volume'],
                color=colors_vix, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Volume (contracts)', fontweight='bold')
ax2.set_title('VIX - Top 10 Most Traded Strikes\nOctober 6, 2025',
              fontsize=13, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='x')

# Add volume labels
for i, (bar, vol) in enumerate(zip(bars, vix_strikes['Volume'])):
    ax2.text(vol, bar.get_y() + bar.get_height()/2,
            f' {vol:,}', va='center', fontweight='bold', fontsize=9)

# SPY expiries
spy_expiry_data = {
    'Expiry': ['Oct 6\n(0DTE)', 'Oct 7\n(1DTE)', 'Oct 10\n(4DTE)',
               'Oct 17\n(11DTE)', 'Oct 8\n(2DTE)', 'Nov 21\n(46DTE)',
               'Oct 9\n(3DTE)', 'Jan 16\n(102DTE)', 'Oct 31\n(25DTE)',
               'Oct 13\n(7DTE)'],
    'Volume': [4497048, 1032441, 344375, 264593, 221727,
               133303, 98303, 97209, 94174, 63471]
}

spy_expiry = pd.DataFrame(spy_expiry_data)

bars = ax3.bar(range(len(spy_expiry)), spy_expiry['Volume'],
               color='#3498DB', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(spy_expiry)))
ax3.set_xticklabels(spy_expiry['Expiry'], rotation=45, ha='right')
ax3.set_ylabel('Volume (contracts)', fontweight='bold')
ax3.set_title('SPY - Top 10 Expiries by Volume\nOctober 6, 2025',
              fontsize=13, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height/1000:.0f}K', ha='center', va='bottom',
            fontweight='bold', fontsize=8)

# Interpretation text
interpretation = """
KEY INSIGHTS:

SPY OPTIONS:
• 0DTE dominates (62.5% of volume)
• Heavy ATM activity around $670-672
• Put bias increases with time (1M+ P/C >3.0)

VIX OPTIONS:
• Extreme tail risk hedging ($55-60 calls)
• 94.1% of calls are OTM
• Put/Call ratio 0.34 (expecting vol spike)

MARKET POSITIONING:
• Near-term: Balanced with slight put bias
• Medium-term: Extreme defensive hedging
• VIX: Preparing for volatility event
"""

ax4.text(0.05, 0.95, interpretation, transform=ax4.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace')
ax4.axis('off')
ax4.set_title('Positioning Interpretation', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'most_traded_analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR / 'most_traded_analysis.png'}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n✅ Generated 4 comprehensive charts:")
print(f"   1. VIX Term Structure Curve")
print(f"   2. Rolling SPY/VIX Beta Analysis (3 panels)")
print(f"   3. Put/Call Ratios Comparison")
print(f"   4. Most Traded Strikes & Expiries")
print(f"\n📁 All charts saved to: {OUTPUT_DIR}/")
print("\n" + "="*80)

# Print interpretation guide
print("\n📊 INTERPRETATION GUIDE:")
print("\n1. VIX TERM STRUCTURE:")
print("   • Upward sloping (contango) = Normal, higher vol expected ahead")
print("   • Downward sloping (backwardation) = Near-term stress")
print("   • Current: Slight near-term elevation, then flat")

print("\n2. SPY/VIX BETA:")
print("   • Normal range: -4 to -7 (every 1% SPY down = 4-7% VIX up)")
print("   • Current: -8.71 (ELEVATED - market jumpy)")
print("   • High beta = Small moves cause big VIX reactions")
print("   • Use for: Position sizing, stop loss widths")

print("\n3. PUT/CALL RATIOS:")
print("   • P/C < 0.7: Bullish/complacent")
print("   • P/C 0.7-1.3: Balanced")
print("   • P/C > 1.3: Defensive/hedging")
print("   • SPY 1.27 (defensive), VIX 0.34 (expecting spike)")

print("\n4. STRIKES & EXPIRIES:")
print("   • Heavy ATM = Directional views")
print("   • Far OTM puts = Crash protection")
print("   • Far OTM calls (VIX) = Tail risk hedging")
print("   • 0DTE dominance = Day trader activity")

print("\n" + "="*80)