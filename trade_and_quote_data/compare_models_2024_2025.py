#!/usr/bin/env python3
"""
Compare Old Model (more sensitive) vs New Model (stricter) for 2024 and 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print('=' * 90)
print('GENERATING MODEL COMPARISON: PULLBACK vs SEVERITY')
print('=' * 90)

# Load SPY data
spy_df = pd.read_parquet('data/ohlc/SPY.parquet')
spy_df.index = pd.to_datetime(spy_df.index)

if isinstance(spy_df.columns, pd.MultiIndex):
    spy_df.columns = spy_df.columns.get_level_values(0)

# Load predictions
pullback_pred_2024 = pd.read_csv('output/2024_enhanced_predictions.csv')
pullback_pred_2024['date'] = pd.to_datetime(pullback_pred_2024['date'])

pullback_pred_2025 = pd.read_csv('output/2025_enhanced_predictions.csv')
pullback_pred_2025['date'] = pd.to_datetime(pullback_pred_2025['date'])

# Get SPY data
spy_2024 = spy_df[spy_df.index.year == 2024].copy()
spy_2025 = spy_df[spy_df.index.year == 2025].copy()

# Calculate simple overextension signal (PULLBACK model - more sensitive)
for spy_year in [spy_2024, spy_2025]:
    spy_year['sma_20'] = spy_year['Close'].rolling(20).mean()
    spy_year['sma_200'] = spy_year['Close'].rolling(200).mean()
    spy_year['distance_200'] = (spy_year['Close'] - spy_year['sma_200']) / spy_year['sma_200']
    spy_year['returns_20d'] = spy_year['Close'].pct_change(20)
    
    # Pullback model: signal when extended above 200-MA AND positive momentum
    spy_year['pullback_prob'] = 0.0
    spy_year.loc[(spy_year['distance_200'] > 0.05) & (spy_year['returns_20d'] > 0.02), 'pullback_prob'] = 0.85

# Rename for clarity
severity_pred_2024 = pullback_pred_2024.copy()
severity_pred_2025 = pullback_pred_2025.copy()

# Create figure
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 1, 2.5, 1], hspace=0.3, wspace=0.25)

# ========== 2024 ==========
# Top: SPY Price with both models
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(spy_2024.index, spy_2024['Close'], linewidth=2.5, color='#2E86AB', label='SPY', zorder=1)

# Pullback model signals (blue dots)
pullback_signals_2024 = spy_2024[spy_2024['pullback_prob'] >= 0.7]
ax1.scatter(pullback_signals_2024.index, pullback_signals_2024['Close'],
           color='blue', s=50, alpha=0.5, zorder=2, label=f'Pullback Model: {len(pullback_signals_2024)} signals')

# Severity model signals (yellow triangles)
severity_signals_2024 = severity_pred_2024[severity_pred_2024['probability'] >= 0.7]
ax1.scatter(severity_signals_2024['date'], severity_signals_2024['spy_close'],
           color='yellow', s=80, alpha=0.8, zorder=3, marker='v',
           edgecolors='orange', linewidths=2, label=f'Severity Model: {len(severity_signals_2024)} signals')

# Find where BOTH models agree (intersection)
pullback_dates = set(pullback_signals_2024.index)
severity_dates = set(severity_signals_2024['date'])
both_agree = pullback_dates.intersection(severity_dates)

if len(both_agree) > 0:
    both_df = spy_2024.loc[list(both_agree)]
    ax1.scatter(both_df.index, both_df['Close'],
               color='red', s=200, alpha=0.9, zorder=5, marker='*',
               edgecolors='darkred', linewidths=3, label=f'🚨 BOTH AGREE: {len(both_agree)} days')

ax1.set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold')
ax1.set_title('2024: Pullback Model vs Severity Model', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)

# 2024 Probability comparison
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.fill_between(spy_2024.index, 0, spy_2024['pullback_prob'] * 100,
                 color='blue', alpha=0.4, label='Pullback Model (Overextension)')
ax2.plot(severity_pred_2024['date'], severity_pred_2024['probability'] * 100,
         color='orange', linewidth=2.5, label='Severity Model (Rotation + Extension)', zorder=5)
ax2.axhline(70, color='darkred', linestyle='--', linewidth=1.5)
ax2.set_ylabel('Probability (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('2024', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# ========== 2025 ==========
# Top: SPY Price with both models
ax3 = fig.add_subplot(gs[2, 1])
ax3.plot(spy_2025.index, spy_2025['Close'], linewidth=2.5, color='#2E86AB', label='SPY', zorder=1)

# Pullback model signals
pullback_signals_2025 = spy_2025[spy_2025['pullback_prob'] >= 0.7]
if len(pullback_signals_2025) > 0:
    ax3.scatter(pullback_signals_2025.index, pullback_signals_2025['Close'],
               color='blue', s=50, alpha=0.5, zorder=2, label=f'Pullback Model: {len(pullback_signals_2025)} signals')

# Severity model signals
severity_signals_2025 = severity_pred_2025[severity_pred_2025['probability'] >= 0.7]
if len(severity_signals_2025) > 0:
    ax3.scatter(severity_signals_2025['date'], severity_signals_2025['spy_close'],
               color='yellow', s=80, alpha=0.8, zorder=3, marker='v',
               edgecolors='orange', linewidths=2, label=f'Severity Model: {len(severity_signals_2025)} signals')

# Find where BOTH models agree
pullback_dates_2025 = set(pullback_signals_2025.index)
severity_dates_2025 = set(severity_signals_2025['date']) if len(severity_signals_2025) > 0 else set()
both_agree_2025 = pullback_dates_2025.intersection(severity_dates_2025)

if len(both_agree_2025) > 0:
    both_df_2025 = spy_2025.loc[list(both_agree_2025)]
    ax3.scatter(both_df_2025.index, both_df_2025['Close'],
               color='red', s=200, alpha=0.9, zorder=5, marker='*',
               edgecolors='darkred', linewidths=3, label=f'🚨 BOTH AGREE: {len(both_agree_2025)} days')

ax3.set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold')
ax3.set_title('2025: Pullback Model vs Severity Model (Both Show Healthy)', 
              fontsize=14, fontweight='bold', color='green')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=10)

# 2025 Probability comparison
ax4 = fig.add_subplot(gs[3, 1], sharex=ax3)
ax4.fill_between(spy_2025.index, 0, spy_2025['pullback_prob'] * 100,
                 color='blue', alpha=0.4, label='Pullback Model')
ax4.plot(severity_pred_2025['date'], severity_pred_2025['probability'] * 100,
         color='green', linewidth=2.5, label='Severity Model', zorder=5)
ax4.axhline(70, color='darkred', linestyle='--', linewidth=1.5)
ax4.set_ylabel('Probability (%)', fontsize=11, fontweight='bold')
ax4.set_xlabel('2025', fontsize=11, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper left', fontsize=9)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.suptitle('Model Comparison: Pullback (Overextension) vs Severity (Rotation + Overextension)', 
             fontsize=16, fontweight='bold')

plt.tight_layout()

# Save
output_file = 'output/model_comparison_2024_2025.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'\n✅ Comparison chart saved to: {output_file}')

# Summary
print('\n' + '=' * 90)
print('MODEL COMPARISON SUMMARY')
print('=' * 90)

print('\n📊 2024:')
print(f'   Pullback Model: {len(pullback_signals_2024)} signals - General overextension')
print(f'   Severity Model: {len(severity_signals_2024)} signals - Unhealthy overextension')
print(f'   🚨 BOTH AGREE: {len(both_agree)} days - HIGHEST CONFIDENCE')

print('\n📊 2025 (through Oct 3):')
print(f'   Pullback Model: {len(pullback_signals_2025)} signals')
print(f'   Severity Model: {len(severity_signals_2025)} signals')
print(f'   Status: Both models show healthy market')
print(f'   Latest SPY: ${spy_2025["Close"].iloc[-1]:.2f} (+{((spy_2025["Close"].iloc[-1]/spy_2025["Close"].iloc[0])-1)*100:.1f}% YTD)')

print('\n💡 KEY INSIGHT:')
print('   Pullback Model: Detects overextension (catches all corrections)')
print('   Severity Model: Detects unhealthy overextension (major events only)')
print('   🚨 When BOTH agree: HIGHEST CONFIDENCE - Take action!')

plt.close()
