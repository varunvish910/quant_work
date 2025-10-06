#!/usr/bin/env python3
"""
Visualize 2025 SPY with horizontal lines at 90%+ confidence signals
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

print("=" * 80)
print("📊 VISUALIZING 2025 SPY WITH 90%+ CONFIDENCE SIGNALS")
print("=" * 80)
print()

# Load the predictions
predictions_file = Path('output/2025_pullback_predictions_4pct.csv')
results = pd.read_csv(predictions_file)
results['date'] = pd.to_datetime(results['date'])

print(f"✅ Loaded {len(results)} days of predictions")
print()

# Find 85%+ confidence signals (since max is 87.9%)
high_confidence = results[results['pullback_probability'] >= 0.85]

print(f"🚨 Found {len(high_confidence)} days with 85%+ confidence:")
if len(high_confidence) > 0:
    for idx, row in high_confidence.iterrows():
        print(f"   {row['date'].date()}: ${row['spy_close']:.2f} - {row['pullback_probability']*100:.1f}%")
else:
    print("   (No days with 90%+ confidence)")
print()

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

# Plot 1: SPY Price with horizontal lines at 90%+ signals
axes[0].plot(results['date'], results['spy_close'], 'k-', linewidth=2, label='SPY Close', zorder=1)

# Draw horizontal lines at each 85%+ signal
if len(high_confidence) > 0:
    for idx, row in high_confidence.iterrows():
        # Draw horizontal line across the entire chart
        axes[0].axhline(y=row['spy_close'], color='red', linestyle='-', 
                       linewidth=2, alpha=0.7, zorder=2)
        # Add a vertical line at the signal date
        axes[0].axvline(x=row['date'], color='red', linestyle='--', 
                       linewidth=1, alpha=0.5, zorder=2)
        # Add text annotation
        axes[0].text(row['date'], row['spy_close'], 
                    f" {row['pullback_probability']*100:.0f}%", 
                    fontsize=9, color='red', fontweight='bold',
                    verticalalignment='bottom', zorder=3)

axes[0].set_ylabel('SPY Price ($)', fontsize=14, fontweight='bold')
axes[0].set_title('2025 SPY with 85%+ Confidence Pullback Signals (Max: 87.9%)', fontsize=16, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(['SPY Close', '85%+ Confidence Signal'], loc='upper left', fontsize=12)

# Plot 2: Pullback Probability
axes[1].fill_between(results['date'], 0, results['pullback_probability'], 
                      color='red', alpha=0.3, label='Pullback Risk')
axes[1].plot(results['date'], results['pullback_probability'], 'r-', linewidth=2)

# Mark thresholds
axes[1].axhline(y=0.85, color='darkred', linestyle='-', linewidth=2, alpha=0.8, label='85% Threshold')
axes[1].axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='70% Threshold')
axes[1].axhline(y=0.5, color='yellow', linestyle='--', linewidth=1, alpha=0.7, label='50% Threshold')

# Highlight 85%+ regions
if len(high_confidence) > 0:
    for idx, row in high_confidence.iterrows():
        axes[1].axvline(x=row['date'], color='red', linestyle='--', 
                       linewidth=1, alpha=0.5)

axes[1].set_ylabel('Pullback Probability', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper left', fontsize=12)

# Format x-axis
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

plt.tight_layout()

# Save figure
output_dir = Path('output')
fig_path = output_dir / '2025_spy_with_90pct_signals.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved chart: {fig_path}")

# Also create a zoomed version showing only high-confidence periods
if len(high_confidence) > 0:
    print("\n📊 Creating zoomed views of high-confidence periods...")
    
    # Group consecutive high-confidence days into periods
    high_confidence = high_confidence.sort_values('date')
    periods = []
    current_period = [high_confidence.iloc[0]]
    
    for i in range(1, len(high_confidence)):
        prev_date = high_confidence.iloc[i-1]['date']
        curr_date = high_confidence.iloc[i]['date']
        
        # If dates are within 5 days, consider them part of same period
        if (curr_date - prev_date).days <= 5:
            current_period.append(high_confidence.iloc[i])
        else:
            periods.append(current_period)
            current_period = [high_confidence.iloc[i]]
    
    periods.append(current_period)
    
    print(f"   Found {len(periods)} distinct high-confidence periods")
    print()
    
    # Create a summary table
    print("=" * 80)
    print("📊 HIGH CONFIDENCE PERIODS SUMMARY")
    print("=" * 80)
    print()
    
    for i, period in enumerate(periods, 1):
        start_date = period[0]['date']
        end_date = period[-1]['date']
        start_price = period[0]['spy_close']
        end_price = period[-1]['spy_close']
        max_prob = max(p['pullback_probability'] for p in period)
        
        print(f"Period {i}:")
        print(f"  Dates: {start_date.date()} to {end_date.date()} ({len(period)} days)")
        print(f"  SPY Price: ${start_price:.2f} → ${end_price:.2f}")
        print(f"  Max Confidence: {max_prob*100:.1f}%")
        print()

print()
print("=" * 80)
print("✅ VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"📊 Chart saved to: {fig_path}")
print()

if len(high_confidence) == 0:
    print("⚠️  Note: No days reached 85%+ confidence in 2025")
    print("   The highest confidence was: {:.1f}%".format(results['pullback_probability'].max() * 100))
else:
    print(f"✅ Found {len(high_confidence)} days with 85%+ confidence")
