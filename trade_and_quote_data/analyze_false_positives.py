#!/usr/bin/env python3
"""
Analyze False Positive Rate in 2025 Predictions

Check how many of the high-confidence signals were actually correct
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

print("=" * 80)
print("📊 ANALYZING FALSE POSITIVE RATE")
print("=" * 80)
print()

# Load predictions
predictions_file = Path('output/2025_pullback_predictions_4pct.csv')
results = pd.read_csv(predictions_file)
results['date'] = pd.to_datetime(results['date'])

print(f"✅ Loaded {len(results)} days of predictions")
print()

# Load actual SPY data to verify what happened
spy = yf.download('SPY', start='2025-01-01', end='2025-12-31', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print(f"✅ Loaded {len(spy)} days of actual SPY data")
print()

# For each prediction, check if a 4%+ pullback actually occurred within 30 days
results['actual_pullback'] = False

for idx, row in results.iterrows():
    current_date = row['date']
    current_price = row['spy_close']
    
    # Look ahead 30 days
    future_dates = spy.loc[current_date:current_date + pd.Timedelta(days=30)]
    
    if len(future_dates) > 1:
        # Check if price dropped 4%+ from current price
        future_low = future_dates['Low'].min()
        drawdown = (future_low / current_price - 1)
        
        if drawdown <= -0.04:
            results.at[idx, 'actual_pullback'] = True

print("✅ Calculated actual pullbacks")
print()

# Analyze different confidence thresholds
thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.87]

print("=" * 80)
print("📊 FALSE POSITIVE ANALYSIS BY CONFIDENCE THRESHOLD")
print("=" * 80)
print()

summary_data = []

for threshold in thresholds:
    high_conf = results[results['pullback_probability'] >= threshold]
    
    if len(high_conf) > 0:
        # Calculate metrics
        true_positives = high_conf[high_conf['actual_pullback'] == True]
        false_positives = high_conf[high_conf['actual_pullback'] == False]
        
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        total_signals = len(high_conf)
        
        precision = tp_count / total_signals if total_signals > 0 else 0
        false_positive_rate = fp_count / total_signals if total_signals > 0 else 0
        
        summary_data.append({
            'Threshold': f'{threshold*100:.0f}%',
            'Total Signals': total_signals,
            'True Positives': tp_count,
            'False Positives': fp_count,
            'Precision': f'{precision*100:.1f}%',
            'FP Rate': f'{false_positive_rate*100:.1f}%'
        })
        
        print(f"Threshold: {threshold*100:.0f}%+")
        print(f"  Total signals: {total_signals}")
        print(f"  True positives: {tp_count} (pullback actually happened)")
        print(f"  False positives: {fp_count} (no pullback)")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  False Positive Rate: {false_positive_rate*100:.1f}%")
        print()

# Create summary table
summary_df = pd.DataFrame(summary_data)
print("=" * 80)
print("📊 SUMMARY TABLE")
print("=" * 80)
print()
print(summary_df.to_string(index=False))
print()

# Analyze the 85%+ signals specifically
print("=" * 80)
print("🔍 DETAILED ANALYSIS OF 85%+ SIGNALS")
print("=" * 80)
print()

high_conf_85 = results[results['pullback_probability'] >= 0.85].copy()

# Show false positives
false_positives_85 = high_conf_85[high_conf_85['actual_pullback'] == False]

if len(false_positives_85) > 0:
    print(f"❌ FALSE POSITIVES (85%+ confidence, but NO 4%+ pullback): {len(false_positives_85)} days")
    print()
    for idx, row in false_positives_85.head(20).iterrows():
        # Calculate what actually happened
        current_date = row['date']
        current_price = row['spy_close']
        future_dates = spy.loc[current_date:current_date + pd.Timedelta(days=30)]
        
        if len(future_dates) > 1:
            future_low = future_dates['Low'].min()
            future_high = future_dates['High'].max()
            actual_dd = (future_low / current_price - 1) * 100
            actual_gain = (future_high / current_price - 1) * 100
            
            print(f"   {row['date'].date()}: ${current_price:.2f} - {row['pullback_probability']*100:.1f}% confidence")
            print(f"      → Actual: {actual_dd:.1f}% max drawdown, {actual_gain:.1f}% max gain")
    
    if len(false_positives_85) > 20:
        print(f"   ... and {len(false_positives_85) - 20} more")
    print()

# Show true positives
true_positives_85 = high_conf_85[high_conf_85['actual_pullback'] == True]

if len(true_positives_85) > 0:
    print(f"✅ TRUE POSITIVES (85%+ confidence, 4%+ pullback occurred): {len(true_positives_85)} days")
    print()
    for idx, row in true_positives_85.head(10).iterrows():
        current_date = row['date']
        current_price = row['spy_close']
        future_dates = spy.loc[current_date:current_date + pd.Timedelta(days=30)]
        
        if len(future_dates) > 1:
            future_low = future_dates['Low'].min()
            actual_dd = (future_low / current_price - 1) * 100
            
            print(f"   {row['date'].date()}: ${current_price:.2f} - {row['pullback_probability']*100:.1f}% confidence")
            print(f"      → Actual: {actual_dd:.1f}% drawdown ✓")
    
    if len(true_positives_85) > 10:
        print(f"   ... and {len(true_positives_85) - 10} more")
    print()

# Overall statistics
print("=" * 80)
print("📊 OVERALL 2025 STATISTICS")
print("=" * 80)
print()

total_days = len(results)
days_with_pullback = results['actual_pullback'].sum()
days_without_pullback = total_days - days_with_pullback

print(f"Total days analyzed: {total_days}")
print(f"Days where 4%+ pullback occurred within 30 days: {days_with_pullback} ({days_with_pullback/total_days*100:.1f}%)")
print(f"Days where NO 4%+ pullback occurred: {days_without_pullback} ({days_without_pullback/total_days*100:.1f}%)")
print()

# Model's overall prediction rate
high_conf_days = len(results[results['pullback_probability'] >= 0.85])
print(f"Model flagged 85%+ confidence: {high_conf_days} days ({high_conf_days/total_days*100:.1f}%)")
print()

print("=" * 80)
print("🎯 CONCLUSION")
print("=" * 80)
print()

if len(false_positives_85) > len(true_positives_85):
    print("❌ PROBLEM IDENTIFIED:")
    print(f"   The model has a HIGH FALSE POSITIVE RATE at 85%+ confidence")
    print(f"   False positives: {len(false_positives_85)}")
    print(f"   True positives: {len(true_positives_85)}")
    print(f"   Precision: {len(true_positives_85)/len(high_conf_85)*100:.1f}%")
    print()
    print("   This means the model is TOO SENSITIVE and flags risk too often.")
    print()
    print("   Recommendations:")
    print("   1. Increase confidence threshold to 90%+ (if any exist)")
    print("   2. Retrain with better class balancing")
    print("   3. Add more discriminative features")
    print("   4. Use a different target definition")
else:
    print("✅ Model performance is acceptable at 85%+ confidence")
    print(f"   Precision: {len(true_positives_85)/len(high_conf_85)*100:.1f}%")

# Save detailed results
output_dir = Path('output')
results.to_csv(output_dir / '2025_predictions_with_actuals.csv', index=False)
print()
print(f"💾 Saved detailed results to: output/2025_predictions_with_actuals.csv")
