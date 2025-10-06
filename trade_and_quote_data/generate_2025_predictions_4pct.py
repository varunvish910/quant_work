#!/usr/bin/env python3
"""
Generate 2025 Predictions with New 4%+ Pullback Model

Show daily probabilities for all of 2025
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.features import FeatureEngine

print("=" * 80)
print("📊 GENERATING 2025 PREDICTIONS - 4%+ PULLBACK MODEL")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("📊 Loading data for 2025...")

loader = DataLoader(start_date='2024-12-01', end_date='2025-12-31')

spy_data = loader.load_spy_data()
sector_data = loader.load_sector_data()
currency_data = loader.load_currency_data()
volatility_data = loader.load_volatility_data()

print(f"✅ Loaded {len(spy_data)} days of data")

# ============================================================================
# CREATE FEATURES
# ============================================================================
print("\n📊 Creating features...")

feature_engine = FeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])

spy_with_features = feature_engine.calculate_features(
    spy_data=spy_data,
    sector_data=sector_data,
    currency_data=currency_data,
    volatility_data=volatility_data
)

# Load options features
options_features = pd.read_parquet('data/options_chains/enhanced_options_features.parquet')
options_features = options_features.set_index('date')

# Merge
existing_cols = set(spy_with_features.columns)
options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]

if options_cols_to_add:
    spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')

all_features = feature_engine.feature_columns + options_cols_to_add

print(f"✅ Created {len(all_features)} features")

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n📊 Loading trained model...")

model = lgb.Booster(model_file='models/trained/early_warning_4pct_30d.txt')

print("✅ Loaded model")

# ============================================================================
# GENERATE PREDICTIONS FOR 2025
# ============================================================================
print("\n📊 Generating predictions for 2025...")

# Filter for 2025 data
data_2025 = spy_with_features[spy_with_features.index >= '2025-01-01']

if len(data_2025) == 0:
    print("⚠️  No 2025 data available yet, using all available data...")
    data_2025 = spy_with_features

print(f"   Generating predictions for {len(data_2025)} days")
print(f"   Date range: {data_2025.index.min().date()} to {data_2025.index.max().date()}")

# Prepare features
X = data_2025[all_features].fillna(0)

# Generate predictions
print("\n🔄 Running predictions...")
pred_proba = model.predict(X, num_iteration=model.best_iteration)

# Create results DataFrame
results = pd.DataFrame({
    'date': data_2025.index,
    'spy_close': data_2025['Close'],
    'pullback_probability': pred_proba
})

print("✅ Predictions complete")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("📊 2025 DAILY PULLBACK PROBABILITIES (4%+ within 30 days)")
print("=" * 80)
print()
print(results.to_string(index=False))
print()

# Summary statistics
print("=" * 80)
print("📊 SUMMARY STATISTICS")
print("=" * 80)
print()
print(f"Date range: {results['date'].min().date()} to {results['date'].max().date()}")
print(f"Total days: {len(results)}")
print()
print("Probability Statistics:")
print(f"  Mean:   {results['pullback_probability'].mean():.2%}")
print(f"  Median: {results['pullback_probability'].median():.2%}")
print(f"  Max:    {results['pullback_probability'].max():.2%} on {results.loc[results['pullback_probability'].idxmax(), 'date'].date()}")
print(f"  Min:    {results['pullback_probability'].min():.2%} on {results.loc[results['pullback_probability'].idxmin(), 'date'].date()}")
print()

# Risk level breakdown
high_risk = results[results['pullback_probability'] > 0.7]
elevated_risk = results[(results['pullback_probability'] > 0.5) & (results['pullback_probability'] <= 0.7)]
moderate_risk = results[(results['pullback_probability'] > 0.3) & (results['pullback_probability'] <= 0.5)]
low_risk = results[results['pullback_probability'] <= 0.3]

print("Risk Level Breakdown:")
print(f"  🚨 HIGH (>70%):      {len(high_risk):3d} days ({len(high_risk)/len(results)*100:5.1f}%)")
print(f"  ⚠️  ELEVATED (50-70%): {len(elevated_risk):3d} days ({len(elevated_risk)/len(results)*100:5.1f}%)")
print(f"  📊 MODERATE (30-50%): {len(moderate_risk):3d} days ({len(moderate_risk)/len(results)*100:5.1f}%)")
print(f"  ✅ LOW (<30%):       {len(low_risk):3d} days ({len(low_risk)/len(results)*100:5.1f}%)")
print()

# ============================================================================
# HIGH RISK PERIODS
# ============================================================================
if len(high_risk) > 0:
    print("=" * 80)
    print("🚨 HIGH RISK PERIODS (>70% probability)")
    print("=" * 80)
    print()
    for idx, row in high_risk.iterrows():
        print(f"   {row['date'].date()}: ${row['spy_close']:.2f} - {row['pullback_probability']*100:.1f}%")
    print()

if len(elevated_risk) > 0:
    print("=" * 80)
    print("⚠️  ELEVATED RISK PERIODS (50-70% probability)")
    print("=" * 80)
    print()
    for idx, row in elevated_risk.iterrows():
        print(f"   {row['date'].date()}: ${row['spy_close']:.2f} - {row['pullback_probability']*100:.1f}%")
    print()

# ============================================================================
# MONTHLY SUMMARY
# ============================================================================
print("=" * 80)
print("📅 MONTHLY SUMMARY")
print("=" * 80)
print()

results['month'] = results['date'].dt.to_period('M')
monthly_summary = results.groupby('month').agg({
    'pullback_probability': ['mean', 'max', 'min'],
    'spy_close': ['first', 'last']
}).round(4)

monthly_summary.columns = ['Avg Prob', 'Max Prob', 'Min Prob', 'Start Price', 'End Price']
monthly_summary['Price Change'] = ((monthly_summary['End Price'] / monthly_summary['Start Price'] - 1) * 100).round(2)

print(monthly_summary.to_string())
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("=" * 80)
print("💾 SAVING RESULTS")
print("=" * 80)
print()

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Save CSV
csv_path = output_dir / '2025_pullback_predictions_4pct.csv'
results.to_csv(csv_path, index=False)
print(f"✅ Saved CSV: {csv_path}")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("\n📊 Creating visualization...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Plot 1: SPY Price
axes[0].plot(results['date'], results['spy_close'], 'k-', linewidth=2, label='SPY Close')
axes[0].set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold')
axes[0].set_title('2025 Pullback Risk Monitor (4%+ within 30 days)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper left')

# Plot 2: Pullback Probability
axes[1].fill_between(results['date'], 0, results['pullback_probability'], 
                      color='red', alpha=0.3, label='Pullback Risk')
axes[1].plot(results['date'], results['pullback_probability'], 'r-', linewidth=2)
axes[1].axhline(y=0.7, color='darkred', linestyle='--', linewidth=1, alpha=0.7, label='High Risk (>70%)')
axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Elevated Risk (>50%)')
axes[1].set_ylabel('Pullback Probability', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper left')

# Format x-axis
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

plt.tight_layout()

# Save figure
fig_path = output_dir / '2025_pullback_predictions_4pct.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved chart: {fig_path}")

print()
print("=" * 80)
print("✅ 2025 PREDICTIONS COMPLETE")
print("=" * 80)
print()
print(f"📊 Results saved to:")
print(f"   CSV:   {csv_path}")
print(f"   Chart: {fig_path}")
print()
