#!/usr/bin/env python3
"""
Generate 2025 Market Condition Oscillator

Creates daily predictions for all three models:
1. Crash probability
2. Gradual pullback probability
3. Time correction probability
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
print("📊 GENERATING 2025 MARKET CONDITION OSCILLATOR")
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
# LOAD MODELS
# ============================================================================
print("\n📊 Loading trained models...")

model_crash = lgb.Booster(model_file='models/trained/crash_detection.txt')
model_gradual = lgb.Booster(model_file='models/trained/gradual_pullback.txt')
model_time = lgb.Booster(model_file='models/trained/time_correction.txt')

print("✅ Loaded all 3 models")

# ============================================================================
# GENERATE PREDICTIONS FOR 2025
# ============================================================================
print("\n📊 Generating predictions for 2025...")

# Filter for 2025 data
data_2025 = spy_with_features[spy_with_features.index >= '2025-01-01']

if len(data_2025) == 0:
    print("❌ No 2025 data available yet")
    print("   Using all available data instead...")
    data_2025 = spy_with_features

print(f"   Generating predictions for {len(data_2025)} days")
print(f"   Date range: {data_2025.index.min().date()} to {data_2025.index.max().date()}")

# Prepare features
X = data_2025[all_features].fillna(0)

# Generate predictions
print("\n🔄 Running predictions...")
pred_crash = model_crash.predict(X, num_iteration=model_crash.best_iteration)
pred_gradual = model_gradual.predict(X, num_iteration=model_gradual.best_iteration)
pred_time = model_time.predict(X, num_iteration=model_time.best_iteration)

# Create results DataFrame
results = pd.DataFrame({
    'date': data_2025.index,
    'spy_close': data_2025['Close'],
    'prob_crash': pred_crash,
    'prob_gradual_pullback': pred_gradual,
    'prob_time_correction': pred_time
})

print("✅ Predictions complete")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n💾 Saving results...")

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Save CSV
csv_path = output_dir / '2025_market_oscillator.csv'
results.to_csv(csv_path, index=False)
print(f"✅ Saved CSV: {csv_path}")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("📊 2025 MARKET CONDITION OSCILLATOR")
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
print(f"  Crash:")
print(f"    Mean: {results['prob_crash'].mean():.4f}")
print(f"    Max:  {results['prob_crash'].max():.4f} on {results.loc[results['prob_crash'].idxmax(), 'date'].date()}")
print(f"    Min:  {results['prob_crash'].min():.4f}")
print()
print(f"  Gradual Pullback:")
print(f"    Mean: {results['prob_gradual_pullback'].mean():.4f}")
print(f"    Max:  {results['prob_gradual_pullback'].max():.4f} on {results.loc[results['prob_gradual_pullback'].idxmax(), 'date'].date()}")
print(f"    Min:  {results['prob_gradual_pullback'].min():.4f}")
print()
print(f"  Time Correction:")
print(f"    Mean: {results['prob_time_correction'].mean():.4f}")
print(f"    Max:  {results['prob_time_correction'].max():.4f} on {results.loc[results['prob_time_correction'].idxmax(), 'date'].date()}")
print(f"    Min:  {results['prob_time_correction'].min():.4f}")
print()

# High risk days
print("=" * 80)
print("🚨 HIGH RISK DAYS (Probability > 0.5)")
print("=" * 80)
print()

high_crash = results[results['prob_crash'] > 0.5]
high_gradual = results[results['prob_gradual_pullback'] > 0.5]
high_time = results[results['prob_time_correction'] > 0.5]

print(f"💥 High Crash Risk: {len(high_crash)} days")
if len(high_crash) > 0:
    print(high_crash[['date', 'spy_close', 'prob_crash']].to_string(index=False))
print()

print(f"📉 High Gradual Pullback Risk: {len(high_gradual)} days")
if len(high_gradual) > 0:
    print(high_gradual[['date', 'spy_close', 'prob_gradual_pullback']].to_string(index=False))
print()

print(f"⏸️  High Time Correction Risk: {len(high_time)} days")
if len(high_time) > 0:
    print(high_time[['date', 'spy_close', 'prob_time_correction']].to_string(index=False))
print()

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("=" * 80)
print("📊 CREATING VISUALIZATION")
print("=" * 80)
print()

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

# Plot 1: SPY Price
axes[0].plot(results['date'], results['spy_close'], 'k-', linewidth=2, label='SPY Close')
axes[0].set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold')
axes[0].set_title('2025 Market Condition Oscillator', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper left')

# Plot 2: Crash Probability
axes[1].fill_between(results['date'], 0, results['prob_crash'], 
                      color='red', alpha=0.3, label='Crash Risk')
axes[1].plot(results['date'], results['prob_crash'], 'r-', linewidth=2)
axes[1].axhline(y=0.5, color='darkred', linestyle='--', linewidth=1, alpha=0.5, label='High Risk (>0.5)')
axes[1].set_ylabel('Crash\nProbability', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper left')

# Plot 3: Gradual Pullback Probability
axes[2].fill_between(results['date'], 0, results['prob_gradual_pullback'], 
                      color='orange', alpha=0.3, label='Gradual Pullback Risk')
axes[2].plot(results['date'], results['prob_gradual_pullback'], color='darkorange', linewidth=2)
axes[2].axhline(y=0.5, color='darkorange', linestyle='--', linewidth=1, alpha=0.5, label='High Risk (>0.5)')
axes[2].set_ylabel('Gradual Pullback\nProbability', fontsize=12, fontweight='bold')
axes[2].set_ylim([0, 1])
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper left')

# Plot 4: Time Correction Probability
axes[3].fill_between(results['date'], 0, results['prob_time_correction'], 
                      color='blue', alpha=0.3, label='Time Correction Risk')
axes[3].plot(results['date'], results['prob_time_correction'], 'b-', linewidth=2)
axes[3].axhline(y=0.5, color='darkblue', linestyle='--', linewidth=1, alpha=0.5, label='High Risk (>0.5)')
axes[3].set_ylabel('Time Correction\nProbability', fontsize=12, fontweight='bold')
axes[3].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[3].set_ylim([0, 1])
axes[3].grid(True, alpha=0.3)
axes[3].legend(loc='upper left')

# Format x-axis
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[3].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

plt.tight_layout()

# Save figure
fig_path = output_dir / '2025_market_oscillator.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved chart: {fig_path}")

print()
print("=" * 80)
print("✅ OSCILLATOR GENERATION COMPLETE")
print("=" * 80)
print()
print(f"📊 Results saved to:")
print(f"   CSV:   {csv_path}")
print(f"   Chart: {fig_path}")
print()
