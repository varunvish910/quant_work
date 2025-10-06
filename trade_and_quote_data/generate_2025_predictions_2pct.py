#!/usr/bin/env python3
"""
Generate 2025 Predictions with 2%+ pullback in 3-5 days model
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.features import FeatureEngine

print("=" * 80)
print("📊 GENERATING 2025 PREDICTIONS - 2%+ PULLBACK IN 3-5 DAYS")
print("=" * 80)
print()

# Load data
loader = DataLoader(start_date='2024-12-01', end_date='2025-12-31')

spy_data = loader.load_spy_data()
sector_data = loader.load_sector_data()
currency_data = loader.load_currency_data()
volatility_data = loader.load_volatility_data()

# Create features
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

existing_cols = set(spy_with_features.columns)
options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]

if options_cols_to_add:
    spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')

all_features = feature_engine.feature_columns + options_cols_to_add

print(f"✅ Created {len(all_features)} features")

# Load model
model = lgb.Booster(model_file='models/trained/early_warning_2pct_3to5d.txt')
print("✅ Loaded model")

# Generate predictions for 2025
data_2025 = spy_with_features[spy_with_features.index >= '2025-01-01']
X = data_2025[all_features].fillna(0)

pred_proba = model.predict(X, num_iteration=model.best_iteration)

results = pd.DataFrame({
    'date': data_2025.index,
    'spy_close': data_2025['Close'],
    'pullback_probability': pred_proba
})

print(f"✅ Generated predictions for {len(results)} days")
print()

# Summary
print("=" * 80)
print("📊 2025 DAILY PREDICTIONS")
print("=" * 80)
print()
print(results.to_string(index=False))
print()

# Statistics
print("=" * 80)
print("📊 SUMMARY STATISTICS")
print("=" * 80)
print()

high_risk = results[results['pullback_probability'] > 0.7]
elevated_risk = results[(results['pullback_probability'] > 0.6) & (results['pullback_probability'] <= 0.7)]
moderate_risk = results[(results['pullback_probability'] > 0.5) & (results['pullback_probability'] <= 0.6)]
low_risk = results[results['pullback_probability'] <= 0.5]

print(f"Risk Level Breakdown:")
print(f"  🚨 HIGH (>70%):      {len(high_risk):3d} days ({len(high_risk)/len(results)*100:5.1f}%)")
print(f"  ⚠️  ELEVATED (60-70%): {len(elevated_risk):3d} days ({len(elevated_risk)/len(results)*100:5.1f}%)")
print(f"  📊 MODERATE (50-60%): {len(moderate_risk):3d} days ({len(moderate_risk)/len(results)*100:5.1f}%)")
print(f"  ✅ LOW (<50%):       {len(low_risk):3d} days ({len(low_risk)/len(results)*100:5.1f}%)")
print()

print(f"Probability Statistics:")
print(f"  Mean:   {results['pullback_probability'].mean():.2%}")
print(f"  Median: {results['pullback_probability'].median():.2%}")
print(f"  Max:    {results['pullback_probability'].max():.2%} on {results.loc[results['pullback_probability'].idxmax(), 'date'].date()}")
print(f"  Min:    {results['pullback_probability'].min():.2%} on {results.loc[results['pullback_probability'].idxmin(), 'date'].date()}")
print()

# High risk dates
if len(high_risk) > 0:
    print("=" * 80)
    print("🚨 HIGH RISK DATES (>70% probability)")
    print("=" * 80)
    print()
    for idx, row in high_risk.iterrows():
        print(f"   {row['date'].date()}: ${row['spy_close']:.2f} - {row['pullback_probability']*100:.1f}%")
    print()

if len(elevated_risk) > 0:
    print("=" * 80)
    print("⚠️  ELEVATED RISK DATES (60-70% probability)")
    print("=" * 80)
    print()
    for idx, row in elevated_risk.iterrows():
        print(f"   {row['date'].date()}: ${row['spy_close']:.2f} - {row['pullback_probability']*100:.1f}%")
    print()

# Save
output_dir = Path('output')
results.to_csv(output_dir / '2025_predictions_2pct_3to5d.csv', index=False)
print(f"💾 Saved: output/2025_predictions_2pct_3to5d.csv")
print()

print("=" * 80)
print("✅ 2025 PREDICTIONS COMPLETE")
print("=" * 80)
