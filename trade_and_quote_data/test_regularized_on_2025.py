#!/usr/bin/env python3
"""
Test Regularized Model on 2025 Data

Compare the regularized model's performance on 2025 vs the original model.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from core.data_loader import DataLoader
from core.features import FeatureEngine
from utils.constants import TEST_END_DATE

print("=" * 80)
print("📊 TESTING REGULARIZED MODEL ON 2025 DATA")
print("=" * 80)
print()

# ============================================================================
# LOAD 2025 DATA
# ============================================================================
print("STEP 1: LOADING 2025 DATA")
print("=" * 80)
print()

data_loader = DataLoader(
    start_date='2025-01-01',
    end_date='2025-10-06'
)

spy_data = data_loader.load_spy_data()
sector_data = data_loader.load_sector_data()
currency_data = data_loader.load_currency_data()
volatility_data = data_loader.load_volatility_data()

# Load options features
options_features_path = 'data/options_chains/enhanced_options_features.parquet'
if os.path.exists(options_features_path):
    options_features = pd.read_parquet(options_features_path)
    options_features['date'] = pd.to_datetime(options_features['date'])
    options_features = options_features.set_index('date')
else:
    options_features = None

print(f"✅ Loaded {len(spy_data)} days of 2025 data")

# ============================================================================
# CREATE FEATURES
# ============================================================================
print("\nSTEP 2: CREATING FEATURES")
print("=" * 80)
print()

feature_engine = FeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])
spy_with_features = feature_engine.calculate_features(
    spy_data=spy_data,
    sector_data=sector_data,
    currency_data=currency_data,
    volatility_data=volatility_data
)

# Merge options features
if options_features is not None:
    existing_cols = set(spy_with_features.columns)
    options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]
    if options_cols_to_add:
        spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')

print(f"✅ Created features for {len(spy_with_features)} days")

# ============================================================================
# LOAD MODELS
# ============================================================================
print("\nSTEP 3: LOADING MODELS")
print("=" * 80)
print()

# Load regularized model
model_reg = lgb.Booster(model_file='models/trained/early_warning_regularized.txt')
with open('models/trained/early_warning_regularized_config.json', 'r') as f:
    config_reg = json.load(f)
print("✅ Loaded regularized model")

# Load original model
model_orig = lgb.Booster(model_file='models/trained/early_warning_2pct_3to5d.txt')
with open('models/trained/early_warning_2pct_3to5d_features.json', 'r') as f:
    config_orig = json.load(f)
print("✅ Loaded original model")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================
print("\nSTEP 4: MAKING PREDICTIONS")
print("=" * 80)
print()

# Prepare features
feature_cols_reg = config_reg['features']
feature_cols_orig = config_orig['features']

X_reg = spy_with_features[feature_cols_reg]
X_orig = spy_with_features[feature_cols_orig]

# Predict
pred_reg = model_reg.predict(X_reg, num_iteration=model_reg.best_iteration)
pred_orig = model_orig.predict(X_orig, num_iteration=model_orig.best_iteration)

# Create results DataFrame
results = pd.DataFrame({
    'date': spy_with_features.index,
    'spy_close': spy_with_features['Close'],
    'prob_regularized': pred_reg,
    'prob_original': pred_orig
})

print(f"✅ Generated predictions for {len(results)} days")

# ============================================================================
# COMPARE MODELS
# ============================================================================
print("\nSTEP 5: COMPARING MODELS")
print("=" * 80)
print()

# Count high-confidence signals
thresholds = [0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]

print("High-Confidence Signal Counts:")
print(f"{'Threshold':<12} {'Original':<12} {'Regularized':<12} {'Change'}")
print("-" * 50)
for thresh in thresholds:
    count_orig = (results['prob_original'] >= thresh).sum()
    count_reg = (results['prob_regularized'] >= thresh).sum()
    change = count_reg - count_orig
    change_str = f"{change:+d}" if change != 0 else "0"
    print(f"{thresh*100:>3.0f}%         {count_orig:<12} {count_reg:<12} {change_str}")

# Probability statistics
print("\nProbability Statistics:")
print(f"{'Metric':<20} {'Original':<12} {'Regularized':<12}")
print("-" * 50)
print(f"{'Max':<20} {results['prob_original'].max()*100:>6.1f}%      {results['prob_regularized'].max()*100:>6.1f}%")
print(f"{'Mean':<20} {results['prob_original'].mean()*100:>6.1f}%      {results['prob_regularized'].mean()*100:>6.1f}%")
print(f"{'Median':<20} {results['prob_original'].median()*100:>6.1f}%      {results['prob_regularized'].median()*100:>6.1f}%")
print(f"{'Std Dev':<20} {results['prob_original'].std()*100:>6.1f}%      {results['prob_regularized'].std()*100:>6.1f}%")

# ============================================================================
# VISUALIZE COMPARISON
# ============================================================================
print("\nSTEP 6: CREATING VISUALIZATION")
print("=" * 80)
print()

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

# Plot 1: SPY Price
axes[0].plot(results['date'], results['spy_close'], 'k-', linewidth=2, label='SPY Close')
axes[0].set_ylabel('SPY Price ($)', fontsize=14, fontweight='bold')
axes[0].set_title('2025 SPY Price', fontsize=16, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper left', fontsize=12)

# Plot 2: Original Model Probability
axes[1].fill_between(results['date'], 0, results['prob_original'], 
                      color='red', alpha=0.3, label='Pullback Risk (Original)')
axes[1].plot(results['date'], results['prob_original'], 'r-', linewidth=2)
axes[1].axhline(y=0.70, color='darkred', linestyle='-', linewidth=2, alpha=0.8, label='70% Threshold')
axes[1].axhline(y=0.60, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='60% Threshold')
axes[1].axhline(y=0.50, color='yellow', linestyle='--', linewidth=1, alpha=0.7, label='50% Threshold')
axes[1].set_ylabel('Probability', fontsize=14, fontweight='bold')
axes[1].set_title('Original Model (High Volatility Weighting)', fontsize=16, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].set_ylim([0, 1])

# Plot 3: Regularized Model Probability
axes[2].fill_between(results['date'], 0, results['prob_regularized'], 
                      color='blue', alpha=0.3, label='Pullback Risk (Regularized)')
axes[2].plot(results['date'], results['prob_regularized'], 'b-', linewidth=2)
axes[2].axhline(y=0.70, color='darkblue', linestyle='-', linewidth=2, alpha=0.8, label='70% Threshold')
axes[2].axhline(y=0.60, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='60% Threshold')
axes[2].axhline(y=0.50, color='lightblue', linestyle='--', linewidth=1, alpha=0.7, label='50% Threshold')
axes[2].set_ylabel('Probability', fontsize=14, fontweight='bold')
axes[2].set_title('Regularized Model (Reduced Volatility Weighting)', fontsize=16, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper left', fontsize=10)
axes[2].set_ylim([0, 1])

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('output/2025_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization to: output/2025_model_comparison.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\nSTEP 7: SAVING RESULTS")
print("=" * 80)
print()

results.to_csv('output/2025_model_comparison.csv', index=False)
print("✅ Saved results to: output/2025_model_comparison.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETE")
print("=" * 80)
print()
print("Key Findings:")
print(f"  1. Original model max confidence: {results['prob_original'].max()*100:.1f}%")
print(f"  2. Regularized model max confidence: {results['prob_regularized'].max()*100:.1f}%")
print(f"  3. Original 70%+ signals: {(results['prob_original'] >= 0.70).sum()} days")
print(f"  4. Regularized 70%+ signals: {(results['prob_regularized'] >= 0.70).sum()} days")
print()
print("The regularized model should have:")
print("  ✅ Fewer false positives")
print("  ✅ More conservative predictions")
print("  ✅ Better precision (when it signals, it's more likely correct)")
print()
print("Files Created:")
print("  - output/2025_model_comparison.png")
print("  - output/2025_model_comparison.csv")
