#!/usr/bin/env python3
"""
Train Early Warning Model with Options Features

Compares performance:
1. Baseline: Technical features only
2. Enhanced: Technical + Options features

Target: early_warning (5%+ crash in 3-13 days)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.features import FeatureEngine
from targets.early_warning import EarlyWarningTarget
from utils.constants import TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE

print("=" * 80)
print("🎯 TRAINING EARLY WARNING MODEL WITH OPTIONS FEATURES")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print()

# Initialize data loader
loader = DataLoader(
    start_date='2016-01-01',  # Options data starts from 2016
    end_date='2024-12-31'
)

# Load SPY data
print("📊 Loading SPY data...")
spy_data = loader.load_spy_data()
print(f"✅ Loaded {len(spy_data)} days of SPY data")

# Load sector data
print("\n📊 Loading sector data...")
sector_data = loader.load_sector_data()
print(f"✅ Loaded {len(sector_data)} sector ETFs")

# Load currency data
print("\n📊 Loading currency data...")
currency_data = loader.load_currency_data()
print(f"✅ Loaded {len(currency_data)} currency pairs")

# Load volatility data
print("\n📊 Loading volatility data...")
volatility_data = loader.load_volatility_data()
print(f"✅ Loaded {len(volatility_data)} volatility indices")

# Load options features
print("\n📊 Loading options features...")
options_features = pd.read_parquet('data/options_chains/enhanced_options_features.parquet')
options_features = options_features.set_index('date')
print(f"✅ Loaded {len(options_features)} days of options features")
print(f"   Features: {len(options_features.columns)} columns")

# ============================================================================
# CREATE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATING FEATURES")
print("=" * 80)
print()

# Initialize feature engine
feature_engine = FeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])

# Create baseline features (technical + sectors + rotation indicators)
print("📊 Creating baseline features...")
spy_with_features = feature_engine.calculate_features(
    spy_data=spy_data,
    sector_data=sector_data,
    currency_data=currency_data,
    volatility_data=volatility_data
)
print(f"✅ Created {len(feature_engine.feature_columns)} baseline features")

# Merge with options features
print("\n📊 Merging options features...")

# Remove duplicate columns from options features
existing_cols = set(spy_with_features.columns)
options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]

if options_cols_to_add:
    spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')
    print(f"✅ Added {len(options_cols_to_add)} new options features")
    
    # Show which columns were skipped due to duplication
    duplicate_cols = [col for col in options_features.columns if col in existing_cols]
    if duplicate_cols:
        print(f"   ℹ️  Skipped {len(duplicate_cols)} duplicate columns: {', '.join(duplicate_cols[:5])}{'...' if len(duplicate_cols) > 5 else ''}")
else:
    print("⚠️  All options features already exist in baseline features")

# Identify which features are options-based
options_feature_cols = options_cols_to_add

# ============================================================================
# CREATE TARGET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATING TARGET")
print("=" * 80)
print()

target_creator = EarlyWarningTarget()
spy_with_target = target_creator.create(spy_with_features)
target_col = target_creator.target_column

print(f"✅ Target created: {target_col}")

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: PREPARING TRAINING DATA")
print("=" * 80)
print()

# Remove rows with missing target
data = spy_with_target.dropna(subset=[target_col]).copy()
print(f"📊 Total samples: {len(data)}")

# Split by date
train_data = data[data.index < '2024-01-01']
val_data = data[(data.index >= '2024-01-01') & (data.index < '2024-07-01')]
test_data = data[data.index >= '2024-07-01']

print(f"\n📊 Data splits:")
print(f"   Train: {len(train_data)} samples ({train_data.index.min().date()} to {train_data.index.max().date()})")
print(f"   Val:   {len(val_data)} samples ({val_data.index.min().date()} to {val_data.index.max().date()})")
print(f"   Test:  {len(test_data)} samples ({test_data.index.min().date()} to {test_data.index.max().date()})")

# Check class balance
print(f"\n📊 Class balance:")
print(f"   Train: {train_data[target_col].mean()*100:.1f}% positive")
print(f"   Val:   {val_data[target_col].mean()*100:.1f}% positive")
print(f"   Test:  {test_data[target_col].mean()*100:.1f}% positive")

# ============================================================================
# TRAIN MODEL 1: BASELINE (TECHNICAL FEATURES ONLY)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: BASELINE (TECHNICAL FEATURES ONLY)")
print("=" * 80)
print()

# Select baseline features (exclude options features)
baseline_features = [col for col in feature_engine.feature_columns if col in data.columns]
print(f"📊 Using {len(baseline_features)} baseline features")

# Prepare data
X_train_baseline = train_data[baseline_features].fillna(0)
y_train = train_data[target_col]

X_val_baseline = val_data[baseline_features].fillna(0)
y_val = val_data[target_col]

X_test_baseline = test_data[baseline_features].fillna(0)
y_test = test_data[target_col]

# Train LightGBM model
print("\n🔄 Training baseline model...")
lgb_train = lgb.Dataset(X_train_baseline, y_train)
lgb_val = lgb.Dataset(X_val_baseline, y_val, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

model_baseline = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
)

# Evaluate baseline model
print("\n📊 BASELINE MODEL PERFORMANCE:")
print("=" * 80)

for split_name, X, y in [('Train', X_train_baseline, y_train), 
                          ('Val', X_val_baseline, y_val), 
                          ('Test', X_test_baseline, y_test)]:
    y_pred_proba = model_baseline.predict(X, num_iteration=model_baseline.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    roc_auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    print(f"\n{split_name}:")
    print(f"   ROC AUC:   {roc_auc*100:.1f}%")
    print(f"   Precision: {precision*100:.1f}%")
    print(f"   Recall:    {recall*100:.1f}%")
    print(f"   F1 Score:  {f1*100:.1f}%")

# ============================================================================
# TRAIN MODEL 2: ENHANCED (TECHNICAL + OPTIONS)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: ENHANCED (TECHNICAL + OPTIONS)")
print("=" * 80)
print()

# Select all features (baseline + options)
all_features = baseline_features + [col for col in options_feature_cols if col in data.columns]
print(f"📊 Using {len(all_features)} features ({len(baseline_features)} baseline + {len(all_features) - len(baseline_features)} options)")

# Prepare data
X_train_enhanced = train_data[all_features].fillna(0)
X_val_enhanced = val_data[all_features].fillna(0)
X_test_enhanced = test_data[all_features].fillna(0)

# Train LightGBM model
print("\n🔄 Training enhanced model...")
lgb_train = lgb.Dataset(X_train_enhanced, y_train)
lgb_val = lgb.Dataset(X_val_enhanced, y_val, reference=lgb_train)

model_enhanced = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
)

# Evaluate enhanced model
print("\n📊 ENHANCED MODEL PERFORMANCE:")
print("=" * 80)

for split_name, X, y in [('Train', X_train_enhanced, y_train), 
                          ('Val', X_val_enhanced, y_val), 
                          ('Test', X_test_enhanced, y_test)]:
    y_pred_proba = model_enhanced.predict(X, num_iteration=model_enhanced.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    roc_auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    print(f"\n{split_name}:")
    print(f"   ROC AUC:   {roc_auc*100:.1f}%")
    print(f"   Precision: {precision*100:.1f}%")
    print(f"   Recall:    {recall*100:.1f}%")
    print(f"   F1 Score:  {f1*100:.1f}%")

# ============================================================================
# COMPARE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("📊 MODEL COMPARISON")
print("=" * 80)
print()

# Get predictions for comparison
baseline_val_pred = model_baseline.predict(X_val_baseline, num_iteration=model_baseline.best_iteration)
enhanced_val_pred = model_enhanced.predict(X_val_enhanced, num_iteration=model_enhanced.best_iteration)

baseline_test_pred = model_baseline.predict(X_test_baseline, num_iteration=model_baseline.best_iteration)
enhanced_test_pred = model_enhanced.predict(X_test_enhanced, num_iteration=model_enhanced.best_iteration)

# Calculate improvements
val_baseline_auc = roc_auc_score(y_val, baseline_val_pred)
val_enhanced_auc = roc_auc_score(y_val, enhanced_val_pred)
val_improvement = (val_enhanced_auc - val_baseline_auc) * 100

test_baseline_auc = roc_auc_score(y_test, baseline_test_pred)
test_enhanced_auc = roc_auc_score(y_test, enhanced_test_pred)
test_improvement = (test_enhanced_auc - test_baseline_auc) * 100

print("Validation Set:")
print(f"   Baseline:  {val_baseline_auc*100:.1f}% ROC AUC")
print(f"   Enhanced:  {val_enhanced_auc*100:.1f}% ROC AUC")
print(f"   Improvement: {val_improvement:+.1f}%")
print()

print("Test Set:")
print(f"   Baseline:  {test_baseline_auc*100:.1f}% ROC AUC")
print(f"   Enhanced:  {test_enhanced_auc*100:.1f}% ROC AUC")
print(f"   Improvement: {test_improvement:+.1f}%")
print()

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📊 TOP 20 FEATURES (ENHANCED MODEL)")
print("=" * 80)
print()

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': model_enhanced.feature_importance(importance_type='gain')
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Mark options features
feature_importance['is_options'] = feature_importance['feature'].isin(options_feature_cols)

print("Top 20 features:")
for i, row in feature_importance.head(20).iterrows():
    feature_type = "📊 OPTIONS" if row['is_options'] else "📈 TECHNICAL"
    print(f"   {i+1:2d}. {row['feature']:40s} {row['importance']:10.0f}  {feature_type}")

# Count options features in top 20
options_in_top20 = feature_importance.head(20)['is_options'].sum()
print(f"\n✅ Options features in top 20: {options_in_top20}/20")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("💾 SAVING MODELS")
print("=" * 80)
print()

output_dir = Path('models/trained')
output_dir.mkdir(parents=True, exist_ok=True)

# Save baseline model
baseline_path = output_dir / 'early_warning_baseline.txt'
model_baseline.save_model(str(baseline_path))
print(f"✅ Saved baseline model: {baseline_path}")

# Save enhanced model
enhanced_path = output_dir / 'early_warning_enhanced.txt'
model_enhanced.save_model(str(enhanced_path))
print(f"✅ Saved enhanced model: {enhanced_path}")

# Save feature lists
import json

feature_config = {
    'baseline_features': baseline_features,
    'options_features': [col for col in options_feature_cols if col in all_features],
    'all_features': all_features,
    'target': target_col
}

config_path = output_dir / 'early_warning_features.json'
with open(config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✅ Saved feature config: {config_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
print()

if test_improvement > 0:
    print(f"🎉 OPTIONS FEATURES IMPROVED MODEL BY {test_improvement:+.1f}% ROC AUC!")
    print(f"   Baseline:  {test_baseline_auc*100:.1f}%")
    print(f"   Enhanced:  {test_enhanced_auc*100:.1f}%")
else:
    print(f"⚠️  Options features did not improve model performance")
    print(f"   Baseline:  {test_baseline_auc*100:.1f}%")
    print(f"   Enhanced:  {test_enhanced_auc*100:.1f}%")
    print(f"   Change:    {test_improvement:+.1f}%")

print()
print("📊 Next steps:")
print("   - Analyze feature importance")
print("   - Check for data quality issues in options features")
print("   - Consider feature engineering improvements")
print()
