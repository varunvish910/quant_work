#!/usr/bin/env python3
"""
Train Models for All Three Market Conditions

1. Crashes (sharp drops): 5%+ in 3-13 days
2. Gradual Pullbacks (slow declines): 4%+ over 30-40 days
3. Time Corrections (sideways): ±3% range over 30-60 days
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.features import FeatureEngine
from targets.early_warning import EarlyWarningTarget
from targets.gradual_pullback import GradualPullbackTarget
from targets.time_correction import TimeCorrectionTarget

print("=" * 80)
print("🎯 TRAINING MODELS FOR ALL THREE MARKET CONDITIONS")
print("=" * 80)
print()
print("1. 💥 CRASHES: Sharp drops (5%+ in 3-13 days)")
print("2. 📉 GRADUAL PULLBACKS: Slow declines (4%+ over 15-30 days)")
print("3. ⏸️  TIME CORRECTIONS: Sideways consolidation (±3% over 30-60 days)")
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print()

loader = DataLoader(start_date='2016-01-01', end_date='2024-12-31')

print("📊 Loading SPY data...")
spy_data = loader.load_spy_data()
print(f"✅ Loaded {len(spy_data)} days of SPY data")

print("\n📊 Loading sector data...")
sector_data = loader.load_sector_data()
print(f"✅ Loaded {len(sector_data)} sector ETFs")

print("\n📊 Loading currency data...")
currency_data = loader.load_currency_data()
print(f"✅ Loaded {len(currency_data)} currency pairs")

print("\n📊 Loading volatility data...")
volatility_data = loader.load_volatility_data()
print(f"✅ Loaded {len(volatility_data)} volatility indices")

print("\n📊 Loading options features...")
options_features = pd.read_parquet('data/options_chains/enhanced_options_features.parquet')
options_features = options_features.set_index('date')
print(f"✅ Loaded {len(options_features)} days of options features")

# ============================================================================
# CREATE FEATURES (ONCE FOR ALL MODELS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATING FEATURES")
print("=" * 80)
print()

feature_engine = FeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])

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
existing_cols = set(spy_with_features.columns)
options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]

if options_cols_to_add:
    spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')
    print(f"✅ Added {len(options_cols_to_add)} new options features")
    duplicate_cols = [col for col in options_features.columns if col in existing_cols]
    if duplicate_cols:
        print(f"   ℹ️  Skipped {len(duplicate_cols)} duplicate columns")

all_features = feature_engine.feature_columns + options_cols_to_add
print(f"\n✅ Total features: {len(all_features)}")

# ============================================================================
# TRAIN MODEL 1: CRASHES (EARLY WARNING)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: CRASHES (EARLY WARNING)")
print("=" * 80)
print()

target_creator_1 = EarlyWarningTarget()
data_with_target_1 = target_creator_1.create(spy_with_features.copy())
target_col_1 = target_creator_1.target_column

# Prepare data
data_1 = data_with_target_1.dropna(subset=[target_col_1]).copy()
train_1 = data_1[data_1.index < '2024-01-01']
test_1 = data_1[data_1.index >= '2024-07-01']

X_train_1 = train_1[all_features].fillna(0)
y_train_1 = train_1[target_col_1]
X_test_1 = test_1[all_features].fillna(0)
y_test_1 = test_1[target_col_1]

print(f"📊 Training samples: {len(train_1)} ({y_train_1.mean()*100:.1f}% positive)")
print(f"📊 Test samples: {len(test_1)} ({y_test_1.mean()*100:.1f}% positive)")

# Train
print("\n🔄 Training crash detection model...")
lgb_train_1 = lgb.Dataset(X_train_1, y_train_1)

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

model_1 = lgb.train(params, lgb_train_1, num_boost_round=200, valid_sets=[lgb_train_1], valid_names=['train'])

# Evaluate
y_pred_1 = model_1.predict(X_test_1, num_iteration=model_1.best_iteration)
roc_auc_1 = roc_auc_score(y_test_1, y_pred_1)

print(f"\n✅ CRASH MODEL: {roc_auc_1*100:.1f}% ROC AUC")

# Save
output_dir = Path('models/trained')
output_dir.mkdir(parents=True, exist_ok=True)
model_1.save_model(str(output_dir / 'crash_detection.txt'))
print(f"💾 Saved to: {output_dir / 'crash_detection.txt'}")

# ============================================================================
# TRAIN MODEL 2: GRADUAL PULLBACKS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: GRADUAL PULLBACKS")
print("=" * 80)
print()

target_creator_2 = GradualPullbackTarget()
data_with_target_2 = target_creator_2.create(spy_with_features.copy())
target_col_2 = target_creator_2.target_column

# Prepare data
data_2 = data_with_target_2.dropna(subset=[target_col_2]).copy()
train_2 = data_2[data_2.index < '2024-01-01']
test_2 = data_2[data_2.index >= '2024-07-01']

X_train_2 = train_2[all_features].fillna(0)
y_train_2 = train_2[target_col_2]
X_test_2 = test_2[all_features].fillna(0)
y_test_2 = test_2[target_col_2]

print(f"📊 Training samples: {len(train_2)} ({y_train_2.mean()*100:.1f}% positive)")
print(f"📊 Test samples: {len(test_2)} ({y_test_2.mean()*100:.1f}% positive)")

# Train
print("\n🔄 Training gradual pullback model...")
lgb_train_2 = lgb.Dataset(X_train_2, y_train_2)
model_2 = lgb.train(params, lgb_train_2, num_boost_round=200, valid_sets=[lgb_train_2], valid_names=['train'])

# Evaluate
y_pred_2 = model_2.predict(X_test_2, num_iteration=model_2.best_iteration)
roc_auc_2 = roc_auc_score(y_test_2, y_pred_2)

print(f"\n✅ GRADUAL PULLBACK MODEL: {roc_auc_2*100:.1f}% ROC AUC")

# Save
model_2.save_model(str(output_dir / 'gradual_pullback.txt'))
print(f"💾 Saved to: {output_dir / 'gradual_pullback.txt'}")

# ============================================================================
# TRAIN MODEL 3: TIME CORRECTIONS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: TIME CORRECTIONS")
print("=" * 80)
print()

target_creator_3 = TimeCorrectionTarget()
data_with_target_3 = target_creator_3.create(spy_with_features.copy())
target_col_3 = target_creator_3.target_column

# Prepare data
data_3 = data_with_target_3.dropna(subset=[target_col_3]).copy()
train_3 = data_3[data_3.index < '2024-01-01']
test_3 = data_3[data_3.index >= '2024-07-01']

X_train_3 = train_3[all_features].fillna(0)
y_train_3 = train_3[target_col_3]
X_test_3 = test_3[all_features].fillna(0)
y_test_3 = test_3[target_col_3]

print(f"📊 Training samples: {len(train_3)} ({y_train_3.mean()*100:.1f}% positive)")
print(f"📊 Test samples: {len(test_3)} ({y_test_3.mean()*100:.1f}% positive)")

# Train
print("\n🔄 Training time correction model...")
lgb_train_3 = lgb.Dataset(X_train_3, y_train_3)
model_3 = lgb.train(params, lgb_train_3, num_boost_round=200, valid_sets=[lgb_train_3], valid_names=['train'])

# Evaluate
y_pred_3 = model_3.predict(X_test_3, num_iteration=model_3.best_iteration)
roc_auc_3 = roc_auc_score(y_test_3, y_pred_3)

print(f"\n✅ TIME CORRECTION MODEL: {roc_auc_3*100:.1f}% ROC AUC")

# Save
model_3.save_model(str(output_dir / 'time_correction.txt'))
print(f"💾 Saved to: {output_dir / 'time_correction.txt'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL THREE MODELS TRAINED")
print("=" * 80)
print()
print(f"1. 💥 CRASH DETECTION:      {roc_auc_1*100:.1f}% ROC AUC")
print(f"2. 📉 GRADUAL PULLBACK:     {roc_auc_2*100:.1f}% ROC AUC")
print(f"3. ⏸️  TIME CORRECTION:      {roc_auc_3*100:.1f}% ROC AUC")
print()
print("📊 Models saved to: models/trained/")
print("   - crash_detection.txt")
print("   - gradual_pullback.txt")
print("   - time_correction.txt")
print()
print("🎯 You now have a complete market condition detection system!")
print()
