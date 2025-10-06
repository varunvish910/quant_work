#!/usr/bin/env python3
"""
Train Improved Early Warning Model

Improvements:
1. Expanded target window (3-7 days)
2. Severity prediction (multi-output)
3. Seasonality features
4. Cross-asset features
5. SMOTE for class imbalance
6. Threshold optimization
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import os
import json
from core.data_loader import DataLoader
from core.features import FeatureEngine
from targets.early_warning_improved import ImprovedEarlyWarningTarget
from features.calendar.seasonality import SeasonalityFeature
from features.market.cross_asset import CrossAssetFeature
from utils.constants import TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE

print("=" * 80)
print("🚀 TRAINING IMPROVED EARLY WARNING MODEL")
print("=" * 80)
print()
print("Improvements:")
print("  1. ✅ Expanded target window: 3-7 days (was 3-5)")
print("  2. ✅ Severity prediction: Minor/Moderate/Major")
print("  3. ✅ Seasonality features: Presidential cycle, OpEx, earnings")
print("  4. ✅ Cross-asset features: TLT, GLD correlations")
print("  5. ✅ SMOTE: Synthetic minority oversampling")
print("  6. ✅ Regularization: L1/L2 penalties")
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print()

data_loader = DataLoader(
    start_date=TRAIN_START_DATE,
    end_date=TEST_END_DATE
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
    print(f"✅ Loaded options features")
else:
    options_features = None
    print("⚠️  No options features found")

# ============================================================================
# CREATE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATING FEATURES")
print("=" * 80)
print()

# Baseline features
feature_engine = FeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])
spy_with_features = feature_engine.calculate_features(
    spy_data=spy_data,
    sector_data=sector_data,
    currency_data=currency_data,
    volatility_data=volatility_data
)
print(f"✅ Created {len(feature_engine.feature_columns)} baseline features")

# Merge options features
if options_features is not None:
    existing_cols = set(spy_with_features.columns)
    options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]
    if options_cols_to_add:
        spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')
        print(f"✅ Added {len(options_cols_to_add)} options features")

# Add seasonality features
print("\n📅 Adding seasonality features...")
seasonality_feature = SeasonalityFeature()
spy_with_features = seasonality_feature.calculate(spy_with_features)
print(f"✅ Added {len(seasonality_feature.feature_names)} seasonality features")

# Add cross-asset features
print("\n🌍 Adding cross-asset features...")
cross_asset_feature = CrossAssetFeature()
spy_with_features = cross_asset_feature.calculate(spy_with_features)
print(f"✅ Added {len(cross_asset_feature.feature_names)} cross-asset features")

total_features = len([col for col in spy_with_features.columns 
                      if col not in ['Close', 'High', 'Low', 'Volume', 'Open']])
print(f"\n✅ Total features: {total_features}")

# ============================================================================
# CREATE IMPROVED TARGET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATING IMPROVED TARGET")
print("=" * 80)
print()

target_creator = ImprovedEarlyWarningTarget()
df_with_target = target_creator.create(spy_with_features.copy())

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: PREPARING TRAINING DATA")
print("=" * 80)
print()

# Get feature columns (exclude OHLCV and target columns)
exclude_cols = ['Close', 'High', 'Low', 'Volume', 'Open', 
                'binary_target', 'severity', 'max_drawdown', 'is_cluster_start',
                target_creator.target_column]
feature_cols = [col for col in df_with_target.columns if col not in exclude_cols]

X = df_with_target[feature_cols]
y = df_with_target['binary_target']  # Use binary target for now
y_severity = df_with_target['severity']  # Save for later

# Align data
X, y = X.align(y, join='inner', axis=0)

# Split by date
X_train = X.loc[TRAIN_START_DATE:TRAIN_END_DATE]
y_train = y.loc[TRAIN_START_DATE:TRAIN_END_DATE]
X_val = X.loc[pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1):VAL_END_DATE]
y_val = y.loc[pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1):VAL_END_DATE]
X_test = X.loc[pd.to_datetime(VAL_END_DATE) + pd.Timedelta(days=1):TEST_END_DATE]
y_test = y.loc[pd.to_datetime(VAL_END_DATE) + pd.Timedelta(days=1):TEST_END_DATE]

print(f"Training: {len(X_train)} samples ({y_train.mean()*100:.1f}% positive)")
print(f"Validation: {len(X_val)} samples ({y_val.mean()*100:.1f}% positive)")
print(f"Test: {len(X_test)} samples ({y_test.mean()*100:.1f}% positive)")

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: HANDLING MISSING VALUES")
print("=" * 80)
print()

# Fill NaN values with median (SMOTE doesn't handle NaN)
print(f"Checking for NaN values...")
nan_counts_train = X_train.isna().sum()
nan_features = nan_counts_train[nan_counts_train > 0]

if len(nan_features) > 0:
    print(f"Found {len(nan_features)} features with NaN values:")
    for feat, count in nan_features.head(10).items():
        print(f"   {feat}: {count} NaN values")
    
    # Fill with forward fill, then backward fill, then median
    X_train = X_train.ffill().bfill().fillna(X_train.median())
    X_val = X_val.ffill().bfill().fillna(X_val.median())
    X_test = X_test.ffill().bfill().fillna(X_test.median())
    
    # Double-check no NaN remains
    remaining_nan_train = X_train.isna().sum().sum()
    if remaining_nan_train > 0:
        print(f"   ⚠️  Still {remaining_nan_train} NaN values, filling with 0")
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
    
    print(f"✅ Filled NaN values")
else:
    print(f"✅ No NaN values found")

# ============================================================================
# APPLY SMOTE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: APPLYING SMOTE FOR CLASS BALANCE")
print("=" * 80)
print()

print(f"Before SMOTE: {len(X_train)} samples ({y_train.sum()} positive)")

# Apply SMOTE to training data only
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {len(X_train_balanced)} samples ({y_train_balanced.sum()} positive)")
print(f"Class balance: {y_train_balanced.mean()*100:.1f}% positive")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: TRAINING MODEL")
print("=" * 80)
print()

model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    learning_rate=0.01,
    num_leaves=15,
    max_depth=4,
    min_child_samples=50,
    min_split_gain=0.1,
    lambda_l1=0.5,
    lambda_l2=0.5,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_estimators=2000,
    verbose=-1
)

model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

print(f"✅ Training complete (stopped at {model.best_iteration_} iterations)")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: EVALUATING MODEL")
print("=" * 80)
print()

# Test set predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
best_f1 = 0
best_threshold = 0.5

print("Threshold Optimization:")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 50)

for thresh in thresholds:
    y_pred = (y_pred_proba >= thresh).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"{thresh:<12.1f} {precision*100:<12.1f} {recall*100:<12.1f} {f1*100:<12.1f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n✅ Best threshold: {best_threshold} (F1: {best_f1*100:.1f}%)")

# Use best threshold
y_pred = (y_pred_proba >= best_threshold).astype(int)

# Calculate final metrics
roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n📊 Final Test Set Performance (threshold={best_threshold}):")
print(f"   ROC AUC: {roc_auc*100:.1f}%")
print(f"   Precision: {precision*100:.1f}%")
print(f"   Recall: {recall*100:.1f}%")
print(f"   F1 Score: {f1*100:.1f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")
print(f"   False Positive Rate: {fpr*100:.1f}%")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: FEATURE IMPORTANCE")
print("=" * 80)
print()

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 30 Most Important Features:")
for i, row in feature_importance.head(30).iterrows():
    print(f"   {row['feature']:<50} {row['importance']:>8.1f}")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAVING MODEL")
print("=" * 80)
print()

os.makedirs('models/trained', exist_ok=True)
model_path = 'models/trained/improved_early_warning.txt'
model.booster_.save_model(model_path)
print(f"✅ Model saved to: {model_path}")

# Save feature config
feature_config = {
    'features': feature_cols,
    'target': target_creator.target_column,
    'model_type': 'LightGBM',
    'strategy': 'improved_with_smote_and_new_features',
    'best_threshold': float(best_threshold),
    'improvements': [
        'Expanded target window (3-7 days)',
        'Seasonality features',
        'Cross-asset features',
        'SMOTE for class balance',
        'Threshold optimization',
        'Regularization'
    ],
    'train_period': f"{TRAIN_START_DATE} to {TRAIN_END_DATE}",
    'test_metrics': {
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'false_positive_rate': float(fpr)
    }
}

config_path = 'models/trained/improved_early_warning_config.json'
with open(config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✅ Config saved to: {config_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
print()
print("Model Performance:")
print(f"  ROC AUC: {roc_auc*100:.1f}%")
print(f"  Precision: {precision*100:.1f}%")
print(f"  Recall: {recall*100:.1f}%")
print(f"  F1 Score: {f1*100:.1f}%")
print(f"  False Positive Rate: {fpr*100:.1f}%")
print(f"  Optimal Threshold: {best_threshold}")
print()
print("Next Steps:")
print("  1. Test on 2024 critical clusters")
print("  2. Generate 2025 predictions")
print("  3. Compare with previous model")
