#!/usr/bin/env python3
"""
Retrain Early Warning Model with Feature Penalties

Better Strategy: Use LightGBM's monotone_constraints and feature penalties
instead of manually scaling features.

Approach:
1. Add monotone constraints (higher VIX = higher risk, but not linearly)
2. Increase min_gain_to_split to prevent overfitting to volatility spikes
3. Add L1/L2 regularization to penalize complex trees
4. Increase min_child_samples to require more evidence
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import json
from core.data_loader import DataLoader
from core.features import FeatureEngine
from targets.early_warning import EarlyWarningTarget
from utils.constants import TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE

print("=" * 80)
print("🔧 RETRAINING WITH FEATURE PENALTIES & REGULARIZATION")
print("=" * 80)
print()
print("Strategy:")
print("  1. Use original features (no scaling)")
print("  2. Add L1/L2 regularization to reduce overfitting")
print("  3. Increase min_gain_to_split (require stronger evidence)")
print("  4. Increase min_child_samples (prevent volatility spike overfitting)")
print("  5. Lower learning rate for more careful learning")
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
print("STEP 2: CREATING FEATURES (ORIGINAL, NO SCALING)")
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
        print(f"✅ Added {len(options_cols_to_add)} options features")

# ============================================================================
# CREATE TARGET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATING TARGET")
print("=" * 80)
print()

target_creator = EarlyWarningTarget()
df_with_target = target_creator.create(spy_with_features.copy())

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: PREPARING TRAINING DATA")
print("=" * 80)
print()

# Get feature columns (exclude OHLCV and target)
feature_cols = [col for col in df_with_target.columns 
                if col not in ['Close', 'High', 'Low', 'Volume', 'Open', target_creator.target_column]]

X = df_with_target[feature_cols]
y = df_with_target[target_creator.target_column]

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
# TRAIN MODEL WITH REGULARIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING MODEL WITH REGULARIZATION")
print("=" * 80)
print()

print("Model Configuration:")
print("  - learning_rate: 0.01 (slower, more careful)")
print("  - lambda_l1: 0.5 (L1 regularization)")
print("  - lambda_l2: 0.5 (L2 regularization)")
print("  - min_gain_to_split: 0.1 (require stronger evidence)")
print("  - min_child_samples: 50 (prevent overfitting to spikes)")
print("  - max_depth: 4 (shallower trees)")
print()

model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    is_unbalance=True,
    learning_rate=0.01,  # Much slower learning
    num_leaves=15,  # Fewer leaves
    max_depth=4,  # Shallower trees
    min_child_samples=50,  # Require more samples per leaf
    min_split_gain=0.1,  # Require stronger evidence for splits
    lambda_l1=0.5,  # L1 regularization
    lambda_l2=0.5,  # L2 regularization
    feature_fraction=0.8,  # Use 80% of features per tree
    bagging_fraction=0.8,  # Use 80% of data per tree
    bagging_freq=5,  # Bagging every 5 iterations
    random_state=42,
    n_estimators=2000,  # More iterations since learning rate is lower
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

print(f"✅ Training complete (stopped at {model.best_iteration_} iterations)")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: EVALUATING MODEL")
print("=" * 80)
print()

# Test set predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"📊 Test Set Performance:")
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
print("STEP 7: FEATURE IMPORTANCE")
print("=" * 80)
print()

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 20 Most Important Features:")
for i, row in feature_importance.head(20).iterrows():
    print(f"   {row['feature']:<40} {row['importance']:>8.1f}")

# Check volatility feature importance
volatility_features = [
    'vix', 'vvix', 'vix_spike', 'vix_regime', 'vix_term_structure',
    'vix_backwardation', 'vix_percentile', 'bb_width', 'bb_width_percentile',
    'bb_squeeze', 'bb_expansion', 'atr', 'atr_percentile', 'vix_level',
    'vvix_level', 'vix_percentile_252d', 'vvix_percentile_252d', 'vix3m_percentile'
]
volatility_importance = feature_importance[feature_importance['feature'].isin(volatility_features)]
total_importance = feature_importance['importance'].sum()
volatility_pct = volatility_importance['importance'].sum() / total_importance * 100

print(f"\n📊 Volatility Features:")
print(f"   Total importance: {volatility_pct:.1f}%")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: SAVING MODEL")
print("=" * 80)
print()

os.makedirs('models/trained', exist_ok=True)
model_path = 'models/trained/early_warning_regularized.txt'
model.booster_.save_model(model_path)
print(f"✅ Model saved to: {model_path}")

# Save feature config
feature_config = {
    'features': feature_cols,
    'target': target_creator.target_column,
    'model_type': 'LightGBM',
    'strategy': 'regularization_and_penalties',
    'hyperparameters': {
        'learning_rate': 0.01,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'min_split_gain': 0.1,
        'min_child_samples': 50,
        'max_depth': 4,
        'feature_fraction': 0.8
    },
    'train_period': f"{TRAIN_START_DATE} to {TRAIN_END_DATE}",
    'test_metrics': {
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'false_positive_rate': float(fpr)
    }
}

config_path = 'models/trained/early_warning_regularized_config.json'
with open(config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✅ Config saved to: {config_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ RETRAINING COMPLETE")
print("=" * 80)
print()
print("Changes Made:")
print("  1. ✅ Added L1/L2 regularization (0.5 each)")
print("  2. ✅ Increased min_gain_to_split (0.1)")
print("  3. ✅ Increased min_child_samples (50)")
print("  4. ✅ Reduced learning rate (0.01)")
print("  5. ✅ Shallower trees (max_depth=4)")
print()
print("Results:")
print(f"  ROC AUC: {roc_auc*100:.1f}%")
print(f"  False Positive Rate: {fpr*100:.1f}%")
print(f"  Volatility feature importance: {volatility_pct:.1f}%")
print()
print("Next Steps:")
print("  1. Run predictions on 2025 data")
print("  2. Compare false positive rate with original model")
print("  3. Visualize new signals")
