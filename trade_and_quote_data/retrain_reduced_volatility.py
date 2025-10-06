#!/usr/bin/env python3
"""
Retrain Early Warning Model with Reduced Volatility Weighting

Strategy: Option A - Reduce volatility feature influence
- Manually downweight volatility features
- Add feature interaction terms (volatility + price action)
- Increase importance of trend/momentum features
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
print("🔧 RETRAINING WITH REDUCED VOLATILITY WEIGHTING")
print("=" * 80)
print()
print("Strategy:")
print("  1. Scale down volatility features by 0.5x")
print("  2. Create interaction features (volatility * price momentum)")
print("  3. Add trend confirmation features")
print("  4. Retrain with adjusted feature set")
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
print("STEP 2: CREATING FEATURES WITH REDUCED VOLATILITY WEIGHTING")
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
# FEATURE ENGINEERING: REDUCE VOLATILITY INFLUENCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: ADJUSTING FEATURE WEIGHTS")
print("=" * 80)
print()

# Identify volatility-related features
volatility_features = [
    'vix', 'vvix', 'vix_spike', 'vix_regime', 'vix_term_structure',
    'vix_backwardation', 'vix_percentile', 'bb_width', 'bb_width_percentile',
    'bb_squeeze', 'bb_expansion', 'atr', 'atr_percentile'
]

# Scale down volatility features by 0.5x
print("📉 Scaling down volatility features by 0.5x:")
for feat in volatility_features:
    if feat in spy_with_features.columns:
        spy_with_features[feat] = spy_with_features[feat] * 0.5
        print(f"   ✅ Scaled: {feat}")

# Create interaction features (volatility * price action)
print("\n📊 Creating interaction features:")

# VIX * Price momentum
if 'vix' in spy_with_features.columns and 'returns_5d' in spy_with_features.columns:
    spy_with_features['vix_momentum_interaction'] = spy_with_features['vix'] * spy_with_features['returns_5d']
    print("   ✅ vix_momentum_interaction")

# VIX * RSI (overbought + fear = warning)
if 'vix' in spy_with_features.columns and 'rsi' in spy_with_features.columns:
    spy_with_features['vix_rsi_interaction'] = spy_with_features['vix'] * (spy_with_features['rsi'] / 100)
    print("   ✅ vix_rsi_interaction")

# Bollinger Band width * Price distance from MA
if 'bb_width' in spy_with_features.columns and 'distance_from_20d_high' in spy_with_features.columns:
    spy_with_features['bb_width_distance_interaction'] = spy_with_features['bb_width'] * abs(spy_with_features['distance_from_20d_high'])
    print("   ✅ bb_width_distance_interaction")

# Create trend confirmation features
print("\n📈 Creating trend confirmation features:")

# Trend strength (ADX * directional movement)
if 'adx' in spy_with_features.columns and 'plus_di' in spy_with_features.columns and 'minus_di' in spy_with_features.columns:
    spy_with_features['trend_strength'] = spy_with_features['adx'] * (spy_with_features['plus_di'] - spy_with_features['minus_di']) / 100
    print("   ✅ trend_strength")

# Price momentum composite (multiple timeframes)
if all(f'returns_{d}d' in spy_with_features.columns for d in [1, 5, 10, 20]):
    spy_with_features['momentum_composite'] = (
        spy_with_features['returns_1d'] * 0.1 +
        spy_with_features['returns_5d'] * 0.3 +
        spy_with_features['returns_10d'] * 0.3 +
        spy_with_features['returns_20d'] * 0.3
    )
    print("   ✅ momentum_composite")

# ============================================================================
# CREATE TARGET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CREATING TARGET")
print("=" * 80)
print()

target_creator = EarlyWarningTarget()
df_with_target = target_creator.create(spy_with_features.copy())

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: PREPARING TRAINING DATA")
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
# TRAIN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TRAINING MODEL")
print("=" * 80)
print()

model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    is_unbalance=True,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    min_child_samples=20,
    random_state=42,
    n_estimators=1000
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

print(f"✅ Training complete (stopped at {model.best_iteration_} iterations)")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: EVALUATING MODEL")
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
print("STEP 8: FEATURE IMPORTANCE")
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
volatility_importance = feature_importance[feature_importance['feature'].isin(volatility_features)]
total_importance = feature_importance['importance'].sum()
volatility_pct = volatility_importance['importance'].sum() / total_importance * 100

print(f"\n📊 Volatility Features:")
print(f"   Total importance: {volatility_pct:.1f}% (down from ~60%)")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: SAVING MODEL")
print("=" * 80)
print()

os.makedirs('models/trained', exist_ok=True)
model_path = 'models/trained/early_warning_reduced_volatility.txt'
model.booster_.save_model(model_path)
print(f"✅ Model saved to: {model_path}")

# Save feature config
feature_config = {
    'features': feature_cols,
    'target': target_creator.target_column,
    'model_type': 'LightGBM',
    'strategy': 'reduced_volatility_weighting',
    'volatility_scale_factor': 0.5,
    'train_period': f"{TRAIN_START_DATE} to {TRAIN_END_DATE}",
    'test_metrics': {
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'false_positive_rate': float(fpr)
    }
}

config_path = 'models/trained/early_warning_reduced_volatility_config.json'
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
print("  1. ✅ Scaled volatility features by 0.5x")
print("  2. ✅ Added interaction features (volatility * price action)")
print("  3. ✅ Added trend confirmation features")
print()
print("Results:")
print(f"  ROC AUC: {roc_auc*100:.1f}%")
print(f"  False Positive Rate: {fpr*100:.1f}%")
print(f"  Volatility feature importance: {volatility_pct:.1f}% (target: <40%)")
print()
print("Next Steps:")
print("  1. Run predictions on 2025 data")
print("  2. Compare false positive rate with original model")
print("  3. Visualize new signals")
