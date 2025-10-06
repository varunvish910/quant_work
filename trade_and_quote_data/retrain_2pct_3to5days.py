#!/usr/bin/env python3
"""
Retrain Early Warning Model: 2%+ pullback within 3-5 days
Much tighter, more actionable signal
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.features import FeatureEngine
from targets.early_warning import EarlyWarningTarget
from utils.constants import TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE

print("=" * 80)
print("🎯 RETRAINING EARLY WARNING MODEL")
print("=" * 80)
print()
print("New Target: 2%+ pullback within 3-5 days")
print("(Much tighter, more actionable signal)")
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print()

loader = DataLoader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE)

spy_data = loader.load_spy_data()
sector_data = loader.load_sector_data()
currency_data = loader.load_currency_data()
volatility_data = loader.load_volatility_data()

print(f"✅ Loaded {len(spy_data)} days of SPY data")

# ============================================================================
# CREATE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATING FEATURES")
print("=" * 80)
print()

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

# Prepare features and target
X = df_with_target[all_features].fillna(0)
y = df_with_target[target_creator.target_column]

# Align
X, y = X.align(y, join='inner', axis=0)

# Split by date
X_train = X.loc[TRAIN_START_DATE:TRAIN_END_DATE]
y_train = y.loc[TRAIN_START_DATE:TRAIN_END_DATE]

X_val = X.loc[pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1):VAL_END_DATE]
y_val = y.loc[pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1):VAL_END_DATE]

X_test = X.loc[pd.to_datetime(VAL_END_DATE) + pd.Timedelta(days=1):TEST_END_DATE]
y_test = y.loc[pd.to_datetime(VAL_END_DATE) + pd.Timedelta(days=1):TEST_END_DATE]

print(f"Training:   {X_train.index.min().date()} to {X_train.index.max().date()}")
print(f"            {len(X_train)} samples, {y_train.sum()} positive ({y_train.mean()*100:.1f}%)")
print()
print(f"Validation: {X_val.index.min().date()} to {X_val.index.max().date()}")
print(f"            {len(X_val)} samples, {y_val.sum()} positive ({y_val.mean()*100:.1f}%)")
print()
print(f"Test:       {X_test.index.min().date()} to {X_test.index.max().date()}")
print(f"            {len(X_test)} samples, {y_test.sum()} positive ({y_test.mean()*100:.1f}%)")
print()

# ============================================================================
# TRAIN MODEL WITH BETTER CLASS BALANCING
# ============================================================================
print("=" * 80)
print("STEP 5: TRAINING MODEL")
print("=" * 80)
print()

print("🔄 Training LightGBM model with class balancing...")
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    is_unbalance=True,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=50,  # Prevent overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

print("✅ Training complete")
print()

# ============================================================================
# EVALUATE ON TEST SET (2024)
# ============================================================================
print("=" * 80)
print("STEP 6: EVALUATING ON 2024 DATA")
print("=" * 80)
print()

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
if len(np.unique(y_test)) > 1:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"📊 ROC AUC: {roc_auc*100:.1f}%")
else:
    print("⚠️  Cannot calculate ROC AUC (only one class in test set)")
    roc_auc = np.nan

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print()
print("Confusion Matrix:")
print(f"                Predicted No    Predicted Yes")
print(f"Actual No       {tn:8d}        {fp:8d}")
print(f"Actual Yes      {fn:8d}        {tp:8d}")
print()
print(f"Metrics:")
print(f"  Precision: {precision:.2%} (When we predict pullback, how often is it real?)")
print(f"  Recall:    {recall:.2%} (Of all real pullbacks, how many did we catch?)")
print(f"  F1 Score:  {f1:.2%}")
print()

# ============================================================================
# TEST ON 2024 KEY DATES
# ============================================================================
print("=" * 80)
print("🎯 TESTING ON KEY 2024 DRAWDOWN DATES")
print("=" * 80)
print()

# Generate predictions for all of 2024
results_2024 = pd.DataFrame({
    'date': X_test.index,
    'spy_close': df_with_target.loc[X_test.index, 'Close'],
    'actual_pullback': y_test.values,
    'predicted_pullback': y_pred,
    'probability': y_pred_proba
})

# Key dates to check
key_dates = {
    'April 1, 2024': '2024-04-01',
    'July 15, 2024': '2024-07-15',
    'October 16, 2024': '2024-10-16',
    'December 5, 2024': '2024-12-05'
}

for description, date_str in key_dates.items():
    print(f"📅 {description}")
    
    try:
        date = pd.to_datetime(date_str)
        
        if date in results_2024['date'].values:
            row = results_2024[results_2024['date'] == date].iloc[0]
        else:
            idx = (results_2024['date'] - date).abs().idxmin()
            row = results_2024.iloc[idx]
            print(f"   ⚠️  Using nearest date: {row['date'].date()}")
        
        print(f"   SPY: ${row['spy_close']:.2f}")
        print(f"   Model probability: {row['probability']*100:.1f}%")
        
        if row['probability'] > 0.7:
            verdict = "🎯 HIGH RISK (>70%)"
        elif row['probability'] > 0.5:
            verdict = "⚠️  ELEVATED (>50%)"
        else:
            verdict = "✅ LOW RISK"
        
        print(f"   Verdict: {verdict}")
        
        if row['actual_pullback'] == 1:
            print(f"   Reality: ✅ 2%+ pullback did occur within 3-5 days")
        else:
            print(f"   Reality: ❌ No 2%+ pullback within 3-5 days")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()

# ============================================================================
# ANALYZE HIGH CONFIDENCE SIGNALS
# ============================================================================
print("=" * 80)
print("📊 HIGH CONFIDENCE SIGNALS IN 2024")
print("=" * 80)
print()

# Different thresholds
for threshold in [0.5, 0.6, 0.7, 0.8]:
    high_conf = results_2024[results_2024['probability'] >= threshold]
    
    if len(high_conf) > 0:
        tp = high_conf[high_conf['actual_pullback'] == 1]
        fp = high_conf[high_conf['actual_pullback'] == 0]
        
        precision_thresh = len(tp) / len(high_conf) if len(high_conf) > 0 else 0
        
        print(f"Threshold {threshold*100:.0f}%+:")
        print(f"  Total signals: {len(high_conf)}")
        print(f"  True positives: {len(tp)} (correct)")
        print(f"  False positives: {len(fp)} (false alarm)")
        print(f"  Precision: {precision_thresh*100:.1f}%")
        print()

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================
print("=" * 80)
print("💾 SAVING MODEL AND RESULTS")
print("=" * 80)
print()

# Save model
os.makedirs('models/trained', exist_ok=True)
model.booster_.save_model('models/trained/early_warning_2pct_3to5d.txt')
print("✅ Saved model: models/trained/early_warning_2pct_3to5d.txt")

# Save feature names
import json
feature_config = {
    'features': all_features,
    'target': target_creator.target_column,
    'params': target_creator.params
}
with open('models/trained/early_warning_2pct_3to5d_features.json', 'w') as f:
    json.dump(feature_config, f, indent=2)
print("✅ Saved feature config: models/trained/early_warning_2pct_3to5d_features.json")

# Save 2024 predictions
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
results_2024.to_csv(output_dir / '2024_predictions_2pct_3to5d.csv', index=False)
print(f"✅ Saved 2024 predictions: output/2024_predictions_2pct_3to5d.csv")

print()
print("=" * 80)
print("✅ RETRAINING COMPLETE")
print("=" * 80)
print()
print(f"Model Performance on 2024:")
print(f"  ROC AUC: {roc_auc*100:.1f}%")
print(f"  Recall: {recall:.1%} (caught {tp}/{tp+fn} events)")
print(f"  Precision: {precision:.1%}")
print()
