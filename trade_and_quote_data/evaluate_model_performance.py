#!/usr/bin/env python3
"""
Comprehensive Model Performance Evaluation

1. Count all missed events in 2024
2. Calculate precision, recall, F1 for each model
3. Evaluate if walk-forward CV would help
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.features import FeatureEngine
from targets.early_warning import EarlyWarningTarget
from targets.gradual_pullback import GradualPullbackTarget
from targets.time_correction import TimeCorrectionTarget
from utils.constants import TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE

print("=" * 80)
print("📊 COMPREHENSIVE MODEL PERFORMANCE EVALUATION")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA AND GENERATE PREDICTIONS
# ============================================================================
print("📊 Loading data...")

loader = DataLoader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE)

spy_data = loader.load_spy_data()
sector_data = loader.load_sector_data()
currency_data = loader.load_currency_data()
volatility_data = loader.load_volatility_data()

print(f"✅ Loaded {len(spy_data)} days of data")

# Create features
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
# CREATE TARGETS AND GENERATE PREDICTIONS FOR EACH MODEL
# ============================================================================
print("\n" + "=" * 80)
print("📊 EVALUATING MODEL 1: CRASH DETECTION")
print("=" * 80)
print()

# Create crash target
target_crash = EarlyWarningTarget()
df_crash = target_crash.create(spy_with_features.copy())

# Load model
model_crash = lgb.Booster(model_file='models/trained/crash_detection.txt')

# Prepare data
X_crash = df_crash[all_features].fillna(0)
y_crash = df_crash[target_crash.target_column]

# Align
X_crash, y_crash = X_crash.align(y_crash, join='inner', axis=0)

# Generate predictions
pred_crash_proba = model_crash.predict(X_crash, num_iteration=model_crash.best_iteration)
pred_crash = (pred_crash_proba > 0.5).astype(int)

# Split by period
test_mask = (X_crash.index >= pd.to_datetime(VAL_END_DATE))
X_test_crash = X_crash[test_mask]
y_test_crash = y_crash[test_mask]
pred_test_crash = pred_crash[test_mask]
pred_test_crash_proba = pred_crash_proba[test_mask]

print(f"Test period: {X_test_crash.index.min().date()} to {X_test_crash.index.max().date()}")
print(f"Test samples: {len(X_test_crash)}")
print(f"Actual crashes: {y_test_crash.sum()}")
print(f"Predicted crashes: {pred_test_crash.sum()}")
print()

# Classification report
print("Classification Report:")
print(classification_report(y_test_crash, pred_test_crash, 
                           target_names=['No Crash', 'Crash'],
                           zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_crash, pred_test_crash)
print("Confusion Matrix:")
print(f"                Predicted No    Predicted Yes")
print(f"Actual No       {cm[0,0]:8d}        {cm[0,1]:8d}")
print(f"Actual Yes      {cm[1,0]:8d}        {cm[1,1]:8d}")
print()

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"📊 Crash Detection Metrics:")
print(f"   True Positives (Caught):  {tp}")
print(f"   False Negatives (Missed): {fn}")
print(f"   False Positives (False Alarms): {fp}")
print(f"   True Negatives (Correct No-Crash): {tn}")
print()
print(f"   Precision: {precision:.2%} (When we predict crash, how often is it real?)")
print(f"   Recall:    {recall:.2%} (Of all real crashes, how many did we catch?)")
print(f"   F1 Score:  {f1:.2%}")
if len(np.unique(y_test_crash)) > 1:
    roc_auc = roc_auc_score(y_test_crash, pred_test_crash_proba)
    print(f"   ROC AUC:   {roc_auc:.2%}")
print()

# Show missed events
if fn > 0:
    missed_crashes = X_test_crash[(y_test_crash == 1) & (pred_test_crash == 0)]
    print(f"🚨 MISSED CRASH EVENTS ({len(missed_crashes)}):")
    for date in missed_crashes.index[:10]:  # Show first 10
        actual_price = df_crash.loc[date, 'Close']
        prob = pred_crash_proba[X_crash.index == date].iloc[0]
        print(f"   {date.date()}: SPY ${actual_price:.2f}, Predicted probability: {prob:.2%}")
    if len(missed_crashes) > 10:
        print(f"   ... and {len(missed_crashes) - 10} more")
    print()

# ============================================================================
# MODEL 2: GRADUAL PULLBACK
# ============================================================================
print("=" * 80)
print("📊 EVALUATING MODEL 2: GRADUAL PULLBACK")
print("=" * 80)
print()

target_gradual = GradualPullbackTarget()
df_gradual = target_gradual.create(spy_with_features.copy())

model_gradual = lgb.Booster(model_file='models/trained/gradual_pullback.txt')

X_gradual = df_gradual[all_features].fillna(0)
y_gradual = df_gradual[target_gradual.target_column]

X_gradual, y_gradual = X_gradual.align(y_gradual, join='inner', axis=0)

pred_gradual_proba = model_gradual.predict(X_gradual, num_iteration=model_gradual.best_iteration)
pred_gradual = (pred_gradual_proba > 0.5).astype(int)

test_mask = (X_gradual.index >= pd.to_datetime(VAL_END_DATE))
X_test_gradual = X_gradual[test_mask]
y_test_gradual = y_gradual[test_mask]
pred_test_gradual = pred_gradual[test_mask]
pred_test_gradual_proba = pred_gradual_proba[test_mask]

print(f"Test period: {X_test_gradual.index.min().date()} to {X_test_gradual.index.max().date()}")
print(f"Test samples: {len(X_test_gradual)}")
print(f"Actual gradual pullbacks: {y_test_gradual.sum()}")
print(f"Predicted gradual pullbacks: {pred_test_gradual.sum()}")
print()

print("Classification Report:")
print(classification_report(y_test_gradual, pred_test_gradual, 
                           target_names=['No Pullback', 'Pullback'],
                           zero_division=0))

cm = confusion_matrix(y_test_gradual, pred_test_gradual)
print("Confusion Matrix:")
print(f"                Predicted No    Predicted Yes")
print(f"Actual No       {cm[0,0]:8d}        {cm[0,1]:8d}")
print(f"Actual Yes      {cm[1,0]:8d}        {cm[1,1]:8d}")
print()

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"📊 Gradual Pullback Metrics:")
print(f"   True Positives (Caught):  {tp}")
print(f"   False Negatives (Missed): {fn}")
print(f"   False Positives (False Alarms): {fp}")
print(f"   True Negatives (Correct No-Pullback): {tn}")
print()
print(f"   Precision: {precision:.2%}")
print(f"   Recall:    {recall:.2%}")
print(f"   F1 Score:  {f1:.2%}")
if len(np.unique(y_test_gradual)) > 1:
    roc_auc = roc_auc_score(y_test_gradual, pred_test_gradual_proba)
    print(f"   ROC AUC:   {roc_auc:.2%}")
print()

if fn > 0:
    missed_gradual = X_test_gradual[(y_test_gradual == 1) & (pred_test_gradual == 0)]
    print(f"🚨 MISSED GRADUAL PULLBACK EVENTS ({len(missed_gradual)}):")
    for date in missed_gradual.index[:10]:
        actual_price = df_gradual.loc[date, 'Close']
        prob = pred_gradual_proba[X_gradual.index == date].iloc[0]
        print(f"   {date.date()}: SPY ${actual_price:.2f}, Predicted probability: {prob:.2%}")
    if len(missed_gradual) > 10:
        print(f"   ... and {len(missed_gradual) - 10} more")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("📊 OVERALL SUMMARY")
print("=" * 80)
print()

print("CRASH DETECTION:")
cm_crash = confusion_matrix(y_test_crash, pred_test_crash)
tn_c, fp_c, fn_c, tp_c = cm_crash.ravel()
print(f"  ✅ Caught: {tp_c}/{tp_c + fn_c} crashes ({tp_c/(tp_c + fn_c)*100:.1f}% recall)")
print(f"  ❌ Missed: {fn_c} crashes")
print(f"  🚨 False Alarms: {fp_c}")
print()

print("GRADUAL PULLBACK:")
cm_gradual = confusion_matrix(y_test_gradual, pred_test_gradual)
tn_g, fp_g, fn_g, tp_g = cm_gradual.ravel()
print(f"  ✅ Caught: {tp_g}/{tp_g + fn_g} pullbacks ({tp_g/(tp_g + fn_g)*100:.1f}% recall)")
print(f"  ❌ Missed: {fn_g} pullbacks")
print(f"  🚨 False Alarms: {fp_g}")
print()

total_events = (tp_c + fn_c) + (tp_g + fn_g)
total_missed = fn_c + fn_g
total_caught = tp_c + tp_g

print(f"COMBINED:")
print(f"  Total events: {total_events}")
print(f"  Total caught: {total_caught} ({total_caught/total_events*100:.1f}%)")
print(f"  Total missed: {total_missed} ({total_missed/total_events*100:.1f}%)")
print()

# ============================================================================
# WALK-FORWARD CV RECOMMENDATION
# ============================================================================
print("=" * 80)
print("🔍 SHOULD WE USE WALK-FORWARD CROSS-VALIDATION?")
print("=" * 80)
print()

print("Current Approach:")
print(f"  Training: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
print(f"  Validation: {TRAIN_END_DATE} to {VAL_END_DATE}")
print(f"  Test: {VAL_END_DATE} to {TEST_END_DATE}")
print(f"  Problem: Single train/test split, no adaptation to regime changes")
print()

print("Walk-Forward CV Benefits:")
print("  ✅ Continuously retrain on most recent data")
print("  ✅ Adapt to changing market regimes")
print("  ✅ More realistic simulation of production deployment")
print("  ✅ Better handle non-stationary time series")
print()

print("Walk-Forward CV Drawbacks:")
print("  ❌ More computationally expensive")
print("  ❌ Requires more careful implementation")
print("  ❌ Risk of overfitting to recent data")
print()

print("RECOMMENDATION:")
if total_missed > total_caught:
    print("  🚨 STRONGLY RECOMMEND WALK-FORWARD CV")
    print(f"     Current model missed {total_missed}/{total_events} events ({total_missed/total_events*100:.1f}%)")
    print("     This suggests the model is not adapting to market regime changes.")
    print()
    print("  Suggested approach:")
    print("     1. Use 3-year rolling training window")
    print("     2. Retrain every quarter (or month)")
    print("     3. Always test on next period forward")
    print("     4. Track performance across different market regimes")
else:
    print("  ⚠️  CONSIDER WALK-FORWARD CV")
    print(f"     Current model caught {total_caught}/{total_events} events ({total_caught/total_events*100:.1f}%)")
    print("     Walk-forward CV could improve adaptation to regime changes.")
print()

# Save detailed results
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

results_df = pd.DataFrame({
    'Model': ['Crash Detection', 'Gradual Pullback'],
    'True Positives': [tp_c, tp_g],
    'False Negatives': [fn_c, fn_g],
    'False Positives': [fp_c, fp_g],
    'True Negatives': [tn_c, tn_g],
    'Precision': [tp_c/(tp_c+fp_c) if (tp_c+fp_c)>0 else 0, 
                  tp_g/(tp_g+fp_g) if (tp_g+fp_g)>0 else 0],
    'Recall': [tp_c/(tp_c+fn_c) if (tp_c+fn_c)>0 else 0,
               tp_g/(tp_g+fn_g) if (tp_g+fn_g)>0 else 0],
    'F1': [2*(tp_c/(tp_c+fp_c))*(tp_c/(tp_c+fn_c))/((tp_c/(tp_c+fp_c))+(tp_c/(tp_c+fn_c))) if (tp_c+fp_c)>0 and (tp_c+fn_c)>0 else 0,
           2*(tp_g/(tp_g+fp_g))*(tp_g/(tp_g+fn_g))/((tp_g/(tp_g+fp_g))+(tp_g/(tp_g+fn_g))) if (tp_g+fp_g)>0 and (tp_g+fn_g)>0 else 0]
})

results_df.to_csv(output_dir / 'model_performance_summary.csv', index=False)
print(f"💾 Saved performance summary to: output/model_performance_summary.csv")
