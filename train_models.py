#!/usr/bin/env python3
"""
Train Early Warning and Pullback Prediction Models
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL TRAINING - EARLY WARNING SYSTEM")
print("="*80)

# Load features
df = pd.read_pickle('data/features_complete.pkl')
print(f"\n📊 Features loaded: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# CREATE TARGET 1: EARLY WARNING (2%+ drops, 3-5 days ahead)
# ============================================================================

print("\n" + "="*80)
print("TARGET 1: EARLY WARNING (2%+ drops, 3-5 days ahead)")
print("="*80)

df['Early_Warning_Target'] = 0

# Look ahead 3-5 days for 2%+ drops
for i in range(len(df) - 5):
    current_price = df['Close'].iloc[i]
    # Check if there's a 2%+ drop in the next 3-5 days
    future_window = df.iloc[i+3:i+6]  # Days 3, 4, 5 ahead
    if len(future_window) > 0:
        future_low = future_window['Low'].min()
        drop = (current_price - future_low) / current_price
        if drop >= 0.02:  # 2%+ drop
            df.iloc[i, df.columns.get_loc('Early_Warning_Target')] = 1

# Remove last 5 days (can't predict for them)
df_early_warning = df.iloc[:-5].copy()

print(f"✅ Early Warning target created")
print(f"   Total samples: {len(df_early_warning)}")
print(f"   Warning events: {df_early_warning['Early_Warning_Target'].sum()} ({df_early_warning['Early_Warning_Target'].mean():.1%})")
print(f"   Normal days: {(df_early_warning['Early_Warning_Target'] == 0).sum()} ({1-df_early_warning['Early_Warning_Target'].mean():.1%})")

# ============================================================================
# CREATE TARGET 2: PULLBACK PREDICTION (5%+ drops, 5-15 days ahead)
# ============================================================================

print("\n" + "="*80)
print("TARGET 2: PULLBACK PREDICTION (5%+ drops, 5-15 days ahead)")
print("="*80)

df['Pullback_Target'] = 0

for i in range(len(df) - 15):
    current_price = df['Close'].iloc[i]
    # Check if there's a 5%+ pullback in days 5-15 ahead
    future_window = df.iloc[i+5:i+16]  # Days 5-15 ahead
    if len(future_window) > 0:
        future_low = future_window['Low'].min()
        pullback = (current_price - future_low) / current_price
        if pullback >= 0.05:  # 5%+ pullback
            df.iloc[i, df.columns.get_loc('Pullback_Target')] = 1

# Remove last 15 days
df_pullback = df.iloc[:-15].copy()

print(f"✅ Pullback target created")
print(f"   Total samples: {len(df_pullback)}")
print(f"   Pullback events: {df_pullback['Pullback_Target'].sum()} ({df_pullback['Pullback_Target'].mean():.1%})")
print(f"   Normal days: {(df_pullback['Pullback_Target'] == 0).sum()} ({1-df_pullback['Pullback_Target'].mean():.1%})")

# ============================================================================
# PREPARE DATA FOR TRAINING
# ============================================================================

# Define feature columns (exclude OHLCV and targets)
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                'Early_Warning_Target', 'Pullback_Target']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n📊 Feature columns: {len(feature_cols)}")

# Split by time periods
train_end = '2022-12-31'
val_end = '2023-12-31'

def split_data(data, target_col):
    """Split data into train/val/test"""
    X = data[feature_cols]
    y = data[target_col]

    X_train = X[X.index <= train_end]
    y_train = y[y.index <= train_end]

    X_val = X[(X.index > train_end) & (X.index <= val_end)]
    y_val = y[(y.index > train_end) & (y.index <= val_end)]

    X_test = X[X.index > val_end]
    y_test = y[y.index > val_end]

    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================================
# TRAIN MODEL 1: EARLY WARNING
# ============================================================================

print("\n" + "="*80)
print("TRAINING EARLY WARNING MODEL")
print("="*80)

X_train, y_train, X_val, y_val, X_test, y_test = split_data(df_early_warning, 'Early_Warning_Target')

print(f"\nData split:")
print(f"  Train: {len(X_train)} samples, {y_train.mean():.1%} positive")
print(f"  Val:   {len(X_val)} samples, {y_val.mean():.1%} positive")
print(f"  Test:  {len(X_test)} samples, {y_test.mean():.1%} positive")

# Build ensemble model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    random_state=42,
    n_jobs=-1
)

early_warning_model = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft',
    weights=[1, 2]
)

print("\n🎓 Training ensemble model...")
early_warning_model.fit(X_train, y_train)
print("✅ Training complete")

# Evaluate
print("\n📊 VALIDATION PERFORMANCE:")
val_pred = early_warning_model.predict(X_val)
val_proba = early_warning_model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, val_pred, target_names=['Normal', 'Warning']))
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.3f}")

print("\n📊 TEST PERFORMANCE (2024):")
test_pred = early_warning_model.predict(X_test)
test_proba = early_warning_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, test_pred, target_names=['Normal', 'Warning']))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.3f}")

# Save model
joblib.dump(early_warning_model, 'models/trained/early_warning_model.pkl')
with open('models/trained/early_warning_features.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("\n💾 Early Warning model saved")

# ============================================================================
# TRAIN MODEL 2: PULLBACK PREDICTION
# ============================================================================

print("\n" + "="*80)
print("TRAINING PULLBACK PREDICTION MODEL")
print("="*80)

X_train, y_train, X_val, y_val, X_test, y_test = split_data(df_pullback, 'Pullback_Target')

print(f"\nData split:")
print(f"  Train: {len(X_train)} samples, {y_train.mean():.1%} positive")
print(f"  Val:   {len(X_val)} samples, {y_val.mean():.1%} positive")
print(f"  Test:  {len(X_test)} samples, {y_test.mean():.1%} positive")

# Build ensemble model
rf2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

xgb2 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    random_state=42,
    n_jobs=-1
)

pullback_model = VotingClassifier(
    estimators=[('rf', rf2), ('xgb', xgb2)],
    voting='soft',
    weights=[1, 2]
)

print("\n🎓 Training ensemble model...")
pullback_model.fit(X_train, y_train)
print("✅ Training complete")

# Evaluate
print("\n📊 VALIDATION PERFORMANCE:")
val_pred = pullback_model.predict(X_val)
val_proba = pullback_model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, val_pred, target_names=['No Pullback', 'Pullback']))
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.3f}")

print("\n📊 TEST PERFORMANCE (2024):")
test_pred = pullback_model.predict(X_test)
test_proba = pullback_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, test_pred, target_names=['No Pullback', 'Pullback']))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.3f}")

# Save model
joblib.dump(pullback_model, 'models/trained/pullback_model.pkl')
with open('models/trained/pullback_features.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("\n💾 Pullback model saved")

# ============================================================================
# CHECK 2024 CRITICAL EVENTS
# ============================================================================

print("\n" + "="*80)
print("2024 CRITICAL EVENTS DETECTION")
print("="*80)

# Get test predictions with dates
test_data = df_early_warning[df_early_warning.index > val_end].copy()
test_data['Early_Warning_Prob'] = early_warning_model.predict_proba(test_data[feature_cols])[:, 1]

# Check specific events
events = [
    ('July Yen Carry Unwind', '2024-07-29', '2024-08-02'),
    ('August VIX Spike', '2024-08-01', '2024-08-05'),
    ('October Correction', '2024-09-24', '2024-09-28')
]

for event_name, start, end in events:
    event_window = test_data[start:end]
    if len(event_window) > 0:
        max_prob = event_window['Early_Warning_Prob'].max()
        max_date = event_window['Early_Warning_Prob'].idxmax()
        detected = max_prob > 0.5
        print(f"\n{event_name}:")
        print(f"  Warning window: {start} to {end}")
        print(f"  Max probability: {max_prob:.1%} on {max_date.date()}")
        print(f"  Detection status: {'✅ DETECTED' if detected else '❌ MISSED'}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
