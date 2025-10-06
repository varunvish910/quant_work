#!/usr/bin/env python3
"""
Train Pullback Prediction Model

Trains a model to predict 5%+ pullbacks within 5-15 days.
This is for risk management and timing exits.
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import ModelTrainer
from targets.pullback_prediction import PullbackPredictionTarget
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score

def train_pullback_model():
    """Train pullback prediction model"""
    
    print("=" * 80)
    print("🎯 TRAINING PULLBACK PREDICTION MODEL")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Target: Predict 5%+ pullback within 5-15 days")
    print("Use case: Risk management, timing exits, hedging decisions")
    print("=" * 80)
    
    # Initialize trainer
    print("\n📦 Initializing trainer...")
    trainer = ModelTrainer(
        model_type='ensemble',  # Use ensemble (RF + XGBoost)
        feature_sets=['baseline', 'currency', 'volatility', 'sectors'],
        start_date='2000-01-01',
        end_date='2024-12-31'
    )
    
    # Load data
    print("\n📥 Loading data...")
    raw_data = trainer.load_data()
    
    # Engineer features
    print("\n🔧 Engineering features...")
    features_data = trainer.engineer_features()
    feature_columns = trainer.feature_engine.get_feature_columns()
    print(f"✅ Created {len(feature_columns)} features")
    
    # Create pullback prediction target
    print("\n🎯 Creating pullback prediction target...")
    target_creator = PullbackPredictionTarget(
        pullback_threshold=0.05,  # 5% pullback
        min_days=5,               # Look 5-15 days ahead
        max_days=15
    )
    
    # Create target on SPY data
    df_with_target = target_creator.create(raw_data['spy'].copy())
    target_column = target_creator.get_target_column()
    
    # Merge target with features
    print("\n🔗 Merging features and target...")
    combined_df = features_data.join(
        df_with_target[target_column],
        how='inner'
    )
    
    print(f"✅ Combined data: {len(combined_df)} records")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Target: {target_column}")
    print(f"   Positive rate: {combined_df[target_column].mean():.1%}")
    
    # Update trainer's data
    trainer.features_data = combined_df
    
    # Split data
    print("\n✂️  Splitting data...")
    trainer.split_data()
    
    print(f"   Train: {len(trainer.train_data)} records")
    print(f"   Val:   {len(trainer.val_data)} records")
    print(f"   Test:  {len(trainer.test_data)} records")
    
    # Prepare training data
    X_train = trainer.train_data[feature_columns]
    y_train = trainer.train_data[target_column]
    
    X_val = trainer.val_data[feature_columns]
    y_val = trainer.val_data[target_column]
    
    X_test = trainer.test_data[feature_columns]
    y_test = trainer.test_data[target_column]
    
    print(f"\n📊 Training set class distribution:")
    print(f"   No pullback: {(~y_train.astype(bool)).sum()} ({(~y_train.astype(bool)).mean():.1%})")
    print(f"   Pullback:    {y_train.sum()} ({y_train.mean():.1%})")
    
    # Train model
    print("\n" + "=" * 80)
    print("🎓 TRAINING MODEL")
    print("=" * 80)
    
    from core.models import EarlyWarningModel
    model = EarlyWarningModel(model_type='ensemble')
    model.fit(X_train, y_train, feature_columns)
    
    print("\n✅ Model trained successfully!")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("📊 VALIDATION SET PERFORMANCE")
    print("=" * 80)
    
    val_predictions = model.predict(X_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_val, val_predictions,
                                target_names=['No Pullback', 'Pullback']))
    
    print("\nConfusion Matrix:")
    cm_val = confusion_matrix(y_val, val_predictions)
    print(cm_val)
    print(f"\nTrue Negatives:  {cm_val[0,0]} (correctly predicted no pullback)")
    print(f"False Positives: {cm_val[0,1]} (false alarm)")
    print(f"False Negatives: {cm_val[1,0]} (missed pullback)")
    print(f"True Positives:  {cm_val[1,1]} (correctly predicted pullback)")
    
    if len(np.unique(y_val)) > 1:
        val_roc_auc = roc_auc_score(y_val, val_probabilities)
        print(f"\n🎯 ROC AUC Score: {val_roc_auc:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("📊 TEST SET PERFORMANCE (2024)")
    print("=" * 80)
    
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions,
                                target_names=['No Pullback', 'Pullback']))
    
    print("\nConfusion Matrix:")
    cm_test = confusion_matrix(y_test, test_predictions)
    print(cm_test)
    print(f"\nTrue Negatives:  {cm_test[0,0]} (correctly predicted no pullback)")
    print(f"False Positives: {cm_test[0,1]} (false alarm - predicted pullback but didn't happen)")
    print(f"False Negatives: {cm_test[1,0]} (missed pullback - didn't predict but happened)")
    print(f"True Positives:  {cm_test[1,1]} (correctly predicted pullback)")
    
    if len(np.unique(y_test)) > 1:
        test_roc_auc = roc_auc_score(y_test, test_probabilities)
        test_precision = precision_score(y_test, test_predictions) if y_test.sum() > 0 else 0
        test_recall = recall_score(y_test, test_predictions) if y_test.sum() > 0 else 0
        
        print(f"\n🎯 ROC AUC Score: {test_roc_auc:.4f}")
        print(f"🎯 Precision: {test_precision:.4f} (when model predicts pullback, how often is it right?)")
        print(f"🎯 Recall: {test_recall:.4f} (of all actual pullbacks, how many did we catch?)")
    
    # Analyze predictions on test set
    print("\n" + "=" * 80)
    print("🔍 DETAILED TEST SET ANALYSIS")
    print("=" * 80)
    
    test_analysis = trainer.test_data[[target_column]].copy()
    test_analysis['predicted'] = test_predictions
    test_analysis['probability'] = test_probabilities
    test_analysis['spy_close'] = raw_data['spy'].loc[test_analysis.index, 'Close']
    
    # Show high-confidence predictions
    print("\n📈 HIGH CONFIDENCE PULLBACK PREDICTIONS (>70%):")
    high_conf = test_analysis[test_analysis['probability'] > 0.7].sort_values('probability', ascending=False)
    if len(high_conf) > 0:
        print(f"\nFound {len(high_conf)} high-confidence predictions:")
        for idx, row in high_conf.head(10).iterrows():
            actual = "✅ CORRECT" if row[target_column] == 1 else "❌ FALSE ALARM"
            print(f"  {idx.date()}: {row['probability']:.1%} confidence - {actual}")
            print(f"    SPY: ${row['spy_close']:.2f}")
    else:
        print("  None found")
    
    # Show missed pullbacks
    print("\n⚠️  MISSED PULLBACKS (actual pullback but low prediction):")
    missed = test_analysis[(test_analysis[target_column] == 1) & (test_analysis['probability'] < 0.5)]
    if len(missed) > 0:
        print(f"\nMissed {len(missed)} pullbacks:")
        for idx, row in missed.head(5).iterrows():
            print(f"  {idx.date()}: {row['probability']:.1%} confidence (missed)")
            print(f"    SPY: ${row['spy_close']:.2f}")
    else:
        print("  None! Model caught all pullbacks ✅")
    
    # Save model
    print("\n" + "=" * 80)
    print("💾 SAVING MODEL")
    print("=" * 80)
    
    model_filename = f"models/trained/pullback_prediction_ensemble_{datetime.now().strftime('%Y%m%d')}.pkl"
    model.save(model_filename)
    print(f"✅ Saved: {model_filename}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel: Pullback Prediction (5%+ in 5-15 days)")
    print(f"Type: Ensemble (Random Forest + XGBoost)")
    print(f"Features: {len(feature_columns)}")
    if len(np.unique(y_test)) > 1:
        print(f"\nTest Performance:")
        print(f"  ROC AUC:   {test_roc_auc:.1%}")
        print(f"  Precision: {test_precision:.1%}")
        print(f"  Recall:    {test_recall:.1%}")
    print(f"\nSaved: {model_filename}")
    print("=" * 80)
    
    return model, test_roc_auc if len(np.unique(y_test)) > 1 else None

if __name__ == "__main__":
    try:
        model, roc_auc = train_pullback_model()
        print("\n🎉 SUCCESS!")
        if roc_auc:
            print(f"🎯 Model achieved {roc_auc:.1%} ROC AUC on 2024 test data")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
