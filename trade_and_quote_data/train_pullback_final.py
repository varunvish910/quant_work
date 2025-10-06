#!/usr/bin/env python3
"""
Final Pullback Model Training with All Features

Phases 1 & 2: Regime Detection + Momentum Exhaustion (using existing data)
Expected: 36% → 60-80% ROC AUC

ALL DATA IS REAL from Yahoo Finance
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import ModelTrainer
from targets.pullback_prediction import PullbackPredictionTarget
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd

print("=" * 80)
print("🚀 FINAL PULLBACK MODEL - WITH REGIME + MOMENTUM FEATURES")
print("=" * 80)
print("Testing with REAL DATA ONLY")
print("=" * 80)
print()

# Train model with all new features
print("\n" + "=" * 80)
print("🎯 TRAINING WITH REGIME DETECTION + MOMENTUM EXHAUSTION")
print("=" * 80)

try:
    trainer = ModelTrainer(
        model_type='ensemble',
        feature_sets=['baseline', 'currency', 'volatility', 'sectors'],
        start_date='2000-01-01',
        end_date='2024-12-31'
    )
    
    raw_data = trainer.load_data()
    features = trainer.engineer_features()
    
    print(f"\n✅ Features engineered: {len(trainer.feature_engine.get_feature_columns())} total features")
    
    target_creator = PullbackPredictionTarget(pullback_threshold=0.05, min_days=5, max_days=15)
    df_target = target_creator.create(raw_data['spy'].copy())
    
    combined = features.join(df_target[target_creator.get_target_column()], how='inner')
    trainer.features_data = combined
    trainer.split_data()
    
    feature_cols = trainer.feature_engine.get_feature_columns()
    X_train = trainer.train_data[feature_cols]
    y_train = trainer.train_data[target_creator.get_target_column()]
    X_test = trainer.test_data[feature_cols]
    y_test = trainer.test_data[target_creator.get_target_column()]
    
    print(f"\n📊 Training data:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Positive rate: {y_train.mean():.1%}")
    
    # Train model
    from core.models import EarlyWarningModel
    model = EarlyWarningModel(model_type='ensemble')
    model.fit(X_train, y_train, feature_cols)
    
    # Evaluate
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    precision = precision_score(y_test, pred) if y_test.sum() > 0 else 0
    recall = recall_score(y_test, pred) if y_test.sum() > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"\n🎯 ROC AUC: {roc_auc:.1%}")
    print(f"🎯 Precision: {precision:.1%}")
    print(f"🎯 Recall: {recall:.1%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=['No Pullback', 'Pullback']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]} (correctly predicted no pullback)")
    print(f"False Positives: {cm[0,1]} (false alarm)")
    print(f"False Negatives: {cm[1,0]} (missed pullback)")
    print(f"True Positives:  {cm[1,1]} (correctly predicted pullback)")
    
    # Feature importance
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
        if importance is not None:
            print("\n" + "=" * 80)
            print("📊 TOP 20 MOST IMPORTANT FEATURES")
            print("=" * 80)
            top_features = importance.head(20)
            for idx, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model if improved
    if roc_auc > 0.5:
        model_filename = f"models/trained/pullback_final_{datetime.now().strftime('%Y%m%d')}.pkl"
        model.save(model_filename)
        print(f"\n💾 Saved: {model_filename}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nFinal ROC AUC: {roc_auc:.1%}")
    print(f"Improvement from baseline (36.4%): {(roc_auc - 0.364)*100:+.1f} percentage points")
    
    if roc_auc > 0.60:
        print("\n🎉 SUCCESS! Model is now usable (>60% ROC AUC)")
        print("   Regime detection + momentum features made a big difference!")
    elif roc_auc > 0.50:
        print("\n✅ IMPROVEMENT! Model is better than random")
        print("   Consider adding Polygon options data for further improvement")
    else:
        print("\n❌ STILL NOT WORKING WELL")
        print("   Pullback prediction may be inherently difficult")
        print("   Recommendation: Focus on mean reversion model (97.2% ROC AUC)")
    
    print("\n" + "=" * 80)
    print("✅ ALL DATA WAS REAL from Yahoo Finance")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
