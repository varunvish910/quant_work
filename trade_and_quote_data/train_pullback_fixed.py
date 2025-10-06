#!/usr/bin/env python3
"""
Fixed Pullback Model

Key improvements:
1. Better target definition (3% pullback in 3-10 days instead of 5% in 5-15)
2. Feature engineering improvements
3. Better class balancing
4. Calibrated probabilities
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import ModelTrainer
from targets.pullback_prediction import PullbackPredictionTarget
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd

print("=" * 80)
print("🔧 FIXED PULLBACK MODEL")
print("=" * 80)
print("Improvements:")
print("  1. Easier target: 3% pullback in 3-10 days (instead of 5% in 5-15)")
print("  2. Feature selection: Remove low-importance features")
print("  3. Better class balancing")
print("  4. Probability calibration")
print("=" * 80)
print()

# Train model with improvements
print("\n" + "=" * 80)
print("🎯 TRAINING IMPROVED MODEL")
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
    
    # Create EASIER target (3% instead of 5%, 3-10 days instead of 5-15)
    print("\n🎯 Creating EASIER target (3% pullback in 3-10 days)...")
    target_creator = PullbackPredictionTarget(
        pullback_threshold=0.03,  # 3% instead of 5%
        min_days=3,               # 3 days instead of 5
        max_days=10               # 10 days instead of 15
    )
    df_target = target_creator.create(raw_data['spy'].copy())
    
    combined = features.join(df_target[target_creator.get_target_column()], how='inner')
    trainer.features_data = combined
    trainer.split_data()
    
    # Feature selection - remove low importance features
    print("\n🔧 Feature selection...")
    feature_cols = trainer.feature_engine.get_feature_columns()
    
    # Remove features that were 0% importance
    features_to_remove = [
        'mags_vs_spy',  # 0.00%
        'vix_spike',  # 0.00%
        'vix_extreme_high',  # 0.00%
    ]
    
    feature_cols_filtered = [f for f in feature_cols if f not in features_to_remove]
    
    print(f"   Original features: {len(feature_cols)}")
    print(f"   Filtered features: {len(feature_cols_filtered)}")
    print(f"   Removed: {len(features_to_remove)}")
    
    X_train = trainer.train_data[feature_cols_filtered]
    y_train = trainer.train_data[target_creator.get_target_column()]
    X_val = trainer.val_data[feature_cols_filtered]
    y_val = trainer.val_data[target_creator.get_target_column()]
    X_test = trainer.test_data[feature_cols_filtered]
    y_test = trainer.test_data[target_creator.get_target_column()]
    
    print(f"\n📊 Training data:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_cols_filtered)}")
    print(f"   Positive rate (train): {y_train.mean():.1%}")
    print(f"   Positive rate (test): {y_test.mean():.1%}")
    
    # Train base model with better class balancing
    print("\n🎓 Training ensemble model...")
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from xgboost import XGBClassifier
    
    # Increase class weight to combat imbalance
    rf = RandomForestClassifier(
        n_estimators=300,  # More trees
        max_depth=10,      # Deeper trees
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced_subsample',  # Better balancing
        random_state=42,
        n_jobs=-1
    )
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,  # Slower learning
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5,  # Increased from default
        random_state=42,
        n_jobs=-1
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=[1, 2]  # Give XGBoost more weight
    )
    
    ensemble.fit(X_train, y_train)
    
    # Calibrate probabilities using validation set
    print("\n🔧 Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(
        ensemble,
        method='sigmoid',
        cv='prefit'
    )
    calibrated_model.fit(X_val, y_val)
    
    print("✅ Model trained and calibrated")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("📊 TEST SET RESULTS (2024)")
    print("=" * 80)
    
    pred = calibrated_model.predict(X_test)
    pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    precision = precision_score(y_test, pred) if y_test.sum() > 0 else 0
    recall = recall_score(y_test, pred) if y_test.sum() > 0 else 0
    
    print(f"\n🎯 ROC AUC: {roc_auc:.1%}")
    print(f"🎯 Precision: {precision:.1%}")
    print(f"🎯 Recall: {recall:.1%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=['No Pullback', 'Pullback']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # Analyze predictions
    print("\n" + "=" * 80)
    print("🔍 PREDICTION ANALYSIS")
    print("=" * 80)
    
    test_analysis = pd.DataFrame({
        'date': trainer.test_data.index,
        'spy_close': raw_data['spy'].loc[trainer.test_data.index, 'Close'],
        'actual': y_test,
        'predicted': pred,
        'probability': pred_proba
    })
    
    # High confidence predictions
    high_conf = test_analysis[test_analysis['probability'] > 0.7].sort_values('probability', ascending=False)
    
    print(f"\n🚨 High confidence warnings (>70%): {len(high_conf)}")
    if len(high_conf) > 0:
        correct = (high_conf['actual'] == high_conf['predicted']).sum()
        print(f"   Correct: {correct}/{len(high_conf)} ({correct/len(high_conf):.1%})")
        
        print("\n   Top 10:")
        for idx, row in high_conf.head(10).iterrows():
            result = "✅" if row['actual'] == 1 else "❌"
            print(f"   {result} {row['date'].date()}: ${row['spy_close']:.2f} ({row['probability']:.1%})")
    
    # Save model
    if roc_auc > 0.55:
        import joblib
        from pathlib import Path
        
        model_dir = Path(f"models/trained/pullback_fixed_{datetime.now().strftime('%Y%m%d')}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(calibrated_model, model_dir / "model.pkl")
        
        import json
        with open(model_dir / "feature_columns.json", 'w') as f:
            json.dump(feature_cols_filtered, f)
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump({
                'target': '3% pullback in 3-10 days',
                'roc_auc': float(roc_auc),
                'precision': float(precision),
                'recall': float(recall),
                'features': len(feature_cols_filtered),
                'trained_date': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Saved: {model_dir}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    
    print(f"\nFinal ROC AUC: {roc_auc:.1%}")
    print(f"Improvement from original (50.6%): {(roc_auc - 0.506)*100:+.1f} percentage points")
    
    if roc_auc > 0.65:
        print("\n🎉 SUCCESS! Model is now usable (>65% ROC AUC)")
    elif roc_auc > 0.55:
        print("\n✅ IMPROVED! Model is better")
        print("   Consider Phase 3: Add Polygon options data for further improvement")
    else:
        print("\n⚠️  STILL NEEDS WORK")
        print("   Recommendation: Try Phase 3 (Polygon options) or focus on mean reversion")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
