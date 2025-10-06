#!/usr/bin/env python3
"""
Complete Model Training with All Features

This script trains models with ALL new features integrated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from training.multi_target_trainer import MultiTargetTrainer

def main():
    print("="*80)
    print("🚀 TRAINING WITH ALL FEATURES")
    print("="*80)
    print()
    print("Features included:")
    print("  ✅ Baseline (8)")
    print("  ✅ Technical (momentum, volatility, MA, volume, trend)")
    print("  ✅ Market (sector rotation, rotation indicators)")
    print("  ✅ Currency (USD/JPY)")
    print("  ✅ Volatility (VIX)")
    print()
    print("Targets:")
    print("  ✅ Early Warning")
    print("  ✅ Mean Reversion")
    print()
    print("This will take 30-60 minutes...")
    print("="*80)
    print()
    
    # Create trainer with all feature sets
    trainer = MultiTargetTrainer(
        model_type='ensemble',
        feature_sets=['baseline', 'currency', 'volatility'],
        start_date='2000-01-01',
        end_date='2024-12-31'
    )
    
    # Train all targets
    results = trainer.train_all_targets(
        targets=['early_warning', 'mean_reversion'],
        save_models=True
    )
    
    print()
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print()
    print("Models saved in: models/trained/")
    print()
    
    # Get best model
    best_target, best_model, best_score = trainer.get_best_model(metric='roc_auc')
    print(f"🏆 Best Model: {best_target}")
    print(f"   ROC AUC: {best_score:.3f}")
    print()
    
    return results

if __name__ == "__main__":
    results = main()
