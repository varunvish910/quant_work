#!/usr/bin/env python3
"""
Train Models for All Targets

This script trains separate models for each target type:
- Early Warning: Predict 5% drawdowns 3-13 days ahead
- Mean Reversion: Predict bounces after pullbacks
- (Add more targets as you implement them)

Usage:
    python3 train_all_targets.py
"""

from training.multi_target_trainer import MultiTargetTrainer

def main():
    print("=" * 80)
    print("🎯 TRAINING MODELS FOR ALL TARGETS")
    print("=" * 80)
    print()
    print("This will train separate models for:")
    print("  1. Early Warning - Predict drawdowns before they happen")
    print("  2. Mean Reversion - Predict bounces after pullbacks")
    print()
    print("Each model will be:")
    print("  - Trained on 2000-2022 data")
    print("  - Validated on 2023 data")
    print("  - Tested on 2024 data")
    print("=" * 80)
    
    # Create multi-target trainer
    trainer = MultiTargetTrainer(
        model_type='ensemble',  # Random Forest + XGBoost
        feature_sets=['baseline', 'currency', 'volatility'],
        start_date='2000-01-01',
        end_date='2024-12-31'
    )
    
    # Train models for all targets
    results = trainer.train_all_targets(
        targets=['early_warning', 'mean_reversion'],
        save_models=True
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    
    for target_type, result in results.items():
        print(f"\n{target_type.upper()}:")
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            metrics = result['test_metrics']
            print(f"  ✅ ROC AUC: {metrics.get('roc_auc', 0):.3f}")
            print(f"  ✅ Precision: {metrics.get('precision', 0):.3f}")
            print(f"  ✅ Recall: {metrics.get('recall', 0):.3f}")
            
            print(f"\n  Top 5 Features:")
            top_features = result['feature_importance'].head(5)
            for idx, row in top_features.iterrows():
                print(f"    {row['feature']}: {row['importance']:.1%}")
    
    # Find best model
    print("\n" + "=" * 80)
    print("🏆 BEST MODEL")
    print("=" * 80)
    
    best_target, best_model, best_score = trainer.get_best_model(metric='roc_auc')
    print(f"Target: {best_target}")
    print(f"ROC AUC: {best_score:.3f}")
    print(f"Model Type: {trainer.model_type}")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\nModels saved in: models/trained/")
    print("\nYou now have:")
    print("  ✅ Separate model for each target")
    print("  ✅ Performance comparison")
    print("  ✅ Feature importance for each")
    print("\nUse the best model for your specific use case!")
    print("=" * 80)
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
