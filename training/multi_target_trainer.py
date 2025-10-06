"""
Multi-Target Model Trainer

Trains separate models for each target type and compares performance.
This allows you to:
1. Train one model per target
2. Compare which targets work best
3. Use different models for different prediction tasks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import joblib
from pathlib import Path

from training.trainer import ModelTrainer
from targets.early_warning import EarlyWarningTarget
from targets.mean_reversion import MeanReversionTarget
# Import other targets as needed


class MultiTargetTrainer:
    """Train and compare models across multiple targets"""
    
    def __init__(self,
                 model_type: str = 'ensemble',
                 feature_sets: List[str] = None,
                 start_date: str = '2000-01-01',
                 end_date: str = '2024-12-31'):
        """
        Initialize multi-target trainer
        
        Args:
            model_type: Type of model ('rf', 'xgboost', 'ensemble')
            feature_sets: Feature sets to use
            start_date: Start date for data
            end_date: End date for data
        """
        self.model_type = model_type
        self.feature_sets = feature_sets or ['baseline', 'currency', 'volatility']
        self.start_date = start_date
        self.end_date = end_date
        
        # Storage for models and results
        self.models = {}
        self.results = {}
        self.trainers = {}
    
    def train_all_targets(self, 
                         targets: List[str] = None,
                         save_models: bool = True) -> Dict:
        """
        Train separate models for each target
        
        Args:
            targets: List of target types to train on
                    Options: ['early_warning', 'mean_reversion', 'pullback']
                    If None, trains on all available targets
            save_models: Whether to save trained models
            
        Returns:
            Dictionary with results for each target
        """
        if targets is None:
            targets = ['early_warning', 'mean_reversion']  # Add more as implemented
        
        print("=" * 80)
        print("🎯 MULTI-TARGET TRAINING")
        print("=" * 80)
        print(f"Training {len(targets)} separate models:")
        for t in targets:
            print(f"   - {t}")
        print("=" * 80)
        
        # Train a model for each target
        for target_type in targets:
            print(f"\n{'='*80}")
            print(f"🎯 TRAINING MODEL FOR: {target_type.upper()}")
            print(f"{'='*80}")
            
            try:
                # Create trainer for this target
                trainer = ModelTrainer(
                    model_type=self.model_type,
                    feature_sets=self.feature_sets,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                # Train model
                model = trainer.train(
                    target_type=target_type,
                    save_model=False  # We'll save with custom names
                )
                
                # Evaluate on test set
                test_metrics = trainer.evaluate_test_set(target_type=target_type)
                
                # Store results
                self.models[target_type] = model
                self.trainers[target_type] = trainer
                self.results[target_type] = {
                    'test_metrics': test_metrics,
                    'feature_importance': trainer.get_feature_importance(top_n=10)
                }
                
                # Save model with target-specific name
                if save_models:
                    self._save_model(model, target_type)
                
                print(f"\n✅ {target_type.upper()} model complete!")
                
            except Exception as e:
                print(f"\n❌ Failed to train {target_type}: {e}")
                self.results[target_type] = {'error': str(e)}
        
        # Print comparison
        self._print_comparison()
        
        return self.results
    
    def _save_model(self, model, target_type: str):
        """Save model with target-specific filename"""
        output_dir = Path('models/trained')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f'{target_type}_{self.model_type}_{datetime.now().strftime("%Y%m%d")}.pkl'
        filepath = output_dir / filename
        
        joblib.dump(model, filepath)
        print(f"💾 Saved: {filepath}")
    
    def _print_comparison(self):
        """Print comparison table of all models"""
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON")
        print("=" * 80)
        
        comparison_data = []
        for target_type, result in self.results.items():
            if 'error' in result:
                continue
            
            metrics = result['test_metrics']
            comparison_data.append({
                'Target': target_type,
                'ROC AUC': f"{metrics.get('roc_auc', 0):.3f}",
                'Precision': f"{metrics.get('precision', 0):.3f}",
                'Recall': f"{metrics.get('recall', 0):.3f}",
            })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
        else:
            print("No successful models to compare")
        
        print("=" * 80)
    
    def get_best_model(self, metric: str = 'roc_auc') -> tuple:
        """
        Get the best performing model based on a metric
        
        Args:
            metric: Metric to optimize ('roc_auc', 'precision', 'recall')
            
        Returns:
            Tuple of (target_type, model, score)
        """
        best_score = -1
        best_target = None
        best_model = None
        
        for target_type, result in self.results.items():
            if 'error' in result:
                continue
            
            score = result['test_metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_target = target_type
                best_model = self.models[target_type]
        
        return best_target, best_model, best_score
    
    def predict_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions from all models
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with predictions from each model
        """
        predictions = pd.DataFrame(index=data.index)
        
        for target_type, model in self.models.items():
            trainer = self.trainers[target_type]
            feature_columns = trainer.feature_engine.get_feature_columns()
            
            X = data[feature_columns]
            proba = model.predict_proba(X)[:, 1]
            
            predictions[f'{target_type}_probability'] = proba
            predictions[f'{target_type}_prediction'] = (proba > 0.5).astype(int)
        
        return predictions


def example_usage():
    """Example of how to use MultiTargetTrainer"""
    print("🎯 Multi-Target Training Example")
    print("=" * 80)
    
    # Create trainer
    trainer = MultiTargetTrainer(
        model_type='ensemble',
        feature_sets=['baseline', 'currency', 'volatility'],
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # Train models for all targets
    results = trainer.train_all_targets(
        targets=['early_warning', 'mean_reversion'],
        save_models=True
    )
    
    # Get best model
    best_target, best_model, best_score = trainer.get_best_model(metric='roc_auc')
    print(f"\n🏆 Best model: {best_target} (ROC AUC: {best_score:.3f})")
    
    # Get predictions from all models
    from core.data_loader import DataLoader
    loader = DataLoader(start_date='2024-01-01', end_date='2024-10-01')
    data = loader.load_all_data()
    
    # Calculate features
    from engines.unified_engine import UnifiedFeatureEngine
    engine = UnifiedFeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])
    features = engine.calculate_features(**data)
    
    # Get predictions from all models
    all_predictions = trainer.predict_all(features)
    print("\n📊 Predictions from all models:")
    print(all_predictions.tail())
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = example_usage()
