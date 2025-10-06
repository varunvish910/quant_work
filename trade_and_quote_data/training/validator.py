"""
Model Validation Module

Backtesting and validation logic for models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from utils.constants import MUST_CATCH_2024_EVENTS


class ModelValidator:
    """Model validation and backtesting"""
    
    def __init__(self, model, features_data: pd.DataFrame, 
                 feature_columns: List[str], target_column: str = 'early_warning_target'):
        """
        Initialize validator
        
        Args:
            model: Trained model
            features_data: DataFrame with features and targets
            feature_columns: List of feature column names
            target_column: Name of target column
        """
        self.model = model
        self.features_data = features_data
        self.feature_columns = feature_columns
        self.target_column = target_column
    
    def validate_2024_events(self) -> Dict:
        """
        Validate that model catches critical 2024 events 3-5 days early
        
        Returns:
            Dictionary of event validation results
        """
        print("=" * 80)
        print("🚨 VALIDATING 2024 CRITICAL EVENTS")
        print("=" * 80)
        
        results = {}
        
        for event_name, event_info in MUST_CATCH_2024_EVENTS.items():
            print(f"\n{event_name}:")
            print(f"  Event date: {event_info['date']}")
            print(f"  Early warning window: {event_info['early_warning_window']}")
            
            # Get predictions for early warning window
            window_start, window_end = event_info['early_warning_window']
            
            # Filter data for this window
            window_data = self.features_data[
                (self.features_data.index >= window_start) & 
                (self.features_data.index <= window_end)
            ]
            
            if len(window_data) == 0:
                print(f"  ⚠️  No data in warning window")
                results[event_name] = {'caught': False, 'reason': 'no_data'}
                continue
            
            # Make predictions
            X = window_data[self.feature_columns]
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            # Check if model warned during window
            max_prob = probabilities.max()
            any_warning = predictions.sum() > 0
            
            results[event_name] = {
                'caught': any_warning,
                'max_probability': float(max_prob),
                'warning_days': int(predictions.sum()),
                'window_start': window_start,
                'window_end': window_end
            }
            
            if any_warning:
                # Find first warning date
                first_warning_idx = np.where(predictions == 1)[0][0]
                first_warning_date = window_data.index[first_warning_idx]
                results[event_name]['first_warning_date'] = str(first_warning_date.date())
                
                print(f"  ✅ CAUGHT! First warning: {first_warning_date.date()}")
                print(f"  📊 Max probability: {max_prob:.2%}")
                print(f"  📅 Warning days: {predictions.sum()}")
            else:
                print(f"  ❌ MISSED! Max probability: {max_prob:.2%}")
        
        # Summary
        caught_count = sum(1 for r in results.values() if r.get('caught', False))
        total_count = len(results)
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: Caught {caught_count}/{total_count} critical events")
        print("=" * 80)
        
        return results
    
    def backtest(self, start_date: str, end_date: str) -> Dict:
        """
        Backtest model over a date range
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary of backtest results
        """
        print(f"📊 Backtesting from {start_date} to {end_date}...")
        
        # Filter data
        backtest_data = self.features_data[
            (self.features_data.index >= start_date) & 
            (self.features_data.index <= end_date)
        ]
        
        if len(backtest_data) == 0:
            print("⚠️  No data in backtest period")
            return {}
        
        # Make predictions
        X = backtest_data[self.feature_columns]
        y_true = backtest_data[self.target_column]
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'total_days': len(backtest_data),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'precision': float(precision_score(y_true, predictions, zero_division=0)),
            'recall': float(recall_score(y_true, predictions, zero_division=0)),
            'f1': float(f1_score(y_true, predictions, zero_division=0))
        }
        
        if len(np.unique(y_true)) > 1:
            results['roc_auc'] = float(roc_auc_score(y_true, probabilities))
        
        print(f"✅ Backtest complete")
        print(f"   Precision: {results['precision']:.2%}")
        print(f"   Recall: {results['recall']:.2%}")
        print(f"   F1: {results['f1']:.2%}")
        if 'roc_auc' in results:
            print(f"   ROC AUC: {results['roc_auc']:.4f}")
        
        return results


if __name__ == "__main__":
    print("ModelValidator module - use with trained models")

