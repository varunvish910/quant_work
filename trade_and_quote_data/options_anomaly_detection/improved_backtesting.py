#!/usr/bin/env python3
"""
Improved backtesting framework that works with verified data ranges
Uses the corrected target creation approach with proper date handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
warnings.filterwarnings('ignore')

import sys
sys.path.append('../')
from target_creator import CorrectionTargetCreator

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ImprovedBacktestingFramework:
    """
    Improved backtesting framework with verified target creation
    """
    
    def __init__(self, correction_threshold: float = 0.04):
        self.correction_threshold = correction_threshold
        self.target_creator = CorrectionTargetCreator(correction_threshold=correction_threshold)
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def create_dataset_for_period(self, start_date: str, end_date: str, period_name: str) -> Optional[pd.DataFrame]:
        """Create a complete dataset for a specific period with features and targets"""
        print(f"\n🔄 Creating dataset for {period_name} ({start_date} to {end_date})...")
        
        try:
            # Step 1: Load price data and create targets
            price_data = self.target_creator.load_price_data(start_date, end_date)
            corrections = self.target_creator.identify_corrections(price_data)
            targets_df = self.target_creator.create_prediction_targets(corrections)
            
            print(f"   📊 {len(price_data)} days, {len(corrections)} corrections, {targets_df['target'].sum()} targets")
            
            if targets_df['target'].sum() == 0:
                print(f"   ⚠️  No positive targets found for {period_name}")
                return None
            
            # Step 2: Create features from price data
            features_df = self.create_features_from_price_data(price_data, targets_df)
            
            return features_df
            
        except Exception as e:
            print(f"   ❌ Error creating dataset for {period_name}: {e}")
            return None
    
    def create_features_from_price_data(self, price_data: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features from price data"""
        
        # Merge price data with targets
        merged_df = pd.merge(price_data, targets_df, on='date', how='inner')
        
        # Sort by date
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        features_list = []
        
        for i, row in merged_df.iterrows():
            if i < 50:  # Need enough history for features
                continue
                
            # Get historical price data for this date
            hist_prices = merged_df.iloc[max(0, i-50):i+1]['underlying_price'].values
            hist_returns = np.diff(hist_prices) / hist_prices[:-1]
            
            features = {
                'date': row['date'],
                'target': row['target'],
                'underlying_price': row['underlying_price'],
                
                # Price momentum features
                'return_1d': hist_returns[-1] if len(hist_returns) >= 1 else 0,
                'return_5d': np.mean(hist_returns[-5:]) if len(hist_returns) >= 5 else 0,
                'return_10d': np.mean(hist_returns[-10:]) if len(hist_returns) >= 10 else 0,
                'return_20d': np.mean(hist_returns[-20:]) if len(hist_returns) >= 20 else 0,
                
                # Volatility features
                'vol_5d': np.std(hist_returns[-5:]) * np.sqrt(252) if len(hist_returns) >= 5 else 0,
                'vol_10d': np.std(hist_returns[-10:]) * np.sqrt(252) if len(hist_returns) >= 10 else 0,
                'vol_20d': np.std(hist_returns[-20:]) * np.sqrt(252) if len(hist_returns) >= 20 else 0,
                
                # Moving averages
                'sma_5': np.mean(hist_prices[-5:]) if len(hist_prices) >= 5 else hist_prices[-1],
                'sma_10': np.mean(hist_prices[-10:]) if len(hist_prices) >= 10 else hist_prices[-1],
                'sma_20': np.mean(hist_prices[-20:]) if len(hist_prices) >= 20 else hist_prices[-1],
                
                # Price position features
                'price_vs_sma5': (hist_prices[-1] / np.mean(hist_prices[-5:])) - 1 if len(hist_prices) >= 5 else 0,
                'price_vs_sma20': (hist_prices[-1] / np.mean(hist_prices[-20:])) - 1 if len(hist_prices) >= 20 else 0,
                
                # High/low features
                'price_vs_high_20d': (hist_prices[-1] / np.max(hist_prices[-20:])) - 1 if len(hist_prices) >= 20 else 0,
                'drawdown_from_high': (hist_prices[-1] / np.max(hist_prices)) - 1,
                
                # RSI-like features
                'rsi_14d': self.calculate_rsi(hist_returns, 14),
                
                # Trend features
                'trend_5d': 1 if len(hist_prices) >= 5 and hist_prices[-1] > hist_prices[-5] else 0,
                'trend_10d': 1 if len(hist_prices) >= 10 and hist_prices[-1] > hist_prices[-10] else 0,
                
                # Target-related features (for analysis)
                'days_to_correction': row.get('days_to_correction', np.nan),
                'correction_magnitude': row.get('correction_magnitude', np.nan),
            }
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        print(f"   ✅ Created {len(features_df)} feature rows with {features_df['target'].sum()} positive targets")
        
        return features_df
    
    def calculate_rsi(self, returns: np.ndarray, periods: int = 14) -> float:
        """Calculate RSI from returns"""
        if len(returns) < periods:
            return 50.0
        
        recent_returns = returns[-periods:]
        gains = recent_returns[recent_returns > 0].sum()
        losses = abs(recent_returns[recent_returns < 0].sum())
        
        if losses == 0:
            return 100.0
        if gains == 0:
            return 0.0
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features_for_ml(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix for machine learning"""
        
        # Select numeric feature columns
        feature_cols = [col for col in df.columns if col not in ['date', 'target', 'underlying_price', 'days_to_correction', 'correction_magnitude']]
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].fillna(0).values
        y = df['target'].values
        
        print(f"   📊 Feature matrix: {X.shape}, Positive targets: {y.sum()}")
        
        return X, y, feature_cols
    
    def train_and_evaluate_models(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict:
        """Train multiple models and evaluate performance"""
        print(f"\n🤖 Training models...")
        
        # Prepare training data
        X_train, y_train, feature_names = self.prepare_features_for_ml(train_df)
        
        if y_train.sum() == 0:
            print("   ❌ No positive targets in training data")
            return {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'logistic': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   🔄 Training {name}...")
            
            try:
                # Train model
                if name == 'logistic':
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                else:
                    importance = np.ones(len(feature_names))
                
                # Evaluate on validation set if provided
                val_metrics = {}
                if val_df is not None:
                    X_val, y_val, _ = self.prepare_features_for_ml(val_df)
                    if y_val.sum() > 0:
                        if name == 'logistic':
                            X_val_processed = self.scaler.transform(X_val)
                        else:
                            X_val_processed = X_val
                        
                        y_pred = model.predict(X_val_processed)
                        y_proba = model.predict_proba(X_val_processed)[:, 1]
                        
                        val_metrics = {
                            'auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5,
                            'precision': precision_score(y_val, y_pred) if y_pred.sum() > 0 else 0,
                            'recall': recall_score(y_val, y_pred) if y_val.sum() > 0 else 0
                        }
                        
                        print(f"      📈 AUC: {val_metrics['auc']:.3f}, Precision: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}")
                
                results[name] = {
                    'model': model,
                    'feature_names': feature_names,
                    'feature_importance': importance,
                    'validation_metrics': val_metrics,
                    'scaler': self.scaler if name == 'logistic' else None
                }
                
            except Exception as e:
                print(f"      ❌ {name} failed: {e}")
        
        # Select best model based on validation AUC
        if results and val_df is not None:
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x]['validation_metrics'].get('auc', 0))
            self.best_model = best_model_name
            print(f"   🏆 Best model: {best_model_name}")
        
        self.models = results
        return results
    
    def predict_period(self, test_df: pd.DataFrame, period_name: str) -> Dict:
        """Make predictions on test period"""
        print(f"\n🔮 Predicting {period_name}...")
        
        if not self.best_model or self.best_model not in self.models:
            print("   ❌ No trained model available")
            return {}
        
        X_test, y_test, _ = self.prepare_features_for_ml(test_df)
        model_info = self.models[self.best_model]
        model = model_info['model']
        
        # Apply scaling if needed
        if model_info['scaler']:
            X_test_processed = model_info['scaler'].transform(X_test)
        else:
            X_test_processed = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Create results
        results_df = test_df[['date']].copy()
        results_df['actual_target'] = y_test
        results_df['predicted_target'] = y_pred
        results_df['correction_probability'] = y_proba
        results_df['risk_level'] = pd.cut(y_proba, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
        
        # Calculate metrics if we have actual targets
        metrics = {}
        if y_test.sum() > 0:
            metrics = {
                'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
                'precision': precision_score(y_test, y_pred) if y_pred.sum() > 0 else 0,
                'recall': recall_score(y_test, y_pred) if y_test.sum() > 0 else 0,
                'true_positives': ((y_test == 1) & (y_pred == 1)).sum(),
                'false_positives': ((y_test == 0) & (y_pred == 1)).sum(),
                'false_negatives': ((y_test == 1) & (y_pred == 0)).sum(),
            }
            
            print(f"   📈 {period_name} Results:")
            print(f"      AUC: {metrics['auc']:.3f}")
            print(f"      Precision: {metrics['precision']:.3f}")  
            print(f"      Recall: {metrics['recall']:.3f}")
            print(f"      True Positives: {metrics['true_positives']}")
            print(f"      False Positives: {metrics['false_positives']}")
        
        return {
            'predictions_df': results_df,
            'metrics': metrics,
            'model_name': self.best_model
        }
    
    def run_complete_backtest(self) -> Dict:
        """Run the complete backtesting process"""
        print("🚀 IMPROVED BACKTESTING FRAMEWORK")
        print("=" * 50)
        
        # Define periods that we know have corrections
        periods = {
            'train_2020': ('2020-01-01', '2020-12-31'),
            'train_2022': ('2022-01-01', '2022-12-31'),
            'val_2024': ('2024-01-01', '2024-12-31'),
            'test_2025': ('2025-01-01', '2025-08-31'),
            'predict_sept': ('2025-09-01', '2025-09-30')
        }
        
        datasets = {}
        
        # Create datasets for each period
        for period_name, (start_date, end_date) in periods.items():
            dataset = self.create_dataset_for_period(start_date, end_date, period_name)
            datasets[period_name] = dataset
        
        # Combine training data
        train_datasets = [datasets['train_2020'], datasets['train_2022']]
        train_datasets = [df for df in train_datasets if df is not None]
        
        if not train_datasets:
            print("❌ No training data available")
            return {}
        
        combined_train = pd.concat(train_datasets, ignore_index=True)
        print(f"\n📊 Combined training data: {len(combined_train)} samples, {combined_train['target'].sum()} positive targets")
        
        # Train models using validation data
        val_df = datasets['val_2024']
        model_results = self.train_and_evaluate_models(combined_train, val_df)
        
        if not model_results:
            print("❌ Model training failed")
            return {}
        
        # Test on 2025 data
        test_results = {}
        if datasets['test_2025'] is not None:
            test_results['test_2025'] = self.predict_period(datasets['test_2025'], 'Test 2025')
        
        # Predict September 2025
        if datasets['predict_sept'] is not None:
            test_results['predict_sept'] = self.predict_period(datasets['predict_sept'], 'September 2025')
        else:
            # Create prediction for Sept even without actual targets
            sept_price_data = self.target_creator.load_price_data('2025-09-01', '2025-09-30')
            if len(sept_price_data) > 0:
                # Create dummy targets (all 0) for prediction
                dummy_targets = pd.DataFrame({
                    'date': sept_price_data['date'],
                    'target': 0,
                    'days_to_correction': np.nan,
                    'correction_magnitude': np.nan
                })
                sept_features = self.create_features_from_price_data(sept_price_data, dummy_targets)
                test_results['predict_sept'] = self.predict_period(sept_features, 'September 2025 (Prediction)')
        
        return {
            'datasets': datasets,
            'models': model_results,
            'predictions': test_results
        }

def main():
    """Run the improved backtesting framework"""
    framework = ImprovedBacktestingFramework(correction_threshold=0.04)
    
    results = framework.run_complete_backtest()
    
    if results:
        print(f"\n🎉 BACKTESTING COMPLETE!")
        print(f"📊 Results Summary:")
        
        if 'predictions' in results:
            for period, pred_results in results['predictions'].items():
                if pred_results and 'metrics' in pred_results:
                    metrics = pred_results['metrics']
                    if metrics:
                        print(f"   {period}: AUC={metrics.get('auc', 0):.3f}, Precision={metrics.get('precision', 0):.3f}")
                
                # Show September predictions
                if period == 'predict_sept' and 'predictions_df' in pred_results:
                    pred_df = pred_results['predictions_df']
                    high_risk_days = (pred_df['risk_level'] == 'High').sum()
                    max_prob = pred_df['correction_probability'].max()
                    print(f"   September 2025: {high_risk_days} high-risk days, max probability: {max_prob:.3f}")
        
        # Export results
        output_dir = Path("analysis/improved_backtesting_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export September predictions
        if 'predictions' in results and 'predict_sept' in results['predictions']:
            sept_results = results['predictions']['predict_sept']
            if 'predictions_df' in sept_results:
                output_file = output_dir / f"september_2025_predictions_{timestamp}.csv"
                sept_results['predictions_df'].to_csv(output_file, index=False)
                print(f"💾 Exported September predictions: {output_file}")
        
        print(f"✅ Improved backtesting framework completed successfully!")
    else:
        print("❌ Backtesting failed")

if __name__ == "__main__":
    main()