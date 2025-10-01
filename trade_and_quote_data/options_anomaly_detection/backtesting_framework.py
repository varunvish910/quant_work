#!/usr/bin/env python3
"""
Comprehensive backtesting framework for SPY options anomaly detection
Uses 2016-2023 for training, 2024 for validation, 2025 for testing, and predicts remaining Sept 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
warnings.filterwarnings('ignore')

from target_creator import CorrectionTargetCreator
# Note: We'll implement our own simplified feature extraction since the existing modules have import issues

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestingFramework:
    """
    Comprehensive backtesting framework for options anomaly detection
    
    Data Split Strategy:
    - Training: 2016-2023 (8 years of data)
    - Validation: 2024 (1 year for hyperparameter tuning)
    - Test: First half of 2025 (real-world performance)
    - Prediction: Remaining September 2025
    """
    
    def __init__(self, data_dir: str = "data", correction_threshold: float = 0.04):
        self.data_dir = Path(data_dir)
        self.correction_threshold = correction_threshold
        
        # Initialize components
        self.target_creator = CorrectionTargetCreator(correction_threshold=correction_threshold)
        
        # Data containers
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.prediction_data = None
        
        # Models
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
        # Results
        self.results = {
            'training': {},
            'validation': {},
            'testing': {},
            'prediction': {}
        }
        
    def prepare_historical_data(self):
        """Load and prepare all historical data with proper time splits"""
        print("📊 PREPARING HISTORICAL DATA")
        print("=" * 50)
        
        # Define date ranges
        date_ranges = {
            'training': ('2016-01-01', '2023-12-31'),
            'validation': ('2024-01-01', '2024-12-31'), 
            'testing': ('2025-01-01', '2025-08-31'),  # First 8 months of 2025
            'prediction': ('2025-09-01', '2025-09-30')  # September 2025
        }
        
        # Load SPY summary data for each period
        all_datasets = {}
        
        for period, (start_date, end_date) in date_ranges.items():
            print(f"\n🔄 Loading {period} data ({start_date} to {end_date})...")
            
            try:
                # Load price data and create targets
                price_data = self.target_creator.load_price_data(start_date, end_date)
                corrections = self.target_creator.identify_corrections(price_data)
                targets_df = self.target_creator.create_prediction_targets(corrections)
                
                print(f"   📈 Price data: {len(price_data)} days")
                print(f"   🎯 Corrections found: {len(corrections)}")
                print(f"   ⚠️  Prediction targets: {targets_df['target'].sum()}")
                
                # Store dataset info
                all_datasets[period] = {
                    'price_data': price_data,
                    'corrections': corrections,
                    'targets': targets_df,
                    'date_range': (start_date, end_date)
                }
                
            except Exception as e:
                print(f"   ❌ Error loading {period} data: {e}")
                all_datasets[period] = None
        
        # Store datasets
        self.train_data = all_datasets['training']
        self.val_data = all_datasets['validation']
        self.test_data = all_datasets['testing']
        self.prediction_data = all_datasets['prediction']
        
        # Summary statistics
        print(f"\n📊 DATA SUMMARY")
        print("-" * 30)
        for period, data in all_datasets.items():
            if data:
                targets = data['targets']['target'].sum()
                total = len(data['targets'])
                ratio = targets / total if total > 0 else 0
                print(f"{period:>10}: {total:>4} days, {targets:>2} targets ({ratio:.1%})")
        
        return all_datasets
    
    def extract_features_for_period(self, period_name: str, data: Dict) -> Optional[pd.DataFrame]:
        """Extract features for a specific time period using summary data"""
        if not data:
            print(f"❌ No data available for {period_name}")
            return None
        
        print(f"🔧 Extracting features for {period_name} period...")
        
        try:
            # Get date range
            start_date, end_date = data['date_range']
            targets_df = data['targets']
            
            # Create feature vectors from summary data (we don't have individual options contracts)
            # Use the price data and summary statistics to create predictive features
            features_list = []
            
            for _, row in targets_df.iterrows():
                date = row['date']
                target = row['target']
                
                # Create features from price action and available data
                features = {
                    'date': date,
                    'target': target,
                    'underlying_price': self._get_price_for_date(data['price_data'], date),
                    
                    # Price-based momentum features
                    'price_momentum_5d': self._calculate_price_momentum(data['price_data'], date, 5),
                    'price_momentum_10d': self._calculate_price_momentum(data['price_data'], date, 10),
                    'price_momentum_20d': self._calculate_price_momentum(data['price_data'], date, 20),
                    
                    # Volatility features
                    'volatility_5d': self._calculate_volatility(data['price_data'], date, 5),
                    'volatility_10d': self._calculate_volatility(data['price_data'], date, 10),
                    'volatility_20d': self._calculate_volatility(data['price_data'], date, 20),
                    
                    # Technical indicators
                    'rsi_14d': self._calculate_rsi(data['price_data'], date, 14),
                    'sma_20d_ratio': self._calculate_sma_ratio(data['price_data'], date, 20),
                    'sma_50d_ratio': self._calculate_sma_ratio(data['price_data'], date, 50),
                    
                    # Distance to correction features
                    'days_to_correction': row.get('days_to_correction', np.nan),
                    'correction_magnitude': row.get('correction_magnitude', np.nan),
                    
                    # Market regime features
                    'price_near_high_20d': self._price_near_high(data['price_data'], date, 20),
                    'price_near_high_50d': self._price_near_high(data['price_data'], date, 50),
                    'drawdown_from_high': self._current_drawdown(data['price_data'], date),
                }
                
                features_list.append(features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            features_df['date'] = pd.to_datetime(features_df['date'])
            
            # Remove rows with insufficient data (NaN features)
            initial_rows = len(features_df)
            features_df = features_df.dropna(subset=[col for col in features_df.columns 
                                                   if col not in ['target', 'date', 'days_to_correction', 'correction_magnitude']])
            
            print(f"   ✅ Extracted features: {len(features_df)} rows (removed {initial_rows - len(features_df)} with missing data)")
            print(f"   📊 Feature count: {len(features_df.columns) - 2}")  # Exclude date and target
            
            return features_df
            
        except Exception as e:
            print(f"   ❌ Error extracting features for {period_name}: {e}")
            return None
    
    def _get_price_for_date(self, price_data: pd.DataFrame, date) -> float:
        """Get SPY price for a specific date"""
        try:
            mask = price_data['date'] == pd.to_datetime(date)
            if mask.any():
                return price_data[mask]['underlying_price'].iloc[0]
        except:
            pass
        return np.nan
    
    def _calculate_price_momentum(self, price_data: pd.DataFrame, date, periods: int) -> float:
        """Calculate price momentum over specified periods"""
        try:
            date_idx = price_data[price_data['date'] == pd.to_datetime(date)].index
            if len(date_idx) == 0 or date_idx[0] < periods:
                return np.nan
            
            current_price = price_data.iloc[date_idx[0]]['underlying_price']
            past_price = price_data.iloc[date_idx[0] - periods]['underlying_price']
            
            return (current_price - past_price) / past_price
        except:
            return np.nan
    
    def _calculate_volatility(self, price_data: pd.DataFrame, date, periods: int) -> float:
        """Calculate price volatility over specified periods"""
        try:
            date_idx = price_data[price_data['date'] == pd.to_datetime(date)].index
            if len(date_idx) == 0 or date_idx[0] < periods:
                return np.nan
            
            end_idx = date_idx[0]
            start_idx = max(0, end_idx - periods)
            
            prices = price_data.iloc[start_idx:end_idx + 1]['underlying_price']
            returns = prices.pct_change().dropna()
            
            return returns.std() * np.sqrt(252)  # Annualized volatility
        except:
            return np.nan
    
    def _calculate_rsi(self, price_data: pd.DataFrame, date, periods: int) -> float:
        """Calculate RSI"""
        try:
            date_idx = price_data[price_data['date'] == pd.to_datetime(date)].index
            if len(date_idx) == 0 or date_idx[0] < periods:
                return np.nan
            
            end_idx = date_idx[0]
            start_idx = max(0, end_idx - periods)
            
            prices = price_data.iloc[start_idx:end_idx + 1]['underlying_price']
            returns = prices.pct_change().dropna()
            
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            
            if losses == 0:
                return 100
            
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return 50
    
    def _calculate_sma_ratio(self, price_data: pd.DataFrame, date, periods: int) -> float:
        """Calculate ratio of current price to SMA"""
        try:
            date_idx = price_data[price_data['date'] == pd.to_datetime(date)].index
            if len(date_idx) == 0 or date_idx[0] < periods:
                return np.nan
            
            end_idx = date_idx[0]
            start_idx = max(0, end_idx - periods)
            
            current_price = price_data.iloc[end_idx]['underlying_price']
            sma = price_data.iloc[start_idx:end_idx + 1]['underlying_price'].mean()
            
            return current_price / sma
        except:
            return 1.0
    
    def _price_near_high(self, price_data: pd.DataFrame, date, periods: int) -> float:
        """Check if price is near recent high"""
        try:
            date_idx = price_data[price_data['date'] == pd.to_datetime(date)].index
            if len(date_idx) == 0 or date_idx[0] < periods:
                return np.nan
            
            end_idx = date_idx[0]
            start_idx = max(0, end_idx - periods)
            
            current_price = price_data.iloc[end_idx]['underlying_price']
            high_price = price_data.iloc[start_idx:end_idx + 1]['underlying_price'].max()
            
            return current_price / high_price
        except:
            return 1.0
    
    def _current_drawdown(self, price_data: pd.DataFrame, date) -> float:
        """Calculate current drawdown from recent high"""
        try:
            date_idx = price_data[price_data['date'] == pd.to_datetime(date)].index
            if len(date_idx) == 0:
                return np.nan
            
            end_idx = date_idx[0]
            
            current_price = price_data.iloc[end_idx]['underlying_price']
            high_price = price_data.iloc[:end_idx + 1]['underlying_price'].max()
            
            return (current_price - high_price) / high_price
        except:
            return 0.0
    
    def train_models(self, train_features: pd.DataFrame) -> Dict:
        """Train multiple models on training data"""
        print("\n🤖 TRAINING MODELS")
        print("=" * 30)
        
        # Prepare features and targets
        feature_cols = [col for col in train_features.columns 
                       if col not in ['date', 'target', 'days_to_correction', 'correction_magnitude']]
        
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features['target']
        
        print(f"📊 Training data: {len(X_train)} samples, {y_train.sum()} positive targets")
        print(f"   Feature count: {len(feature_cols)}")
        print(f"   Target ratio: {y_train.mean():.3f}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define models to train
        model_configs = {
            'logistic': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        
        for name, model in model_configs.items():
            print(f"\n   🔄 Training {name}...")
            
            try:
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
                    importance = np.ones(len(feature_cols))
                
                trained_models[name] = {
                    'model': model,
                    'feature_names': feature_cols,
                    'feature_importance': importance,
                    'scaler': self.scaler if name == 'logistic' else None
                }
                
                print(f"      ✅ {name} trained successfully")
                
            except Exception as e:
                print(f"      ❌ {name} training failed: {e}")
        
        self.models = trained_models
        return trained_models
    
    def validate_models(self, val_features: pd.DataFrame) -> Dict:
        """Validate models on validation data"""
        print("\n🔍 VALIDATING MODELS")
        print("=" * 30)
        
        if not self.models:
            print("❌ No trained models available")
            return {}
        
        # Prepare validation data
        feature_cols = self.models[list(self.models.keys())[0]]['feature_names']
        X_val = val_features[feature_cols].fillna(0)
        y_val = val_features['target']
        
        print(f"📊 Validation data: {len(X_val)} samples, {y_val.sum()} positive targets")
        
        validation_results = {}
        
        for name, model_info in self.models.items():
            print(f"\n   🧪 Validating {name}...")
            
            try:
                model = model_info['model']
                
                # Apply scaling if needed
                if model_info['scaler']:
                    X_val_processed = model_info['scaler'].transform(X_val)
                else:
                    X_val_processed = X_val
                
                # Make predictions
                y_pred = model.predict(X_val_processed)
                y_proba = model.predict_proba(X_val_processed)[:, 1]
                
                # Calculate metrics
                if len(np.unique(y_val)) > 1:
                    auc_score = roc_auc_score(y_val, y_proba)
                    precision = precision_score(y_val, y_pred) if y_pred.sum() > 0 else 0
                    recall = recall_score(y_val, y_pred) if y_val.sum() > 0 else 0
                else:
                    auc_score = 0.5
                    precision = 0
                    recall = 0
                
                validation_results[name] = {
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'true_values': y_val,
                    'auc_score': auc_score,
                    'precision': precision,
                    'recall': recall
                }
                
                print(f"      📈 AUC: {auc_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                
            except Exception as e:
                print(f"      ❌ {name} validation failed: {e}")
        
        # Select best model based on AUC score
        if validation_results:
            best_model_name = max(validation_results.keys(), 
                                key=lambda x: validation_results[x]['auc_score'])
            self.best_model = best_model_name
            print(f"\n🏆 Best model: {best_model_name} (AUC: {validation_results[best_model_name]['auc_score']:.3f})")
        
        self.results['validation'] = validation_results
        return validation_results
    
    def test_model(self, test_features: pd.DataFrame) -> Dict:
        """Test best model on held-out test data"""
        print("\n🎯 TESTING BEST MODEL")
        print("=" * 30)
        
        if not self.best_model or self.best_model not in self.models:
            print("❌ No best model available for testing")
            return {}
        
        # Prepare test data
        feature_cols = self.models[self.best_model]['feature_names']
        X_test = test_features[feature_cols].fillna(0)
        y_test = test_features['target']
        
        print(f"📊 Test data: {len(X_test)} samples, {y_test.sum()} positive targets")
        print(f"🤖 Using model: {self.best_model}")
        
        try:
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
            
            # Calculate metrics
            if len(np.unique(y_test)) > 1:
                auc_score = roc_auc_score(y_test, y_proba)
                print(f"\n📈 Test Results:")
                print(f"   AUC Score: {auc_score:.3f}")
                print(f"   Predictions: {y_pred.sum()} positive out of {len(y_pred)}")
                print(f"   Actual: {y_test.sum()} positive out of {len(y_test)}")
                
                if y_test.sum() > 0 and y_pred.sum() > 0:
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred))
            
            test_results = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'true_values': y_test,
                'features': test_features,
                'model_name': self.best_model
            }
            
            self.results['testing'] = test_results
            return test_results
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return {}
    
    def predict_september_2025(self) -> Dict:
        """Generate predictions for remaining September 2025"""
        print("\n🔮 PREDICTING SEPTEMBER 2025")
        print("=" * 30)
        
        if not self.best_model or not self.prediction_data:
            print("❌ No model or prediction data available")
            return {}
        
        # Extract features for prediction period
        pred_features = self.extract_features_for_period('prediction', self.prediction_data)
        
        if pred_features is None or len(pred_features) == 0:
            print("❌ No features available for prediction period")
            return {}
        
        # Prepare features
        feature_cols = self.models[self.best_model]['feature_names']
        X_pred = pred_features[feature_cols].fillna(0)
        
        print(f"📊 Prediction period: {len(X_pred)} trading days in September 2025")
        
        try:
            model_info = self.models[self.best_model]
            model = model_info['model']
            
            # Apply scaling if needed
            if model_info['scaler']:
                X_pred_processed = model_info['scaler'].transform(X_pred)
            else:
                X_pred_processed = X_pred
            
            # Make predictions
            y_pred = model.predict(X_pred_processed)
            y_proba = model.predict_proba(X_pred_processed)[:, 1]
            
            # Create results DataFrame
            results_df = pred_features[['date']].copy()
            results_df['correction_probability'] = y_proba
            results_df['correction_prediction'] = y_pred
            results_df['risk_level'] = pd.cut(y_proba, bins=[0, 0.3, 0.6, 1.0], 
                                            labels=['Low', 'Medium', 'High'])
            
            print(f"\n📊 September 2025 Predictions:")
            print(f"   Days with correction risk: {y_pred.sum()}")
            print(f"   Highest risk day: {results_df.loc[results_df['correction_probability'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
            print(f"   Max probability: {results_df['correction_probability'].max():.3f}")
            
            # Show risk level distribution
            risk_dist = results_df['risk_level'].value_counts()
            print(f"\n📈 Risk Distribution:")
            for level, count in risk_dist.items():
                print(f"   {level}: {count} days")
            
            prediction_results = {
                'predictions_df': results_df,
                'raw_predictions': y_pred,
                'probabilities': y_proba,
                'model_name': self.best_model
            }
            
            self.results['prediction'] = prediction_results
            return prediction_results
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return {}
    
    def run_complete_backtest(self) -> Dict:
        """Run the complete backtesting pipeline"""
        print("🚀 RUNNING COMPLETE BACKTEST")
        print("=" * 60)
        print("📊 Data Strategy:")
        print("   • Training: 2016-2023 (8 years)")
        print("   • Validation: 2024 (hyperparameter tuning)")
        print("   • Testing: Jan-Aug 2025 (performance validation)")
        print("   • Prediction: September 2025 (live predictions)")
        print()
        
        try:
            # Step 1: Prepare historical data
            datasets = self.prepare_historical_data()
            
            # Step 2: Extract features for training
            if self.train_data:
                train_features = self.extract_features_for_period('training', self.train_data)
                if train_features is not None and len(train_features) > 0:
                    # Step 3: Train models
                    trained_models = self.train_models(train_features)
                    
                    # Step 4: Validate on 2024 data
                    if self.val_data:
                        val_features = self.extract_features_for_period('validation', self.val_data)
                        if val_features is not None and len(val_features) > 0:
                            validation_results = self.validate_models(val_features)
                            
                            # Step 5: Test on 2025 data
                            if self.test_data:
                                test_features = self.extract_features_for_period('testing', self.test_data)
                                if test_features is not None and len(test_features) > 0:
                                    test_results = self.test_model(test_features)
                                    
                                    # Step 6: Predict September 2025
                                    prediction_results = self.predict_september_2025()
                                    
                                    # Step 7: Generate summary report
                                    self.generate_summary_report()
                                    
                                    return {
                                        'datasets': datasets,
                                        'training': trained_models,
                                        'validation': validation_results,
                                        'testing': test_results,
                                        'predictions': prediction_results
                                    }
            
            print("❌ Backtest incomplete - check data availability")
            return {}
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return {}
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n📋 BACKTESTING SUMMARY REPORT")
        print("=" * 50)
        
        # Training summary
        if 'training' in self.results:
            print(f"🏋️  TRAINING PHASE")
            print(f"   Models trained: {len(self.models)}")
            print(f"   Best model: {self.best_model}")
        
        # Validation summary
        if 'validation' in self.results:
            val_results = self.results['validation']
            if val_results and self.best_model in val_results:
                best_val = val_results[self.best_model]
                print(f"\n🔍 VALIDATION PHASE (2024)")
                print(f"   AUC Score: {best_val['auc_score']:.3f}")
                print(f"   Precision: {best_val['precision']:.3f}")
                print(f"   Recall: {best_val['recall']:.3f}")
        
        # Testing summary
        if 'testing' in self.results:
            test_results = self.results['testing']
            if test_results:
                print(f"\n🎯 TESTING PHASE (Jan-Aug 2025)")
                print(f"   Test samples: {len(test_results['true_values'])}")
                print(f"   Actual corrections: {test_results['true_values'].sum()}")
                print(f"   Predicted corrections: {test_results['predictions'].sum()}")
        
        # Prediction summary
        if 'prediction' in self.results:
            pred_results = self.results['prediction']
            if pred_results:
                pred_df = pred_results['predictions_df']
                high_risk_days = (pred_df['risk_level'] == 'High').sum()
                print(f"\n🔮 PREDICTIONS (September 2025)")
                print(f"   Trading days analyzed: {len(pred_df)}")
                print(f"   High-risk days: {high_risk_days}")
                print(f"   Max risk probability: {pred_df['correction_probability'].max():.3f}")
                
                if high_risk_days > 0:
                    high_risk_dates = pred_df[pred_df['risk_level'] == 'High']['date']
                    print(f"   High-risk dates: {', '.join(high_risk_dates.dt.strftime('%Y-%m-%d'))}")
        
        print(f"\n✅ Backtest completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def export_results(self, output_dir: str = "analysis/backtesting_results"):
        """Export all results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export prediction results
            if 'prediction' in self.results and self.results['prediction']:
                pred_df = self.results['prediction']['predictions_df']
                pred_file = output_path / f"september_2025_predictions_{timestamp}.csv"
                pred_df.to_csv(pred_file, index=False)
                print(f"💾 Exported predictions: {pred_file}")
            
            # Export summary statistics
            summary_file = output_path / f"backtest_summary_{timestamp}.json"
            summary_data = {
                'timestamp': timestamp,
                'best_model': self.best_model,
                'correction_threshold': self.correction_threshold,
                'results_summary': {
                    k: {
                        'completed': v is not None and len(v) > 0
                    } for k, v in self.results.items()
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            print(f"📊 Exported summary: {summary_file}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

# Import required metrics
from sklearn.metrics import precision_score, recall_score

def main():
    """Run the complete backtesting framework"""
    print("🎯 SPY OPTIONS ANOMALY DETECTION - BACKTESTING FRAMEWORK")
    print("=" * 70)
    
    # Initialize framework
    framework = BacktestingFramework(
        data_dir="data",
        correction_threshold=0.04  # 4% correction threshold
    )
    
    # Run complete backtest
    results = framework.run_complete_backtest()
    
    if results:
        # Export results
        framework.export_results()
        
        print("\n🎉 BACKTESTING COMPLETE!")
        print("📈 Key outputs:")
        print("   • Model training on 2016-2023 data")
        print("   • Validation on 2024 data")
        print("   • Testing on Jan-Aug 2025 data")
        print("   • Predictions for September 2025")
        print("   • Results exported to analysis/backtesting_results/")
    else:
        print("\n❌ Backtesting failed - check data availability and logs")

if __name__ == "__main__":
    main()