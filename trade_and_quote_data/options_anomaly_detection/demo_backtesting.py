#!/usr/bin/env python3
"""
Demonstration backtesting framework with synthetic data
Shows the complete pipeline for SPY correction prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

class DemoBacktestingFramework:
    """
    Demonstration backtesting framework with synthetic correction data
    Simulates the complete pipeline for SPY 4%+ correction prediction
    """
    
    def __init__(self, correction_threshold: float = 0.04):
        self.correction_threshold = correction_threshold
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def generate_synthetic_spy_data(self, start_date: str, end_date: str, include_corrections: bool = True) -> pd.DataFrame:
        """Generate realistic synthetic SPY price data with corrections"""
        
        date_range = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(date_range)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Generate base price movement (realistic SPY-like behavior)
        base_returns = np.random.normal(0.0005, 0.01, n_days)  # ~12% annual return, 16% volatility
        
        # Add market regime changes
        regime_changes = np.random.choice([0, 1], size=n_days, p=[0.95, 0.05])
        volatility_multiplier = np.where(regime_changes, 3.0, 1.0)  # 3x volatility during stress
        
        returns = base_returns * volatility_multiplier
        
        # Inject specific corrections if requested
        if include_corrections:
            correction_dates = []
            
            # Add major corrections at specific points
            correction_points = [
                (int(n_days * 0.15), -0.08),  # 8% correction early
                (int(n_days * 0.35), -0.06),  # 6% correction mid-period  
                (int(n_days * 0.65), -0.12), # 12% correction later
                (int(n_days * 0.85), -0.05), # 5% correction near end
            ]
            
            for correction_start, magnitude in correction_points:
                if correction_start < n_days - 10:
                    # Create multi-day correction
                    correction_duration = np.random.randint(3, 15)
                    correction_end = min(correction_start + correction_duration, n_days - 1)
                    
                    # Distribute the correction over multiple days
                    correction_returns = np.random.normal(magnitude / correction_duration, 0.02, correction_duration)
                    
                    end_idx = min(correction_start + len(correction_returns), n_days)
                    returns[correction_start:end_idx] = correction_returns[:end_idx - correction_start]
                    
                    correction_dates.append((correction_start, magnitude))
        
        # Calculate prices
        initial_price = 450.0  # Realistic SPY starting price
        prices = [initial_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'underlying_price': prices[1:],  # Remove initial price
            'returns': returns,
            'volume': np.random.randint(50000000, 200000000, n_days),  # Realistic SPY volume
        })
        
        return df
    
    def identify_corrections_in_data(self, price_data: pd.DataFrame) -> List[Dict]:
        """Identify correction events in price data"""
        
        corrections = []
        prices = price_data['underlying_price'].values
        dates = price_data['date'].values
        
        # Simple peak detection with rolling maximum
        window = 20  # 20-day lookback for peaks
        rolling_max = pd.Series(prices).rolling(window=window, min_periods=1).max()
        
        for i in range(len(prices)):
            current_price = prices[i]
            
            # Check if we're at a recent peak
            if i >= window and current_price == rolling_max.iloc[i]:
                peak_price = current_price
                peak_date = dates[i]
                
                # Look forward for significant drawdown
                for j in range(i + 1, min(i + 30, len(prices))):  # Look up to 30 days ahead
                    trough_price = prices[j]
                    drawdown = (peak_price - trough_price) / peak_price
                    
                    if drawdown >= self.correction_threshold:
                        # Find the actual trough
                        trough_idx = j
                        for k in range(j + 1, min(j + 10, len(prices))):
                            if prices[k] < trough_price:
                                trough_price = prices[k]
                                trough_idx = k
                            elif prices[k] >= peak_price * 0.98:  # Recovery threshold
                                break
                        
                        final_drawdown = (peak_price - trough_price) / peak_price
                        
                        correction = {
                            'peak_date': peak_date,
                            'trough_date': dates[trough_idx],
                            'peak_price': peak_price,
                            'trough_price': trough_price,
                            'magnitude': final_drawdown,
                            'duration_days': (pd.to_datetime(dates[trough_idx]) - pd.to_datetime(peak_date)).days,
                            'peak_idx': i,
                            'trough_idx': trough_idx
                        }
                        corrections.append(correction)
                        break
        
        # Remove overlapping corrections
        filtered_corrections = []
        for correction in corrections:
            is_overlapping = False
            for existing in filtered_corrections:
                if (correction['peak_idx'] <= existing['trough_idx'] and 
                    correction['trough_idx'] >= existing['peak_idx']):
                    if correction['magnitude'] > existing['magnitude']:
                        filtered_corrections.remove(existing)
                        filtered_corrections.append(correction)
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_corrections.append(correction)
        
        return filtered_corrections
    
    def create_prediction_targets(self, price_data: pd.DataFrame, corrections: List[Dict]) -> pd.DataFrame:
        """Create prediction targets 1-3 days before corrections"""
        
        targets_df = price_data[['date', 'underlying_price']].copy()
        targets_df['target'] = 0
        targets_df['days_to_correction'] = np.nan
        targets_df['correction_magnitude'] = np.nan
        
        for correction in corrections:
            peak_date = correction['peak_date']
            magnitude = correction['magnitude']
            
            # Find peak date in DataFrame
            peak_mask = targets_df['date'] == peak_date
            if not peak_mask.any():
                continue
            
            peak_idx = targets_df[peak_mask].index[0]
            
            # Mark 1-3 days before as prediction targets
            for days_before in range(1, 4):
                target_idx = peak_idx - days_before
                if target_idx >= 0:
                    targets_df.loc[target_idx, 'target'] = 1
                    targets_df.loc[target_idx, 'days_to_correction'] = days_before
                    targets_df.loc[target_idx, 'correction_magnitude'] = magnitude
        
        return targets_df
    
    def create_features(self, price_data: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for prediction"""
        
        # Merge price and target data
        merged_df = pd.merge(price_data, targets_df[['date', 'target', 'days_to_correction', 'correction_magnitude']], 
                           on='date', how='inner')
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        features_list = []
        
        for i, row in merged_df.iterrows():
            if i < 50:  # Need history for features
                continue
            
            # Get historical data
            hist_prices = merged_df.iloc[max(0, i-50):i+1]['underlying_price'].values
            hist_returns = merged_df.iloc[max(0, i-50):i]['returns'].values if 'returns' in merged_df.columns else np.diff(hist_prices[:-1]) / hist_prices[:-2]
            hist_volumes = merged_df.iloc[max(0, i-20):i+1]['volume'].values if 'volume' in merged_df.columns else np.ones(21)
            
            # Calculate features
            features = {
                'date': row['date'],
                'target': row['target'],
                'underlying_price': row['underlying_price'],
                
                # Price momentum
                'return_1d': hist_returns[-1] if len(hist_returns) >= 1 else 0,
                'return_5d': np.mean(hist_returns[-5:]) if len(hist_returns) >= 5 else 0,
                'return_10d': np.mean(hist_returns[-10:]) if len(hist_returns) >= 10 else 0,
                'return_20d': np.mean(hist_returns[-20:]) if len(hist_returns) >= 20 else 0,
                
                # Volatility
                'vol_5d': np.std(hist_returns[-5:]) * np.sqrt(252) if len(hist_returns) >= 5 else 0,
                'vol_10d': np.std(hist_returns[-10:]) * np.sqrt(252) if len(hist_returns) >= 10 else 0,
                'vol_20d': np.std(hist_returns[-20:]) * np.sqrt(252) if len(hist_returns) >= 20 else 0,
                
                # Moving averages
                'price_vs_sma5': (hist_prices[-1] / np.mean(hist_prices[-5:])) - 1 if len(hist_prices) >= 5 else 0,
                'price_vs_sma10': (hist_prices[-1] / np.mean(hist_prices[-10:])) - 1 if len(hist_prices) >= 10 else 0,
                'price_vs_sma20': (hist_prices[-1] / np.mean(hist_prices[-20:])) - 1 if len(hist_prices) >= 20 else 0,
                
                # Position features
                'price_vs_high_20d': (hist_prices[-1] / np.max(hist_prices[-20:])) - 1 if len(hist_prices) >= 20 else 0,
                'drawdown_current': (hist_prices[-1] / np.max(hist_prices)) - 1,
                
                # Trend features
                'uptrend_5d': 1 if len(hist_prices) >= 5 and hist_prices[-1] > hist_prices[-5] else 0,
                'uptrend_10d': 1 if len(hist_prices) >= 10 and hist_prices[-1] > hist_prices[-10] else 0,
                
                # Volume features (if available)
                'volume_ratio_5d': hist_volumes[-1] / np.mean(hist_volumes[-5:]) if len(hist_volumes) >= 5 else 1.0,
                'volume_trend': 1 if len(hist_volumes) >= 5 and np.mean(hist_volumes[-3:]) > np.mean(hist_volumes[-8:-3]) else 0,
                
                # Advanced features
                'momentum_divergence': self.calculate_momentum_divergence(hist_prices, hist_returns),
                'volatility_regime': 1 if len(hist_returns) >= 20 and np.std(hist_returns[-5:]) > np.std(hist_returns[-20:]) * 1.5 else 0,
                
                # Target features (for analysis)
                'days_to_correction': row.get('days_to_correction', np.nan),
                'correction_magnitude': row.get('correction_magnitude', np.nan),
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def calculate_momentum_divergence(self, prices: np.ndarray, returns: np.ndarray) -> float:
        """Calculate momentum divergence indicator"""
        if len(prices) < 20 or len(returns) < 20:
            return 0.0
        
        # Price momentum vs return momentum
        price_trend = (prices[-1] - prices[-10]) / prices[-10]
        return_momentum = np.mean(returns[-10:])
        
        # Divergence when price goes up but momentum slows
        divergence = price_trend - (return_momentum * 10)
        return np.clip(divergence, -1, 1)
    
    def run_backtest_demo(self) -> Dict:
        """Run complete backtesting demonstration"""
        print("🚀 SPY CORRECTION PREDICTION - DEMO BACKTESTING")
        print("=" * 60)
        print("📊 Simulating realistic SPY data with known corrections")
        print("🎯 Objective: Predict 4%+ corrections 1-3 days in advance")
        print()
        
        # Generate synthetic data for different periods
        periods = {
            'training': ('2020-01-01', '2022-12-31'),
            'validation': ('2023-01-01', '2023-12-31'),
            'testing': ('2024-01-01', '2024-12-31'),
            'prediction': ('2025-09-01', '2025-09-30')
        }
        
        datasets = {}
        
        for period_name, (start_date, end_date) in periods.items():
            print(f"📊 Creating {period_name} dataset ({start_date} to {end_date})...")
            
            # Generate data
            include_corrections = period_name != 'prediction'  # No known corrections for prediction period
            price_data = self.generate_synthetic_spy_data(start_date, end_date, include_corrections)
            
            # Identify corrections
            corrections = self.identify_corrections_in_data(price_data)
            
            # Create targets
            targets_df = self.create_prediction_targets(price_data, corrections)
            
            # Create features
            features_df = self.create_features(price_data, targets_df)
            
            datasets[period_name] = {
                'price_data': price_data,
                'corrections': corrections,
                'features': features_df
            }
            
            target_count = features_df['target'].sum() if 'target' in features_df.columns else 0
            print(f"   ✅ {len(price_data)} days, {len(corrections)} corrections, {target_count} targets")
        
        # Train models on training data
        print(f"\n🤖 Training models...")
        train_features = datasets['training']['features']
        
        # Prepare features for ML
        feature_cols = [col for col in train_features.columns 
                       if col not in ['date', 'target', 'underlying_price', 'days_to_correction', 'correction_magnitude']]
        
        X_train = train_features[feature_cols].fillna(0).values
        y_train = train_features['target'].values
        
        print(f"   📊 Training: {X_train.shape[0]} samples, {y_train.sum()} positive targets")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'logistic': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"   🔄 Training {name}...")
            
            if name == 'logistic':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            model_results[name] = {
                'model': model,
                'feature_names': feature_cols
            }
        
        # Validate on validation data
        print(f"\n🔍 Validating models...")
        val_features = datasets['validation']['features']
        X_val = val_features[feature_cols].fillna(0).values
        y_val = val_features['target'].values
        
        best_auc = 0
        best_model_name = None
        
        for name, model_info in model_results.items():
            model = model_info['model']
            
            if name == 'logistic':
                X_val_processed = self.scaler.transform(X_val)
            else:
                X_val_processed = X_val
            
            y_pred = model.predict(X_val_processed)
            y_proba = model.predict_proba(X_val_processed)[:, 1]
            
            auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5
            precision = precision_score(y_val, y_pred) if y_pred.sum() > 0 else 0
            recall = recall_score(y_val, y_pred) if y_val.sum() > 0 else 0
            
            print(f"   📈 {name}: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model_name = name
        
        self.best_model = best_model_name
        self.models = model_results
        print(f"   🏆 Best model: {best_model_name}")
        
        # Test on test data
        print(f"\n🎯 Testing on 2024 data...")
        test_features = datasets['testing']['features']
        X_test = test_features[feature_cols].fillna(0).values
        y_test = test_features['target'].values
        
        best_model = model_results[best_model_name]['model']
        
        if best_model_name == 'logistic':
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test
        
        y_pred_test = best_model.predict(X_test_processed)
        y_proba_test = best_model.predict_proba(X_test_processed)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_proba_test) if len(np.unique(y_test)) > 1 else 0.5
        test_precision = precision_score(y_test, y_pred_test) if y_pred_test.sum() > 0 else 0
        test_recall = recall_score(y_test, y_pred_test) if y_test.sum() > 0 else 0
        
        print(f"   📈 Test Results: AUC={test_auc:.3f}, Precision={test_precision:.3f}, Recall={test_recall:.3f}")
        print(f"   🎯 True positives: {((y_test == 1) & (y_pred_test == 1)).sum()}")
        print(f"   ⚠️  False positives: {((y_test == 0) & (y_pred_test == 1)).sum()}")
        
        # Predict September 2025
        print(f"\n🔮 Predicting September 2025...")
        pred_features = datasets['prediction']['features']
        X_pred = pred_features[feature_cols].fillna(0).values
        
        if best_model_name == 'logistic':
            X_pred_processed = self.scaler.transform(X_pred)
        else:
            X_pred_processed = X_pred
        
        y_pred_sept = best_model.predict(X_pred_processed)
        y_proba_sept = best_model.predict_proba(X_pred_processed)[:, 1]
        
        # Create results
        sept_results = pred_features[['date']].copy()
        sept_results['correction_probability'] = y_proba_sept
        sept_results['correction_prediction'] = y_pred_sept
        sept_results['risk_level'] = pd.cut(y_proba_sept, bins=[0, 0.3, 0.6, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
        
        high_risk_days = (sept_results['risk_level'] == 'High').sum()
        medium_risk_days = (sept_results['risk_level'] == 'Medium').sum()
        max_prob = sept_results['correction_probability'].max()
        
        print(f"   📊 September 2025 Predictions:")
        print(f"      High risk days: {high_risk_days}")
        print(f"      Medium risk days: {medium_risk_days}")
        print(f"      Max probability: {max_prob:.3f}")
        print(f"      Predicted corrections: {y_pred_sept.sum()}")
        
        if high_risk_days > 0:
            high_risk_dates = sept_results[sept_results['risk_level'] == 'High']['date']
            print(f"      High-risk dates: {', '.join(high_risk_dates.dt.strftime('%Y-%m-%d'))}")
        
        # Export results
        output_dir = Path("analysis/demo_backtesting_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"september_2025_demo_predictions_{timestamp}.csv"
        sept_results.to_csv(output_file, index=False)
        
        print(f"\n💾 Results exported to: {output_file}")
        
        return {
            'datasets': datasets,
            'models': model_results,
            'validation_metrics': {'auc': best_auc, 'best_model': best_model_name},
            'test_metrics': {'auc': test_auc, 'precision': test_precision, 'recall': test_recall},
            'september_predictions': sept_results
        }

def main():
    """Run the demonstration backtesting framework"""
    framework = DemoBacktestingFramework(correction_threshold=0.04)
    
    results = framework.run_backtest_demo()
    
    print(f"\n🎉 DEMO BACKTESTING COMPLETE!")
    print(f"📈 This demonstrates the complete pipeline:")
    print(f"   • Synthetic SPY data generation with realistic corrections")
    print(f"   • Feature engineering from price/volume data")
    print(f"   • Model training and validation (2020-2023)")
    print(f"   • Testing on held-out data (2024)")
    print(f"   • Live predictions for September 2025")
    print(f"   • Performance metrics and risk assessment")
    print(f"\n🔧 Next steps with real data:")
    print(f"   • Replace synthetic data with actual SPY options flow")
    print(f"   • Add institutional flow features (put/call ratios, OI analysis)")
    print(f"   • Incorporate VIX and sentiment indicators")
    print(f"   • Optimize prediction thresholds")
    print(f"   • Build real-time prediction pipeline")

if __name__ == "__main__":
    main()