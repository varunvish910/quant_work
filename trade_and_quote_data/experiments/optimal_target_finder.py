#!/usr/bin/env python3
"""
Optimal Target Finder: Multi-Target Analysis System
Phase 0 implementation from MODEL_IMPROVEMENT_ROADMAP.md

This script systematically tests all possible target combinations to find
the optimal prediction parameters instead of trying to fix the existing system.

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import warnings
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimalTargetFinder:
    """
    Systematic analysis of all possible target combinations to find optimal
    prediction parameters for SPY pullback prediction.
    
    Tests:
    - Pullback magnitudes: 2%, 5%, 10%
    - Time horizons: 5, 10, 15, 20 days
    - Total combinations: 12 different prediction targets
    """
    
    def __init__(self, start_date='2016-01-01', end_date='2024-12-31'):
        self.magnitudes = [0.02, 0.05, 0.10]  # 2%, 5%, 10%
        self.horizons = [5, 10, 15, 20]       # days
        self.start_date = start_date
        self.end_date = end_date
        self.results = []
        self.models = {}
        
        # Create output directory
        self.output_dir = Path('analysis/outputs/optimal_targets')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_spy_data(self):
        """Download SPY data with all required fields"""
        logger.info(f"Downloading SPY data from {self.start_date} to {self.end_date}")
        
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
        spy = spy.dropna()
        
        # Add basic features for initial analysis
        spy['Returns'] = spy['Close'].pct_change()
        spy['SMA_20'] = spy['Close'].rolling(20).mean()
        spy['SMA_50'] = spy['Close'].rolling(50).mean()
        spy['Volatility_20'] = spy['Returns'].rolling(20).std() * np.sqrt(252)
        spy['RSI'] = self._calculate_rsi(spy['Close'])
        spy['BB_Width'] = (spy['Close'].rolling(20).std() * 2) / spy['SMA_20']
        
        # Download VIX data
        vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix = vix_data['Close'].iloc[:, 0]  # Handle MultiIndex columns
        else:
            vix = vix_data['Close']
        vix = vix.reindex(spy.index, method='ffill')
        spy['VIX'] = vix
        spy['VIX_Change'] = spy['VIX'].pct_change()
        spy['VIX_MA5'] = spy['VIX'].rolling(5).mean()
        
        spy = spy.dropna()
        logger.info(f"Downloaded {len(spy)} days of data")
        return spy
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_target_labels(self, spy_data, magnitude, horizon):
        """
        Create binary labels for specific magnitude/horizon combination
        
        Args:
            spy_data: DataFrame with OHLC data
            magnitude: Pullback threshold (e.g., 0.05 for 5%)
            horizon: Days to look forward (e.g., 10)
            
        Returns:
            pandas Series with binary labels (1 = pullback occurs, 0 = no pullback)
        """
        labels = pd.Series(0, index=spy_data.index, name=f'pullback_{int(magnitude*100)}pct_{horizon}d')
        
        for i in range(len(spy_data) - horizon):
            current_date = spy_data.index[i]
            current_price = spy_data['Close'].iloc[i]
            
            # Look forward 'horizon' days
            future_slice = spy_data.iloc[i+1:i+horizon+1]
            if len(future_slice) == 0:
                continue
                
            # Find minimum price in the forward window
            min_future_price = future_slice['Low'].min()
            
            # Calculate the maximum drawdown
            drawdown = (min_future_price / current_price) - 1
            
            # Label as 1 if drawdown exceeds threshold
            if drawdown <= -magnitude:
                labels.iloc[i] = 1
                
        return labels
    
    def prepare_features(self, spy_data):
        """
        Prepare minimal, high-quality feature set based on market logic
        
        Tier 1 Features (Core - 15 features):
        - VIX level and momentum
        - RSI and extremes 
        - Trend strength (ADX proxy)
        - Bollinger Band indicators
        - Volatility measures
        """
        features = pd.DataFrame(index=spy_data.index)
        
        # VIX features (volatility regime)
        features['vix_level'] = spy_data['VIX']
        features['vix_momentum_3d'] = spy_data['VIX'].pct_change(3)
        features['vix_momentum_5d'] = spy_data['VIX'].pct_change(5)
        features['vix_vs_ma5'] = spy_data['VIX'] / spy_data['VIX_MA5'] - 1
        features['vix_spike'] = (spy_data['VIX'] > spy_data['VIX'].rolling(20).quantile(0.8)).astype(int)
        
        # Price momentum and trend
        features['rsi_14'] = spy_data['RSI']
        features['rsi_extreme'] = ((spy_data['RSI'] > 70) | (spy_data['RSI'] < 30)).astype(int)
        features['momentum_5d'] = spy_data['Close'].pct_change(5)
        features['momentum_10d'] = spy_data['Close'].pct_change(10)
        features['momentum_20d'] = spy_data['Close'].pct_change(20)
        
        # Trend and positioning
        features['price_vs_sma20'] = spy_data['Close'] / spy_data['SMA_20'] - 1
        features['price_vs_sma50'] = spy_data['Close'] / spy_data['SMA_50'] - 1
        features['sma20_vs_sma50'] = spy_data['SMA_20'] / spy_data['SMA_50'] - 1
        
        # Volatility and Bollinger Bands
        features['volatility_20d'] = spy_data['Volatility_20']
        features['bb_width'] = spy_data['BB_Width']
        features['bb_position'] = ((spy_data['Close'] - spy_data['SMA_20']) / 
                                  (spy_data['Close'].rolling(20).std() * 2))
        
        # Distance from recent highs/lows
        features['distance_from_high_10d'] = spy_data['Close'] / spy_data['High'].rolling(10).max() - 1
        features['distance_from_high_20d'] = spy_data['Close'] / spy_data['High'].rolling(20).max() - 1
        features['distance_from_low_10d'] = spy_data['Close'] / spy_data['Low'].rolling(10).min() - 1
        
        # Volume-price relationships (using Close as proxy for volume patterns)
        features['price_acceleration'] = spy_data['Returns'].diff()
        features['volatility_regime'] = (spy_data['Volatility_20'] > 
                                        spy_data['Volatility_20'].rolling(60).median()).astype(int)
        
        return features.dropna()
    
    def evaluate_target(self, magnitude, horizon, spy_data, features):
        """
        Train and evaluate model for specific target combination
        
        Args:
            magnitude: Pullback threshold (0.02, 0.05, 0.10)
            horizon: Forward looking days (5, 10, 15, 20)
            spy_data: Price data
            features: Feature matrix
            
        Returns:
            Dictionary with evaluation metrics and trained model
        """
        logger.info(f"Evaluating target: {int(magnitude*100)}% pullback in {horizon} days")
        
        # Create target labels
        labels = self.create_target_labels(spy_data, magnitude, horizon)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        if len(X) == 0:
            logger.warning(f"No common data for target {magnitude}_{horizon}")
            return None
        
        # Time-based split (train on first 70%, validate on next 15%, test on last 15%)
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        
        # Check if we have enough positive cases
        if y_train.sum() < 10:
            logger.warning(f"Too few positive cases ({y_train.sum()}) for {magnitude}_{horizon}")
            return None
        
        # Train LightGBM model
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            class_weight='balanced'
        )
        
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Model training failed for {magnitude}_{horizon}: {e}")
            return None
        
        # Get predictions on validation and test sets
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        target_name = f'{int(magnitude*100)}pct_{horizon}d'
        
        # Validation metrics for threshold selection
        val_precisions, val_recalls, val_thresholds = precision_recall_curve(y_val, val_pred_proba)
        val_f1_scores = 2 * (val_precisions * val_recalls) / (val_precisions + val_recalls + 1e-10)
        optimal_idx = np.argmax(val_f1_scores)
        optimal_threshold = val_thresholds[optimal_idx] if optimal_idx < len(val_thresholds) else 0.5
        
        # Test metrics with optimal threshold
        test_pred_binary = (test_pred_proba >= optimal_threshold).astype(int)
        
        try:
            test_roc_auc = roc_auc_score(y_test, test_pred_proba)
        except:
            test_roc_auc = 0.5
        
        # Precision at different confidence levels
        precision_80 = 0
        precision_90 = 0
        if len(test_pred_proba[test_pred_proba >= 0.80]) > 0:
            precision_80 = precision_score(y_test[test_pred_proba >= 0.80], 
                                         (test_pred_proba >= 0.80)[test_pred_proba >= 0.80])
        if len(test_pred_proba[test_pred_proba >= 0.90]) > 0:
            precision_90 = precision_score(y_test[test_pred_proba >= 0.90], 
                                         (test_pred_proba >= 0.90)[test_pred_proba >= 0.90])
        
        metrics = {
            'magnitude': magnitude,
            'horizon': horizon,
            'target_name': target_name,
            'train_period': f"{X_train.index[0].date()} to {X_train.index[-1].date()}",
            'test_period': f"{X_test.index[0].date()} to {X_test.index[-1].date()}",
            'positive_rate_train': y_train.mean(),
            'positive_rate_test': y_test.mean(),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_positive_train': y_train.sum(),
            'n_positive_test': y_test.sum(),
            'roc_auc': test_roc_auc,
            'optimal_threshold': optimal_threshold,
            'precision_at_optimal': precision_score(y_test, test_pred_binary),
            'recall_at_optimal': recall_score(y_test, test_pred_binary),
            'f1_at_optimal': val_f1_scores[optimal_idx],
            'precision_at_80pct': precision_80,
            'precision_at_90pct': precision_90,
            'feature_importance_top5': list(zip(X.columns[:5], model.feature_importances_[:5])),
            'validation_date_cutoff': X_val.index[0].date(),
            'test_date_cutoff': X_test.index[0].date()
        }
        
        return {
            'metrics': metrics,
            'model': model,
            'predictions': {
                'test_dates': X_test.index,
                'test_probabilities': test_pred_proba,
                'test_labels': y_test
            }
        }
    
    def run_full_analysis(self):
        """
        Run complete multi-target analysis across all combinations
        """
        logger.info("Starting comprehensive multi-target analysis")
        
        # Download data
        spy_data = self.download_spy_data()
        
        # Prepare features
        features = self.prepare_features(spy_data)
        
        logger.info(f"Prepared {len(features.columns)} features for {len(features)} observations")
        
        # Test all target combinations
        for magnitude in self.magnitudes:
            for horizon in self.horizons:
                result = self.evaluate_target(magnitude, horizon, spy_data, features)
                
                if result is not None:
                    self.results.append(result['metrics'])
                    self.models[f'{int(magnitude*100)}pct_{horizon}d'] = result
                    
                    # Save individual model results
                    model_file = self.output_dir / f'model_{int(magnitude*100)}pct_{horizon}d.json'
                    with open(model_file, 'w') as f:
                        json.dump(result['metrics'], f, indent=2, default=str)
        
        logger.info(f"Completed analysis of {len(self.results)} target combinations")
        
        return self.create_results_dashboard()
    
    def create_results_dashboard(self):
        """Create comprehensive comparison of all targets"""
        if not self.results:
            logger.error("No results to display")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Calculate composite scores
        df['major_event_score'] = df['recall_at_optimal'] * df['precision_at_optimal']
        df['trading_score'] = df['f1_at_optimal'] * (1 - df['positive_rate_test'])
        df['risk_mgmt_score'] = df['precision_at_80pct']
        
        # Sort and display results
        print("=" * 100)
        print("OPTIMAL TARGET ANALYSIS RESULTS")
        print("=" * 100)
        print()
        
        print("SUMMARY STATISTICS:")
        print(f"Total targets tested: {len(df)}")
        print(f"Average ROC AUC: {df['roc_auc'].mean():.3f}")
        print(f"Average F1 Score: {df['f1_at_optimal'].mean():.3f}")
        print()
        
        print("TOP 3 FOR CATCHING MAJOR EVENTS (Recall × Precision):")
        top_events = df.nlargest(3, 'major_event_score')[
            ['target_name', 'major_event_score', 'recall_at_optimal', 'precision_at_optimal', 'roc_auc']
        ]
        print(top_events.to_string(index=False))
        print()
        
        print("TOP 3 FOR ACTIVE TRADING (F1 × Signal Frequency):")
        top_trading = df.nlargest(3, 'trading_score')[
            ['target_name', 'trading_score', 'f1_at_optimal', 'positive_rate_test', 'precision_at_optimal']
        ]
        print(top_trading.to_string(index=False))
        print()
        
        print("TOP 3 FOR RISK MANAGEMENT (High Confidence Precision):")
        top_risk = df.nlargest(3, 'risk_mgmt_score')[
            ['target_name', 'risk_mgmt_score', 'precision_at_80pct', 'precision_at_90pct']
        ]
        print(top_risk.to_string(index=False))
        print()
        
        print("DETAILED RESULTS FOR ALL TARGETS:")
        detailed_cols = ['target_name', 'roc_auc', 'f1_at_optimal', 'precision_at_optimal', 
                        'recall_at_optimal', 'positive_rate_test', 'n_positive_test']
        print(df[detailed_cols].sort_values('f1_at_optimal', ascending=False).to_string(index=False))
        
        # Save comprehensive results
        results_file = self.output_dir / f'optimal_targets_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save dashboard summary
        summary_file = self.output_dir / f'dashboard_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(summary_file, index=False)
        
        logger.info(f"Results saved to {results_file} and {summary_file}")
        
        return df
    
    def analyze_2024_performance(self):
        """
        Analyze how each target performs specifically on 2024 data
        """
        logger.info("Analyzing 2024 performance for all targets")
        
        performance_2024 = []
        
        for target_name, result in self.models.items():
            if result is None:
                continue
                
            predictions = result['predictions']
            test_dates = predictions['test_dates']
            test_probs = predictions['test_probabilities']
            test_labels = predictions['test_labels']
            
            # Filter for 2024 data
            mask_2024 = test_dates >= pd.Timestamp('2024-01-01')
            if not mask_2024.any():
                continue
                
            dates_2024 = test_dates[mask_2024]
            probs_2024 = test_probs[mask_2024]
            labels_2024 = test_labels[mask_2024]
            
            # Calculate 2024-specific metrics
            if len(labels_2024) > 0 and labels_2024.sum() > 0:
                try:
                    roc_2024 = roc_auc_score(labels_2024, probs_2024)
                except:
                    roc_2024 = 0.5
                
                # High confidence signals in 2024
                high_conf_mask = probs_2024 >= 0.8
                if high_conf_mask.sum() > 0:
                    precision_high_conf = precision_score(labels_2024[high_conf_mask], 
                                                        (probs_2024 >= 0.8)[high_conf_mask])
                else:
                    precision_high_conf = 0
                
                performance_2024.append({
                    'target_name': target_name,
                    'n_days_2024': len(labels_2024),
                    'n_positive_2024': labels_2024.sum(),
                    'positive_rate_2024': labels_2024.mean(),
                    'roc_auc_2024': roc_2024,
                    'n_high_conf_signals': high_conf_mask.sum(),
                    'precision_high_conf_2024': precision_high_conf,
                    'max_probability_2024': probs_2024.max(),
                    'mean_probability_2024': probs_2024.mean()
                })
        
        if performance_2024:
            df_2024 = pd.DataFrame(performance_2024)
            print("\n" + "="*80)
            print("2024 PERFORMANCE ANALYSIS")
            print("="*80)
            print(df_2024.sort_values('roc_auc_2024', ascending=False).to_string(index=False))
            
            # Save 2024 analysis
            analysis_2024_file = self.output_dir / f'2024_performance_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df_2024.to_csv(analysis_2024_file, index=False)
            logger.info(f"2024 analysis saved to {analysis_2024_file}")
            
            return df_2024
        
        return None


def main():
    """Main execution function"""
    print("Starting Optimal Target Finder Analysis")
    print("Phase 0: Clean Slate Multi-Target Analysis")
    print("="*60)
    
    # Initialize analyzer
    finder = OptimalTargetFinder(start_date='2016-01-01', end_date='2024-12-31')
    
    # Run full analysis
    results_df = finder.run_full_analysis()
    
    if results_df is not None:
        # Analyze 2024 performance specifically
        finder.analyze_2024_performance()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nKey Recommendations:")
        
        # Get top performers
        top_overall = results_df.nlargest(1, 'f1_at_optimal').iloc[0]
        top_precision = results_df.nlargest(1, 'precision_at_80pct').iloc[0]
        
        print(f"1. Best Overall Performance: {top_overall['target_name']}")
        print(f"   - F1 Score: {top_overall['f1_at_optimal']:.3f}")
        print(f"   - ROC AUC: {top_overall['roc_auc']:.3f}")
        print(f"   - Precision: {top_overall['precision_at_optimal']:.3f}")
        
        print(f"\n2. Best High-Confidence Signals: {top_precision['target_name']}")
        print(f"   - Precision at 80% confidence: {top_precision['precision_at_80pct']:.3f}")
        print(f"   - ROC AUC: {top_precision['roc_auc']:.3f}")
        
        print("\n3. Next Steps:")
        print("   - Review detailed results in analysis/outputs/optimal_targets/")
        print("   - Select top 2-3 targets for ensemble modeling")
        print("   - Implement selected targets in production pipeline")
        
    else:
        print("Analysis failed - check logs for details")


if __name__ == "__main__":
    main()