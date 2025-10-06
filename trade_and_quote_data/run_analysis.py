#!/usr/bin/env python3
"""
Run Multi-Target Analysis - Simplified Version
Actually train models and report performance

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
import json
from pathlib import Path
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplifiedTargetAnalyzer:
    """Simplified version that actually runs and trains models"""
    
    def __init__(self):
        self.results = []
        self.magnitudes = [0.02, 0.05, 0.10]  # 2%, 5%, 10%
        self.horizons = [5, 10, 15, 20]       # days
        
    def download_data(self):
        """Download SPY and VIX data"""
        logger.info("Downloading market data...")
        
        # Download complete dataset for comprehensive analysis
        spy_data = yf.download('SPY', start='2016-01-01', end='2025-01-01', progress=False)
        vix_data = yf.download('^VIX', start='2016-01-01', end='2025-01-01', progress=False)
        
        # Handle multiindex column structure for both SPY and VIX
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy = spy_data.xs('SPY', axis=1, level=1)
        else:
            spy = spy_data
            
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix = vix_data['Close']['^VIX']  # Extract single column
        else:
            vix = vix_data['Close']
            
        return spy, vix
    
    def create_features(self, spy, vix):
        """Create comprehensive 54-feature matrix"""
        logger.info("Creating comprehensive 54-feature matrix...")
        
        features = pd.DataFrame(index=spy.index)
        vix_aligned = vix.reindex(spy.index, method='ffill')
        
        # === TIER 1: Core Price & Volume Features (15) ===
        features['returns'] = spy['Close'].pct_change()
        features['returns_2d'] = spy['Close'].pct_change(2)
        features['returns_5d'] = spy['Close'].pct_change(5)
        features['returns_10d'] = spy['Close'].pct_change(10)
        features['returns_20d'] = spy['Close'].pct_change(20)
        
        features['sma_5'] = spy['Close'].rolling(5).mean()
        features['sma_10'] = spy['Close'].rolling(10).mean()
        features['sma_20'] = spy['Close'].rolling(20).mean()
        features['sma_50'] = spy['Close'].rolling(50).mean()
        features['sma_200'] = spy['Close'].rolling(200).mean()
        
        features['price_vs_sma5'] = spy['Close'] / features['sma_5'] - 1
        features['price_vs_sma20'] = spy['Close'] / features['sma_20'] - 1
        features['price_vs_sma50'] = spy['Close'] / features['sma_50'] - 1
        
        features['volume_sma_20'] = spy['Volume'].rolling(20).mean()
        features['volume_ratio'] = spy['Volume'] / features['volume_sma_20']
        
        # === TIER 2: Volatility & Technical Indicators (15) ===
        features['volatility_5d'] = features['returns'].rolling(5).std() * np.sqrt(252)
        features['volatility_10d'] = features['returns'].rolling(10).std() * np.sqrt(252)
        features['volatility_20d'] = features['returns'].rolling(20).std() * np.sqrt(252)
        features['volatility_60d'] = features['returns'].rolling(60).std() * np.sqrt(252)
        
        features['rsi_14'] = self.calculate_rsi(spy['Close'], 14)
        features['rsi_7'] = self.calculate_rsi(spy['Close'], 7)
        features['rsi_21'] = self.calculate_rsi(spy['Close'], 21)
        
        # Bollinger Bands
        bb_std = features['returns'].rolling(20).std()
        bb_mean = features['sma_20']
        features['bb_upper'] = bb_mean + (2 * bb_std * spy['Close'])
        features['bb_lower'] = bb_mean - (2 * bb_std * spy['Close'])
        features['bb_position'] = (spy['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Distance indicators
        features['distance_from_high_5d'] = spy['Close'] / spy['High'].rolling(5).max() - 1
        features['distance_from_high_20d'] = spy['Close'] / spy['High'].rolling(20).max() - 1
        features['distance_from_high_60d'] = spy['Close'] / spy['High'].rolling(60).max() - 1
        features['distance_from_low_20d'] = spy['Close'] / spy['Low'].rolling(20).min() - 1
        
        # === TIER 3: VIX & Fear/Greed Indicators (12) ===
        features['vix_level'] = vix_aligned
        features['vix_returns'] = vix_aligned.pct_change()
        features['vix_returns_5d'] = vix_aligned.pct_change(5)
        features['vix_returns_10d'] = vix_aligned.pct_change(10)
        
        features['vix_sma_10'] = vix_aligned.rolling(10).mean()
        features['vix_sma_20'] = vix_aligned.rolling(20).mean()
        features['vix_vs_sma10'] = vix_aligned / features['vix_sma_10'] - 1
        features['vix_vs_sma20'] = vix_aligned / features['vix_sma_20'] - 1
        
        features['vix_percentile_60d'] = vix_aligned.rolling(60).rank(pct=True)
        features['vix_percentile_252d'] = vix_aligned.rolling(252).rank(pct=True)
        
        # VIX-SPY correlation
        features['vix_spy_corr_20d'] = features['returns'].rolling(20).corr(features['vix_returns'])
        features['vix_spy_corr_60d'] = features['returns'].rolling(60).corr(features['vix_returns'])
        
        # === TIER 4: Advanced Momentum & Mean Reversion (12) ===
        # MACD
        ema_12 = spy['Close'].ewm(span=12).mean()
        ema_26 = spy['Close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Williams %R
        highest_high = spy['High'].rolling(14).max()
        lowest_low = spy['Low'].rolling(14).min()
        features['williams_r'] = -100 * (highest_high - spy['Close']) / (highest_high - lowest_low)
        
        # Stochastic oscillator
        features['stoch_k'] = 100 * (spy['Close'] - lowest_low) / (highest_high - lowest_low)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # Rate of change
        features['roc_5d'] = (spy['Close'] / spy['Close'].shift(5) - 1) * 100
        features['roc_10d'] = (spy['Close'] / spy['Close'].shift(10) - 1) * 100
        features['roc_20d'] = (spy['Close'] / spy['Close'].shift(20) - 1) * 100
        
        # Money Flow Index approximation
        typical_price = (spy['High'] + spy['Low'] + spy['Close']) / 3
        money_flow = typical_price * spy['Volume']
        features['money_flow_20d'] = money_flow.rolling(20).sum()
        features['mfi_ratio'] = features['money_flow_20d'] / features['money_flow_20d'].shift(1)
        features['price_volume_trend'] = ((spy['Close'] - spy['Close'].shift(1)) / spy['Close'].shift(1) * spy['Volume']).cumsum()
        
        logger.info(f"Created {len(features.columns)} features")
        return features.dropna()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def create_target(self, spy, magnitude, horizon):
        """Create pullback target"""
        target = pd.Series(0, index=spy.index, name=f'pullback_{int(magnitude*100)}pct_{horizon}d')
        
        for i in range(len(spy) - horizon):
            current_price = spy['Close'].iloc[i]
            future_lows = spy['Low'].iloc[i+1:i+horizon+1]
            
            if len(future_lows) > 0:
                min_future = future_lows.min()
                if pd.notna(min_future) and pd.notna(current_price) and current_price > 0:
                    drawdown = (min_future / current_price) - 1
                    if drawdown <= -magnitude:
                        target.iloc[i] = 1
        
        return target
    
    def create_vix_spike_targets(self, vix):
        """Create VIX spike targets (calm → storm transitions)"""
        targets = {}
        
        # VIX thresholds for calm/storm classification
        vix_low = vix.quantile(0.25)    # Calm market
        vix_high = vix.quantile(0.75)   # Storm market
        
        # Create different VIX spike targets
        for horizon in [3, 5, 10]:
            target = pd.Series(0, index=vix.index, name=f'vix_spike_{horizon}d')
            
            for i in range(len(vix) - horizon):
                current_vix = vix.iloc[i]
                future_vix_max = vix.iloc[i+1:i+horizon+1].max()
                
                # Calm → Storm transition
                if current_vix <= vix_low and future_vix_max >= vix_high:
                    target.iloc[i] = 1
                # Large VIX spike (50%+ increase)
                elif pd.notna(current_vix) and pd.notna(future_vix_max) and current_vix > 0:
                    if (future_vix_max / current_vix) >= 1.5:
                        target.iloc[i] = 1
            
            targets[f'vix_spike_{horizon}d'] = target
        
        return targets
    
    def train_and_evaluate_vix(self, features, target, target_name, horizon):
        """Train model and evaluate performance for VIX targets"""
        logger.info(f"Training VIX model for {target_name}")
        
        # Align data
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]
        
        if len(X) < 200:
            logger.warning(f"Insufficient data for {target_name}: {len(X)} samples")
            return None
        
        # Time-based split (70% train, 30% test)
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_test = y.iloc[split_point:]
        
        if y_train.sum() < 3:
            logger.warning(f"Too few positive samples for {target_name}: {y_train.sum()}")
            return None
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
            
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # High confidence metrics
        high_conf_mask = y_pred_proba >= 0.8
        precision_80 = 0
        if high_conf_mask.sum() > 0:
            precision_80 = precision_score(y_test[high_conf_mask], 
                                         (y_pred_proba >= 0.8)[high_conf_mask], 
                                         zero_division=0)
        
        return {
            'target_name': target_name,
            'magnitude': 'vix_spike',
            'horizon': horizon,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_positive_train': int(y_train.sum()),
            'n_positive_test': int(y_test.sum()),
            'positive_rate_train': float(y_train.mean()),
            'positive_rate_test': float(y_test.mean()),
            'roc_auc': float(roc_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'precision_at_80pct': float(precision_80),
            'n_signals_80pct': int(high_conf_mask.sum()),
            'roc_auc_2024': 0.5,  # Will be calculated separately if needed
            'precision_2024': 0.0,
            'n_samples_2024': 0,
            'test_period_start': str(X_test.index[0].date()),
            'test_period_end': str(X_test.index[-1].date())
        }
    
    def train_and_evaluate(self, features, target, magnitude, horizon):
        """Train model and evaluate performance"""
        target_name = f'{int(magnitude*100)}pct_{horizon}d'
        logger.info(f"Training model for {target_name}")
        
        # Align data
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]
        
        if len(X) < 200:
            logger.warning(f"Insufficient data for {target_name}: {len(X)} samples")
            return None
        
        # Time-based split (70% train, 30% test)
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_test = y.iloc[split_point:]
        
        if y_train.sum() < 5:
            logger.warning(f"Too few positive samples for {target_name}: {y_train.sum()}")
            return None
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
            
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # High confidence metrics
        high_conf_mask = y_pred_proba >= 0.8
        precision_80 = 0
        if high_conf_mask.sum() > 0:
            precision_80 = precision_score(y_test[high_conf_mask], 
                                         (y_pred_proba >= 0.8)[high_conf_mask], 
                                         zero_division=0)
        
        # Check 2024 performance
        test_dates = X_test.index
        mask_2024 = test_dates >= pd.Timestamp('2024-01-01')
        
        roc_2024 = 0.5
        precision_2024 = 0
        if mask_2024.sum() > 0:
            y_test_2024 = y_test[mask_2024]
            y_pred_2024 = y_pred_proba[mask_2024]
            
            if y_test_2024.sum() > 0:
                try:
                    roc_2024 = roc_auc_score(y_test_2024, y_pred_2024)
                except:
                    roc_2024 = 0.5
                    
                high_conf_2024 = y_pred_2024 >= 0.8
                if high_conf_2024.sum() > 0:
                    precision_2024 = precision_score(y_test_2024[high_conf_2024],
                                                   (y_pred_2024 >= 0.8)[high_conf_2024],
                                                   zero_division=0)
        
        return {
            'target_name': target_name,
            'magnitude': magnitude,
            'horizon': horizon,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_positive_train': int(y_train.sum()),
            'n_positive_test': int(y_test.sum()),
            'positive_rate_train': float(y_train.mean()),
            'positive_rate_test': float(y_test.mean()),
            'roc_auc': float(roc_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'precision_at_80pct': float(precision_80),
            'n_signals_80pct': int(high_conf_mask.sum()),
            'roc_auc_2024': float(roc_2024),
            'precision_2024': float(precision_2024),
            'n_samples_2024': int(mask_2024.sum()),
            'test_period_start': str(X_test.index[0].date()),
            'test_period_end': str(X_test.index[-1].date())
        }
    
    def run_analysis(self):
        """Run complete analysis"""
        logger.info("Starting complete multi-target analysis")
        
        # Download data
        spy, vix = self.download_data()
        logger.info(f"Downloaded {len(spy)} days of SPY data")
        
        # Create features
        features = self.create_features(spy, vix)
        logger.info(f"Created features: {len(features.columns)} features, {len(features)} observations")
        
        # Test all pullback target combinations
        for magnitude in self.magnitudes:
            for horizon in self.horizons:
                # Create target
                target = self.create_target(spy, magnitude, horizon)
                
                # Train and evaluate
                result = self.train_and_evaluate(features, target, magnitude, horizon)
                
                if result is not None:
                    self.results.append(result)
                    logger.info(f"Completed {result['target_name']}: "
                              f"ROC AUC={result['roc_auc']:.3f}, "
                              f"F1={result['f1_score']:.3f}, "
                              f"Precision@80%={result['precision_at_80pct']:.3f}")
        
        # Test VIX spike targets
        logger.info("Creating VIX spike targets...")
        vix_targets = self.create_vix_spike_targets(vix)
        
        for target_name, target in vix_targets.items():
            # Extract horizon for training
            horizon = int(target_name.split('_')[-1].replace('d', ''))
            
            # Train and evaluate with special handling for VIX targets
            result = self.train_and_evaluate_vix(features, target, target_name, horizon)
            
            if result is not None:
                self.results.append(result)
                logger.info(f"Completed {result['target_name']}: "
                          f"ROC AUC={result['roc_auc']:.3f}, "
                          f"F1={result['f1_score']:.3f}, "
                          f"Precision@80%={result['precision_at_80pct']:.3f}")
        
        return self.results
    
    def create_report(self):
        """Create comprehensive report"""
        if not self.results:
            return "No results available"
        
        df = pd.DataFrame(self.results)
        
        report = []
        report.append("=" * 80)
        report.append("MULTI-TARGET ANALYSIS RESULTS")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"Targets analyzed: {len(df)}")
        report.append(f"Average ROC AUC: {df['roc_auc'].mean():.3f} ± {df['roc_auc'].std():.3f}")
        report.append(f"Average F1 Score: {df['f1_score'].mean():.3f} ± {df['f1_score'].std():.3f}")
        report.append(f"Average Precision: {df['precision'].mean():.3f} ± {df['precision'].std():.3f}")
        report.append("")
        
        # Top performers
        report.append("TOP 5 PERFORMERS (by F1 Score):")
        top_f1 = df.nlargest(5, 'f1_score')
        for _, row in top_f1.iterrows():
            report.append(f"{row['target_name']}: F1={row['f1_score']:.3f}, "
                         f"ROC AUC={row['roc_auc']:.3f}, "
                         f"Precision={row['precision']:.3f}, "
                         f"Recall={row['recall']:.3f}")
        report.append("")
        
        # High confidence performance
        report.append("HIGH CONFIDENCE SIGNALS (80% threshold):")
        high_conf = df[df['precision_at_80pct'] > 0]
        if len(high_conf) > 0:
            best_high_conf = high_conf.nlargest(3, 'precision_at_80pct')
            for _, row in best_high_conf.iterrows():
                report.append(f"{row['target_name']}: "
                             f"Precision@80%={row['precision_at_80pct']:.3f}, "
                             f"Signals={row['n_signals_80pct']}")
        else:
            report.append("No targets with reliable high confidence signals")
        report.append("")
        
        # 2024 performance
        report.append("2024 PERFORMANCE:")
        df_2024 = df[df['n_samples_2024'] > 0]
        if len(df_2024) > 0:
            report.append(f"Targets with 2024 data: {len(df_2024)}")
            report.append(f"Average 2024 ROC AUC: {df_2024['roc_auc_2024'].mean():.3f}")
            
            best_2024 = df_2024.nlargest(3, 'roc_auc_2024')
            report.append("Top 2024 performers:")
            for _, row in best_2024.iterrows():
                report.append(f"  {row['target_name']}: ROC AUC 2024={row['roc_auc_2024']:.3f}")
        else:
            report.append("No 2024 data available for analysis")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        
        # Best overall
        best_overall = df.loc[df['f1_score'].idxmax()]
        report.append(f"Best Overall Target: {best_overall['target_name']}")
        report.append(f"  - F1 Score: {best_overall['f1_score']:.3f}")
        report.append(f"  - ROC AUC: {best_overall['roc_auc']:.3f}")
        report.append(f"  - Signal Rate: {best_overall['positive_rate_test']:.3f}")
        
        # Ensemble recommendation
        top_3 = df.nlargest(3, 'f1_score')
        report.append(f"\nRecommended Ensemble (Top 3):")
        for _, row in top_3.iterrows():
            report.append(f"  - {row['target_name']} (F1: {row['f1_score']:.3f})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('analysis/outputs/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        if self.results:
            df = pd.DataFrame(self.results)
            results_file = output_dir / f'target_analysis_results_{timestamp}.csv'
            df.to_csv(results_file, index=False)
            
            # Save JSON for programmatic access
            json_file = output_dir / f'target_analysis_results_{timestamp}.json'
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            return results_file, json_file
        
        return None, None


def main():
    """Main execution"""
    print("🚀 Running Multi-Target Model Analysis")
    print("Training models and evaluating performance...")
    print("=" * 60)
    
    # Run analysis
    analyzer = SimplifiedTargetAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        print(f"\n✅ Analysis Complete! Trained {len(results)} models")
        
        # Generate and display report
        report = analyzer.create_report()
        print(report)
        
        # Save results
        csv_file, json_file = analyzer.save_results()
        if csv_file:
            print(f"\n📊 Results saved to:")
            print(f"  CSV: {csv_file}")
            print(f"  JSON: {json_file}")
        
        return True
    else:
        print("\n❌ Analysis failed - no results generated")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Multi-target analysis completed successfully!")
        print("Models have been trained and performance evaluated.")
    else:
        print("\n💥 Analysis failed!")