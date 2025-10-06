#!/usr/bin/env python3
"""
2024 Signal Analysis
Analyze what dates the system flagged in 2024 and their outcomes

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Signal2024Analyzer:
    """Analyze 2024 signals and their outcomes"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.features = None
        self.targets = {}
        
    def download_2024_data(self):
        """Download complete dataset including 2024"""
        logger.info("Downloading market data for 2024 analysis...")
        
        # Download data from 2016 to get enough history for features
        spy_data = yf.download('SPY', start='2016-01-01', end='2025-01-01', progress=False)
        vix_data = yf.download('^VIX', start='2016-01-01', end='2025-01-01', progress=False)
        
        # Handle multiindex columns
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy = spy_data.xs('SPY', axis=1, level=1)
        else:
            spy = spy_data
            
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix = vix_data['Close']['^VIX']
        else:
            vix = vix_data['Close']
            
        logger.info(f"Downloaded SPY: {len(spy)} days")
        logger.info(f"Downloaded VIX: {len(vix)} days")
        
        return spy, vix
    
    def create_enhanced_features(self, spy, vix):
        """Create the 53-feature comprehensive matrix"""
        logger.info("Creating 53-feature comprehensive matrix...")
        
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
        
        features['vix_spy_corr_20d'] = features['returns'].rolling(20).corr(features['vix_returns'])
        features['vix_spy_corr_60d'] = features['returns'].rolling(60).corr(features['vix_returns'])
        
        # === TIER 4: Advanced Momentum & Mean Reversion (11) ===
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
        
        logger.info(f"Created {len(features.columns)} features")
        return features.dropna()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def create_targets(self, spy, vix):
        """Create prediction targets"""
        logger.info("Creating prediction targets...")
        
        targets = {}
        
        # Best performing targets from previous analysis
        best_targets = [
            (0.02, 20),  # 2pct_20d - champion
            (0.02, 15),  # 2pct_15d - strong performer
            (0.02, 10),  # 2pct_10d - good performer
            (0.02, 5),   # 2pct_5d - decent
        ]
        
        for magnitude, horizon in best_targets:
            target = pd.Series(0, index=spy.index, name=f'{int(magnitude*100)}pct_{horizon}d')
            
            for i in range(len(spy) - horizon):
                current_price = spy['Close'].iloc[i]
                future_lows = spy['Low'].iloc[i+1:i+horizon+1]
                
                if len(future_lows) > 0:
                    min_future = future_lows.min()
                    if pd.notna(min_future) and pd.notna(current_price) and current_price > 0:
                        drawdown = (min_future / current_price) - 1
                        if drawdown <= -magnitude:
                            target.iloc[i] = 1
            
            targets[f'{int(magnitude*100)}pct_{horizon}d'] = target
        
        # VIX spike target
        vix_target = pd.Series(0, index=vix.index, name='vix_spike_10d')
        vix_low = vix.quantile(0.25)
        vix_high = vix.quantile(0.75)
        
        for i in range(len(vix) - 10):
            current_vix = vix.iloc[i]
            future_vix_max = vix.iloc[i+1:i+11].max()
            
            if current_vix <= vix_low and future_vix_max >= vix_high:
                vix_target.iloc[i] = 1
            elif pd.notna(current_vix) and pd.notna(future_vix_max) and current_vix > 0:
                if (future_vix_max / current_vix) >= 1.5:
                    vix_target.iloc[i] = 1
        
        targets['vix_spike_10d'] = vix_target
        
        return targets
    
    def train_models_pre_2024(self, features, targets):
        """Train models on pre-2024 data"""
        logger.info("Training models on pre-2024 data...")
        
        # Split at 2024
        cutoff_date = pd.Timestamp('2024-01-01')
        
        for target_name, target in targets.items():
            logger.info(f"Training model for {target_name}")
            
            # Align data
            common_index = features.index.intersection(target.index)
            X = features.loc[common_index]
            y = target.loc[common_index]
            
            # Pre-2024 training data
            train_mask = X.index < cutoff_date
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            if y_train.sum() < 10:
                logger.warning(f"Insufficient positive samples for {target_name}: {y_train.sum()}")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            self.models[target_name] = model
            self.scalers[target_name] = scaler
            
            logger.info(f"Trained {target_name}: {len(X_train)} samples, {y_train.sum()} positive")
    
    def analyze_2024_signals(self, features, targets):
        """Analyze signals generated in 2024"""
        logger.info("Analyzing 2024 signals...")
        
        # 2024 data only
        start_2024 = pd.Timestamp('2024-01-01')
        end_2024 = pd.Timestamp('2024-12-31')
        
        signal_results = {}
        
        for target_name in self.models.keys():
            logger.info(f"Analyzing 2024 signals for {target_name}")
            
            model = self.models[target_name]
            scaler = self.scalers[target_name]
            target = targets[target_name]
            
            # Get 2024 data - align indices first
            common_index = features.index.intersection(target.index)
            X_aligned = features.loc[common_index]
            y_aligned = target.loc[common_index]
            
            mask_2024 = (X_aligned.index >= start_2024) & (X_aligned.index <= end_2024)
            X_2024 = X_aligned[mask_2024]
            y_2024 = y_aligned[mask_2024]
            
            if len(X_2024) == 0:
                continue
            
            # Generate predictions
            X_2024_scaled = scaler.transform(X_2024)
            predictions = model.predict_proba(X_2024_scaled)[:, 1]
            
            # Create signal DataFrame
            signal_df = pd.DataFrame({
                'date': X_2024.index,
                'prediction': predictions,
                'actual': y_2024.values,
                'signal_60': (predictions >= 0.6).astype(int),
                'signal_70': (predictions >= 0.7).astype(int),
                'signal_80': (predictions >= 0.8).astype(int)
            })
            
            # Identify signal dates
            signal_dates_60 = signal_df[signal_df['signal_60'] == 1]['date'].tolist()
            signal_dates_70 = signal_df[signal_df['signal_70'] == 1]['date'].tolist()
            signal_dates_80 = signal_df[signal_df['signal_80'] == 1]['date'].tolist()
            
            # Calculate outcomes for each signal threshold
            outcomes_60 = self.calculate_signal_outcomes(signal_df, 'signal_60')
            outcomes_70 = self.calculate_signal_outcomes(signal_df, 'signal_70')
            outcomes_80 = self.calculate_signal_outcomes(signal_df, 'signal_80')
            
            signal_results[target_name] = {
                'signal_data': signal_df,
                'dates_60': signal_dates_60,
                'dates_70': signal_dates_70,
                'dates_80': signal_dates_80,
                'outcomes_60': outcomes_60,
                'outcomes_70': outcomes_70,
                'outcomes_80': outcomes_80,
                'total_signals_60': len(signal_dates_60),
                'total_signals_70': len(signal_dates_70),
                'total_signals_80': len(signal_dates_80)
            }
            
            logger.info(f"{target_name} 2024 signals: 60%={len(signal_dates_60)}, 70%={len(signal_dates_70)}, 80%={len(signal_dates_80)}")
        
        return signal_results
    
    def calculate_signal_outcomes(self, signal_df, signal_col):
        """Calculate outcomes for signals"""
        signals = signal_df[signal_df[signal_col] == 1]
        
        if len(signals) == 0:
            return {
                'total_signals': 0,
                'correct_signals': 0,
                'accuracy': 0.0,
                'false_positive_rate': 0.0
            }
        
        correct_signals = signals[signals['actual'] == 1]
        
        return {
            'total_signals': len(signals),
            'correct_signals': len(correct_signals),
            'accuracy': len(correct_signals) / len(signals) if len(signals) > 0 else 0,
            'false_positive_rate': (len(signals) - len(correct_signals)) / len(signals) if len(signals) > 0 else 0
        }
    
    def create_detailed_report(self, signal_results, spy):
        """Create detailed report of 2024 signals"""
        logger.info("Creating detailed 2024 signal report...")
        
        report = []
        report.append("=" * 80)
        report.append("2024 SIGNAL ANALYSIS - DETAILED REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Get SPY 2024 data for context
        spy_2024 = spy[spy.index >= pd.Timestamp('2024-01-01')]
        spy_start = spy_2024['Close'].iloc[0] if len(spy_2024) > 0 else 0
        spy_end = spy_2024['Close'].iloc[-1] if len(spy_2024) > 0 else 0
        spy_return = (spy_end / spy_start - 1) * 100 if spy_start > 0 else 0
        
        report.append("2024 MARKET CONTEXT:")
        report.append("-" * 30)
        report.append(f"SPY Start: ${spy_start:.2f}")
        report.append(f"SPY End: ${spy_end:.2f}")
        report.append(f"SPY Return: {spy_return:+.1f}%")
        report.append("")
        
        # Summary by model
        for target_name, results in signal_results.items():
            report.append(f"{target_name.upper()} SIGNALS:")
            report.append("-" * 40)
            
            # Signal counts
            report.append(f"Total Signals @ 60%: {results['total_signals_60']}")
            report.append(f"Total Signals @ 70%: {results['total_signals_70']}")
            report.append(f"Total Signals @ 80%: {results['total_signals_80']}")
            
            # Accuracy at different thresholds
            for threshold in [60, 70, 80]:
                outcomes = results[f'outcomes_{threshold}']
                if outcomes['total_signals'] > 0:
                    report.append(f"Accuracy @ {threshold}%: {outcomes['accuracy']:.1%} "
                                f"({outcomes['correct_signals']}/{outcomes['total_signals']})")
            
            report.append("")
            
            # List actual signal dates for 70% threshold
            if results['dates_70']:
                report.append(f"SIGNAL DATES @ 70% CONFIDENCE:")
                for i, date in enumerate(results['dates_70'][:10]):  # Show first 10
                    # Get actual outcome
                    signal_row = results['signal_data'][results['signal_data']['date'] == date]
                    if len(signal_row) > 0:
                        actual = signal_row['actual'].iloc[0]
                        prediction = signal_row['prediction'].iloc[0]
                        outcome = "✅ CORRECT" if actual == 1 else "❌ FALSE POSITIVE"
                        report.append(f"  {date.strftime('%Y-%m-%d')}: {prediction:.3f} - {outcome}")
                
                if len(results['dates_70']) > 10:
                    report.append(f"  ... and {len(results['dates_70']) - 10} more dates")
                report.append("")
        
        # Major market events analysis
        report.append("MAJOR 2024 MARKET EVENTS ANALYSIS:")
        report.append("-" * 40)
        
        # Identify major down days
        if len(spy_2024) > 0:
            spy_2024_returns = spy_2024['Close'].pct_change()
            major_down_days = spy_2024_returns[spy_2024_returns < -0.02].index  # 2%+ down days
            
            report.append(f"Major Down Days (>2% decline): {len(major_down_days)}")
            
            for target_name, results in signal_results.items():
                if '2pct' in target_name:  # Only for pullback models
                    predicted_count = 0
                    for down_day in major_down_days:
                        # Check if we had a signal in the days leading up
                        signal_window = results['signal_data'][
                            (results['signal_data']['date'] >= down_day - pd.Timedelta(days=20)) &
                            (results['signal_data']['date'] <= down_day) &
                            (results['signal_data']['signal_70'] == 1)
                        ]
                        if len(signal_window) > 0:
                            predicted_count += 1
                    
                    if len(major_down_days) > 0:
                        prediction_rate = predicted_count / len(major_down_days)
                        report.append(f"  {target_name}: Predicted {predicted_count}/{len(major_down_days)} "
                                    f"major declines ({prediction_rate:.1%})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_results(self, signal_results, report):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('analysis/outputs/2024_signals')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_file = output_dir / f'2024_signals_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save detailed signal data
        for target_name, results in signal_results.items():
            csv_file = output_dir / f'2024_signals_{target_name}_{timestamp}.csv'
            results['signal_data'].to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {output_dir}")
        return output_dir


def main():
    """Main execution function"""
    print("📅 2024 Signal Analysis")
    print("Analyzing what dates the system flagged in 2024")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = Signal2024Analyzer()
    
    # Download data
    spy, vix = analyzer.download_2024_data()
    
    # Create features
    features = analyzer.create_enhanced_features(spy, vix)
    
    # Create targets
    targets = analyzer.create_targets(spy, vix)
    
    # Train models on pre-2024 data
    analyzer.train_models_pre_2024(features, targets)
    
    # Analyze 2024 signals
    signal_results = analyzer.analyze_2024_signals(features, targets)
    
    # Create detailed report
    report = analyzer.create_detailed_report(signal_results, spy)
    
    # Save results
    output_dir = analyzer.save_detailed_results(signal_results, report)
    
    # Display report
    print(report)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"Results saved to: {output_dir}")
    
    total_signals = 0
    for target_name, results in signal_results.items():
        signals_70 = results['total_signals_70']
        total_signals += signals_70
        accuracy_70 = results['outcomes_70']['accuracy']
        print(f"{target_name}: {signals_70} signals @ 70%, {accuracy_70:.1%} accuracy")
    
    print(f"\nTotal signals across all models: {total_signals}")
    
    return signal_results


if __name__ == "__main__":
    results = main()