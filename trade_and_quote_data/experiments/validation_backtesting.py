#!/usr/bin/env python3
"""
Validation & Backtesting Framework
Phase 5: Comprehensive validation and historical backtesting

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')


class ValidationBacktesting:
    """Comprehensive validation and backtesting framework"""
    
    def __init__(self):
        self.backtest_results = {}
        self.walk_forward_results = {}
        self.regime_analysis = {}
        self.drawdown_analysis = {}
        
    def walk_forward_validation(self, features, targets, models, train_window=252*2, test_window=63):
        """Walk-forward validation with rolling windows"""
        logger.info(f"Starting walk-forward validation...")
        logger.info(f"Train window: {train_window} days, Test window: {test_window} days")
        
        results = {}
        
        for target_name in targets.columns:
            logger.info(f"Walk-forward validation for {target_name}")
            
            target = targets[target_name]
            
            # Skip if insufficient data
            if target.sum() < 20:
                logger.warning(f"Skipping {target_name}: insufficient positive samples")
                continue
            
            # Align data
            common_index = features.index.intersection(target.index)
            X = features.loc[common_index].fillna(method='ffill').fillna(method='bfill')
            y = target.loc[common_index]
            
            # Walk-forward splits
            predictions = []
            actuals = []
            dates = []
            performance_over_time = []
            
            start_idx = train_window
            while start_idx + test_window < len(X):
                # Define windows
                train_start = start_idx - train_window
                train_end = start_idx
                test_start = start_idx
                test_end = min(start_idx + test_window, len(X))
                
                # Split data
                X_train = X.iloc[train_start:train_end]
                y_train = y.iloc[train_start:train_end]
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]
                
                # Skip if insufficient positive samples in training
                if y_train.sum() < 10:
                    start_idx += test_window
                    continue
                
                # Train a simple Random Forest for this window
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Store results
                predictions.extend(y_pred_proba)
                actuals.extend(y_test.values)
                dates.extend(y_test.index)
                
                # Calculate window performance
                if len(y_test) > 0 and y_test.sum() > 0:
                    window_metrics = {
                        'start_date': y_test.index[0],
                        'end_date': y_test.index[-1],
                        'roc_auc': roc_auc_score(y_test, y_pred_proba),
                        'precision': precision_score(y_test, (y_pred_proba > 0.5).astype(int), zero_division=0),
                        'recall': recall_score(y_test, (y_pred_proba > 0.5).astype(int), zero_division=0),
                        'f1_score': f1_score(y_test, (y_pred_proba > 0.5).astype(int), zero_division=0),
                        'n_samples': len(y_test),
                        'n_positive': y_test.sum()
                    }
                    performance_over_time.append(window_metrics)
                
                start_idx += test_window
            
            # Overall walk-forward performance
            if len(predictions) > 0:
                overall_predictions = np.array(predictions)
                overall_actuals = np.array(actuals)
                
                overall_metrics = {
                    'roc_auc': roc_auc_score(overall_actuals, overall_predictions),
                    'precision': precision_score(overall_actuals, (overall_predictions > 0.5).astype(int), zero_division=0),
                    'recall': recall_score(overall_actuals, (overall_predictions > 0.5).astype(int), zero_division=0),
                    'f1_score': f1_score(overall_actuals, (overall_predictions > 0.5).astype(int), zero_division=0),
                    'total_predictions': len(predictions),
                    'total_positives': np.sum(overall_actuals)
                }
                
                results[target_name] = {
                    'overall_metrics': overall_metrics,
                    'window_performance': performance_over_time,
                    'predictions': overall_predictions,
                    'actuals': overall_actuals,
                    'dates': dates
                }
                
                logger.info(f"{target_name} walk-forward results:")
                logger.info(f"  ROC AUC: {overall_metrics['roc_auc']:.3f}")
                logger.info(f"  F1 Score: {overall_metrics['f1_score']:.3f}")
                logger.info(f"  Total predictions: {overall_metrics['total_predictions']}")
        
        self.walk_forward_results = results
        return results
    
    def regime_based_analysis(self, features, targets, predictions_dict):
        """Analyze performance across different market regimes"""
        logger.info("Performing regime-based analysis...")
        
        # Define market regimes based on VIX and SPY momentum
        regime_indicators = pd.DataFrame(index=features.index)
        
        # VIX-based regimes
        if 'vix_level' in features.columns:
            vix = features['vix_level']
            vix_percentile = vix.rolling(252).rank(pct=True)
            
            regime_indicators['vix_regime'] = 'Normal'
            regime_indicators.loc[vix_percentile < 0.25, 'vix_regime'] = 'Low_VIX'
            regime_indicators.loc[vix_percentile > 0.75, 'vix_regime'] = 'High_VIX'
            regime_indicators.loc[vix_percentile > 0.95, 'vix_regime'] = 'Crisis'
        
        # Momentum-based regimes
        if 'returns_20d' in features.columns:
            momentum = features['returns_20d']
            
            regime_indicators['momentum_regime'] = 'Neutral'
            regime_indicators.loc[momentum > 0.05, 'momentum_regime'] = 'Bull'
            regime_indicators.loc[momentum < -0.05, 'momentum_regime'] = 'Bear'
            regime_indicators.loc[momentum < -0.15, 'momentum_regime'] = 'Crash'
        
        # Volatility-based regimes
        if 'volatility_20d' in features.columns:
            volatility = features['volatility_20d']
            vol_percentile = volatility.rolling(252).rank(pct=True)
            
            regime_indicators['vol_regime'] = 'Normal_Vol'
            regime_indicators.loc[vol_percentile < 0.25, 'vol_regime'] = 'Low_Vol'
            regime_indicators.loc[vol_percentile > 0.75, 'vol_regime'] = 'High_Vol'
        
        # Analyze performance by regime
        regime_results = {}
        
        for target_name in targets.columns:
            if target_name not in predictions_dict:
                continue
            
            target = targets[target_name]
            predictions = predictions_dict[target_name]
            
            target_regime_results = {}
            
            for regime_type in ['vix_regime', 'momentum_regime', 'vol_regime']:
                if regime_type not in regime_indicators.columns:
                    continue
                
                regime_performance = {}
                
                for regime in regime_indicators[regime_type].unique():
                    mask = regime_indicators[regime_type] == regime
                    
                    if mask.sum() > 10:  # Minimum samples per regime
                        regime_target = target[mask]
                        regime_pred = predictions[mask] if hasattr(predictions, '__getitem__') else predictions
                        
                        if len(regime_target) > 0 and len(regime_pred) > 0:
                            try:
                                regime_performance[regime] = {
                                    'n_samples': len(regime_target),
                                    'positive_rate': regime_target.mean(),
                                    'roc_auc': roc_auc_score(regime_target, regime_pred),
                                    'precision': precision_score(regime_target, (regime_pred > 0.5).astype(int), zero_division=0),
                                    'recall': recall_score(regime_target, (regime_pred > 0.5).astype(int), zero_division=0),
                                    'f1_score': f1_score(regime_target, (regime_pred > 0.5).astype(int), zero_division=0)
                                }
                            except:
                                continue
                
                target_regime_results[regime_type] = regime_performance
            
            regime_results[target_name] = target_regime_results
        
        self.regime_analysis = regime_results
        return regime_results
    
    def drawdown_analysis(self, features, targets, predictions_dict):
        """Analyze performance during historical drawdowns"""
        logger.info("Performing drawdown analysis...")
        
        # Identify major drawdowns
        if 'returns' in features.columns:
            returns = features['returns']
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns / rolling_max) - 1
            
            # Identify major drawdown periods (>5%)
            major_drawdowns = drawdown < -0.05
            
            # Find drawdown events
            drawdown_events = []
            in_drawdown = False
            start_date = None
            
            for date, is_drawdown in major_drawdowns.items():
                if is_drawdown and not in_drawdown:
                    start_date = date
                    in_drawdown = True
                elif not is_drawdown and in_drawdown:
                    end_date = date
                    max_dd = drawdown[start_date:end_date].min()
                    
                    if abs(max_dd) > 0.05:  # At least 5% drawdown
                        drawdown_events.append({
                            'start': start_date,
                            'end': end_date,
                            'max_drawdown': max_dd,
                            'duration': (end_date - start_date).days
                        })
                    
                    in_drawdown = False
            
            logger.info(f"Identified {len(drawdown_events)} major drawdown events")
            
            # Analyze performance during drawdowns
            drawdown_results = {}
            
            for target_name in targets.columns:
                if target_name not in predictions_dict:
                    continue
                
                target = targets[target_name]
                predictions = predictions_dict[target_name]
                
                event_performance = []
                
                for i, event in enumerate(drawdown_events):
                    # Get data during drawdown
                    mask = (target.index >= event['start']) & (target.index <= event['end'])
                    
                    if mask.sum() > 5:  # Minimum samples
                        event_target = target[mask]
                        event_pred = predictions[mask] if hasattr(predictions, '__getitem__') else predictions
                        
                        if len(event_target) > 0 and event_target.sum() > 0:
                            try:
                                event_metrics = {
                                    'event_id': i,
                                    'start_date': event['start'],
                                    'end_date': event['end'],
                                    'max_drawdown': event['max_drawdown'],
                                    'duration_days': event['duration'],
                                    'n_samples': len(event_target),
                                    'positive_rate': event_target.mean(),
                                    'roc_auc': roc_auc_score(event_target, event_pred),
                                    'precision': precision_score(event_target, (event_pred > 0.5).astype(int), zero_division=0),
                                    'recall': recall_score(event_target, (event_pred > 0.5).astype(int), zero_division=0),
                                    'f1_score': f1_score(event_target, (event_pred > 0.5).astype(int), zero_division=0)
                                }
                                event_performance.append(event_metrics)
                            except:
                                continue
                
                drawdown_results[target_name] = {
                    'drawdown_events': drawdown_events,
                    'performance_during_drawdowns': event_performance
                }
            
            self.drawdown_analysis = drawdown_results
            return drawdown_results
        
        return {}
    
    def stress_testing(self, features, targets, predictions_dict):
        """Perform stress testing under extreme conditions"""
        logger.info("Performing stress testing...")
        
        stress_results = {}
        
        for target_name in targets.columns:
            if target_name not in predictions_dict:
                continue
            
            target = targets[target_name]
            predictions = predictions_dict[target_name]
            
            stress_scenarios = {}
            
            # High volatility stress test
            if 'volatility_20d' in features.columns:
                vol = features['volatility_20d']
                high_vol_mask = vol > vol.quantile(0.95)
                
                if high_vol_mask.sum() > 10:
                    stress_target = target[high_vol_mask]
                    stress_pred = predictions[high_vol_mask] if hasattr(predictions, '__getitem__') else predictions
                    
                    stress_scenarios['high_volatility'] = {
                        'n_samples': len(stress_target),
                        'condition': 'Volatility > 95th percentile',
                        'roc_auc': roc_auc_score(stress_target, stress_pred) if stress_target.sum() > 0 else 0.5,
                        'precision': precision_score(stress_target, (stress_pred > 0.5).astype(int), zero_division=0),
                        'f1_score': f1_score(stress_target, (stress_pred > 0.5).astype(int), zero_division=0)
                    }
            
            # Market crash stress test
            if 'returns_5d' in features.columns:
                returns = features['returns_5d']
                crash_mask = returns < returns.quantile(0.05)
                
                if crash_mask.sum() > 10:
                    stress_target = target[crash_mask]
                    stress_pred = predictions[crash_mask] if hasattr(predictions, '__getitem__') else predictions
                    
                    stress_scenarios['market_crash'] = {
                        'n_samples': len(stress_target),
                        'condition': '5-day returns < 5th percentile',
                        'roc_auc': roc_auc_score(stress_target, stress_pred) if stress_target.sum() > 0 else 0.5,
                        'precision': precision_score(stress_target, (stress_pred > 0.5).astype(int), zero_division=0),
                        'f1_score': f1_score(stress_target, (stress_pred > 0.5).astype(int), zero_division=0)
                    }
            
            # VIX spike stress test
            if 'vix_level' in features.columns:
                vix = features['vix_level']
                vix_spike_mask = vix > vix.quantile(0.9)
                
                if vix_spike_mask.sum() > 10:
                    stress_target = target[vix_spike_mask]
                    stress_pred = predictions[vix_spike_mask] if hasattr(predictions, '__getitem__') else predictions
                    
                    stress_scenarios['vix_spike'] = {
                        'n_samples': len(stress_target),
                        'condition': 'VIX > 90th percentile',
                        'roc_auc': roc_auc_score(stress_target, stress_pred) if stress_target.sum() > 0 else 0.5,
                        'precision': precision_score(stress_target, (stress_pred > 0.5).astype(int), zero_division=0),
                        'f1_score': f1_score(stress_target, (stress_pred > 0.5).astype(int), zero_division=0)
                    }
            
            stress_results[target_name] = stress_scenarios
        
        return stress_results
    
    def create_performance_visualizations(self, walk_forward_results):
        """Create comprehensive performance visualizations"""
        logger.info("Creating performance visualizations...")
        
        output_dir = Path('analysis/outputs/validation_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for target_name, results in walk_forward_results.items():
            if 'window_performance' not in results:
                continue
            
            performance_df = pd.DataFrame(results['window_performance'])
            if len(performance_df) == 0:
                continue
            
            # Performance over time
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Walk-Forward Performance: {target_name}', fontsize=16)
            
            # ROC AUC over time
            axes[0, 0].plot(performance_df['start_date'], performance_df['roc_auc'], marker='o')
            axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('ROC AUC Over Time')
            axes[0, 0].set_ylabel('ROC AUC')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # F1 Score over time
            axes[0, 1].plot(performance_df['start_date'], performance_df['f1_score'], marker='o', color='green')
            axes[0, 1].set_title('F1 Score Over Time')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Precision over time
            axes[1, 0].plot(performance_df['start_date'], performance_df['precision'], marker='o', color='orange')
            axes[1, 0].set_title('Precision Over Time')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Recall over time
            axes[1, 1].plot(performance_df['start_date'], performance_df['recall'], marker='o', color='purple')
            axes[1, 1].set_title('Recall Over Time')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'walk_forward_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Performance visualizations saved to {output_dir}")
        return output_dir
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("Generating comprehensive validation report...")
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE VALIDATION & BACKTESTING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Walk-forward results summary
        if self.walk_forward_results:
            report.append("WALK-FORWARD VALIDATION RESULTS:")
            report.append("=" * 50)
            
            for target_name, results in self.walk_forward_results.items():
                metrics = results['overall_metrics']
                report.append(f"\n{target_name}:")
                report.append(f"  ROC AUC: {metrics['roc_auc']:.3f}")
                report.append(f"  F1 Score: {metrics['f1_score']:.3f}")
                report.append(f"  Precision: {metrics['precision']:.3f}")
                report.append(f"  Recall: {metrics['recall']:.3f}")
                report.append(f"  Total Predictions: {metrics['total_predictions']}")
                report.append(f"  Positive Samples: {metrics['total_positives']}")
                
                # Time stability
                window_performance = results['window_performance']
                if len(window_performance) > 1:
                    window_df = pd.DataFrame(window_performance)
                    roc_std = window_df['roc_auc'].std()
                    f1_std = window_df['f1_score'].std()
                    report.append(f"  Time Stability (ROC AUC std): {roc_std:.3f}")
                    report.append(f"  Time Stability (F1 std): {f1_std:.3f}")
        
        # Regime analysis summary
        if self.regime_analysis:
            report.append("\n\nREGIME-BASED PERFORMANCE:")
            report.append("=" * 50)
            
            for target_name, regime_results in self.regime_analysis.items():
                report.append(f"\n{target_name}:")
                
                for regime_type, regimes in regime_results.items():
                    report.append(f"  {regime_type}:")
                    for regime, metrics in regimes.items():
                        report.append(f"    {regime}: ROC AUC={metrics['roc_auc']:.3f}, "
                                    f"F1={metrics['f1_score']:.3f}, "
                                    f"Samples={metrics['n_samples']}")
        
        # Drawdown analysis summary
        if self.drawdown_analysis:
            report.append("\n\nDRAWDOWN ANALYSIS:")
            report.append("=" * 50)
            
            for target_name, dd_results in self.drawdown_analysis.items():
                report.append(f"\n{target_name}:")
                
                events = dd_results['performance_during_drawdowns']
                if events:
                    avg_roc = np.mean([e['roc_auc'] for e in events])
                    avg_f1 = np.mean([e['f1_score'] for e in events])
                    report.append(f"  Drawdown Events Analyzed: {len(events)}")
                    report.append(f"  Average ROC AUC during drawdowns: {avg_roc:.3f}")
                    report.append(f"  Average F1 Score during drawdowns: {avg_f1:.3f}")
                    
                    for event in events:
                        report.append(f"    {event['start_date'].date()} to {event['end_date'].date()}: "
                                    f"Max DD={event['max_drawdown']:.1%}, "
                                    f"ROC AUC={event['roc_auc']:.3f}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def run_comprehensive_validation(self, features, targets, predictions_dict=None):
        """Run comprehensive validation pipeline"""
        logger.info("Starting comprehensive validation pipeline...")
        
        # If no predictions provided, run walk-forward validation
        if predictions_dict is None:
            predictions_dict = {}
            wf_results = self.walk_forward_validation(features, targets)
            
            for target_name, results in wf_results.items():
                predictions_dict[target_name] = results['predictions']
        
        # Regime-based analysis
        regime_results = self.regime_based_analysis(features, targets, predictions_dict)
        
        # Drawdown analysis
        drawdown_results = self.drawdown_analysis(features, targets, predictions_dict)
        
        # Stress testing
        stress_results = self.stress_testing(features, targets, predictions_dict)
        
        # Create visualizations
        viz_dir = self.create_performance_visualizations(self.walk_forward_results)
        
        # Generate report
        report = self.generate_validation_report()
        
        return {
            'walk_forward': self.walk_forward_results,
            'regime_analysis': regime_results,
            'drawdown_analysis': drawdown_results,
            'stress_testing': stress_results,
            'validation_report': report,
            'visualization_dir': viz_dir
        }
    
    def save_validation_results(self, results):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('analysis/outputs/validation_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results_file = output_dir / f'validation_results_{timestamp}.json'
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            if key == 'validation_report':
                clean_results[key] = value
            elif key == 'visualization_dir':
                clean_results[key] = str(value)
            else:
                clean_results[key] = self._clean_for_json(value)
        
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        # Save validation report
        report_file = output_dir / f'validation_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(results['validation_report'])
        
        logger.info(f"Validation results saved to {results_file}")
        logger.info(f"Validation report saved to {report_file}")
        
        return output_dir
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def main():
    """Main execution function"""
    print("📊 Starting Validation & Backtesting")
    print("Phase 5: Comprehensive Validation and Historical Backtesting")
    print("=" * 60)
    
    # Load data
    from run_analysis import SimplifiedTargetAnalyzer
    
    analyzer = SimplifiedTargetAnalyzer()
    spy, vix = analyzer.download_data()
    features = analyzer.create_features(spy, vix)
    
    # Create target matrix
    targets = pd.DataFrame(index=features.index)
    
    # Add best performing targets
    best_targets = [(0.02, 20), (0.02, 15), (0.02, 10)]
    
    for magnitude, horizon in best_targets:
        target = analyzer.create_target(spy, magnitude, horizon)
        targets[f'{int(magnitude*100)}pct_{horizon}d'] = target
    
    # Add VIX spike target
    vix_targets = analyzer.create_vix_spike_targets(vix)
    targets['vix_spike_10d'] = vix_targets['vix_spike_10d']
    
    # Initialize validation framework
    validator = ValidationBacktesting()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation(features, targets)
    
    # Save results
    output_dir = validator.save_validation_results(results)
    
    # Print results
    print("\n✅ Comprehensive Validation Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {results['visualization_dir']}")
    
    # Print validation report
    print("\n" + results['validation_report'])
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n🎉 Phase 5 Validation & Backtesting Complete!")