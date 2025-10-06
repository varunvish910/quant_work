#!/usr/bin/env python3
"""
Walk-Forward Validation Framework
Phase 0 implementation from MODEL_IMPROVEMENT_ROADMAP.md

Implements proper walk-forward validation to avoid overfitting and ensure
models generalize across different market regimes.

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, precision_score, 
    recall_score, f1_score, accuracy_score
)
import lightgbm as lgb
import warnings
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Callable

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Implement proper walk-forward validation to avoid overfitting
    
    WARNING: Do NOT optimize specifically for 2024 events!
    This framework ensures robust validation across multiple time periods
    and market regimes to build generalizable models.
    """
    
    def __init__(self, 
                 train_window_years: int = 5,
                 test_window_months: int = 6,
                 step_months: int = 6,
                 min_train_samples: int = 1000):
        """
        Initialize walk-forward validation parameters
        
        Args:
            train_window_years: Years of data for training (default: 5)
            test_window_months: Months of data for testing (default: 6) 
            step_months: Months to step forward each iteration (default: 6)
            min_train_samples: Minimum samples required for training
        """
        self.train_window_years = train_window_years
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.min_train_samples = min_train_samples
        
        self.results = []
        self.fold_details = []
        self.predictions_by_fold = {}
        
        # Create output directory
        self.output_dir = Path('analysis/outputs/walk_forward_validation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_target(self, 
                       features: pd.DataFrame,
                       target_creator: Callable,
                       model_params: Dict = None,
                       start_date: str = '2010-01-01',
                       end_date: str = '2024-12-31') -> pd.DataFrame:
        """
        Run walk-forward validation for a specific target
        
        Args:
            features: Feature matrix with datetime index
            target_creator: Function that creates target labels from data
            model_params: Parameters for LightGBM model
            start_date: Start date for validation period
            end_date: End date for validation period
            
        Returns:
            DataFrame with validation results for each fold
        """
        logger.info(f"Starting walk-forward validation from {start_date} to {end_date}")
        
        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1,
                'class_weight': 'balanced'
            }
        
        # Filter data to validation period
        validation_data = features[(features.index >= start_date) & (features.index <= end_date)]
        
        if len(validation_data) < self.min_train_samples * 2:
            logger.error(f"Insufficient data for validation: {len(validation_data)} samples")
            return pd.DataFrame()
        
        # Generate time windows
        windows = self._generate_time_windows(validation_data.index, start_date, end_date)
        
        logger.info(f"Generated {len(windows)} validation windows")
        
        fold_results = []
        
        for fold_num, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Fold {fold_num + 1}/{len(windows)}: "
                       f"Train {train_start.date()}-{train_end.date()}, "
                       f"Test {test_start.date()}-{test_end.date()}")
            
            # Extract fold data
            train_data = validation_data[train_start:train_end]
            test_data = validation_data[test_start:test_end]
            
            if len(train_data) < self.min_train_samples:
                logger.warning(f"Insufficient training data in fold {fold_num + 1}: {len(train_data)}")
                continue
                
            if len(test_data) == 0:
                logger.warning(f"No test data in fold {fold_num + 1}")
                continue
            
            # Create targets for this fold
            try:
                train_labels = target_creator(train_data.index, train_data)
                test_labels = target_creator(test_data.index, test_data) 
                
                # Align with feature data
                train_features = train_data.loc[train_labels.index]
                test_features = test_data.loc[test_labels.index]
                
            except Exception as e:
                logger.error(f"Target creation failed for fold {fold_num + 1}: {e}")
                continue
            
            # Validate fold has sufficient positive cases
            if train_labels.sum() < 10:
                logger.warning(f"Too few positive cases in fold {fold_num + 1}: {train_labels.sum()}")
                continue
            
            # Train model for this fold
            fold_result = self._train_and_evaluate_fold(
                fold_num, train_features, train_labels, test_features, test_labels,
                model_params, train_start, train_end, test_start, test_end
            )
            
            if fold_result is not None:
                fold_results.append(fold_result)
        
        if not fold_results:
            logger.error("No successful validation folds")
            return pd.DataFrame()
        
        # Compile results
        results_df = pd.DataFrame(fold_results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(results_df)
        
        # Save detailed results
        self._save_validation_results(results_df, summary_stats)
        
        logger.info(f"Walk-forward validation completed: {len(fold_results)} successful folds")
        
        return results_df
    
    def _generate_time_windows(self, 
                              data_index: pd.DatetimeIndex,
                              start_date: str,
                              end_date: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate train/test windows for walk-forward validation"""
        
        windows = []
        current_date = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(end_date)
        
        while current_date < end_timestamp:
            # Define training window
            train_start = current_date
            train_end = train_start + pd.DateOffset(years=self.train_window_years)
            
            # Define test window  
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_window_months)
            
            # Check if we have enough data for this window
            if test_end > end_timestamp:
                break
                
            # Ensure we have data in both windows
            train_mask = (data_index >= train_start) & (data_index < train_end)
            test_mask = (data_index >= test_start) & (data_index < test_end)
            
            if train_mask.sum() >= self.min_train_samples and test_mask.sum() > 0:
                windows.append((train_start, train_end, test_start, test_end))
            
            # Step forward
            current_date += pd.DateOffset(months=self.step_months)
        
        return windows
    
    def _train_and_evaluate_fold(self,
                                fold_num: int,
                                train_features: pd.DataFrame,
                                train_labels: pd.Series,
                                test_features: pd.DataFrame,
                                test_labels: pd.Series,
                                model_params: Dict,
                                train_start: pd.Timestamp,
                                train_end: pd.Timestamp,
                                test_start: pd.Timestamp,
                                test_end: pd.Timestamp) -> Optional[Dict]:
        """Train model and evaluate on a single fold"""
        
        try:
            # Train model
            model = lgb.LGBMClassifier(**model_params)
            model.fit(train_features, train_labels)
            
            # Get predictions
            test_pred_proba = model.predict_proba(test_features)[:, 1]
            
            # Calculate optimal threshold using validation approach
            # (Use a small validation set from end of training period for threshold selection)
            val_size = min(len(train_features) // 5, 500)  # 20% of training or max 500 samples
            val_features = train_features.iloc[-val_size:]
            val_labels = train_labels.iloc[-val_size:]
            
            val_pred_proba = model.predict_proba(val_features)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(val_labels, val_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Apply optimal threshold to test predictions
            test_pred_binary = (test_pred_proba >= optimal_threshold).astype(int)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_fold_metrics(
                test_labels, test_pred_proba, test_pred_binary, optimal_threshold
            )
            
            # Add fold metadata
            fold_result = {
                'fold_number': fold_num + 1,
                'train_start': train_start.date(),
                'train_end': train_end.date(),
                'test_start': test_start.date(),
                'test_end': test_end.date(),
                'n_train': len(train_features),
                'n_test': len(test_features),
                'n_positive_train': train_labels.sum(),
                'n_positive_test': test_labels.sum(),
                'positive_rate_train': train_labels.mean(),
                'positive_rate_test': test_labels.mean(),
                **metrics
            }
            
            # Store predictions for later analysis
            self.predictions_by_fold[fold_num] = {
                'test_dates': test_features.index,
                'test_probabilities': test_pred_proba,
                'test_labels': test_labels,
                'optimal_threshold': optimal_threshold
            }
            
            return fold_result
            
        except Exception as e:
            logger.error(f"Fold {fold_num + 1} evaluation failed: {e}")
            return None
    
    def _calculate_fold_metrics(self, 
                               true_labels: pd.Series,
                               pred_probabilities: np.ndarray,
                               pred_binary: np.ndarray,
                               optimal_threshold: float) -> Dict:
        """Calculate comprehensive metrics for a single fold"""
        
        metrics = {}
        
        # Basic classification metrics
        try:
            metrics['roc_auc'] = roc_auc_score(true_labels, pred_probabilities)
        except:
            metrics['roc_auc'] = 0.5
            
        metrics['accuracy'] = accuracy_score(true_labels, pred_binary)
        metrics['precision'] = precision_score(true_labels, pred_binary, zero_division=0)
        metrics['recall'] = recall_score(true_labels, pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(true_labels, pred_binary, zero_division=0)
        metrics['optimal_threshold'] = optimal_threshold
        
        # High confidence metrics
        for confidence_level in [0.7, 0.8, 0.9]:
            high_conf_mask = pred_probabilities >= confidence_level
            n_high_conf = high_conf_mask.sum()
            
            metrics[f'n_signals_{int(confidence_level*100)}pct'] = n_high_conf
            
            if n_high_conf > 0:
                high_conf_precision = precision_score(
                    true_labels[high_conf_mask], 
                    (pred_probabilities >= confidence_level)[high_conf_mask],
                    zero_division=0
                )
                metrics[f'precision_{int(confidence_level*100)}pct'] = high_conf_precision
            else:
                metrics[f'precision_{int(confidence_level*100)}pct'] = 0
        
        # Distribution metrics
        metrics['mean_probability'] = pred_probabilities.mean()
        metrics['std_probability'] = pred_probabilities.std()
        metrics['max_probability'] = pred_probabilities.max()
        metrics['min_probability'] = pred_probabilities.min()
        
        return metrics
    
    def _calculate_summary_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate summary statistics across all folds"""
        
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'total_folds': len(results_df),
            'avg_metrics': results_df[numeric_cols].mean().to_dict(),
            'std_metrics': results_df[numeric_cols].std().to_dict(),
            'min_metrics': results_df[numeric_cols].min().to_dict(),
            'max_metrics': results_df[numeric_cols].max().to_dict(),
        }
        
        # Key performance indicators
        summary['stability_score'] = 1 - (results_df['roc_auc'].std() / results_df['roc_auc'].mean())
        summary['consistency_score'] = (results_df['roc_auc'] > 0.55).mean()  # Fraction above random
        
        return summary
    
    def _save_validation_results(self, results_df: pd.DataFrame, summary_stats: Dict):
        """Save validation results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f'walk_forward_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        # Save summary statistics
        summary_file = self.output_dir / f'walk_forward_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        # Save predictions
        predictions_file = self.output_dir / f'walk_forward_predictions_{timestamp}.json'
        predictions_serializable = {}
        for fold_num, pred_data in self.predictions_by_fold.items():
            predictions_serializable[fold_num] = {
                'test_dates': pred_data['test_dates'].strftime('%Y-%m-%d').tolist(),
                'test_probabilities': pred_data['test_probabilities'].tolist(),
                'test_labels': pred_data['test_labels'].tolist(),
                'optimal_threshold': pred_data['optimal_threshold']
            }
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions_serializable, f, indent=2)
        
        logger.info(f"Validation results saved to {results_file}")
        logger.info(f"Summary statistics saved to {summary_file}")
        logger.info(f"Predictions saved to {predictions_file}")
    
    def analyze_time_stability(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze model stability over time
        Check if performance degrades or improves over different periods
        """
        if len(results_df) == 0:
            return {}
        
        # Convert dates and sort by test period
        results_df['test_start_dt'] = pd.to_datetime(results_df['test_start'])
        results_df = results_df.sort_values('test_start_dt')
        
        # Calculate rolling performance metrics
        window_size = min(5, len(results_df) // 2)  # Use 5-fold rolling window or half the data
        
        if window_size < 2:
            return {'error': 'Insufficient data for time stability analysis'}
        
        rolling_metrics = {}
        for metric in ['roc_auc', 'precision', 'recall', 'f1_score']:
            rolling_metrics[f'{metric}_rolling_mean'] = results_df[metric].rolling(window_size).mean()
            rolling_metrics[f'{metric}_rolling_std'] = results_df[metric].rolling(window_size).std()
        
        # Detect trends
        roc_trend = np.polyfit(range(len(results_df)), results_df['roc_auc'], 1)[0]
        precision_trend = np.polyfit(range(len(results_df)), results_df['precision'], 1)[0]
        
        # Market regime analysis
        regime_performance = {}
        results_df['year'] = results_df['test_start_dt'].dt.year
        
        for year in results_df['year'].unique():
            year_data = results_df[results_df['year'] == year]
            regime_performance[f'year_{year}'] = {
                'n_folds': len(year_data),
                'avg_roc_auc': year_data['roc_auc'].mean(),
                'avg_precision': year_data['precision'].mean(),
                'avg_recall': year_data['recall'].mean()
            }
        
        stability_analysis = {
            'roc_auc_trend': roc_trend,
            'precision_trend': precision_trend,
            'performance_stability': results_df['roc_auc'].std(),
            'regime_performance': regime_performance,
            'best_period': {
                'test_start': results_df.loc[results_df['roc_auc'].idxmax(), 'test_start'],
                'roc_auc': results_df['roc_auc'].max()
            },
            'worst_period': {
                'test_start': results_df.loc[results_df['roc_auc'].idxmin(), 'test_start'],
                'roc_auc': results_df['roc_auc'].min()
            }
        }
        
        return stability_analysis
    
    def generate_performance_report(self, results_df: pd.DataFrame) -> str:
        """Generate comprehensive performance report"""
        
        if len(results_df) == 0:
            return "No validation results available"
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(results_df)
        time_stability = self.analyze_time_stability(results_df)
        
        report = []
        report.append("=" * 80)
        report.append("WALK-FORWARD VALIDATION PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("OVERVIEW:")
        report.append(f"Total validation folds: {summary_stats['total_folds']}")
        report.append(f"Average ROC AUC: {summary_stats['avg_metrics']['roc_auc']:.3f} ± {summary_stats['std_metrics']['roc_auc']:.3f}")
        report.append(f"Average Precision: {summary_stats['avg_metrics']['precision']:.3f} ± {summary_stats['std_metrics']['precision']:.3f}")
        report.append(f"Average Recall: {summary_stats['avg_metrics']['recall']:.3f} ± {summary_stats['std_metrics']['recall']:.3f}")
        report.append(f"Average F1 Score: {summary_stats['avg_metrics']['f1_score']:.3f} ± {summary_stats['std_metrics']['f1_score']:.3f}")
        report.append("")
        
        # Model stability
        report.append("MODEL STABILITY:")
        report.append(f"Stability Score: {summary_stats['stability_score']:.3f} (1.0 = perfect stability)")
        report.append(f"Consistency Score: {summary_stats['consistency_score']:.3f} (fraction of folds above random)")
        report.append("")
        
        # High confidence performance
        report.append("HIGH CONFIDENCE SIGNALS:")
        for conf_level in [70, 80, 90]:
            avg_signals = summary_stats['avg_metrics'].get(f'n_signals_{conf_level}pct', 0)
            avg_precision = summary_stats['avg_metrics'].get(f'precision_{conf_level}pct', 0)
            report.append(f"At {conf_level}% confidence: {avg_signals:.1f} signals/fold, {avg_precision:.3f} precision")
        report.append("")
        
        # Time stability analysis
        if 'roc_auc_trend' in time_stability:
            report.append("TIME STABILITY ANALYSIS:")
            trend_direction = "improving" if time_stability['roc_auc_trend'] > 0 else "declining"
            report.append(f"ROC AUC trend: {trend_direction} ({time_stability['roc_auc_trend']:.6f}/fold)")
            report.append(f"Performance stability (std): {time_stability['performance_stability']:.3f}")
            report.append("")
            
            report.append("Best performing period:")
            report.append(f"  Date: {time_stability['best_period']['test_start']}")
            report.append(f"  ROC AUC: {time_stability['best_period']['roc_auc']:.3f}")
            report.append("")
            
            report.append("Worst performing period:")
            report.append(f"  Date: {time_stability['worst_period']['test_start']}")
            report.append(f"  ROC AUC: {time_stability['worst_period']['roc_auc']:.3f}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        avg_roc = summary_stats['avg_metrics']['roc_auc']
        consistency = summary_stats['consistency_score']
        
        if avg_roc > 0.65 and consistency > 0.8:
            report.append("✓ Model shows strong and consistent performance")
        elif avg_roc > 0.6 and consistency > 0.7:
            report.append("✓ Model shows good performance with room for improvement")
        elif avg_roc > 0.55:
            report.append("⚠ Model shows marginal performance - consider feature engineering")
        else:
            report.append("✗ Model performance is poor - major revision needed")
        
        if summary_stats['std_metrics']['roc_auc'] > 0.1:
            report.append("⚠ High performance variability - consider ensemble methods")
        
        if time_stability.get('roc_auc_trend', 0) < -0.001:
            report.append("⚠ Performance declining over time - check for regime changes")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def test_walk_forward_validator():
    """Test the walk-forward validation framework"""
    logger.info("Testing Walk-Forward Validator")
    
    # Create dummy data for testing
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='D')
    n_features = 10
    
    # Create synthetic features
    features = pd.DataFrame(
        np.random.randn(len(dates), n_features),
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some trend to make it more realistic
    features['feature_0'] = np.cumsum(np.random.randn(len(dates)) * 0.01)
    
    # Create simple target creator for testing
    def simple_target_creator(date_index, feature_data):
        """Create binary target based on feature_0 > threshold"""
        target = (feature_data['feature_0'] > feature_data['feature_0'].rolling(20).mean()).astype(int)
        return target.reindex(date_index).fillna(0)
    
    # Initialize validator
    validator = WalkForwardValidator(
        train_window_years=3,
        test_window_months=6,
        step_months=6
    )
    
    # Run validation
    results = validator.validate_target(
        features=features,
        target_creator=simple_target_creator,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    if len(results) > 0:
        print(f"\nValidation completed successfully!")
        print(f"Number of folds: {len(results)}")
        print(f"Average ROC AUC: {results['roc_auc'].mean():.3f}")
        
        # Generate report
        report = validator.generate_performance_report(results)
        print("\n" + report)
        
        return True
    else:
        print("Validation failed!")
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    success = test_walk_forward_validator()
    
    if success:
        print("\nWalk-forward validation framework test completed successfully!")
    else:
        print("\nWalk-forward validation framework test failed!")