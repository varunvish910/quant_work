#!/usr/bin/env python3
"""
Multi-Target Labeling System
Phase 0 implementation from MODEL_IMPROVEMENT_ROADMAP.md

Creates labels for all 12 different magnitude/horizon combinations systematically.
Supports both pullback and momentum-based targets.

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import logging
from datetime import datetime
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MultiTargetLabeler:
    """
    Systematic target labeling for multiple magnitude/horizon combinations
    
    Creates binary labels for:
    - Pullback magnitudes: 2%, 5%, 10%
    - Time horizons: 5, 10, 15, 20 days
    - Total combinations: 12 different prediction targets
    
    Also supports additional target types for comparison:
    - Momentum continuation targets
    - Volatility spike targets
    - Mean reversion targets
    """
    
    def __init__(self):
        self.pullback_magnitudes = [0.02, 0.05, 0.10]  # 2%, 5%, 10%
        self.time_horizons = [5, 10, 15, 20]           # days
        
        # Additional target configurations
        self.momentum_magnitudes = [0.03, 0.05, 0.08]  # 3%, 5%, 8% upward moves
        self.volatility_thresholds = [1.5, 2.0, 3.0]   # VIX spike multiples
        
        self.target_cache = {}
        
        # Output directory for target analysis
        self.output_dir = Path('analysis/outputs/target_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_pullback_labels(self, 
                              price_data: pd.DataFrame,
                              magnitude: float,
                              horizon: int,
                              use_intraday_lows: bool = True) -> pd.Series:
        """
        Create binary labels for pullback prediction
        
        Args:
            price_data: DataFrame with OHLC data
            magnitude: Pullback threshold (e.g., 0.05 for 5%)
            horizon: Days to look forward
            use_intraday_lows: If True, use Low prices; if False, use Close prices
            
        Returns:
            Binary series (1 = pullback occurs, 0 = no pullback)
        """
        target_name = f'pullback_{int(magnitude*100)}pct_{horizon}d'
        
        # Check cache
        cache_key = f"{target_name}_{id(price_data)}"
        if cache_key in self.target_cache:
            return self.target_cache[cache_key]
        
        labels = pd.Series(0, index=price_data.index, name=target_name)
        
        for i in range(len(price_data) - horizon):
            current_date = price_data.index[i]
            current_price = price_data['Close'].iloc[i]
            
            # Look forward 'horizon' days
            future_slice = price_data.iloc[i+1:i+horizon+1]
            if len(future_slice) == 0:
                continue
            
            # Find minimum price in the forward window
            if use_intraday_lows:
                min_future_price = future_slice['Low'].min()
            else:
                min_future_price = future_slice['Close'].min()
            
            # Calculate maximum drawdown
            drawdown = (min_future_price / current_price) - 1
            
            # Label as 1 if drawdown exceeds threshold
            if drawdown <= -magnitude:
                labels.iloc[i] = 1
        
        # Cache result
        self.target_cache[cache_key] = labels
        return labels
    
    def create_momentum_labels(self,
                              price_data: pd.DataFrame,
                              magnitude: float,
                              horizon: int) -> pd.Series:
        """
        Create labels for momentum continuation (upward moves)
        
        Args:
            price_data: DataFrame with OHLC data
            magnitude: Minimum upward move (e.g., 0.05 for 5%)
            horizon: Days to look forward
            
        Returns:
            Binary series (1 = momentum continues, 0 = no significant upward move)
        """
        target_name = f'momentum_{int(magnitude*100)}pct_{horizon}d'
        labels = pd.Series(0, index=price_data.index, name=target_name)
        
        for i in range(len(price_data) - horizon):
            current_price = price_data['Close'].iloc[i]
            
            # Look forward 'horizon' days
            future_slice = price_data.iloc[i+1:i+horizon+1]
            if len(future_slice) == 0:
                continue
            
            # Find maximum price in the forward window
            max_future_price = future_slice['High'].max()
            
            # Calculate maximum gain
            gain = (max_future_price / current_price) - 1
            
            # Label as 1 if gain exceeds threshold
            if gain >= magnitude:
                labels.iloc[i] = 1
        
        return labels
    
    def create_volatility_spike_labels(self,
                                      price_data: pd.DataFrame,
                                      vix_data: pd.Series,
                                      threshold_multiple: float,
                                      horizon: int) -> pd.Series:
        """
        Create labels for volatility spike prediction
        
        Args:
            price_data: DataFrame with OHLC data
            vix_data: VIX time series
            threshold_multiple: Multiple of VIX moving average (e.g., 1.5)
            horizon: Days to look forward
            
        Returns:
            Binary series (1 = volatility spike occurs, 0 = no spike)
        """
        target_name = f'vol_spike_{threshold_multiple}x_{horizon}d'
        labels = pd.Series(0, index=price_data.index, name=target_name)
        
        # Align VIX data with price data
        aligned_vix = vix_data.reindex(price_data.index, method='ffill')
        vix_ma = aligned_vix.rolling(20).mean()
        
        for i in range(len(price_data) - horizon):
            current_vix_threshold = vix_ma.iloc[i] * threshold_multiple
            
            # Look forward 'horizon' days
            future_vix = aligned_vix.iloc[i+1:i+horizon+1]
            if len(future_vix) == 0:
                continue
            
            # Check if VIX exceeds threshold in the forward window
            if future_vix.max() >= current_vix_threshold:
                labels.iloc[i] = 1
        
        return labels
    
    def create_mean_reversion_labels(self,
                                   price_data: pd.DataFrame,
                                   horizon: int,
                                   reversion_threshold: float = 0.02) -> pd.Series:
        """
        Create labels for mean reversion after extreme moves
        
        Args:
            price_data: DataFrame with OHLC data
            horizon: Days to look forward
            reversion_threshold: Minimum reversion (e.g., 0.02 for 2%)
            
        Returns:
            Binary series (1 = mean reversion occurs, 0 = no reversion)
        """
        target_name = f'mean_revert_{int(reversion_threshold*100)}pct_{horizon}d'
        labels = pd.Series(0, index=price_data.index, name=target_name)
        
        # Calculate distance from 20-day moving average
        sma20 = price_data['Close'].rolling(20).mean()
        distance_from_mean = (price_data['Close'] / sma20) - 1
        
        for i in range(len(price_data) - horizon):
            current_distance = distance_from_mean.iloc[i]
            
            # Only label if currently extended from mean (>2% above or below)
            if abs(current_distance) < 0.02:
                continue
            
            current_price = price_data['Close'].iloc[i]
            mean_price = sma20.iloc[i]
            
            # Look forward 'horizon' days
            future_prices = price_data['Close'].iloc[i+1:i+horizon+1]
            if len(future_prices) == 0:
                continue
            
            # Check for reversion toward mean
            if current_distance > 0:  # Above mean, look for move down
                min_future = future_prices.min()
                reversion = (current_price - min_future) / current_price
            else:  # Below mean, look for move up
                max_future = future_prices.max()
                reversion = (max_future - current_price) / current_price
            
            if reversion >= reversion_threshold:
                labels.iloc[i] = 1
        
        return labels
    
    def create_all_pullback_targets(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create all 12 pullback target combinations
        
        Returns:
            Dictionary mapping target names to label series
        """
        logger.info("Creating all pullback target combinations")
        
        targets = {}
        
        for magnitude in self.pullback_magnitudes:
            for horizon in self.time_horizons:
                target_name = f'pullback_{int(magnitude*100)}pct_{horizon}d'
                logger.info(f"Creating {target_name}")
                
                targets[target_name] = self.create_pullback_labels(
                    price_data, magnitude, horizon
                )
        
        logger.info(f"Created {len(targets)} pullback targets")
        return targets
    
    def create_all_momentum_targets(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create all momentum target combinations"""
        logger.info("Creating momentum target combinations")
        
        targets = {}
        
        for magnitude in self.momentum_magnitudes:
            for horizon in self.time_horizons:
                target_name = f'momentum_{int(magnitude*100)}pct_{horizon}d'
                logger.info(f"Creating {target_name}")
                
                targets[target_name] = self.create_momentum_labels(
                    price_data, magnitude, horizon
                )
        
        logger.info(f"Created {len(targets)} momentum targets")
        return targets
    
    def analyze_target_characteristics(self, 
                                     targets: Dict[str, pd.Series],
                                     price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze characteristics of different targets
        
        Returns:
            DataFrame with target statistics
        """
        logger.info("Analyzing target characteristics")
        
        analysis_results = []
        
        for target_name, target_series in targets.items():
            # Parse target parameters
            parts = target_name.split('_')
            target_type = parts[0]
            magnitude = int(parts[1].replace('pct', ''))
            horizon = int(parts[2].replace('d', ''))
            
            # Calculate basic statistics
            total_samples = len(target_series)
            positive_samples = target_series.sum()
            positive_rate = positive_samples / total_samples if total_samples > 0 else 0
            
            # Calculate clustering (consecutive positive labels)
            clusters = self._find_label_clusters(target_series)
            
            # Calculate temporal distribution
            temporal_dist = self._analyze_temporal_distribution(target_series)
            
            # Calculate market context
            market_context = self._analyze_market_context(target_series, price_data)
            
            analysis_results.append({
                'target_name': target_name,
                'target_type': target_type,
                'magnitude_pct': magnitude,
                'horizon_days': horizon,
                'total_samples': total_samples,
                'positive_samples': positive_samples,
                'positive_rate': positive_rate,
                'n_clusters': len(clusters),
                'avg_cluster_size': np.mean([c['size'] for c in clusters]) if clusters else 0,
                'max_cluster_size': max([c['size'] for c in clusters]) if clusters else 0,
                'avg_gap_between_clusters': np.mean([c['gap_after'] for c in clusters[:-1]]) if len(clusters) > 1 else 0,
                **temporal_dist,
                **market_context
            })
        
        results_df = pd.DataFrame(analysis_results)
        
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.output_dir / f'target_characteristics_{timestamp}.csv'
        results_df.to_csv(analysis_file, index=False)
        
        logger.info(f"Target analysis saved to {analysis_file}")
        
        return results_df
    
    def _find_label_clusters(self, target_series: pd.Series) -> List[Dict]:
        """Find clusters of consecutive positive labels"""
        clusters = []
        in_cluster = False
        cluster_start = None
        cluster_size = 0
        
        for i, label in enumerate(target_series):
            if label == 1:
                if not in_cluster:
                    in_cluster = True
                    cluster_start = i
                    cluster_size = 1
                else:
                    cluster_size += 1
            else:
                if in_cluster:
                    # End of cluster
                    gap_after = 0
                    # Calculate gap to next cluster
                    for j in range(i, len(target_series)):
                        if target_series.iloc[j] == 1:
                            gap_after = j - i
                            break
                        elif j == len(target_series) - 1:
                            gap_after = len(target_series) - i
                    
                    clusters.append({
                        'start': cluster_start,
                        'size': cluster_size,
                        'gap_after': gap_after
                    })
                    
                    in_cluster = False
                    cluster_size = 0
        
        # Handle case where series ends in cluster
        if in_cluster:
            clusters.append({
                'start': cluster_start,
                'size': cluster_size,
                'gap_after': 0
            })
        
        return clusters
    
    def _analyze_temporal_distribution(self, target_series: pd.Series) -> Dict:
        """Analyze temporal distribution of positive labels"""
        positive_indices = target_series[target_series == 1].index
        
        if len(positive_indices) == 0:
            return {
                'seasonal_month_effect': 0,
                'seasonal_quarter_effect': 0,
                'day_of_week_effect': 0
            }
        
        # Monthly distribution
        months = positive_indices.month.value_counts()
        month_bias = months.std() / months.mean() if len(months) > 0 else 0
        
        # Quarterly distribution
        quarters = positive_indices.quarter.value_counts()
        quarter_bias = quarters.std() / quarters.mean() if len(quarters) > 0 else 0
        
        # Day of week distribution
        dow = positive_indices.dayofweek.value_counts()
        dow_bias = dow.std() / dow.mean() if len(dow) > 0 else 0
        
        return {
            'seasonal_month_effect': month_bias,
            'seasonal_quarter_effect': quarter_bias,
            'day_of_week_effect': dow_bias
        }
    
    def _analyze_market_context(self, target_series: pd.Series, price_data: pd.DataFrame) -> Dict:
        """Analyze market context when labels are positive"""
        positive_dates = target_series[target_series == 1].index
        
        if len(positive_dates) == 0:
            return {
                'avg_volatility_context': 0,
                'avg_momentum_context': 0,
                'avg_distance_from_high': 0
            }
        
        # Calculate market metrics at positive label dates
        returns = price_data['Close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        momentum_10d = price_data['Close'].pct_change(10)
        distance_from_high = price_data['Close'] / price_data['High'].rolling(20).max() - 1
        
        # Get average values at positive label dates
        avg_vol = volatility.loc[positive_dates].mean()
        avg_momentum = momentum_10d.loc[positive_dates].mean()
        avg_dist_high = distance_from_high.loc[positive_dates].mean()
        
        return {
            'avg_volatility_context': avg_vol,
            'avg_momentum_context': avg_momentum,
            'avg_distance_from_high': avg_dist_high
        }
    
    def create_target_comparison_report(self, analysis_df: pd.DataFrame) -> str:
        """Generate comprehensive target comparison report"""
        
        report = []
        report.append("=" * 80)
        report.append("MULTI-TARGET ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        total_targets = len(analysis_df)
        avg_positive_rate = analysis_df['positive_rate'].mean()
        report.append(f"Total targets analyzed: {total_targets}")
        report.append(f"Average positive rate: {avg_positive_rate:.3f}")
        report.append("")
        
        # Best targets by different criteria
        report.append("TOP TARGETS BY CRITERIA:")
        
        # Most frequent signals
        top_frequent = analysis_df.nlargest(3, 'positive_rate')
        report.append("Most frequent signals:")
        for _, row in top_frequent.iterrows():
            report.append(f"  {row['target_name']}: {row['positive_rate']:.3f} positive rate")
        report.append("")
        
        # Least clustered (more independent signals)
        analysis_df['clustering_score'] = analysis_df['avg_cluster_size'] / analysis_df['positive_rate']
        least_clustered = analysis_df.nsmallest(3, 'clustering_score')
        report.append("Least clustered signals (more independent):")
        for _, row in least_clustered.iterrows():
            report.append(f"  {row['target_name']}: {row['clustering_score']:.2f} clustering score")
        report.append("")
        
        # Balanced frequency
        balanced_targets = analysis_df[
            (analysis_df['positive_rate'] >= 0.05) & 
            (analysis_df['positive_rate'] <= 0.20)
        ].copy()
        
        if len(balanced_targets) > 0:
            report.append("Balanced frequency targets (5-20% positive rate):")
            for _, row in balanced_targets.iterrows():
                report.append(f"  {row['target_name']}: {row['positive_rate']:.3f} positive rate")
        report.append("")
        
        # Magnitude vs Horizon analysis
        report.append("MAGNITUDE vs HORIZON ANALYSIS:")
        pullback_targets = analysis_df[analysis_df['target_type'] == 'pullback'].copy()
        
        if len(pullback_targets) > 0:
            for magnitude in [2, 5, 10]:
                mag_targets = pullback_targets[pullback_targets['magnitude_pct'] == magnitude]
                if len(mag_targets) > 0:
                    report.append(f"{magnitude}% pullback targets:")
                    for _, row in mag_targets.iterrows():
                        report.append(f"  {row['horizon_days']}d: {row['positive_rate']:.3f} positive rate, "
                                    f"{row['n_clusters']} clusters")
                    report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        
        # Find optimal balanced target
        if len(balanced_targets) > 0:
            optimal = balanced_targets.loc[balanced_targets['positive_rate'].idxmax()]
            report.append(f"Best balanced target: {optimal['target_name']}")
            report.append(f"  - Positive rate: {optimal['positive_rate']:.3f}")
            report.append(f"  - Number of clusters: {optimal['n_clusters']}")
            report.append(f"  - Average cluster size: {optimal['avg_cluster_size']:.1f}")
        
        # Strategy recommendations
        high_freq = analysis_df[analysis_df['positive_rate'] > 0.15]
        low_freq = analysis_df[analysis_df['positive_rate'] < 0.05]
        
        if len(high_freq) > 0:
            best_high_freq = high_freq.loc[high_freq['positive_rate'].idxmax()]
            report.append(f"\nFor active trading: {best_high_freq['target_name']}")
            report.append(f"  - High signal frequency: {best_high_freq['positive_rate']:.3f}")
        
        if len(low_freq) > 0:
            best_low_freq = low_freq.loc[low_freq['n_clusters'].idxmax()]
            report.append(f"\nFor crash detection: {best_low_freq['target_name']}")
            report.append(f"  - Low but significant signals: {best_low_freq['positive_rate']:.3f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def test_multi_target_labeler():
    """Test the multi-target labeling system"""
    logger.info("Testing Multi-Target Labeler")
    
    # Download SPY data for testing
    spy_data = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    vix_data = yf.download('^VIX', start='2020-01-01', end='2024-12-31', progress=False)['Close']
    
    # Initialize labeler
    labeler = MultiTargetLabeler()
    
    # Test pullback targets
    print("Creating pullback targets...")
    pullback_targets = labeler.create_all_pullback_targets(spy_data)
    print(f"Created {len(pullback_targets)} pullback targets")
    
    # Test momentum targets  
    print("Creating momentum targets...")
    momentum_targets = labeler.create_all_momentum_targets(spy_data)
    print(f"Created {len(momentum_targets)} momentum targets")
    
    # Test individual target types
    print("\nTesting individual target creation...")
    
    # Test volatility spike target
    vol_spike = labeler.create_volatility_spike_labels(spy_data, vix_data, 1.5, 10)
    print(f"Volatility spike target: {vol_spike.sum()} positive labels")
    
    # Test mean reversion target
    mean_revert = labeler.create_mean_reversion_labels(spy_data, 10)
    print(f"Mean reversion target: {mean_revert.sum()} positive labels")
    
    # Analyze all targets
    all_targets = {**pullback_targets, **momentum_targets}
    
    print(f"\nAnalyzing {len(all_targets)} targets...")
    analysis_df = labeler.analyze_target_characteristics(all_targets, spy_data)
    
    # Generate report
    report = labeler.create_target_comparison_report(analysis_df)
    print(report)
    
    # Show sample statistics
    print("\nSample Target Statistics:")
    print(analysis_df[['target_name', 'positive_rate', 'n_clusters', 'avg_cluster_size']].head(10))
    
    return True


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    success = test_multi_target_labeler()
    
    if success:
        print("\nMulti-target labeling system test completed successfully!")
    else:
        print("\nMulti-target labeling system test failed!")