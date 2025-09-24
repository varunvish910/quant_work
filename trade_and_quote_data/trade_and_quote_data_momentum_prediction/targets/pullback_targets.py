#!/usr/bin/env python3
"""
Multi-Horizon Pullback Target Creator

Creates pullback prediction targets for momentum-based trading systems.
Supports multiple thresholds and horizons with comprehensive validation.

USAGE:
======
from targets.pullback_targets import PullbackTargetCreator

# Single target
creator = PullbackTargetCreator(threshold=0.05, horizon=10)
df_with_targets = creator.create_targets(df)

# Multiple thresholds and horizons
creator = PullbackTargetCreator()
targets = creator.create_multi_horizon_targets(
    df, 
    thresholds=[0.02, 0.05, 0.10],
    horizons=[5, 10, 15, 20]
)

TARGETS CREATED:
===============
1. pullback_{threshold}_{horizon}: Primary binary targets
2. pullback_probability: Probability of any pullback
3. max_decline_magnitude: Maximum decline within horizon
4. days_to_pullback: Time until pullback occurs
5. volatility_adjusted_pullback: Risk-adjusted targets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PullbackTargetCreator:
    """
    Creates pullback prediction targets for momentum-based trading systems.
    Supports multiple thresholds, horizons, and volatility adjustments.
    """
    
    def __init__(self, threshold: float = 0.05, horizon: int = 10):
        """
        Initialize pullback target creator.
        
        Args:
            threshold: Default pullback threshold (e.g., 0.05 = 5% decline)
            horizon: Default prediction horizon in days
        """
        self.threshold = threshold
        self.horizon = horizon
        self._validate_parameters(threshold, horizon)
        
        logger.info(f"PullbackTargetCreator initialized: "
                   f"threshold={self.threshold:.1%}, horizon={self.horizon}d")
    
    def create_targets(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Create pullback targets for single threshold/horizon.
        
        Args:
            df: DataFrame with price data and timestamp
            price_col: Name of price column
            
        Returns:
            DataFrame with pullback targets added
        """
        print(f"\n🎯 Creating pullback targets "
              f"(threshold={self.threshold:.1%}, horizon={self.horizon}d)...")
        
        # Validate and prepare data
        df_processed = self._prepare_data(df, price_col)
        price_col = self._find_price_column(df_processed, price_col)
        
        # Generate targets
        df_targets = self._generate_single_target(df_processed, price_col, 
                                                 self.threshold, self.horizon)
        
        # Add volatility-adjusted targets
        df_targets = self._add_volatility_adjusted_targets(df_targets, price_col)
        
        # Print statistics and validate
        self._print_target_statistics(df_targets, self.threshold, self.horizon)
        
        print(f"   ✅ Target creation completed successfully")
        return df_targets
    
    def create_multi_horizon_targets(self, df: pd.DataFrame, 
                                   thresholds: List[float] = [0.02, 0.05, 0.10],
                                   horizons: List[int] = [5, 10, 15, 20],
                                   price_col: str = 'close') -> pd.DataFrame:
        """
        Create multiple pullback targets for different thresholds and horizons.
        
        Args:
            df: DataFrame with price data
            thresholds: List of pullback thresholds (e.g., [0.02, 0.05, 0.10])
            horizons: List of prediction horizons in days
            price_col: Name of price column
            
        Returns:
            DataFrame with multiple pullback targets
        """
        print(f"\n🎯 Creating multi-horizon pullback targets...")
        print(f"   • Thresholds: {[f'{t:.1%}' for t in thresholds]}")
        print(f"   • Horizons: {horizons} days")
        
        # Validate and prepare data
        df_processed = self._prepare_data(df, price_col)
        price_col = self._find_price_column(df_processed, price_col)
        
        # Generate all target combinations
        for threshold in thresholds:
            for horizon in horizons:
                print(f"   • Generating {threshold:.1%} pullback @ {horizon}d...")
                df_processed = self._generate_single_target(
                    df_processed, price_col, threshold, horizon)
        
        # Add aggregate targets
        df_processed = self._add_aggregate_targets(df_processed, thresholds, horizons)
        
        # Add volatility-adjusted targets
        df_processed = self._add_volatility_adjusted_targets(df_processed, price_col)
        
        # Print comprehensive statistics
        self._print_multi_target_statistics(df_processed, thresholds, horizons)
        
        print(f"   ✅ Multi-horizon target creation completed")
        return df_processed
    
    def _prepare_data(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Prepare and validate input data."""
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None")
        
        # Sort by timestamp
        timestamp_col = self._find_timestamp_column(df)
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True).copy()
        
        return df_sorted
    
    def _find_timestamp_column(self, df: pd.DataFrame) -> str:
        """Find the timestamp column."""
        timestamp_candidates = [
            'sip_timestamp', 'timestamp', 'date', 'datetime', 'time',
            'Date', 'index'
        ]
        
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                return candidate
        
        # Use index if it's datetime-like
        if hasattr(df.index, 'to_pydatetime'):
            return df.index.name or 'index'
        
        raise ValueError(f"No timestamp column found. Tried: {timestamp_candidates}")
    
    def _find_price_column(self, df: pd.DataFrame, price_col: str) -> str:
        """Find and validate price column."""
        if price_col in df.columns:
            return price_col
        
        price_candidates = [
            'close', 'Close', 'daily_close', 'underlying_price', 
            'close_price', 'adj_close', 'adjusted_close'
        ]
        
        for candidate in price_candidates:
            if candidate in df.columns:
                print(f"   • Using price column: {candidate}")
                return candidate
        
        raise ValueError(f"No price column found. Tried: {[price_col] + price_candidates}")
    
    def _generate_single_target(self, df: pd.DataFrame, price_col: str,
                               threshold: float, horizon: int) -> pd.DataFrame:
        """Generate targets for single threshold/horizon combination."""
        target_name = f"pullback_{threshold:.0%}_{horizon}d".replace('%', 'pct')
        
        # Calculate future returns for this horizon
        future_returns = []
        for h in range(1, horizon + 1):
            future_price = df[price_col].shift(-h)
            return_col = f'future_return_{threshold:.0%}_{h}d'.replace('%', 'pct')
            df[return_col] = (future_price / df[price_col]) - 1
            future_returns.append(return_col)
        
        # Create pullback target: True if ANY future period shows required decline
        pullback_conditions = [df[col] < -threshold for col in future_returns]
        df[target_name] = np.any(pullback_conditions, axis=0).astype(int)
        
        # Add maximum decline magnitude
        decline_col = f'max_decline_{threshold:.0%}_{horizon}d'.replace('%', 'pct')
        df[decline_col] = np.minimum.reduce([df[col] for col in future_returns])
        
        # Add days to pullback
        days_col = f'days_to_pullback_{threshold:.0%}_{horizon}d'.replace('%', 'pct')
        df[days_col] = np.nan
        
        for i, row in df.iterrows():
            for h, ret_col in enumerate(future_returns, 1):
                if row[ret_col] < -threshold:
                    df.at[i, days_col] = h
                    break
        
        return df
    
    def _add_aggregate_targets(self, df: pd.DataFrame, 
                             thresholds: List[float], horizons: List[int]) -> pd.DataFrame:
        """Add aggregate targets across multiple thresholds/horizons."""
        
        # Any pullback target (combines all thresholds/horizons)
        pullback_cols = []
        for threshold in thresholds:
            for horizon in horizons:
                col_name = f"pullback_{threshold:.0%}_{horizon}d".replace('%', 'pct')
                if col_name in df.columns:
                    pullback_cols.append(col_name)
        
        if pullback_cols:
            df['pullback_any'] = df[pullback_cols].any(axis=1).astype(int)
            df['pullback_probability'] = df[pullback_cols].mean(axis=1)
        
        # Severity targets (small, medium, large pullbacks)
        small_threshold = min(thresholds) if thresholds else 0.02
        large_threshold = max(thresholds) if thresholds else 0.10
        
        if len(thresholds) >= 3:
            medium_threshold = sorted(thresholds)[len(thresholds)//2]
            
            df['pullback_small'] = df[f"pullback_{small_threshold:.0%}_10d".replace('%', 'pct')]
            df['pullback_medium'] = df[f"pullback_{medium_threshold:.0%}_10d".replace('%', 'pct')]
            df['pullback_large'] = df[f"pullback_{large_threshold:.0%}_10d".replace('%', 'pct')]
        
        return df
    
    def _add_volatility_adjusted_targets(self, df: pd.DataFrame, 
                                       price_col: str) -> pd.DataFrame:
        """Add volatility-adjusted pullback targets."""
        try:
            # Calculate realized volatility
            returns = df[price_col].pct_change()
            df['realized_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
            
            # Volatility-adjusted thresholds
            vol_adjusted_threshold = df['realized_vol_20d'] * 0.5  # Half of 20-day vol
            
            # Create volatility-adjusted targets for first available pullback target
            pullback_cols = [col for col in df.columns if col.startswith('pullback_') and col.endswith('d')]
            if pullback_cols:
                # Use the decline magnitude for vol-adjusted target
                decline_cols = [col for col in df.columns if col.startswith('max_decline_')]
                if decline_cols:
                    decline_col = decline_cols[0]
                    df['pullback_vol_adjusted'] = (
                        np.abs(df[decline_col]) > vol_adjusted_threshold
                    ).astype(int)
            
        except Exception as e:
            logger.warning(f"Could not create volatility-adjusted targets: {e}")
        
        return df
    
    def _print_target_statistics(self, df: pd.DataFrame, threshold: float, horizon: int):
        """Print statistics for single target."""
        target_name = f"pullback_{threshold:.0%}_{horizon}d".replace('%', 'pct')
        
        if target_name not in df.columns:
            return
        
        target_series = df[target_name].dropna()
        if len(target_series) == 0:
            return
        
        pullback_count = target_series.sum()
        pullback_rate = target_series.mean()
        
        print(f"   📈 {target_name}:")
        print(f"      • Events: {pullback_count:,} ({pullback_rate:.1%})")
        print(f"      • Valid observations: {len(target_series):,}")
        
        if pullback_rate < 0.05:
            print(f"   ⚠️  Warning: Low event rate ({pullback_rate:.1%})")
        elif pullback_rate > 0.50:
            print(f"   ⚠️  Warning: High event rate ({pullback_rate:.1%})")
        else:
            print(f"   ✅ Good balance for ML training")
    
    def _print_multi_target_statistics(self, df: pd.DataFrame, 
                                     thresholds: List[float], horizons: List[int]):
        """Print statistics for multiple targets."""
        print(f"\n   📊 Multi-Target Statistics:")
        
        for threshold in thresholds:
            print(f"\n   {threshold:.1%} Pullbacks:")
            for horizon in horizons:
                target_name = f"pullback_{threshold:.0%}_{horizon}d".replace('%', 'pct')
                if target_name in df.columns:
                    target_series = df[target_name].dropna()
                    if len(target_series) > 0:
                        rate = target_series.mean()
                        count = target_series.sum()
                        print(f"      • {horizon}d: {count:,} events ({rate:.1%})")
        
        # Aggregate statistics
        if 'pullback_any' in df.columns:
            any_rate = df['pullback_any'].mean()
            print(f"\n   📈 Any pullback rate: {any_rate:.1%}")
        
        if 'pullback_probability' in df.columns:
            avg_prob = df['pullback_probability'].mean()
            print(f"   📈 Average pullback probability: {avg_prob:.1%}")
    
    def _validate_parameters(self, threshold: float, horizon: int):
        """Validate input parameters."""
        if threshold <= 0 or threshold >= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        if horizon <= 0 or horizon > 252:
            raise ValueError(f"Horizon must be between 1 and 252 days, got {horizon}")
    
    def get_target_columns(self, thresholds: List[float] = None, 
                          horizons: List[int] = None) -> List[str]:
        """
        Get list of target column names that would be created.
        
        Args:
            thresholds: List of thresholds (uses default if None)
            horizons: List of horizons (uses default if None)
            
        Returns:
            List of target column names
        """
        if thresholds is None:
            thresholds = [self.threshold]
        if horizons is None:
            horizons = [self.horizon]
        
        columns = []
        
        # Individual targets
        for threshold in thresholds:
            for horizon in horizons:
                target_name = f"pullback_{threshold:.0%}_{horizon}d".replace('%', 'pct')
                columns.append(target_name)
                
                # Related columns
                decline_col = f'max_decline_{threshold:.0%}_{horizon}d'.replace('%', 'pct')
                days_col = f'days_to_pullback_{threshold:.0%}_{horizon}d'.replace('%', 'pct')
                columns.extend([decline_col, days_col])
        
        # Aggregate columns (only for multi-target)
        if len(thresholds) > 1 or len(horizons) > 1:
            columns.extend(['pullback_any', 'pullback_probability'])
            
            if len(thresholds) >= 3:
                columns.extend(['pullback_small', 'pullback_medium', 'pullback_large'])
        
        # Volatility-adjusted target
        columns.append('pullback_vol_adjusted')
        
        return columns
    
    def __str__(self) -> str:
        return (f"PullbackTargetCreator(threshold={self.threshold:.1%}, "
                f"horizon={self.horizon}d)")
    
    def __repr__(self) -> str:
        return (f"PullbackTargetCreator(threshold={self.threshold}, "
                f"horizon={self.horizon})")