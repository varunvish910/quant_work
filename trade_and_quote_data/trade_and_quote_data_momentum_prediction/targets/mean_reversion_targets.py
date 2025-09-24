#!/usr/bin/env python3
"""
Mean Reversion Target Creator

Creates mean reversion prediction targets for momentum-based trading systems.
Predicts probability of price returning to various moving averages.

USAGE:
======
from targets.mean_reversion_targets import MeanReversionTargetCreator

# Basic usage
creator = MeanReversionTargetCreator()
df_with_targets = creator.create_targets(df)

# Custom SMA periods and horizons
creator = MeanReversionTargetCreator(sma_periods=[20, 50, 100, 200])
targets = creator.create_targets(df, horizons=[5, 10, 20])

TARGETS CREATED:
===============
1. mean_revert_smaX_Yd: Binary target for reversion to SMA-X within Y days
2. reversion_probability_smaX: Probability of reverting to SMA-X
3. distance_from_smaX: Current distance from moving average (%)
4. reversion_magnitude: How much price needs to move to reach SMA
5. vol_adjusted_reversion: Risk-adjusted reversion targets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MeanReversionTargetCreator:
    """
    Creates mean reversion targets for momentum-based trading systems.
    Predicts when price will revert to key moving averages.
    """
    
    def __init__(self, sma_periods: List[int] = [20, 50, 100, 200],
                 reversion_threshold: float = 0.01):
        """
        Initialize mean reversion target creator.
        
        Args:
            sma_periods: List of SMA periods to calculate reversion for
            reversion_threshold: Threshold for considering reversion (1% = 0.01)
        """
        self.sma_periods = sma_periods
        self.reversion_threshold = reversion_threshold
        
        if not sma_periods:
            raise ValueError("At least one SMA period must be specified")
        
        logger.info(f"MeanReversionTargetCreator initialized: "
                   f"SMAs={self.sma_periods}, threshold={self.reversion_threshold:.2%}")
    
    def create_targets(self, df: pd.DataFrame, 
                      horizons: List[int] = [5, 10, 15, 20],
                      price_col: str = 'close') -> pd.DataFrame:
        """
        Create mean reversion targets for specified horizons.
        
        Args:
            df: DataFrame with price data
            horizons: List of prediction horizons in days
            price_col: Name of price column
            
        Returns:
            DataFrame with mean reversion targets added
        """
        print(f"\n🎯 Creating mean reversion targets...")
        print(f"   • SMA periods: {self.sma_periods}")
        print(f"   • Horizons: {horizons} days")
        print(f"   • Reversion threshold: {self.reversion_threshold:.2%}")
        
        # Prepare data
        df_processed = self._prepare_data(df, price_col)
        price_col = self._find_price_column(df_processed, price_col)
        
        # Calculate SMAs and distance metrics
        df_processed = self._calculate_sma_features(df_processed, price_col)
        
        # Generate reversion targets for each SMA/horizon combination
        for sma_period in self.sma_periods:
            for horizon in horizons:
                df_processed = self._generate_reversion_target(
                    df_processed, price_col, sma_period, horizon)
        
        # Add aggregate and probability targets
        df_processed = self._add_aggregate_reversion_targets(df_processed, horizons)
        
        # Add volatility-adjusted targets
        df_processed = self._add_volatility_adjusted_reversion(df_processed, price_col)
        
        # Add momentum-based reversion signals
        df_processed = self._add_momentum_reversion_signals(df_processed, price_col)
        
        # Print statistics
        self._print_reversion_statistics(df_processed, horizons)
        
        print(f"   ✅ Mean reversion target creation completed")
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
    
    def _calculate_sma_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Calculate SMAs and distance metrics."""
        for period in self.sma_periods:
            sma_col = f'sma_{period}'
            distance_col = f'distance_from_sma{period}'
            distance_pct_col = f'distance_from_sma{period}_pct'
            
            # Calculate SMA
            df[sma_col] = df[price_col].rolling(window=period).mean()
            
            # Distance from SMA (absolute)
            df[distance_col] = df[price_col] - df[sma_col]
            
            # Distance from SMA (percentage)
            df[distance_pct_col] = (df[price_col] / df[sma_col] - 1) * 100
            
            # Above/below SMA indicator
            df[f'above_sma{period}'] = (df[price_col] > df[sma_col]).astype(int)
            
        return df
    
    def _generate_reversion_target(self, df: pd.DataFrame, price_col: str,
                                  sma_period: int, horizon: int) -> pd.DataFrame:
        """Generate reversion target for specific SMA/horizon combination."""
        sma_col = f'sma_{sma_period}'
        target_col = f'mean_revert_sma{sma_period}_{horizon}d'
        prob_col = f'reversion_probability_sma{sma_period}_{horizon}d'
        
        if sma_col not in df.columns:
            logger.warning(f"SMA column {sma_col} not found")
            return df
        
        # For each observation, check if price reverts to SMA within horizon
        reversion_occurred = []
        reversion_probabilities = []
        
        for i in range(len(df)):
            current_price = df.loc[i, price_col]
            current_sma = df.loc[i, sma_col]
            
            if pd.isna(current_price) or pd.isna(current_sma):
                reversion_occurred.append(np.nan)
                reversion_probabilities.append(np.nan)
                continue
            
            # Check if already near SMA (within threshold)
            distance_pct = abs(current_price / current_sma - 1)
            if distance_pct <= self.reversion_threshold:
                reversion_occurred.append(1)
                reversion_probabilities.append(1.0)
                continue
            
            # Look ahead for reversion within horizon
            reverted = False
            reversion_count = 0
            valid_periods = 0
            
            for h in range(1, horizon + 1):
                if i + h >= len(df):
                    break
                
                future_price = df.loc[i + h, price_col]
                future_sma = df.loc[i + h, sma_col]
                
                if pd.isna(future_price) or pd.isna(future_sma):
                    continue
                
                valid_periods += 1
                
                # Check if price has reverted (within threshold of SMA)
                future_distance_pct = abs(future_price / future_sma - 1)
                if future_distance_pct <= self.reversion_threshold:
                    reversion_count += 1
                    reverted = True
            
            reversion_occurred.append(1 if reverted else 0)
            reversion_probabilities.append(
                reversion_count / valid_periods if valid_periods > 0 else 0
            )
        
        df[target_col] = reversion_occurred
        df[prob_col] = reversion_probabilities
        
        # Add magnitude of reversion needed
        reversion_magnitude_col = f'reversion_magnitude_sma{sma_period}'
        df[reversion_magnitude_col] = abs(df[price_col] / df[sma_col] - 1) * 100
        
        return df
    
    def _add_aggregate_reversion_targets(self, df: pd.DataFrame, 
                                       horizons: List[int]) -> pd.DataFrame:
        """Add aggregate reversion targets across SMAs."""
        
        for horizon in horizons:
            # Any reversion target (to any SMA)
            reversion_cols = [f'mean_revert_sma{p}_{horizon}d' for p in self.sma_periods]
            existing_cols = [col for col in reversion_cols if col in df.columns]
            
            if existing_cols:
                df[f'mean_revert_any_{horizon}d'] = df[existing_cols].any(axis=1).astype(int)
                df[f'mean_revert_probability_{horizon}d'] = df[existing_cols].mean(axis=1)
            
            # Reversion to key SMAs (20, 50, 200 if available)
            key_smas = [20, 50, 200]
            key_reversion_cols = []
            for sma in key_smas:
                if sma in self.sma_periods:
                    col = f'mean_revert_sma{sma}_{horizon}d'
                    if col in df.columns:
                        key_reversion_cols.append(col)
            
            if key_reversion_cols:
                df[f'mean_revert_key_smas_{horizon}d'] = df[key_reversion_cols].any(axis=1).astype(int)
        
        return df
    
    def _add_volatility_adjusted_reversion(self, df: pd.DataFrame, 
                                         price_col: str) -> pd.DataFrame:
        """Add volatility-adjusted reversion targets."""
        try:
            # Calculate realized volatility
            returns = df[price_col].pct_change()
            df['realized_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
            
            # Volatility-adjusted reversion threshold
            for period in self.sma_periods:
                distance_col = f'distance_from_sma{period}_pct'
                vol_adj_col = f'vol_adjusted_reversion_sma{period}'
                
                if distance_col in df.columns:
                    # Significant reversion = distance > 1 standard deviation
                    df[vol_adj_col] = (
                        abs(df[distance_col]) > df['realized_vol_20d'] * 100
                    ).astype(int)
            
        except Exception as e:
            logger.warning(f"Could not create volatility-adjusted reversion targets: {e}")
        
        return df
    
    def _add_momentum_reversion_signals(self, df: pd.DataFrame, 
                                      price_col: str) -> pd.DataFrame:
        """Add momentum-based reversion signals."""
        try:
            # RSI-based reversion signals
            rsi = self._calculate_rsi(df[price_col], 14)
            df['rsi_oversold_reversion'] = (rsi < 30).astype(int)
            df['rsi_overbought_reversion'] = (rsi > 70).astype(int)
            
            # Bollinger Band reversion signals
            bb_data = self._calculate_bollinger_bands(df[price_col], 20, 2)
            df['bb_lower_touch'] = (df[price_col] <= bb_data['lower']).astype(int)
            df['bb_upper_touch'] = (df[price_col] >= bb_data['upper']).astype(int)
            
            # Price extension signals (far from SMA20)
            if 'distance_from_sma20_pct' in df.columns:
                df['extended_above_sma20'] = (df['distance_from_sma20_pct'] > 5).astype(int)
                df['extended_below_sma20'] = (df['distance_from_sma20_pct'] < -5).astype(int)
            
        except Exception as e:
            logger.warning(f"Could not create momentum reversion signals: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev)
        }
    
    def _print_reversion_statistics(self, df: pd.DataFrame, horizons: List[int]):
        """Print comprehensive reversion statistics."""
        print(f"\n   📊 Mean Reversion Statistics:")
        
        # Individual SMA reversion rates
        for period in self.sma_periods:
            print(f"\n   SMA-{period} Reversion Rates:")
            for horizon in horizons:
                target_col = f'mean_revert_sma{period}_{horizon}d'
                if target_col in df.columns:
                    target_series = df[target_col].dropna()
                    if len(target_series) > 0:
                        rate = target_series.mean()
                        count = target_series.sum()
                        print(f"      • {horizon}d: {count:,} events ({rate:.1%})")
        
        # Aggregate statistics
        for horizon in horizons:
            any_col = f'mean_revert_any_{horizon}d'
            if any_col in df.columns:
                any_rate = df[any_col].mean()
                print(f"\n   📈 Any reversion @ {horizon}d: {any_rate:.1%}")
        
        # Distance statistics
        for period in self.sma_periods:
            distance_col = f'distance_from_sma{period}_pct'
            if distance_col in df.columns:
                distances = df[distance_col].dropna()
                if len(distances) > 0:
                    avg_distance = abs(distances).mean()
                    print(f"   📏 Avg distance from SMA-{period}: {avg_distance:.1f}%")
                    break  # Only print for first SMA to avoid clutter
    
    def get_reversion_columns(self, horizons: List[int] = [5, 10, 15, 20]) -> List[str]:
        """
        Get list of mean reversion column names that would be created.
        
        Args:
            horizons: List of prediction horizons
            
        Returns:
            List of column names
        """
        columns = []
        
        # SMA and distance columns
        for period in self.sma_periods:
            columns.extend([
                f'sma_{period}',
                f'distance_from_sma{period}',
                f'distance_from_sma{period}_pct',
                f'above_sma{period}',
                f'reversion_magnitude_sma{period}'
            ])
        
        # Individual reversion targets
        for period in self.sma_periods:
            for horizon in horizons:
                columns.extend([
                    f'mean_revert_sma{period}_{horizon}d',
                    f'reversion_probability_sma{period}_{horizon}d'
                ])
        
        # Aggregate targets
        for horizon in horizons:
            columns.extend([
                f'mean_revert_any_{horizon}d',
                f'mean_revert_probability_{horizon}d',
                f'mean_revert_key_smas_{horizon}d'
            ])
        
        # Volatility-adjusted targets
        for period in self.sma_periods:
            columns.append(f'vol_adjusted_reversion_sma{period}')
        
        # Momentum reversion signals
        columns.extend([
            'rsi_oversold_reversion',
            'rsi_overbought_reversion',
            'bb_lower_touch',
            'bb_upper_touch',
            'extended_above_sma20',
            'extended_below_sma20',
            'realized_vol_20d'
        ])
        
        return columns
    
    def __str__(self) -> str:
        return (f"MeanReversionTargetCreator(sma_periods={self.sma_periods}, "
                f"threshold={self.reversion_threshold:.2%})")
    
    def __repr__(self) -> str:
        return (f"MeanReversionTargetCreator(sma_periods={self.sma_periods}, "
                f"reversion_threshold={self.reversion_threshold})")