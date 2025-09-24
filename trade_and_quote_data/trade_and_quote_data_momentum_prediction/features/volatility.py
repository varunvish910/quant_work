#!/usr/bin/env python3
"""
Volatility Features Engine

Extracts volatility-based features for pullback prediction systems.
Focuses on volatility regimes, expansion/contraction, and volatility momentum.

USAGE:
======
from features.volatility import VolatilityFeatureEngine

engine = VolatilityFeatureEngine()
df_with_features = engine.add_features(df)

# Get feature names
feature_names = engine.get_feature_names()
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityFeatureEngine:
    """
    Volatility feature engine for pullback prediction.
    
    Creates volatility indicators focused on:
    - Realized volatility across multiple timeframes
    - Volatility momentum and rate of change
    - Volatility regimes and expansion/contraction
    - ATR and Bollinger Band features
    """
    
    def __init__(self, name: str = "Volatility Features"):
        """Initialize volatility feature engine."""
        self.name = name
        logger.info(f"Initialized {self.name}")
    
    def add_features(self, df: pd.DataFrame, 
                    price_col: str = 'close',
                    high_col: str = None,
                    low_col: str = None) -> pd.DataFrame:
        """
        Add volatility features to DataFrame.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            high_col: Name of high column (optional)
            low_col: Name of low column (optional)
            
        Returns:
            DataFrame with volatility features added
        """
        print(f"\n📊 Adding {self.name}...")
        
        df_processed = df.copy()
        
        # Find price columns
        price_col = self._find_price_column(df_processed, price_col)
        high_col = self._find_ohlc_column(df_processed, high_col, 'high')
        low_col = self._find_ohlc_column(df_processed, low_col, 'low')
        
        # Sort data by timestamp
        df_processed = self._prepare_data(df_processed)
        
        # Add volatility features
        df_processed = self._add_realized_volatility(df_processed, price_col)
        df_processed = self._add_volatility_momentum(df_processed)
        df_processed = self._add_volatility_regimes(df_processed)
        df_processed = self._add_atr_features(df_processed, price_col, high_col, low_col)
        df_processed = self._add_bollinger_band_features(df_processed, price_col)
        df_processed = self._add_volatility_signals(df_processed, price_col)
        df_processed = self._add_advanced_volatility_features(df_processed, price_col)
        
        print(f"   ✅ Added {len(self.get_feature_names())} volatility features")
        
        return df_processed
    
    def _find_price_column(self, df: pd.DataFrame, price_col: str) -> str:
        """Find and validate price column."""
        if price_col in df.columns:
            return price_col
        
        price_candidates = [
            'close', 'Close', 'daily_close', 'underlying_price', 
            'close_price', 'adj_close', 'adjusted_close', 'price'
        ]
        
        for candidate in price_candidates:
            if candidate in df.columns:
                print(f"   • Using price column: {candidate}")
                return candidate
        
        raise ValueError(f"No price column found. Tried: {[price_col] + price_candidates}")
    
    def _find_ohlc_column(self, df: pd.DataFrame, col: Optional[str], col_type: str) -> Optional[str]:
        """Find OHLC columns if available."""
        if col and col in df.columns:
            return col
        
        candidates = {
            'high': ['high', 'High', 'daily_high'],
            'low': ['low', 'Low', 'daily_low'],
            'open': ['open', 'Open', 'daily_open']
        }
        
        for candidate in candidates.get(col_type, []):
            if candidate in df.columns:
                return candidate
        
        return None
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and sort data."""
        timestamp_candidates = [
            'sip_timestamp', 'timestamp', 'date', 'datetime', 'Date'
        ]
        
        timestamp_col = None
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                timestamp_col = candidate
                break
        
        if timestamp_col:
            return df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            return df.reset_index(drop=True)
    
    def _add_realized_volatility(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add realized volatility features across multiple timeframes."""
        
        # Calculate daily returns
        df['daily_return'] = df[price_col].pct_change()
        
        # Realized Volatility (annualized)
        df['rv_5d'] = df['daily_return'].rolling(5).std() * np.sqrt(252)
        df['rv_10d'] = df['daily_return'].rolling(10).std() * np.sqrt(252)
        df['rv_20d'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
        df['rv_60d'] = df['daily_return'].rolling(60).std() * np.sqrt(252)
        df['rv_252d'] = df['daily_return'].rolling(252).std() * np.sqrt(252)  # 1-year
        
        # Intraday return (if we have high/low data)
        if 'high' in df.columns and 'low' in df.columns:
            df['intraday_return'] = (df['high'] - df['low']) / df[price_col]
            df['intraday_volatility'] = df['intraday_return'].rolling(20).mean()
        
        # Overnight gaps (if we have previous close)
        df['overnight_gap'] = df[price_col] / df[price_col].shift(1) - 1
        df['overnight_volatility'] = abs(df['overnight_gap']).rolling(20).mean()
        
        return df
    
    def _add_volatility_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility momentum and rate of change features."""
        
        # Volatility Rate of Change (key for pullback prediction!)
        df['rv_5d_roc_5d'] = df['rv_5d'].pct_change(5)   # Short-term vol momentum
        df['rv_20d_roc_5d'] = df['rv_20d'].pct_change(5) # Medium-term vol momentum  
        df['rv_20d_roc_10d'] = df['rv_20d'].pct_change(10) # Longer-term vol momentum
        df['rv_60d_roc_20d'] = df['rv_60d'].pct_change(20) # Long-term vol momentum
        
        # Volatility acceleration (change in vol momentum)
        df['vol_acceleration_5d'] = df['rv_5d_roc_5d'].diff()
        df['vol_acceleration_20d'] = df['rv_20d_roc_5d'].diff()
        
        # Volatility trend
        df['vol_trend_5d'] = (df['rv_5d'] > df['rv_5d'].shift(5)).astype(int)
        df['vol_trend_20d'] = (df['rv_20d'] > df['rv_20d'].shift(10)).astype(int)
        
        return df
    
    def _add_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime and comparison features."""
        
        # Volatility Ratios (regime detection)
        df['rv_ratio_5_20'] = df['rv_5d'] / df['rv_20d']   # Short vs medium-term
        df['rv_ratio_20_60'] = df['rv_20d'] / df['rv_60d'] # Medium vs long-term
        df['rv_ratio_5_60'] = df['rv_5d'] / df['rv_60d']   # Short vs long-term
        
        # Volatility percentiles (regime identification)
        df['rv_20d_percentile'] = df['rv_20d'].rolling(252).rank(pct=True)
        df['rv_5d_percentile'] = df['rv_5d'].rolling(60).rank(pct=True)
        
        # Volatility regime indicators
        df['low_vol_regime'] = (df['rv_20d_percentile'] < 0.25).astype(int)
        df['high_vol_regime'] = (df['rv_20d_percentile'] > 0.75).astype(int)
        df['vol_regime_normal'] = (
            (df['rv_20d_percentile'] >= 0.25) & 
            (df['rv_20d_percentile'] <= 0.75)
        ).astype(int)
        
        # Volatility breakouts
        df['vol_breakout_high'] = (df['rv_5d'] > df['rv_60d'] * 1.5).astype(int)
        df['vol_breakout_low'] = (df['rv_5d'] < df['rv_60d'] * 0.5).astype(int)
        
        return df
    
    def _add_atr_features(self, df: pd.DataFrame, price_col: str, 
                         high_col: Optional[str], low_col: Optional[str]) -> pd.DataFrame:
        """Add Average True Range features."""
        
        # Calculate ATR
        if high_col and low_col:
            df['atr_14'] = self._calculate_true_atr(df, high_col, low_col, price_col, 14)
            df['atr_20'] = self._calculate_true_atr(df, high_col, low_col, price_col, 20)
        else:
            # Use price-based ATR approximation
            df['atr_14'] = self._calculate_price_atr(df, price_col, 14)
            df['atr_20'] = self._calculate_price_atr(df, price_col, 20)
        
        # ATR normalized by price
        df['atr_14_normalized'] = df['atr_14'] / df[price_col]
        df['atr_20_normalized'] = df['atr_20'] / df[price_col]
        
        # ATR momentum
        df['atr_momentum'] = df['atr_14'].pct_change(5)
        df['atr_expansion'] = (df['atr_14'] > df['atr_14'].rolling(20).mean()).astype(int)
        
        # ATR efficiency ratio
        df['atr_efficiency'] = abs(df['daily_return']) / df['atr_14_normalized']
        
        return df
    
    def _add_bollinger_band_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add Bollinger Band volatility features."""
        
        # Calculate Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df[price_col], 20, 2.0)
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        
        # Bollinger Band features
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band squeeze (low volatility)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8).astype(int)
        
        # Bollinger Band expansion (high volatility)
        df['bb_expansion'] = (df['bb_width'] > df['bb_width'].rolling(20).mean() * 1.2).astype(int)
        
        # Bollinger Band touches
        df['bb_upper_touch'] = (df[price_col] >= df['bb_upper']).astype(int)
        df['bb_lower_touch'] = (df[price_col] <= df['bb_lower']).astype(int)
        
        # Bollinger Band momentum
        df['bb_width_momentum'] = df['bb_width'].pct_change(5)
        
        return df
    
    def _add_volatility_signals(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add volatility-based trading signals."""
        
        # Volatility expansion (often precedes pullbacks)
        df['vol_expansion'] = (df['rv_5d'] > df['rv_20d'] * 1.5).astype(int)
        df['vol_spike'] = (df['rv_5d'] > df['rv_60d'] * 2.0).astype(int)
        
        # Volatility contraction (potential breakout setup)
        df['vol_contraction'] = (df['rv_5d'] < df['rv_20d'] * 0.7).astype(int)
        df['vol_compression'] = (df['rv_20d'] < df['rv_60d'] * 0.8).astype(int)
        
        # Volatility mean reversion signals
        df['vol_mean_reversion_high'] = (
            (df['rv_5d'] > df['rv_20d'] * 1.5) & 
            (df['rv_5d_roc_5d'] < 0)  # Vol starting to decline
        ).astype(int)
        
        df['vol_mean_reversion_low'] = (
            (df['rv_5d'] < df['rv_20d'] * 0.7) & 
            (df['rv_5d_roc_5d'] > 0)  # Vol starting to increase
        ).astype(int)
        
        # Volatility divergence with price
        price_direction = (df[price_col] > df[price_col].shift(10)).astype(int)
        vol_direction = (df['rv_20d'] > df['rv_20d'].shift(10)).astype(int)
        
        df['vol_price_divergence'] = (price_direction != vol_direction).astype(int)
        
        return df
    
    def _add_advanced_volatility_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add advanced volatility features."""
        
        # Volatility skew (upside vs downside volatility)
        upside_returns = df['daily_return'].where(df['daily_return'] > 0, 0)
        downside_returns = df['daily_return'].where(df['daily_return'] < 0, 0)
        
        df['upside_volatility'] = upside_returns.rolling(20).std() * np.sqrt(252)
        df['downside_volatility'] = abs(downside_returns).rolling(20).std() * np.sqrt(252)
        df['vol_skew'] = df['downside_volatility'] / df['upside_volatility']
        
        # Volatility clustering (GARCH-like effect)
        df['vol_clustering'] = (
            (df['rv_5d'] > df['rv_20d']) & 
            (df['rv_5d'].shift(1) > df['rv_20d'].shift(1))
        ).astype(int)
        
        # Volatility persistence
        vol_change = df['rv_20d'].diff()
        df['vol_persistence'] = (
            (vol_change > 0) & (vol_change.shift(1) > 0)
        ).astype(int)
        
        # Relative volatility strength
        df['relative_vol_strength'] = df['rv_20d'] / df['rv_252d']
        
        # Volatility-adjusted returns
        df['vol_adj_return_5d'] = df['momentum_5d'] / df['rv_20d'] if 'momentum_5d' in df.columns else np.nan
        df['vol_adj_return_20d'] = df['momentum_20d'] / df['rv_20d'] if 'momentum_20d' in df.columns else np.nan
        
        return df
    
    def _calculate_true_atr(self, df: pd.DataFrame, high_col: str, low_col: str, 
                           close_col: str, window: int) -> pd.Series:
        """Calculate true Average True Range using OHLC data."""
        high = df[high_col]
        low = df[low_col]
        close_prev = df[close_col].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window).mean()
    
    def _calculate_price_atr(self, df: pd.DataFrame, price_col: str, window: int) -> pd.Series:
        """Calculate ATR approximation using only price data."""
        returns = df[price_col].pct_change()
        return abs(returns).rolling(window).mean() * df[price_col]
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, 
                                  std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev)
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all volatility feature names."""
        return [
            # Basic returns and realized volatility
            'daily_return', 
            'rv_5d', 'rv_10d', 'rv_20d', 'rv_60d', 'rv_252d',
            'intraday_return', 'intraday_volatility',
            'overnight_gap', 'overnight_volatility',
            
            # Volatility momentum
            'rv_5d_roc_5d', 'rv_20d_roc_5d', 'rv_20d_roc_10d', 'rv_60d_roc_20d',
            'vol_acceleration_5d', 'vol_acceleration_20d',
            'vol_trend_5d', 'vol_trend_20d',
            
            # Volatility regimes
            'rv_ratio_5_20', 'rv_ratio_20_60', 'rv_ratio_5_60',
            'rv_20d_percentile', 'rv_5d_percentile',
            'low_vol_regime', 'high_vol_regime', 'vol_regime_normal',
            'vol_breakout_high', 'vol_breakout_low',
            
            # ATR features
            'atr_14', 'atr_20',
            'atr_14_normalized', 'atr_20_normalized',
            'atr_momentum', 'atr_expansion', 'atr_efficiency',
            
            # Bollinger Band features
            'bb_upper', 'bb_middle', 'bb_lower',
            'bb_width', 'bb_position',
            'bb_squeeze', 'bb_expansion',
            'bb_upper_touch', 'bb_lower_touch',
            'bb_width_momentum',
            
            # Volatility signals
            'vol_expansion', 'vol_spike',
            'vol_contraction', 'vol_compression',
            'vol_mean_reversion_high', 'vol_mean_reversion_low',
            'vol_price_divergence',
            
            # Advanced volatility features
            'upside_volatility', 'downside_volatility', 'vol_skew',
            'vol_clustering', 'vol_persistence',
            'relative_vol_strength',
            'vol_adj_return_5d', 'vol_adj_return_20d'
        ]
    
    def __str__(self) -> str:
        return f"VolatilityFeatureEngine({self.name})"
    
    def __repr__(self) -> str:
        return f"VolatilityFeatureEngine(name='{self.name}')"