#!/usr/bin/env python3
"""
Momentum Features Engine

Extracts momentum-based features for pullback prediction systems.
Focuses on rate of change, momentum acceleration, and trend momentum.

USAGE:
======
from features.momentum import MomentumFeatureEngine

engine = MomentumFeatureEngine()
df_with_features = engine.add_features(df)

# Get feature names
feature_names = engine.get_feature_names()
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MomentumFeatureEngine:
    """
    Momentum feature engine for pullback prediction.
    
    Creates momentum indicators focused on:
    - Price momentum across multiple timeframes
    - Moving average momentum (trend line slopes)
    - Momentum acceleration and divergence
    - Volatility-adjusted momentum
    """
    
    def __init__(self, name: str = "Momentum Features"):
        """Initialize momentum feature engine."""
        self.name = name
        logger.info(f"Initialized {self.name}")
    
    def add_features(self, df: pd.DataFrame, 
                    price_col: str = 'close') -> pd.DataFrame:
        """
        Add momentum features to DataFrame.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            
        Returns:
            DataFrame with momentum features added
        """
        print(f"\n🚀 Adding {self.name}...")
        
        df_processed = df.copy()
        
        # Find price column
        price_col = self._find_price_column(df_processed, price_col)
        
        # Sort data by timestamp
        df_processed = self._prepare_data(df_processed)
        
        # Add momentum features
        df_processed = self._add_price_momentum(df_processed, price_col)
        df_processed = self._add_moving_average_momentum(df_processed, price_col)
        df_processed = self._add_momentum_acceleration(df_processed, price_col)
        df_processed = self._add_momentum_strength_indicators(df_processed, price_col)
        df_processed = self._add_macd_momentum(df_processed, price_col)
        df_processed = self._add_rsi_momentum(df_processed, price_col)
        df_processed = self._add_volatility_adjusted_momentum(df_processed, price_col)
        
        print(f"   ✅ Added {len(self.get_feature_names())} momentum features")
        
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
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and sort data."""
        # Find timestamp column
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
            # Assume data is already sorted
            return df.reset_index(drop=True)
    
    def _add_price_momentum(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add basic price momentum features."""
        
        # Price momentum (rate of change)
        df['momentum_1d'] = df[price_col].pct_change(1)    # 1-day momentum
        df['momentum_5d'] = df[price_col].pct_change(5)    # 5-day momentum  
        df['momentum_10d'] = df[price_col].pct_change(10)  # 10-day momentum
        df['momentum_20d'] = df[price_col].pct_change(20)  # 20-day momentum
        df['momentum_60d'] = df[price_col].pct_change(60)  # 3-month momentum
        
        # Price momentum strength (absolute values)
        df['momentum_strength_5d'] = abs(df['momentum_5d'])
        df['momentum_strength_20d'] = abs(df['momentum_20d'])
        
        # Momentum direction consistency
        df['momentum_positive_5d'] = (df['momentum_5d'] > 0).astype(int)
        df['momentum_positive_20d'] = (df['momentum_20d'] > 0).astype(int)
        
        return df
    
    def _add_moving_average_momentum(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add moving average momentum features."""
        
        # Calculate SMAs
        df['sma_9'] = df[price_col].rolling(9).mean()
        df['sma_20'] = df[price_col].rolling(20).mean()
        df['sma_50'] = df[price_col].rolling(50).mean()
        df['sma_200'] = df[price_col].rolling(200).mean()
        
        # Calculate EMAs
        df['ema_9'] = df[price_col].ewm(span=9).mean()
        df['ema_21'] = df[price_col].ewm(span=21).mean()
        
        # SMA Rate of Change (momentum of trend lines)
        df['sma_9_roc_1d'] = df['sma_9'].pct_change(1)
        df['sma_20_roc_1d'] = df['sma_20'].pct_change(1)
        df['sma_20_roc_5d'] = df['sma_20'].pct_change(5)
        df['sma_50_roc_5d'] = df['sma_50'].pct_change(5)
        df['sma_200_roc_10d'] = df['sma_200'].pct_change(10)
        
        # EMA Rate of Change (faster moving average momentum)
        df['ema_9_roc_1d'] = df['ema_9'].pct_change(1)
        df['ema_9_roc_5d'] = df['ema_9'].pct_change(5)
        df['ema_21_roc_5d'] = df['ema_21'].pct_change(5)
        
        # Price vs Moving Average momentum
        df['price_vs_sma9'] = df[price_col] / df['sma_9'] - 1
        df['price_vs_sma20'] = df[price_col] / df['sma_20'] - 1
        df['price_vs_sma50'] = df[price_col] / df['sma_50'] - 1
        df['price_vs_ema9'] = df[price_col] / df['ema_9'] - 1
        df['price_vs_ema21'] = df[price_col] / df['ema_21'] - 1
        
        # Rate of change of price vs MA ratios
        df['price_sma20_ratio'] = df[price_col] / df['sma_20']
        df['price_sma20_ratio_roc_5d'] = df['price_sma20_ratio'].pct_change(5)
        
        return df
    
    def _add_momentum_acceleration(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add momentum acceleration features."""
        
        # Momentum acceleration (change in momentum)
        df['momentum_accel_5d'] = df['momentum_5d'] - df['momentum_10d']
        df['momentum_accel_10d'] = df['momentum_10d'] - df['momentum_20d']
        df['momentum_accel_20d'] = df['momentum_20d'] - df['momentum_60d']
        
        # Short vs Long momentum comparison
        df['momentum_short_vs_long'] = df['momentum_5d'] - df['momentum_20d']
        
        # Momentum divergence (price vs momentum direction)
        df['price_momentum_divergence'] = (
            (df['momentum_5d'] > 0) != (df['momentum_1d'] > 0)
        ).astype(int)
        
        # Momentum deceleration detection
        df['momentum_deceleration'] = (
            (df['momentum_accel_5d'] < 0) & (df['momentum_5d'] > 0)
        ).astype(int)
        
        return df
    
    def _add_momentum_strength_indicators(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add momentum strength and quality indicators."""
        
        # Moving average alignment (all MAs in same direction)
        df['ma_alignment_bullish'] = (
            (df['sma_9'] > df['sma_20']) & 
            (df['sma_20'] > df['sma_50']) & 
            (df[price_col] > df['sma_9'])
        ).astype(int)
        
        df['ma_alignment_bearish'] = (
            (df['sma_9'] < df['sma_20']) & 
            (df['sma_20'] < df['sma_50']) & 
            (df[price_col] < df['sma_9'])
        ).astype(int)
        
        # Trend strength
        df['trend_strength_short'] = (df['sma_9'] - df['sma_20']) / df['sma_20']
        df['trend_strength_medium'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Price extension from trend (momentum exhaustion signals)
        df['price_extension_sma9'] = abs(df['price_vs_sma9'])
        df['price_extension_sma20'] = abs(df['price_vs_sma20'])
        
        # Momentum persistence (how long momentum has been in same direction)
        df['momentum_up_streak'] = (df['momentum_5d'] > 0).astype(int)
        df['momentum_down_streak'] = (df['momentum_5d'] < 0).astype(int)
        
        # Calculate streak lengths
        df['momentum_up_streak'] = df['momentum_up_streak'].groupby(
            (df['momentum_up_streak'] != df['momentum_up_streak'].shift()).cumsum()
        ).cumsum()
        
        df['momentum_down_streak'] = df['momentum_down_streak'].groupby(
            (df['momentum_down_streak'] != df['momentum_down_streak'].shift()).cumsum()
        ).cumsum()
        
        return df
    
    def _add_macd_momentum(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add MACD-based momentum features."""
        
        # Calculate MACD
        macd_data = self._calculate_macd(df[price_col])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # MACD momentum features
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_histogram_positive'] = (df['macd_histogram'] > 0).astype(int)
        
        # MACD histogram momentum (rate of change)
        df['macd_histogram_roc'] = df['macd_histogram'].pct_change(1)
        df['macd_histogram_acceleration'] = df['macd_histogram_roc'].diff()
        
        # MACD divergence signals
        df['macd_bullish_divergence'] = (
            (df['macd_histogram'] > df['macd_histogram'].shift(5)) &
            (df[price_col] < df[price_col].shift(5))
        ).astype(int)
        
        df['macd_bearish_divergence'] = (
            (df['macd_histogram'] < df['macd_histogram'].shift(5)) &
            (df[price_col] > df[price_col].shift(5))
        ).astype(int)
        
        return df
    
    def _add_rsi_momentum(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add RSI-based momentum features."""
        
        # Calculate RSI
        df['rsi_14'] = self._calculate_rsi(df[price_col], 14)
        df['rsi_9'] = self._calculate_rsi(df[price_col], 9)
        
        # RSI momentum
        df['rsi_momentum'] = df['rsi_14'].diff()
        df['rsi_acceleration'] = df['rsi_momentum'].diff()
        
        # RSI divergence
        df['rsi_bullish_divergence'] = (
            (df['rsi_14'] > df['rsi_14'].shift(10)) &
            (df[price_col] < df[price_col].shift(10))
        ).astype(int)
        
        df['rsi_bearish_divergence'] = (
            (df['rsi_14'] < df['rsi_14'].shift(10)) &
            (df[price_col] > df[price_col].shift(10))
        ).astype(int)
        
        # RSI momentum zones
        df['rsi_momentum_bullish'] = (
            (df['rsi_14'] > 50) & (df['rsi_momentum'] > 0)
        ).astype(int)
        
        df['rsi_momentum_bearish'] = (
            (df['rsi_14'] < 50) & (df['rsi_momentum'] < 0)
        ).astype(int)
        
        return df
    
    def _add_volatility_adjusted_momentum(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add volatility-adjusted momentum features."""
        
        # Calculate realized volatility
        returns = df[price_col].pct_change()
        df['rv_20d'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Volatility-adjusted momentum
        df['vol_adj_momentum_5d'] = df['momentum_5d'] / df['rv_20d']
        df['vol_adj_momentum_20d'] = df['momentum_20d'] / df['rv_20d']
        
        # Volatility-adjusted distance from moving averages
        df['vol_adj_distance_sma20'] = df['price_vs_sma20'] / df['rv_20d']
        df['vol_adj_distance_ema21'] = df['price_vs_ema21'] / df['rv_20d']
        
        # Risk-adjusted momentum signals
        df['high_momentum_low_vol'] = (
            (abs(df['momentum_5d']) > 0.02) & (df['rv_20d'] < 0.15)
        ).astype(int)
        
        df['momentum_vol_expansion'] = (
            (abs(df['momentum_5d']) > 0.01) & 
            (df['rv_20d'] > df['rv_20d'].rolling(20).mean())
        ).astype(int)
        
        return df
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_feature_names(self) -> List[str]:
        """Get list of all momentum feature names."""
        return [
            # Basic price momentum
            'momentum_1d', 'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
            'momentum_strength_5d', 'momentum_strength_20d',
            'momentum_positive_5d', 'momentum_positive_20d',
            
            # Moving averages
            'sma_9', 'sma_20', 'sma_50', 'sma_200',
            'ema_9', 'ema_21',
            
            # Moving average momentum
            'sma_9_roc_1d', 'sma_20_roc_1d', 'sma_20_roc_5d', 'sma_50_roc_5d', 'sma_200_roc_10d',
            'ema_9_roc_1d', 'ema_9_roc_5d', 'ema_21_roc_5d',
            
            # Price vs MA ratios
            'price_vs_sma9', 'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema9', 'price_vs_ema21',
            'price_sma20_ratio', 'price_sma20_ratio_roc_5d',
            
            # Momentum acceleration
            'momentum_accel_5d', 'momentum_accel_10d', 'momentum_accel_20d',
            'momentum_short_vs_long', 'price_momentum_divergence', 'momentum_deceleration',
            
            # Momentum strength indicators
            'ma_alignment_bullish', 'ma_alignment_bearish',
            'trend_strength_short', 'trend_strength_medium',
            'price_extension_sma9', 'price_extension_sma20',
            'momentum_up_streak', 'momentum_down_streak',
            
            # MACD momentum
            'macd', 'macd_signal', 'macd_histogram',
            'macd_bullish', 'macd_histogram_positive',
            'macd_histogram_roc', 'macd_histogram_acceleration',
            'macd_bullish_divergence', 'macd_bearish_divergence',
            
            # RSI momentum
            'rsi_14', 'rsi_9',
            'rsi_momentum', 'rsi_acceleration',
            'rsi_bullish_divergence', 'rsi_bearish_divergence',
            'rsi_momentum_bullish', 'rsi_momentum_bearish',
            
            # Volatility-adjusted momentum
            'rv_20d',
            'vol_adj_momentum_5d', 'vol_adj_momentum_20d',
            'vol_adj_distance_sma20', 'vol_adj_distance_ema21',
            'high_momentum_low_vol', 'momentum_vol_expansion'
        ]
    
    def __str__(self) -> str:
        return f"MomentumFeatureEngine({self.name})"
    
    def __repr__(self) -> str:
        return f"MomentumFeatureEngine(name='{self.name}')"