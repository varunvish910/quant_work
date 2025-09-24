#!/usr/bin/env python3
"""
Feature Engine Orchestrator

Coordinates all feature engines for momentum-based pullback prediction.
Provides unified interface for feature generation and management.

USAGE:
======
from features.feature_engine import FeatureEngine

# Create all features
engine = FeatureEngine()
df_with_features = engine.create_all_features(df)

# Custom feature selection
engine = FeatureEngine(include=['momentum', 'volatility'])
df_with_features = engine.create_features(df)

# Get feature names
feature_names = engine.get_all_feature_names()
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set
import logging
import time

from .momentum import MomentumFeatureEngine
from .volatility import VolatilityFeatureEngine

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Orchestrates all feature engines for momentum-based pullback prediction.
    Provides unified interface for feature creation and management.
    """
    
    def __init__(self, include: Optional[List[str]] = None):
        """
        Initialize feature engine orchestrator.
        
        Args:
            include: List of feature types to include. If None, includes all.
                    Options: ['momentum', 'volatility', 'mean_reversion']
        """
        self.available_engines = {
            'momentum': MomentumFeatureEngine,
            'volatility': VolatilityFeatureEngine,
        }
        
        # Determine which engines to use
        if include is None:
            self.active_engines = list(self.available_engines.keys())
        else:
            self.active_engines = [eng for eng in include if eng in self.available_engines]
            if not self.active_engines:
                raise ValueError(f"No valid engines specified. Available: {list(self.available_engines.keys())}")
        
        # Initialize engines
        self.engines = {}
        for engine_name in self.active_engines:
            self.engines[engine_name] = self.available_engines[engine_name]()
        
        logger.info(f"FeatureEngine initialized with engines: {self.active_engines}")
    
    def create_all_features(self, df: pd.DataFrame, 
                           price_col: str = 'close',
                           high_col: str = None,
                           low_col: str = None) -> pd.DataFrame:
        """
        Create all features using active engines.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            high_col: Name of high column (optional)
            low_col: Name of low column (optional)
            
        Returns:
            DataFrame with all features added
        """
        print(f"\n🎯 Creating features using FeatureEngine...")
        print(f"   • Active engines: {self.active_engines}")
        
        start_time = time.time()
        df_processed = df.copy()
        feature_counts = {}
        
        # Apply each engine
        for engine_name, engine in self.engines.items():
            print(f"\n   🔧 Running {engine_name} engine...")
            engine_start_time = time.time()
            
            try:
                if engine_name == 'volatility':
                    df_processed = engine.add_features(
                        df_processed, 
                        price_col=price_col,
                        high_col=high_col, 
                        low_col=low_col
                    )
                else:
                    df_processed = engine.add_features(df_processed, price_col=price_col)
                
                feature_count = len(engine.get_feature_names())
                feature_counts[engine_name] = feature_count
                
                engine_time = time.time() - engine_start_time
                print(f"      ✅ {engine_name}: {feature_count} features in {engine_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Error in {engine_name} engine: {e}")
                print(f"      ❌ {engine_name}: Failed - {e}")
                continue
        
        # Add combined features
        df_processed = self._add_combined_features(df_processed)
        
        # Print summary
        total_features = sum(feature_counts.values())
        total_time = time.time() - start_time
        
        print(f"\n   📊 Feature Creation Summary:")
        for engine_name, count in feature_counts.items():
            print(f"      • {engine_name}: {count} features")
        print(f"      • Combined features: {len(self._get_combined_feature_names())}")
        print(f"      • Total features: {total_features + len(self._get_combined_feature_names())}")
        print(f"      • Total time: {total_time:.1f}s")
        
        # Validate features
        validation_results = self._validate_features(df_processed)
        if validation_results['warnings']:
            print(f"   ⚠️  Warnings: {len(validation_results['warnings'])}")
        
        print(f"   ✅ Feature creation completed successfully!")
        
        return df_processed
    
    def create_features_subset(self, df: pd.DataFrame, 
                              feature_types: List[str],
                              price_col: str = 'close') -> pd.DataFrame:
        """
        Create subset of features based on specified types.
        
        Args:
            df: DataFrame with price data
            feature_types: List of feature types to create
            price_col: Name of price column
            
        Returns:
            DataFrame with specified features added
        """
        # Filter engines to only requested types
        subset_engines = {k: v for k, v in self.engines.items() if k in feature_types}
        
        if not subset_engines:
            raise ValueError(f"No engines available for types: {feature_types}")
        
        print(f"\n🎯 Creating feature subset: {list(subset_engines.keys())}")
        
        df_processed = df.copy()
        
        for engine_name, engine in subset_engines.items():
            df_processed = engine.add_features(df_processed, price_col=price_col)
        
        return df_processed
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get features grouped by importance for model interpretation.
        
        Returns:
            Dictionary mapping importance groups to feature lists
        """
        groups = {
            'high_importance': [],
            'medium_importance': [], 
            'low_importance': []
        }
        
        # High importance features (most predictive for pullbacks)
        if 'momentum' in self.engines:
            groups['high_importance'].extend([
                'momentum_5d', 'momentum_20d', 'momentum_accel_5d',
                'sma_20_roc_5d', 'price_vs_sma20', 'macd_histogram',
                'rsi_bullish_divergence', 'rsi_bearish_divergence',
                'momentum_deceleration'
            ])
        
        if 'volatility' in self.engines:
            groups['high_importance'].extend([
                'rv_5d_roc_5d', 'rv_20d_roc_5d', 'vol_expansion', 
                'vol_spike', 'bb_width', 'atr_14_normalized',
                'vol_mean_reversion_high'
            ])
        
        # Medium importance features
        if 'momentum' in self.engines:
            groups['medium_importance'].extend([
                'ema_9_roc_5d', 'trend_strength_short', 'ma_alignment_bullish',
                'price_extension_sma20', 'momentum_strength_5d'
            ])
        
        if 'volatility' in self.engines:
            groups['medium_importance'].extend([
                'rv_ratio_5_20', 'bb_position', 'vol_clustering',
                'vol_skew', 'atr_momentum'
            ])
        
        # Low importance features (supplementary)
        if 'momentum' in self.engines:
            groups['low_importance'].extend([
                'momentum_up_streak', 'momentum_down_streak', 'rsi_9',
                'momentum_positive_5d'
            ])
        
        if 'volatility' in self.engines:
            groups['low_importance'].extend([
                'overnight_volatility', 'intraday_volatility', 
                'vol_persistence', 'bb_squeeze'
            ])
        
        # Filter to only existing features
        all_feature_names = self.get_all_feature_names()
        for group in groups:
            groups[group] = [f for f in groups[group] if f in all_feature_names]
        
        return groups
    
    def _add_combined_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that combine signals from multiple engines."""
        
        print(f"   🔀 Adding combined features...")
        
        try:
            # Momentum-Volatility combinations
            if 'momentum_5d' in df.columns and 'rv_20d' in df.columns:
                # Risk-adjusted momentum
                df['risk_adjusted_momentum'] = df['momentum_5d'] / df['rv_20d']
                
                # High momentum in low vol environment (bullish)
                df['momentum_low_vol_signal'] = (
                    (abs(df['momentum_5d']) > 0.02) & (df['rv_20d'] < 0.15)
                ).astype(int)
                
                # Momentum exhaustion with vol expansion (bearish)
                df['momentum_exhaustion_signal'] = (
                    (df['momentum_5d'] > 0.05) & (df['rv_5d_roc_5d'] > 0.2)
                ).astype(int)
            
            # Multi-timeframe momentum alignment
            if all(col in df.columns for col in ['momentum_5d', 'momentum_20d', 'sma_20_roc_5d']):
                df['momentum_alignment_bullish'] = (
                    (df['momentum_5d'] > 0) & 
                    (df['momentum_20d'] > 0) & 
                    (df['sma_20_roc_5d'] > 0)
                ).astype(int)
                
                df['momentum_alignment_bearish'] = (
                    (df['momentum_5d'] < 0) & 
                    (df['momentum_20d'] < 0) & 
                    (df['sma_20_roc_5d'] < 0)
                ).astype(int)
            
            # Volatility regime with momentum
            if 'high_vol_regime' in df.columns and 'momentum_strength_5d' in df.columns:
                df['high_vol_high_momentum'] = (
                    df['high_vol_regime'] & (df['momentum_strength_5d'] > 0.03)
                ).astype(int)
                
                df['low_vol_low_momentum'] = (
                    df['low_vol_regime'] & (df['momentum_strength_5d'] < 0.01)
                ).astype(int)
            
            # Mean reversion signals
            if all(col in df.columns for col in ['rsi_14', 'bb_position', 'price_vs_sma20']):
                # Oversold mean reversion signal
                df['oversold_mean_reversion'] = (
                    (df['rsi_14'] < 30) & 
                    (df['bb_position'] < 0.2) & 
                    (df['price_vs_sma20'] < -0.05)
                ).astype(int)
                
                # Overbought mean reversion signal  
                df['overbought_mean_reversion'] = (
                    (df['rsi_14'] > 70) & 
                    (df['bb_position'] > 0.8) & 
                    (df['price_vs_sma20'] > 0.05)
                ).astype(int)
            
            # Pullback warning signals
            if all(col in df.columns for col in ['momentum_deceleration', 'vol_expansion', 'rsi_14']):
                df['pullback_warning_signal'] = (
                    df['momentum_deceleration'] | 
                    df['vol_expansion'] | 
                    (df['rsi_14'] > 80)
                ).astype(int)
            
        except Exception as e:
            logger.warning(f"Error creating combined features: {e}")
        
        return df
    
    def _get_combined_feature_names(self) -> List[str]:
        """Get names of combined features."""
        return [
            'risk_adjusted_momentum',
            'momentum_low_vol_signal',
            'momentum_exhaustion_signal',
            'momentum_alignment_bullish',
            'momentum_alignment_bearish',
            'high_vol_high_momentum',
            'low_vol_low_momentum',
            'oversold_mean_reversion',
            'overbought_mean_reversion',
            'pullback_warning_signal'
        ]
    
    def _validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate created features."""
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        feature_names = self.get_all_feature_names()
        
        for feature_name in feature_names:
            if feature_name not in df.columns:
                continue
            
            feature_data = df[feature_name]
            
            # Check for all NaN
            if feature_data.isnull().all():
                results['warnings'].append(f"Feature {feature_name} is all NaN")
                continue
            
            # Check for excessive NaNs
            nan_pct = feature_data.isnull().mean()
            if nan_pct > 0.5:
                results['warnings'].append(f"Feature {feature_name} has {nan_pct:.1%} NaN values")
            
            # Check for infinite values
            if np.isinf(feature_data).any():
                results['warnings'].append(f"Feature {feature_name} contains infinite values")
            
            # Basic statistics
            valid_data = feature_data.dropna()
            if len(valid_data) > 0:
                results['statistics'][feature_name] = {
                    'count': len(valid_data),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max())
                }
        
        return results
    
    def get_all_feature_names(self) -> List[str]:
        """Get names of all features that would be created."""
        all_features = []
        
        # Add features from each engine
        for engine_name, engine in self.engines.items():
            all_features.extend(engine.get_feature_names())
        
        # Add combined features
        all_features.extend(self._get_combined_feature_names())
        
        return all_features
    
    def get_engine_feature_names(self, engine_name: str) -> List[str]:
        """Get feature names for specific engine."""
        if engine_name in self.engines:
            return self.engines[engine_name].get_feature_names()
        else:
            raise ValueError(f"Engine {engine_name} not available. Available: {list(self.engines.keys())}")
    
    def filter_features_by_importance(self, feature_names: List[str], 
                                    importance_level: str = 'high') -> List[str]:
        """
        Filter features by importance level.
        
        Args:
            feature_names: List of feature names to filter
            importance_level: 'high', 'medium', 'low', or 'all'
            
        Returns:
            Filtered list of feature names
        """
        if importance_level == 'all':
            return feature_names
        
        importance_groups = self.get_feature_importance_groups()
        
        if importance_level == 'high':
            important_features = set(importance_groups['high_importance'])
        elif importance_level == 'medium':
            important_features = set(importance_groups['high_importance'] + importance_groups['medium_importance'])
        elif importance_level == 'low':
            important_features = set(importance_groups['high_importance'] + 
                                   importance_groups['medium_importance'] + 
                                   importance_groups['low_importance'])
        else:
            raise ValueError("importance_level must be 'high', 'medium', 'low', or 'all'")
        
        return [f for f in feature_names if f in important_features]
    
    def __str__(self) -> str:
        return f"FeatureEngine(engines={self.active_engines})"
    
    def __repr__(self) -> str:
        return f"FeatureEngine(active_engines={self.active_engines})"