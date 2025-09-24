#!/usr/bin/env python3
"""
Target Factory

Orchestrates creation of all target types for momentum-based trading systems.
Provides unified interface for creating pullback and mean reversion targets.

USAGE:
======
from targets.target_factory import TargetFactory

# Create all default targets
factory = TargetFactory()
df_with_targets = factory.create_all_targets(df)

# Custom configuration
config = {
    "pullback_targets": {
        "thresholds": [0.02, 0.05, 0.10],
        "horizons": [5, 10, 15, 20]
    },
    "mean_reversion_targets": {
        "sma_periods": [20, 50, 100, 200],
        "horizons": [5, 10, 20]
    }
}
df_with_targets = factory.create_targets_from_config(df, config)

# Individual target types
df = factory.create_pullback_targets(df, thresholds=[0.03, 0.07])
df = factory.create_mean_reversion_targets(df, sma_periods=[20, 50])
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from .pullback_targets import PullbackTargetCreator
from .mean_reversion_targets import MeanReversionTargetCreator

logger = logging.getLogger(__name__)


class TargetFactory:
    """
    Factory for creating all types of targets for momentum-based trading systems.
    Provides unified interface and configuration management.
    """
    
    def __init__(self):
        """Initialize target factory with default configurations."""
        self.default_config = {
            "pullback_targets": {
                "thresholds": [0.02, 0.05, 0.10],
                "horizons": [5, 10, 15, 20]
            },
            "mean_reversion_targets": {
                "sma_periods": [20, 50, 100, 200],
                "horizons": [5, 10, 15, 20],
                "reversion_threshold": 0.01
            }
        }
        
        logger.info("TargetFactory initialized with default configurations")
    
    def create_all_targets(self, df: pd.DataFrame, 
                          price_col: str = 'close',
                          config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all target types using default or provided configuration.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            config: Custom configuration (uses defaults if None)
            
        Returns:
            DataFrame with all targets added
        """
        print(f"\n🎯 Creating all targets using TargetFactory...")
        
        if config is None:
            config = self.default_config
        
        df_processed = df.copy()
        
        # Create pullback targets
        if "pullback_targets" in config:
            print(f"\n📉 Creating pullback targets...")
            df_processed = self.create_pullback_targets(
                df_processed, 
                price_col=price_col,
                **config["pullback_targets"]
            )
        
        # Create mean reversion targets
        if "mean_reversion_targets" in config:
            print(f"\n📈 Creating mean reversion targets...")
            df_processed = self.create_mean_reversion_targets(
                df_processed,
                price_col=price_col,
                **config["mean_reversion_targets"]
            )
        
        # Create combined targets
        df_processed = self._create_combined_targets(df_processed)
        
        # Print summary statistics
        self._print_target_summary(df_processed)
        
        print(f"\n✅ All targets created successfully!")
        return df_processed
    
    def create_pullback_targets(self, df: pd.DataFrame,
                               thresholds: List[float] = [0.02, 0.05, 0.10],
                               horizons: List[int] = [5, 10, 15, 20],
                               price_col: str = 'close') -> pd.DataFrame:
        """
        Create pullback targets only.
        
        Args:
            df: DataFrame with price data
            thresholds: List of pullback thresholds
            horizons: List of prediction horizons
            price_col: Name of price column
            
        Returns:
            DataFrame with pullback targets added
        """
        creator = PullbackTargetCreator()
        return creator.create_multi_horizon_targets(
            df, thresholds=thresholds, horizons=horizons, price_col=price_col
        )
    
    def create_mean_reversion_targets(self, df: pd.DataFrame,
                                    sma_periods: List[int] = [20, 50, 100, 200],
                                    horizons: List[int] = [5, 10, 15, 20],
                                    reversion_threshold: float = 0.01,
                                    price_col: str = 'close') -> pd.DataFrame:
        """
        Create mean reversion targets only.
        
        Args:
            df: DataFrame with price data
            sma_periods: List of SMA periods
            horizons: List of prediction horizons
            reversion_threshold: Threshold for considering reversion
            price_col: Name of price column
            
        Returns:
            DataFrame with mean reversion targets added
        """
        creator = MeanReversionTargetCreator(
            sma_periods=sma_periods, 
            reversion_threshold=reversion_threshold
        )
        return creator.create_targets(df, horizons=horizons, price_col=price_col)
    
    def create_targets_from_config(self, df: pd.DataFrame, 
                                  config: Dict[str, Any],
                                  price_col: str = 'close') -> pd.DataFrame:
        """
        Create targets based on configuration dictionary.
        
        Args:
            df: DataFrame with price data
            config: Configuration dictionary
            price_col: Name of price column
            
        Returns:
            DataFrame with configured targets added
        """
        return self.create_all_targets(df, price_col=price_col, config=config)
    
    def create_spy_targets(self, df: pd.DataFrame, 
                          price_col: str = 'close') -> pd.DataFrame:
        """
        Create targets optimized for SPY trading.
        
        Args:
            df: DataFrame with SPY price data
            price_col: Name of price column
            
        Returns:
            DataFrame with SPY-optimized targets
        """
        spy_config = {
            "pullback_targets": {
                "thresholds": [0.02, 0.03, 0.05, 0.07, 0.10],  # More granular for SPY
                "horizons": [5, 10, 15, 20, 30]  # Include monthly horizon
            },
            "mean_reversion_targets": {
                "sma_periods": [9, 20, 50, 100, 200],  # Include fast 9-day SMA
                "horizons": [5, 10, 15, 20],
                "reversion_threshold": 0.005  # Tighter threshold for liquid ETF
            }
        }
        
        print(f"\n🎯 Creating SPY-optimized targets...")
        return self.create_all_targets(df, price_col=price_col, config=spy_config)
    
    def create_volatile_stock_targets(self, df: pd.DataFrame,
                                    price_col: str = 'close') -> pd.DataFrame:
        """
        Create targets optimized for volatile individual stocks.
        
        Args:
            df: DataFrame with stock price data
            price_col: Name of price column
            
        Returns:
            DataFrame with targets optimized for volatility
        """
        volatile_config = {
            "pullback_targets": {
                "thresholds": [0.05, 0.10, 0.15, 0.20],  # Larger thresholds
                "horizons": [3, 5, 10, 15]  # Shorter horizons
            },
            "mean_reversion_targets": {
                "sma_periods": [10, 20, 50],  # Focus on shorter-term SMAs
                "horizons": [3, 5, 10],
                "reversion_threshold": 0.02  # Looser threshold
            }
        }
        
        print(f"\n🎯 Creating volatile stock targets...")
        return self.create_all_targets(df, price_col=price_col, config=volatile_config)
    
    def _create_combined_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined targets that merge pullback and mean reversion signals."""
        
        # Find available pullback and mean reversion columns
        pullback_cols = [col for col in df.columns if col.startswith('pullback_') and col.endswith('d')]
        reversion_cols = [col for col in df.columns if col.startswith('mean_revert_') and col.endswith('d')]
        
        if not pullback_cols or not reversion_cols:
            logger.warning("Cannot create combined targets - missing pullback or reversion targets")
            return df
        
        print(f"   🔀 Creating combined targets...")
        
        # Combined signal: Either pullback OR mean reversion expected
        try:
            # Get targets for same horizons
            horizons = [5, 10, 15, 20]
            for horizon in horizons:
                pullback_horizon_cols = [col for col in pullback_cols if f'_{horizon}d' in col]
                reversion_horizon_cols = [col for col in reversion_cols if f'_{horizon}d' in col]
                
                if pullback_horizon_cols and reversion_horizon_cols:
                    # Any movement expected (pullback or reversion)
                    df[f'any_movement_{horizon}d'] = (
                        df[pullback_horizon_cols].any(axis=1) | 
                        df[reversion_horizon_cols].any(axis=1)
                    ).astype(int)
                    
                    # Directional bias (more pullback signals vs reversion signals)
                    pullback_sum = df[pullback_horizon_cols].sum(axis=1)
                    reversion_sum = df[reversion_horizon_cols].sum(axis=1)
                    
                    df[f'pullback_bias_{horizon}d'] = (pullback_sum > reversion_sum).astype(int)
                    df[f'reversion_bias_{horizon}d'] = (reversion_sum > pullback_sum).astype(int)
            
            # Market regime indicators
            if 'distance_from_sma20_pct' in df.columns and 'distance_from_sma200_pct' in df.columns:
                # Bull/Bear market regime
                df['bull_market_regime'] = (df['distance_from_sma200_pct'] > 0).astype(int)
                df['bear_market_regime'] = (df['distance_from_sma200_pct'] < -5).astype(int)
                
                # Short-term overbought/oversold
                df['short_term_overbought'] = (df['distance_from_sma20_pct'] > 3).astype(int)
                df['short_term_oversold'] = (df['distance_from_sma20_pct'] < -3).astype(int)
        
        except Exception as e:
            logger.warning(f"Error creating combined targets: {e}")
        
        return df
    
    def _print_target_summary(self, df: pd.DataFrame):
        """Print summary of all created targets."""
        pullback_cols = [col for col in df.columns if col.startswith('pullback_') and '_' in col and col.endswith('d')]
        reversion_cols = [col for col in df.columns if col.startswith('mean_revert_') and '_' in col and col.endswith('d')]
        combined_cols = [col for col in df.columns if col.startswith(('any_movement_', 'pullback_bias_', 'reversion_bias_'))]
        
        print(f"\n   📋 Target Summary:")
        print(f"      • Pullback targets: {len(pullback_cols)}")
        print(f"      • Mean reversion targets: {len(reversion_cols)}")
        print(f"      • Combined targets: {len(combined_cols)}")
        print(f"      • Total target columns: {len(pullback_cols + reversion_cols + combined_cols)}")
        
        # Sample a few key targets for statistics
        key_targets = []
        
        # Find representative targets
        for threshold in ['2pct', '5pct', '10pct']:
            for horizon in ['5d', '10d', '20d']:
                col_name = f'pullback_{threshold}_{horizon}'
                if col_name in df.columns:
                    key_targets.append(col_name)
                    if len(key_targets) >= 3:  # Limit output
                        break
            if len(key_targets) >= 3:
                break
        
        # Add a mean reversion target
        for sma in [20, 50]:
            col_name = f'mean_revert_sma{sma}_10d'
            if col_name in df.columns:
                key_targets.append(col_name)
                break
        
        if key_targets:
            print(f"\n   📊 Key Target Rates:")
            for target in key_targets:
                if target in df.columns:
                    rate = df[target].dropna().mean()
                    count = df[target].dropna().sum()
                    print(f"      • {target}: {count:,} events ({rate:.1%})")
    
    def get_all_target_columns(self, config: Optional[Dict] = None) -> List[str]:
        """
        Get list of all target column names that would be created.
        
        Args:
            config: Configuration dictionary (uses defaults if None)
            
        Returns:
            List of all target column names
        """
        if config is None:
            config = self.default_config
        
        columns = []
        
        # Pullback target columns
        if "pullback_targets" in config:
            pb_creator = PullbackTargetCreator()
            pb_columns = pb_creator.get_target_columns(
                thresholds=config["pullback_targets"]["thresholds"],
                horizons=config["pullback_targets"]["horizons"]
            )
            columns.extend(pb_columns)
        
        # Mean reversion target columns  
        if "mean_reversion_targets" in config:
            mr_creator = MeanReversionTargetCreator(
                sma_periods=config["mean_reversion_targets"]["sma_periods"],
                reversion_threshold=config["mean_reversion_targets"]["reversion_threshold"]
            )
            mr_columns = mr_creator.get_reversion_columns(
                horizons=config["mean_reversion_targets"]["horizons"]
            )
            columns.extend(mr_columns)
        
        # Combined target columns
        horizons = config.get("pullback_targets", {}).get("horizons", [5, 10, 15, 20])
        for horizon in horizons:
            columns.extend([
                f'any_movement_{horizon}d',
                f'pullback_bias_{horizon}d',
                f'reversion_bias_{horizon}d'
            ])
        
        # Market regime columns
        columns.extend([
            'bull_market_regime',
            'bear_market_regime', 
            'short_term_overbought',
            'short_term_oversold'
        ])
        
        return columns
    
    def validate_targets(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate created targets and return quality metrics.
        
        Args:
            df: DataFrame with targets
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {}
        }
        
        # Find target columns
        target_cols = [col for col in df.columns 
                      if any(col.startswith(prefix) for prefix in ['pullback_', 'mean_revert_', 'any_movement_'])]
        
        if not target_cols:
            results["valid"] = False
            results["errors"].append("No target columns found")
            return results
        
        # Validate each target
        for col in target_cols:
            if col not in df.columns:
                continue
                
            target_data = df[col].dropna()
            
            if len(target_data) == 0:
                results["warnings"].append(f"No valid data for target {col}")
                continue
            
            # Calculate statistics
            event_rate = target_data.mean()
            event_count = target_data.sum()
            
            results["statistics"][col] = {
                "event_rate": event_rate,
                "event_count": int(event_count),
                "total_observations": len(target_data)
            }
            
            # Check for extreme imbalance
            if event_rate < 0.01:
                results["warnings"].append(f"Very low event rate for {col}: {event_rate:.3%}")
            elif event_rate > 0.99:
                results["warnings"].append(f"Very high event rate for {col}: {event_rate:.1%}")
            
            # Check for minimum data
            if len(target_data) < 100:
                results["warnings"].append(f"Insufficient data for {col}: {len(target_data)} observations")
        
        return results
    
    def __str__(self) -> str:
        return "TargetFactory(pullback + mean_reversion + combined targets)"
    
    def __repr__(self) -> str:
        return f"TargetFactory(default_config={self.default_config})"