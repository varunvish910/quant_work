"""
Target Creation Module

Unified target generation for all prediction tasks.
Consolidates logic from phase files and rally_analyzer targets.

CRITICAL: Targets must signal BEFORE events, not during!
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.constants import (
    DRAWDOWN_THRESHOLD, EARLY_WARNING_DAYS, LOOKFORWARD_DAYS,
    MEAN_REVERSION_THRESHOLD, MEAN_REVERSION_DAYS
)


class TargetCreator:
    """Unified target creation for all prediction tasks"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize TargetCreator
        
        Args:
            data: DataFrame with datetime index and OHLC columns
        """
        self.data = data.copy()
    
    def create_early_warning_target(self,
                                   drawdown_threshold: float = DRAWDOWN_THRESHOLD,
                                   min_lead_days: int = EARLY_WARNING_DAYS,
                                   max_lead_days: int = LOOKFORWARD_DAYS + EARLY_WARNING_DAYS) -> pd.DataFrame:
        """
        Create target that signals BEFORE drawdowns happen (not during!)
        This is KEY to actionable early warning system.
        
        The target looks 3-13 days ahead and marks dates where a significant
        drawdown will occur in the near future. This gives actionable warning.
        
        Args:
            drawdown_threshold: Minimum drawdown to qualify (default 5%)
            min_lead_days: Minimum days ahead to signal (default 3)
            max_lead_days: Maximum days ahead to signal (default 13)
            
        Returns:
            DataFrame with 'early_warning_target' column (0 or 1)
        """
        print(
            f"🎯 Creating early warning target: {drawdown_threshold*100}% drawdown, "
            f"{min_lead_days}-{max_lead_days} days ahead"
        )
        
        df = self.data.copy()
        df['early_warning_target'] = 0
        
        # Calculate rolling future drawdowns
        for lead_days in range(min_lead_days, max_lead_days + 1):
            # Look ahead 'lead_days' and calculate max drawdown over next 10 days
            future_prices = df['Close'].shift(-lead_days)
            future_low = df['Low'].shift(-lead_days).rolling(LOOKFORWARD_DAYS, min_periods=1).min()
            
            # Calculate drawdown from current price to future low
            future_drawdown = (future_low - df['Close']) / df['Close']
            
            # Mark as target if significant drawdown occurs in future
            df['early_warning_target'] |= (future_drawdown <= -drawdown_threshold)
        
        # Convert to int
        df['early_warning_target'] = df['early_warning_target'].astype(int)
        
        # Remove last few days (can't predict future for them)
        df = df.iloc[:-min_lead_days]
        
        target_rate = df['early_warning_target'].mean()
        total_events = df['early_warning_target'].sum()
        
        print(f"✅ Early warning target created: {target_rate:.1%} positive rate ({total_events} events)")
        
        return df
    
    def create_mean_reversion_target(self,
                                    bounce_threshold: float = MEAN_REVERSION_THRESHOLD,
                                    bounce_days: int = MEAN_REVERSION_DAYS) -> pd.DataFrame:
        """
        Create target for mean reversion after pullbacks
        
        Identifies opportunities where a pullback is followed by a bounce.
        Useful for rally continuation and mean reversion strategies.
        
        Args:
            bounce_threshold: Minimum bounce percentage (default 3%)
            bounce_days: Days to detect bounce (default 5)
            
        Returns:
            DataFrame with 'mean_reversion_target' column (0 or 1)
        """
        print(
            f"🎯 Creating mean reversion target: {bounce_threshold*100}% bounce "
            f"within {bounce_days} days"
        )
        
        df = self.data.copy()
        df['mean_reversion_target'] = 0
        
        # Calculate future returns over bounce_days window
        future_high = df['High'].shift(-1).rolling(bounce_days, min_periods=1).max()
        future_return = (future_high - df['Close']) / df['Close']
        
        # Mark as target if bounce occurs
        df['mean_reversion_target'] = (future_return >= bounce_threshold).astype(int)
        
        # Remove last few days
        df = df.iloc[:-bounce_days]
        
        target_rate = df['mean_reversion_target'].mean()
        total_events = df['mean_reversion_target'].sum()
        
        print(f"✅ Mean reversion target created: {target_rate:.1%} positive rate ({total_events} events)")
        
        return df
    
    def create_pullback_target(self,
                              pullback_threshold: float = 0.02,
                              bounce_threshold: float = 0.03,
                              max_pullback_days: int = 10,
                              max_bounce_days: int = 10) -> pd.DataFrame:
        """
        Create target for pullback-then-bounce patterns
        
        From rally_analyzer: Identifies healthy pullbacks during uptrends
        that lead to continuation moves.
        
        Args:
            pullback_threshold: Minimum pullback percentage (default 2%)
            bounce_threshold: Minimum bounce percentage (default 3%)
            max_pullback_days: Maximum days for pullback (default 10)
            max_bounce_days: Maximum days for bounce (default 10)
            
        Returns:
            DataFrame with 'pullback_target' column (0 or 1)
        """
        print(
            f"🎯 Creating pullback target: {pullback_threshold*100}% pullback "
            f"→ {bounce_threshold*100}% bounce"
        )
        
        df = self.data.copy()
        df['pullback_target'] = 0
        
        # Find pullback lows
        rolling_min = df['Low'].rolling(max_pullback_days, min_periods=1).min()
        current_pullback = (rolling_min - df['Close']) / df['Close']
        
        # Find subsequent bounce
        future_high = df['High'].shift(-1).rolling(max_bounce_days, min_periods=1).max()
        future_bounce = (future_high - df['Close']) / df['Close']
        
        # Mark as target if pullback followed by bounce
        df['pullback_target'] = (
            (current_pullback <= -pullback_threshold) & 
            (future_bounce >= bounce_threshold)
        ).astype(int)
        
        # Remove last few days
        df = df.iloc[:-max_bounce_days]
        
        target_rate = df['pullback_target'].mean()
        total_events = df['pullback_target'].sum()
        
        print(f"✅ Pullback target created: {target_rate:.1%} positive rate ({total_events} events)")
        
        return df
    
    def create_volatility_expansion_target(self,
                                          volatility_threshold: float = 1.5,
                                          lead_days: int = 3) -> pd.DataFrame:
        """
        Create target for volatility expansion events
        
        Identifies periods where volatility is about to spike, useful
        for options strategies and risk management.
        
        Args:
            volatility_threshold: Multiple of current volatility (default 1.5x)
            lead_days: Days ahead to predict (default 3)
            
        Returns:
            DataFrame with 'volatility_expansion_target' column (0 or 1)
        """
        print(f"🎯 Creating volatility expansion target: {volatility_threshold}x vol spike")
        
        df = self.data.copy()
        
        # Calculate current and future realized volatility
        df['returns'] = df['Close'].pct_change()
        current_vol = df['returns'].rolling(20).std()
        future_vol = df['returns'].shift(-lead_days).rolling(20).std()
        
        # Mark as target if volatility expands
        df['volatility_expansion_target'] = (
            future_vol >= volatility_threshold * current_vol
        ).fillna(0).astype(int)
        
        # Remove last few days
        df = df.iloc[:-lead_days]
        
        target_rate = df['volatility_expansion_target'].mean()
        total_events = df['volatility_expansion_target'].sum()
        
        print(f"✅ Volatility expansion target created: {target_rate:.1%} positive rate ({total_events} events)")
        
        return df
    
    def create_all_targets(self) -> pd.DataFrame:
        """
        Create all target types in one DataFrame
        
        Returns:
            DataFrame with all target columns
        """
        print("=" * 80)
        print("🎯 CREATING ALL TARGETS")
        print("=" * 80)
        
        # Create each target type
        df = self.create_early_warning_target()
        
        # For other targets, we need to re-create from original data
        # since each method returns truncated data
        df_mr = TargetCreator(self.data).create_mean_reversion_target()
        df_pb = TargetCreator(self.data).create_pullback_target()
        df_ve = TargetCreator(self.data).create_volatility_expansion_target()
        
        # Merge targets on index
        df = df.join(df_mr['mean_reversion_target'], how='left')
        df = df.join(df_pb['pullback_target'], how='left')
        df = df.join(df_ve['volatility_expansion_target'], how='left')
        
        # Fill NaN values with 0
        target_columns = [
            'mean_reversion_target', 'pullback_target', 'volatility_expansion_target'
        ]
        for col in target_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        print("=" * 80)
        print("✅ ALL TARGETS CREATED")
        print("=" * 80)
        
        return df


if __name__ == "__main__":
    # Test target creation
    import yfinance as yf
    
    print("Testing TargetCreator...")
    spy_data = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    
    creator = TargetCreator(spy_data)
    df_with_targets = creator.create_all_targets()
    
    print("\n" + "=" * 80)
    print("TARGET CREATION TEST COMPLETE")
    print("=" * 80)
    print(f"Total records: {len(df_with_targets)}")
    print("\nTarget summary:")
    for col in df_with_targets.columns:
        if 'target' in col:
            rate = df_with_targets[col].mean()
            events = df_with_targets[col].sum()
            print(f"  {col}: {rate:.1%} ({events} events)")

