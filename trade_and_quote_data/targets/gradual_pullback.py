"""
Gradual Pullback Target

Detects slow, steady declines over a medium-term horizon (15-30 days).
This is different from crashes (sharp drops) and represents a grinding bear market.
"""

import pandas as pd
import numpy as np
from targets.base import BaseTarget
from typing import Dict, Any


class GradualPullbackTarget(BaseTarget):
    """
    Detects gradual pullbacks: 4%+ decline over 15-30 days.
    
    Characteristics:
    - Steady downward trend (negative slope)
    - Lower daily volatility than crashes
    - Sustained selling pressure
    """
    
    def __init__(self, 
                 decline_threshold: float = 0.04,  # 4% decline
                 min_lookforward_days: int = 15,
                 max_lookforward_days: int = 30,
                 max_daily_volatility: float = 0.02):  # 2% max daily moves
        """
        Initialize Gradual Pullback Target
        
        Args:
            decline_threshold: Minimum decline to qualify (e.g., 0.04 = 4%)
            min_lookforward_days: Minimum days for decline to occur
            max_lookforward_days: Maximum days for decline to occur
            max_daily_volatility: Maximum daily volatility to qualify as "gradual"
        """
        self.decline_threshold = decline_threshold
        self.min_lookforward_days = min_lookforward_days
        self.max_lookforward_days = max_lookforward_days
        self.max_daily_volatility = max_daily_volatility
        
        params = {
            'decline_threshold': decline_threshold,
            'min_lookforward_days': min_lookforward_days,
            'max_lookforward_days': max_lookforward_days,
            'max_daily_volatility': max_daily_volatility
        }
        
        super().__init__(
            name=f"gradual_pullback_{int(decline_threshold*100)}pct",
            params=params
        )
    
    def create(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create gradual pullback target
        
        A gradual pullback is flagged if:
        1. Price declines by decline_threshold or more
        2. Decline occurs over min_lookforward_days to max_lookforward_days
        3. Daily volatility is below max_daily_volatility (to ensure it's gradual)
        4. Negative slope (consistent downward trend)
        
        Args:
            data: DataFrame with 'Close' prices
            
        Returns:
            DataFrame with target column added
        """
        df = data.copy()
        
        print(f"🎯 Creating Gradual Pullback Target")
        print(f"   Decline threshold: {self.decline_threshold*100:.1f}%")
        print(f"   Lookforward window: {self.min_lookforward_days}-{self.max_lookforward_days} days")
        print(f"   Max daily volatility: {self.max_daily_volatility*100:.1f}%")
        
        target_series = pd.Series(0, index=df.index, dtype=int)
        
        for i in range(len(df) - self.max_lookforward_days):
            current_close = df['Close'].iloc[i]
            
            # Define the future window
            start_idx = i + self.min_lookforward_days
            end_idx = i + self.max_lookforward_days + 1
            
            if start_idx >= len(df):
                break
            
            future_window = df.iloc[start_idx:end_idx]
            
            if len(future_window) == 0:
                continue
            
            # Check for decline
            min_in_window = future_window['Close'].min()
            decline = (min_in_window / current_close - 1)
            
            if decline <= -self.decline_threshold:
                # Check if it's gradual (low daily volatility)
                # Calculate daily returns in the window from current to min point
                full_window = df.iloc[i:end_idx]
                daily_returns = full_window['Close'].pct_change().dropna()
                
                if len(daily_returns) > 0:
                    daily_volatility = daily_returns.std()
                    
                    # Check for negative slope (consistent downward trend)
                    # Use linear regression slope
                    x = np.arange(len(full_window))
                    y = full_window['Close'].values
                    
                    if len(x) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        
                        # Flag as gradual pullback if:
                        # 1. Daily volatility is low (gradual)
                        # 2. Slope is negative (downward trend)
                        if daily_volatility <= self.max_daily_volatility and slope < 0:
                            target_series.iloc[i] = 1
        
        df[self.target_column] = target_series
        
        positive_rate = df[self.target_column].mean() * 100
        num_events = df[self.target_column].sum()
        num_no_events = len(df) - num_events
        
        print(f"\n🎯 GRADUAL PULLBACK TARGET CREATED")
        print(f"   Total samples: {len(df)}")
        print(f"   Gradual pullback events: {num_events} ({positive_rate:.1f}%)")
        print(f"   No gradual pullback: {num_no_events} ({100-positive_rate:.1f}%)")
        print(f"   Class balance: {num_events / len(df):.2f}")
        
        return df
