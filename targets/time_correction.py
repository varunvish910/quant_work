"""
Time Correction Target

Detects sideways consolidation periods where the market moves horizontally
rather than up or down. These are "time corrections" rather than "price corrections".
"""

import pandas as pd
import numpy as np
from targets.base import BaseTarget
from typing import Dict, Any


class TimeCorrectionTarget(BaseTarget):
    """
    Detects time corrections: sideways consolidation over 30-60 days.
    
    Characteristics:
    - Price stays within a narrow range (e.g., ±3%)
    - Low volatility
    - No significant trend (flat slope)
    - Duration: 30-60 days
    """
    
    def __init__(self, 
                 max_range_pct: float = 0.03,  # ±3% range
                 min_lookforward_days: int = 30,
                 max_lookforward_days: int = 60,
                 max_abs_slope: float = 0.001):  # Nearly flat slope
        """
        Initialize Time Correction Target
        
        Args:
            max_range_pct: Maximum price range to qualify as consolidation (e.g., 0.03 = ±3%)
            min_lookforward_days: Minimum consolidation duration
            max_lookforward_days: Maximum consolidation duration
            max_abs_slope: Maximum absolute slope to qualify as "flat"
        """
        self.max_range_pct = max_range_pct
        self.min_lookforward_days = min_lookforward_days
        self.max_lookforward_days = max_lookforward_days
        self.max_abs_slope = max_abs_slope
        
        params = {
            'max_range_pct': max_range_pct,
            'min_lookforward_days': min_lookforward_days,
            'max_lookforward_days': max_lookforward_days,
            'max_abs_slope': max_abs_slope
        }
        
        super().__init__(
            name=f"time_correction_{int(max_range_pct*100)}pct",
            params=params
        )
    
    def create(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create time correction target
        
        A time correction is flagged if:
        1. Price stays within max_range_pct of starting price
        2. Consolidation lasts min_lookforward_days to max_lookforward_days
        3. Slope is nearly flat (abs(slope) < max_abs_slope)
        4. Low volatility (price compression)
        
        Args:
            data: DataFrame with 'Close', 'High', 'Low' prices
            
        Returns:
            DataFrame with target column added
        """
        df = data.copy()
        
        print(f"🎯 Creating Time Correction Target")
        print(f"   Max range: ±{self.max_range_pct*100:.1f}%")
        print(f"   Lookforward window: {self.min_lookforward_days}-{self.max_lookforward_days} days")
        print(f"   Max absolute slope: {self.max_abs_slope}")
        
        target_series = pd.Series(0, index=df.index, dtype=int)
        
        for i in range(len(df) - self.max_lookforward_days):
            current_close = df['Close'].iloc[i]
            
            # Check multiple window sizes within the range
            for window_size in range(self.min_lookforward_days, self.max_lookforward_days + 1, 5):
                if i + window_size >= len(df):
                    break
                
                future_window = df.iloc[i:i+window_size+1]
                
                if len(future_window) < self.min_lookforward_days:
                    continue
                
                # Calculate price range
                max_price = future_window['High'].max()
                min_price = future_window['Low'].min()
                price_range = (max_price - min_price) / current_close
                
                # Check if price stays within range
                if price_range <= self.max_range_pct * 2:  # *2 because it's ±range
                    # Check slope (should be nearly flat)
                    x = np.arange(len(future_window))
                    y = future_window['Close'].values
                    
                    if len(x) > 1:
                        # Normalize slope by dividing by starting price
                        slope = np.polyfit(x, y, 1)[0] / current_close
                        
                        # Check if slope is nearly flat
                        if abs(slope) <= self.max_abs_slope:
                            # Calculate volatility (should be low)
                            daily_returns = future_window['Close'].pct_change().dropna()
                            if len(daily_returns) > 0:
                                volatility = daily_returns.std()
                                
                                # Flag as time correction if volatility is low
                                # (less than 1% daily volatility)
                                if volatility <= 0.01:
                                    target_series.iloc[i] = 1
                                    break  # Found a consolidation, move to next day
        
        df[self.target_column] = target_series
        
        positive_rate = df[self.target_column].mean() * 100
        num_events = df[self.target_column].sum()
        num_no_events = len(df) - num_events
        
        print(f"\n🎯 TIME CORRECTION TARGET CREATED")
        print(f"   Total samples: {len(df)}")
        print(f"   Time correction events: {num_events} ({positive_rate:.1f}%)")
        print(f"   No time correction: {num_no_events} ({100-positive_rate:.1f}%)")
        print(f"   Class balance: {num_events / len(df):.2f}")
        
        return df
