"""
Regime-Specific Target

Adaptive thresholds based on volatility regime.
"""

import pandas as pd
import numpy as np
from targets.base import BaseTarget

class RegimeSpecificTarget(BaseTarget):
    """Target with adaptive thresholds for different volatility regimes"""
    
    def __init__(self):
        super().__init__("regime_specific")
    
    def create(self, data, **kwargs):
        df = data.copy()
        
        # Calculate volatility regime
        returns = df['Close'].pct_change()
        vol = returns.rolling(20).std()
        vol_percentile = vol.rolling(252, min_periods=20).rank(pct=True)
        
        # Define regimes
        df['regime'] = 'medium'
        df.loc[vol_percentile < 0.33, 'regime'] = 'low_vol'
        df.loc[vol_percentile > 0.67, 'regime'] = 'high_vol'
        
        # Adaptive thresholds
        df[self.target_column] = 0
        for i in range(len(df) - 10):
            regime = df.iloc[i]['regime']
            current_price = df.iloc[i]['Close']
            future_low = df.iloc[i:i+10]['Low'].min()
            drawdown = (current_price - future_low) / current_price
            
            # Different thresholds for different regimes
            if regime == 'low_vol':
                threshold = 0.03  # 3% in low vol
            elif regime == 'high_vol':
                threshold = 0.07  # 7% in high vol
            else:
                threshold = 0.05  # 5% in medium vol
            
            if drawdown >= threshold:
                df.iloc[i, df.columns.get_loc(self.target_column)] = 1
        
        df = df.iloc[:-10]
        print(f"✅ Regime-Specific Target: {df[self.target_column].sum()} events")
        return df
