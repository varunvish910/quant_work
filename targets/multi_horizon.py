"""Multi-horizon prediction targets"""
import pandas as pd
from targets.base import BaseTarget

class MultiHorizonTarget(BaseTarget):
    """Predict risk at multiple time horizons"""
    
    def __init__(self, horizons=[3, 5, 10, 20], threshold=0.03):
        super().__init__("multi_horizon")
        self.horizons = horizons
        self.threshold = threshold
    
    def create(self, data, **kwargs):
        df = data.copy()
        
        for days in self.horizons:
            future_low = df['Low'].shift(-days).rolling(days, min_periods=1).min()
            drawdown = (future_low - df['Close']) / df['Close']
            df[f'risk_{days}d'] = (drawdown < -self.threshold).astype(int)
        
        # Truncate last N days
        max_horizon = max(self.horizons)
        df = df.iloc[:-max_horizon]
        
        return df
