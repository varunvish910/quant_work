"""
Mean Reversion Target

Identifies opportunities where pullbacks are followed by bounces.
"""

import pandas as pd
import numpy as np
from targets.base import ForwardLookingTarget


class MeanReversionTarget(ForwardLookingTarget):
    """
    Mean reversion target for bounce prediction after pullbacks.
    """
    
    def __init__(self, 
                 bounce_threshold: float = 0.03,
                 bounce_days: int = 5,
                 min_lead_days: int = 1,
                 max_lead_days: int = 5):
        super().__init__(
            name="mean_reversion",
            min_lead_days=min_lead_days,
            max_lead_days=max_lead_days,
            params={
                'bounce_threshold': bounce_threshold,
                'bounce_days': bounce_days
            }
        )
        self.bounce_threshold = bounce_threshold
        self.bounce_days = bounce_days
        self.required_columns = ['Close', 'High']
    
    def create(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create mean reversion target"""
        self.validate_data(data)
        
        print(f"🎯 Creating Mean Reversion Target")
        print(f"   Bounce threshold: {self.bounce_threshold*100}%")
        print(f"   Bounce window: {self.bounce_days} days")
        
        df = data.copy()
        
        # Calculate future returns
        future_high = df['High'].shift(-1).rolling(self.bounce_days, min_periods=1).max()
        future_return = (future_high - df['Close']) / df['Close']
        
        # Mark as target if bounce occurs
        df[self.target_column] = (future_return >= self.bounce_threshold).astype(int)
        
        # Remove rows we can't predict for
        df = df.iloc[:-self.bounce_days]
        
        # Print summary
        self._print_target_summary(df[self.target_column])
        
        return df
