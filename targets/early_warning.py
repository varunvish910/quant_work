"""
Early Warning Target

Predicts 2%+ drawdowns within the next 3-5 days.
"""

import pandas as pd
import numpy as np
from targets.base import ForwardLookingTarget


class EarlyWarningTarget(ForwardLookingTarget):
    """
    Early warning target for drawdown prediction.
    
    Signals BEFORE drawdowns happen (not during) to enable actionable trading decisions.
    """
    
    def __init__(self, 
                 drawdown_threshold: float = 0.02,
                 min_lead_days: int = 3,
                 max_lead_days: int = 5,
                 lookforward_window: int = 5):
        super().__init__(
            name="early_warning",
            min_lead_days=min_lead_days,
            max_lead_days=max_lead_days,
            params={
                'drawdown_threshold': drawdown_threshold,
                'lookforward_window': lookforward_window
            }
        )
        self.drawdown_threshold = drawdown_threshold
        self.lookforward_window = lookforward_window
        self.required_columns = ['Close', 'Low']
    
    def create(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create early warning target.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with early_warning_target column
        """
        self.validate_data(data)
        
        print(f"🎯 Creating Early Warning Target")
        print(f"   Drawdown threshold: {self.drawdown_threshold*100}%")
        print(f"   Lead time: {self.min_lead_days}-{self.max_lead_days} days")
        print(f"   Lookforward window: {self.lookforward_window} days")
        
        df = data.copy()
        df[self.target_column] = 0
        
        # Look ahead and mark dates where significant drawdown will occur
        for lead_days in range(self.min_lead_days, self.max_lead_days + 1):
            # Future prices starting lead_days ahead
            future_low = df['Low'].shift(-lead_days).rolling(
                self.lookforward_window, min_periods=1
            ).min()
            
            # Calculate drawdown from current price to future low
            future_drawdown = (future_low - df['Close']) / df['Close']
            
            # Mark as target if significant drawdown occurs
            df[self.target_column] |= (future_drawdown <= -self.drawdown_threshold)
        
        # Convert to int
        df[self.target_column] = df[self.target_column].astype(int)
        
        # Remove rows we can't predict for
        df = self._truncate_for_prediction_horizon(df)
        
        # Print summary
        self._print_target_summary(df[self.target_column])
        
        return df


if __name__ == "__main__":
    # Test early warning target
    import yfinance as yf
    
    print("Testing EarlyWarningTarget...")
    spy = yf.download('SPY', start='2020-01-01', end='2024-10-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    target = EarlyWarningTarget(drawdown_threshold=0.05)
    result = target.create(spy)
    
    print(f"\n✅ Test complete: {len(result)} rows")
    print(f"Target stats: {target.get_target_stats()}")
