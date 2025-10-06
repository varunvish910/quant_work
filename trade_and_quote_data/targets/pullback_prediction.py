"""
Pullback Prediction Target

Predicts if a 5%+ pullback will occur within the next 5-15 days.
This is useful for:
- Risk management
- Timing exits
- Hedging decisions
- Reducing exposure before drops
"""

import pandas as pd
import numpy as np
from targets.base import BaseTarget


class PullbackPredictionTarget(BaseTarget):
    """
    Predict if a 5%+ pullback will occur in the next 5-15 days.
    
    This target looks FORWARD to identify upcoming pullbacks, giving you
    time to reduce risk, hedge, or take profits before the drop.
    """
    
    def __init__(self, 
                 pullback_threshold: float = 0.05,
                 min_days: int = 5,
                 max_days: int = 15):
        """
        Initialize pullback prediction target
        
        Args:
            pullback_threshold: Minimum pullback size (default 5%)
            min_days: Minimum days ahead to look (default 5)
            max_days: Maximum days ahead to look (default 15)
        """
        super().__init__("pullback_prediction", params={
            'pullback_threshold': pullback_threshold,
            'min_days': min_days,
            'max_days': max_days
        })
        self.pullback_threshold = pullback_threshold
        self.min_days = min_days
        self.max_days = max_days
        self.required_columns = ['Close', 'Low']
    
    def create(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Create pullback prediction target
        
        For each day, looks 5-15 days ahead and checks if price drops 5%+
        from current level.
        """
        print(f"🎯 Creating Pullback Prediction Target")
        print(f"   Pullback threshold: {self.pullback_threshold*100}%")
        print(f"   Lookforward window: {self.min_days}-{self.max_days} days")
        
        self.validate_data(data)
        df = data.copy()
        
        # Initialize target
        df[self.target_column] = 0
        
        # For each day, look forward 5-15 days
        for i in range(len(df) - self.max_days):
            current_price = df['Close'].iloc[i]
            
            # Get the window of future prices (days 5-15 ahead)
            start_idx = i + self.min_days
            end_idx = min(i + self.max_days + 1, len(df))
            
            if start_idx < len(df):
                future_window = df.iloc[start_idx:end_idx]
                
                # Find the lowest price in this window
                future_low = future_window['Low'].min()
                
                # Calculate pullback from current price
                pullback = (current_price - future_low) / current_price
                
                # Mark as target if pullback >= threshold
                if pullback >= self.pullback_threshold:
                    df.iloc[i, df.columns.get_loc(self.target_column)] = 1
        
        # Remove last N days (can't predict for them)
        df = df.iloc[:-self.max_days]
        
        # Print statistics
        total_samples = len(df)
        positive_samples = df[self.target_column].sum()
        positive_rate = positive_samples / total_samples if total_samples > 0 else 0
        
        print(f"\n🎯 PULLBACK PREDICTION TARGET CREATED")
        print(f"   Total samples: {total_samples}")
        print(f"   Pullback events: {positive_samples} ({positive_rate:.1%})")
        print(f"   No pullback: {total_samples - positive_samples} ({(1-positive_rate):.1%})")
        
        # Calculate class balance
        if positive_rate > 0:
            class_balance = min(positive_rate, 1 - positive_rate) / max(positive_rate, 1 - positive_rate)
            print(f"   Class balance: {class_balance:.2f}")
        
        return df


class PullbackSeverityTarget(BaseTarget):
    """
    Multi-class target for pullback severity prediction.
    
    Classes:
    0 = No pullback (<3%)
    1 = Small pullback (3-5%)
    2 = Medium pullback (5-7%)
    3 = Large pullback (>7%)
    """
    
    def __init__(self, min_days: int = 5, max_days: int = 15):
        super().__init__("pullback_severity", params={
            'min_days': min_days,
            'max_days': max_days
        })
        self.min_days = min_days
        self.max_days = max_days
    
    def create(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create multi-class pullback severity target"""
        df = data.copy()
        df[self.target_column] = 0
        
        for i in range(len(df) - self.max_days):
            current_price = df['Close'].iloc[i]
            start_idx = i + self.min_days
            end_idx = min(i + self.max_days + 1, len(df))
            
            if start_idx < len(df):
                future_window = df.iloc[start_idx:end_idx]
                future_low = future_window['Low'].min()
                pullback = (current_price - future_low) / current_price
                
                # Classify severity
                if pullback >= 0.07:
                    severity = 3  # Large
                elif pullback >= 0.05:
                    severity = 2  # Medium
                elif pullback >= 0.03:
                    severity = 1  # Small
                else:
                    severity = 0  # No pullback
                
                df.iloc[i, df.columns.get_loc(self.target_column)] = severity
        
        df = df.iloc[:-self.max_days]
        
        print(f"\n🎯 PULLBACK SEVERITY TARGET CREATED")
        print(f"   Class distribution:")
        for severity, count in df[self.target_column].value_counts().sort_index().items():
            severity_names = {0: 'No pullback', 1: 'Small (3-5%)', 2: 'Medium (5-7%)', 3: 'Large (>7%)'}
            print(f"     {severity_names.get(severity, severity)}: {count} ({count/len(df):.1%})")
        
        return df


if __name__ == "__main__":
    # Test the target
    print("Testing PullbackPredictionTarget...")
    
    import yfinance as yf
    
    # Download test data
    spy = yf.download('SPY', start='2020-01-01', end='2024-10-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # Create target
    target = PullbackPredictionTarget(
        pullback_threshold=0.05,
        min_days=5,
        max_days=15
    )
    
    result = target.create(spy)
    
    print(f"\n✅ Target created successfully")
    print(f"Target column: {target.get_target_column()}")
    print(f"\nSample data (last 10 rows):")
    print(result[[target.get_target_column()]].tail(10))
