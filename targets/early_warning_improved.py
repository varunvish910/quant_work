"""
Improved Early Warning Target

Key Improvements:
1. Expanded window: 3-7 days (not 3-5)
2. Severity prediction: Predict drawdown magnitude
3. Cluster-aware: Mark first signal in cluster
4. Multiple thresholds: 2%, 3%, 5% for different severity levels
"""

import pandas as pd
import numpy as np
from targets.base import ForwardLookingTarget


class ImprovedEarlyWarningTarget(ForwardLookingTarget):
    """
    Improved early warning target with:
    - Expanded prediction window (3-7 days)
    - Severity classification (minor/moderate/major)
    - Cluster detection
    """
    
    def __init__(self, 
                 drawdown_threshold: float = 0.02,
                 min_lead_days: int = 3,
                 max_lead_days: int = 7,  # Extended from 5 to 7
                 lookforward_window: int = 7):  # Extended from 5 to 7
        super().__init__(
            name="improved_early_warning",
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
        Create improved early warning target with severity classification.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with multiple target columns:
            - binary_target: 1 if any pullback >= threshold
            - severity: 0=none, 1=minor (2-3%), 2=moderate (3-5%), 3=major (5%+)
            - max_drawdown: Actual maximum drawdown in window
            - is_cluster_start: 1 if first signal in a cluster
        """
        self.validate_data(data)
        
        print(f"🎯 Creating Improved Early Warning Target")
        print(f"   Drawdown threshold: {self.drawdown_threshold*100}%")
        print(f"   Lead time: {self.min_lead_days}-{self.max_lead_days} days")
        print(f"   Lookforward window: {self.lookforward_window} days")
        
        df = data.copy()
        
        # Initialize target columns
        df['binary_target'] = 0
        df['max_drawdown'] = 0.0
        df['severity'] = 0
        
        # Look ahead and calculate maximum drawdown
        for lead_days in range(self.min_lead_days, self.max_lead_days + 1):
            # Future prices starting lead_days ahead
            future_low = df['Low'].shift(-lead_days).rolling(
                self.lookforward_window, min_periods=1
            ).min()
            
            # Calculate drawdown from current price to future low
            future_drawdown = (future_low - df['Close']) / df['Close']
            
            # Update max_drawdown if this is worse
            df['max_drawdown'] = np.minimum(df['max_drawdown'], future_drawdown)
            
            # Mark as target if significant drawdown occurs
            df['binary_target'] |= (future_drawdown <= -self.drawdown_threshold)
        
        # Convert binary target to int
        df['binary_target'] = df['binary_target'].astype(int)
        
        # Classify severity
        df.loc[df['max_drawdown'] <= -0.05, 'severity'] = 3  # Major: 5%+
        df.loc[(df['max_drawdown'] > -0.05) & (df['max_drawdown'] <= -0.03), 'severity'] = 2  # Moderate: 3-5%
        df.loc[(df['max_drawdown'] > -0.03) & (df['max_drawdown'] <= -0.02), 'severity'] = 1  # Minor: 2-3%
        df.loc[df['max_drawdown'] > -0.02, 'severity'] = 0  # None
        
        # Detect cluster starts (first signal after 5+ days of no signal)
        df['is_cluster_start'] = 0
        signal_indices = df[df['binary_target'] == 1].index
        
        if len(signal_indices) > 0:
            # First signal is always a cluster start
            df.loc[signal_indices[0], 'is_cluster_start'] = 1
            
            # Mark subsequent cluster starts (5+ days gap)
            for i in range(1, len(signal_indices)):
                days_since_last = (signal_indices[i] - signal_indices[i-1]).days
                if days_since_last >= 5:
                    df.loc[signal_indices[i], 'is_cluster_start'] = 1
        
        # Set the main target column (for compatibility)
        df[self.target_column] = df['binary_target']
        
        # Remove rows we can't predict for
        df = self._truncate_for_prediction_horizon(df)
        
        # Print summary
        self._print_target_summary(df['binary_target'])
        
        # Additional statistics
        print(f"\n📊 Severity Distribution:")
        severity_counts = df['severity'].value_counts().sort_index()
        severity_labels = {0: 'None', 1: 'Minor (2-3%)', 2: 'Moderate (3-5%)', 3: 'Major (5%+)'}
        for sev, count in severity_counts.items():
            pct = count / len(df) * 100
            print(f"   {severity_labels.get(sev, f'Level {sev}')}: {count} ({pct:.1f}%)")
        
        cluster_starts = df['is_cluster_start'].sum()
        total_signals = df['binary_target'].sum()
        print(f"\n📊 Cluster Analysis:")
        print(f"   Total signals: {total_signals}")
        print(f"   Cluster starts: {cluster_starts}")
        if cluster_starts > 0:
            print(f"   Avg signals per cluster: {total_signals / cluster_starts:.1f}")
        
        return df


if __name__ == "__main__":
    # Test improved early warning target
    import yfinance as yf
    
    print("Testing ImprovedEarlyWarningTarget...")
    spy = yf.download('SPY', start='2024-01-01', end='2024-12-31', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    target = ImprovedEarlyWarningTarget(drawdown_threshold=0.02)
    result = target.create(spy)
    
    print(f"\n✅ Test complete: {len(result)} rows")
    print(f"\nTarget columns created:")
    for col in ['binary_target', 'severity', 'max_drawdown', 'is_cluster_start']:
        if col in result.columns:
            print(f"   ✅ {col}")
