"""
VIX Features

Volatility index analysis and regime detection.
"""

import pandas as pd
import numpy as np
from features.volatility_indices.base import BaseVolatilityIndexFeature


class VIXFeature(BaseVolatilityIndexFeature):
    """VIX level, momentum, and regime features"""
    
    def __init__(self):
        super().__init__("VIX")
        self.required_indices = ['VIX']
    
    def calculate(self, data: pd.DataFrame, volatility_data: dict = None, **kwargs) -> pd.DataFrame:
        if volatility_data is None or 'VIX' not in volatility_data:
            return data
        
        self.validate_volatility_data(volatility_data)
        df = data.copy()
        
        vix = volatility_data['VIX']['Close']
        
        # Basic VIX metrics
        df['vix_level'] = vix
        df['vix_percentile_252d'] = vix.rolling(252, min_periods=20).rank(pct=True)
        
        # VIX momentum
        df['vix_momentum_3d'] = vix.pct_change(3) * 100
        df['vix_momentum_5d'] = vix.pct_change(5) * 100
        df['vix_momentum_10d'] = vix.pct_change(10) * 100
        
        # VIX regime
        df['vix_regime'] = pd.cut(vix, bins=[0, 15, 25, 35, 100], 
                                  labels=[1, 2, 3, 4], include_lowest=True).astype(float)
        
        # VIX spike detection
        df['vix_spike'] = (vix.pct_change() > 15).astype(int)
        
        # VIX mean reversion
        vix_ma20 = vix.rolling(20).mean()
        df['vix_vs_ma20'] = ((vix - vix_ma20) / vix_ma20) * 100
        df['vix_extreme_high'] = (df['vix_vs_ma20'] > 50).astype(int)
        
        self.feature_names = [
            'vix_level', 'vix_percentile_252d', 'vix_momentum_3d', 'vix_momentum_5d',
            'vix_momentum_10d', 'vix_regime', 'vix_spike', 'vix_vs_ma20', 'vix_extreme_high'
        ]
        return df
