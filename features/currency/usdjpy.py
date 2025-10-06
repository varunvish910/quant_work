"""
USD/JPY Features

Critical for carry trade unwind detection.
"""

import pandas as pd
import numpy as np
from features.currency.base import BaseCurrencyFeature


class USDJPYFeature(BaseCurrencyFeature):
    """USD/JPY momentum and carry trade signals"""
    
    def __init__(self):
        super().__init__("USDJPY")
        self.required_currencies = ['USDJPY']
    
    def calculate(self, data: pd.DataFrame, currency_data: dict = None, **kwargs) -> pd.DataFrame:
        if currency_data is None or 'USDJPY' not in currency_data:
            return data
        
        self.validate_currency_data(currency_data)
        df = data.copy()
        
        usdjpy = currency_data['USDJPY']['Close']
        
        df['usdjpy_level'] = usdjpy
        df['usdjpy_momentum_3d'] = usdjpy.pct_change(3) * 100
        df['usdjpy_momentum_5d'] = usdjpy.pct_change(5) * 100
        df['usdjpy_momentum_10d'] = usdjpy.pct_change(10) * 100
        df['usdjpy_acceleration_3d'] = df['usdjpy_momentum_3d'].diff()
        df['usdjpy_volatility'] = usdjpy.pct_change().rolling(20).std() * 100
        
        # Carry trade unwind risk
        df['yen_carry_unwind_risk'] = (
            (df['usdjpy_momentum_5d'] < -1.0) &
            (df['usdjpy_acceleration_3d'] < 0)
        ).astype(int)
        
        self.feature_names = [
            'usdjpy_level', 'usdjpy_momentum_3d', 'usdjpy_momentum_5d',
            'usdjpy_momentum_10d', 'usdjpy_acceleration_3d', 'usdjpy_volatility',
            'yen_carry_unwind_risk'
        ]
        return df
