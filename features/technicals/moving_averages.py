"""
Moving Average Features

Calculates various moving averages and their relationships.
"""

import pandas as pd
import numpy as np
from features.technicals.base import BaseTechnicalFeature


class SMAFeature(BaseTechnicalFeature):
    """Calculate Simple Moving Averages"""
    
    def __init__(self, windows: list = None):
        if windows is None:
            windows = [20, 50, 200]
        super().__init__("SMA", params={'windows': windows})
        self.windows = windows
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        for window in self.windows:
            col_name = f'sma_{window}'
            df[col_name] = df['Close'].rolling(window, min_periods=max(1, window//4)).mean()
            self.feature_names.append(col_name)
        
        return df


class EMAFeature(BaseTechnicalFeature):
    """Calculate Exponential Moving Averages"""
    
    def __init__(self, windows: list = None):
        if windows is None:
            windows = [12, 26, 50]
        super().__init__("EMA", params={'windows': windows})
        self.windows = windows
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        for window in self.windows:
            col_name = f'ema_{window}'
            df[col_name] = df['Close'].ewm(span=window).mean()
            self.feature_names.append(col_name)
        
        return df


class MADistanceFeature(BaseTechnicalFeature):
    """Calculate distance from moving averages"""
    
    def __init__(self, windows: list = None):
        if windows is None:
            windows = [50, 200]
        super().__init__("MADistance", params={'windows': windows})
        self.windows = windows
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        for window in self.windows:
            sma = df['Close'].rolling(window, min_periods=max(1, window//4)).mean()
            col_name = f'price_vs_sma{window}'
            df[col_name] = ((df['Close'] - sma) / sma) * 100
            self.feature_names.append(col_name)
        
        return df
