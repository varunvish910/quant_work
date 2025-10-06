"""
Volatility Features

Calculates volatility-based indicators.
"""

import pandas as pd
import numpy as np
from features.technicals.base import BaseTechnicalFeature


class VolatilityFeature(BaseTechnicalFeature):
    """Calculate realized volatility"""
    
    def __init__(self, window: int = 20):
        super().__init__("Volatility", params={'window': window})
        self.window = window
        self.required_columns = ['Close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        returns = df['Close'].pct_change()
        df[f'volatility_{self.window}d'] = returns.rolling(self.window).std() * np.sqrt(252)
        
        self.feature_names = [f'volatility_{self.window}d']
        return df


class ATRFeature(BaseTechnicalFeature):
    """Calculate Average True Range"""
    
    def __init__(self, period: int = 14):
        super().__init__("ATR", params={'period': period})
        self.period = period
        self.required_columns = ['High', 'Low', 'Close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{self.period}'] = true_range.rolling(self.period).mean()
        
        self.feature_names = [f'atr_{self.period}']
        return df


class BollingerBandsFeature(BaseTechnicalFeature):
    """Calculate Bollinger Bands"""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__("BollingerBands", params={'window': window, 'num_std': num_std})
        self.window = window
        self.num_std = num_std
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        sma = df['Close'].rolling(self.window).mean()
        std = df['Close'].rolling(self.window).std()
        
        df['bb_upper'] = sma + (std * self.num_std)
        df['bb_lower'] = sma - (std * self.num_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        self.feature_names = ['bb_upper', 'bb_lower', 'bb_width', 'bb_position']
        return df
