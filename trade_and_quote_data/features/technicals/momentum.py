"""
Momentum Features

Calculates momentum-based technical indicators across multiple timeframes.
"""

import pandas as pd
import numpy as np
from features.technicals.base import BaseTechnicalFeature


class MomentumFeature(BaseTechnicalFeature):
    """Calculate momentum (rate of change) across multiple periods"""
    
    def __init__(self, windows: list = None):
        if windows is None:
            windows = [5, 10, 20, 50]
        super().__init__("Momentum", params={'windows': windows})
        self.windows = windows
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        for window in self.windows:
            col_name = f'momentum_{window}d'
            df[col_name] = df['Close'].pct_change(window) * 100
            self.feature_names.append(col_name)
        
        return df


class RSIFeature(BaseTechnicalFeature):
    """Calculate Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        super().__init__("RSI", params={'period': period})
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.period).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        self.feature_names = ['rsi']
        return df


class MACDFeature(BaseTechnicalFeature):
    """Calculate MACD indicator"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD", params={'fast': fast, 'slow': slow, 'signal': signal})
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        df = data.copy()
        
        ema_fast = df['Close'].ewm(span=self.fast).mean()
        ema_slow = df['Close'].ewm(span=self.slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        self.feature_names = ['macd', 'macd_signal', 'macd_histogram']
        return df
