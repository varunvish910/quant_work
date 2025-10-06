"""Trend strength and direction features"""
import pandas as pd
import numpy as np
from features.technicals.base import BaseTechnicalFeature

class TrendFeature(BaseTechnicalFeature):
    """Trend analysis features"""
    
    def __init__(self):
        super().__init__("Trend")
        self.required_columns = ['High', 'Low', 'Close']
    
    def calculate(self, data, **kwargs):
        df = data.copy()
        
        # Higher highs and higher lows
        df['higher_highs'] = (df['High'] > df['High'].shift(1)).rolling(5).sum()
        df['higher_lows'] = (df['Low'] > df['Low'].shift(1)).rolling(5).sum()
        df['uptrend_score'] = (df['higher_highs'] + df['higher_lows']) / 10
        
        # Lower highs and lower lows
        df['lower_highs'] = (df['High'] < df['High'].shift(1)).rolling(5).sum()
        df['lower_lows'] = (df['Low'] < df['Low'].shift(1)).rolling(5).sum()
        df['downtrend_score'] = (df['lower_highs'] + df['lower_lows']) / 10
        
        # Trend strength
        df['trend_strength'] = abs(df['uptrend_score'] - df['downtrend_score'])
        
        self.feature_names = ['higher_highs', 'higher_lows', 'uptrend_score', 
                             'lower_highs', 'lower_lows', 'downtrend_score', 'trend_strength']
        return df
