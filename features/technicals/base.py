"""
Base Classes for Technical Features

Technical features are calculated from OHLCV data only.
"""

from features.base import BaseFeature
import pandas as pd
from typing import List

class BaseTechnicalFeature(BaseFeature):
    """Base class for technical indicators requiring OHLCV data"""
    
    def __init__(self, name: str, params: dict = None):
        super().__init__(name, params)
        self.required_columns = ['Close']  # Minimum requirement
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate OHLCV data"""
        super().validate_data(data)
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"{self.name} requires DatetimeIndex")
        
        return True
