"""
Base Classes for Volatility Index Features

Volatility index features analyze VIX, VVIX, and related indices.
"""

from features.base import BaseFeature
import pandas as pd
from typing import Dict

class BaseVolatilityIndexFeature(BaseFeature):
    """Base class for volatility index features"""
    
    def __init__(self, name: str, params: dict = None):
        super().__init__(name, params)
        self.required_indices = []
    
    def validate_volatility_data(self, volatility_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate volatility index data availability"""
        if not volatility_data:
            raise ValueError(f"{self.name} requires volatility_data")
        
        missing = [v for v in self.required_indices if v not in volatility_data]
        if missing:
            raise ValueError(f"{self.name} requires indices: {missing}")
        
        return True
