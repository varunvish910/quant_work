"""
Base Classes for Currency Features

Currency features analyze FX pairs and currency strength.
"""

from features.base import BaseFeature
import pandas as pd
from typing import Dict

class BaseCurrencyFeature(BaseFeature):
    """Base class for currency features"""
    
    def __init__(self, name: str, params: dict = None):
        super().__init__(name, params)
        self.required_currencies = []
    
    def validate_currency_data(self, currency_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate currency data availability"""
        if not currency_data:
            raise ValueError(f"{self.name} requires currency_data")
        
        missing = [c for c in self.required_currencies if c not in currency_data]
        if missing:
            raise ValueError(f"{self.name} requires currencies: {missing}")
        
        return True
