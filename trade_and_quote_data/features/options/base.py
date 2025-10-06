"""
Base Classes for Options Features

Options features analyze options chains, greeks, and flow.
"""

from features.base import BaseFeature
import pandas as pd

class BaseOptionsFeature(BaseFeature):
    """Base class for options features"""
    
    def __init__(self, name: str, params: dict = None):
        super().__init__(name, params)
        self.required_columns = ['strike', 'expiration', 'option_type']
    
    def validate_options_data(self, options_data: pd.DataFrame) -> bool:
        """Validate options data"""
        if options_data is None or len(options_data) == 0:
            raise ValueError(f"{self.name} requires options_data")
        
        missing = [c for c in self.required_columns if c not in options_data.columns]
        if missing:
            raise ValueError(f"{self.name} requires columns: {missing}")
        
        return True
