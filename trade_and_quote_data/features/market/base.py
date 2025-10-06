"""
Base Classes for Market Features

Market features require sector/market data in addition to primary instrument.
"""

from features.base import BaseFeature
import pandas as pd
from typing import Dict

class BaseMarketFeature(BaseFeature):
    """Base class for market-level features requiring sector data"""
    
    def __init__(self, name: str, params: dict = None):
        super().__init__(name, params)
        self.required_sectors = []  # Can be specified by subclass
    
    def validate_sector_data(self, sector_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate sector data availability"""
        if not sector_data:
            raise ValueError(f"{self.name} requires sector_data")
        
        missing_sectors = [s for s in self.required_sectors if s not in sector_data]
        if missing_sectors:
            raise ValueError(f"{self.name} requires sectors: {missing_sectors}")
        
        return True
