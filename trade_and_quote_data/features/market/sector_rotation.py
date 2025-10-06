"""
Sector Rotation Features

Analyzes sector rotation and defensive positioning.
"""

import pandas as pd
import numpy as np
from features.market.base import BaseMarketFeature


class SectorRotationFeature(BaseMarketFeature):
    """Calculate sector rotation signals"""
    
    def __init__(self):
        super().__init__("SectorRotation")
        self.required_sectors = ['XLU', 'XLK', 'XLV']
    
    def calculate(self, data: pd.DataFrame, sector_data: dict = None, **kwargs) -> pd.DataFrame:
        if sector_data is None:
            return data
        
        self.validate_sector_data(sector_data)
        df = data.copy()
        
        # Calculate 20-day returns for each sector
        xlu_return = sector_data['XLU']['Close'].pct_change(20) * 100
        xlk_return = sector_data['XLK']['Close'].pct_change(20) * 100
        xlv_return = sector_data['XLV']['Close'].pct_change(20) * 100
        
        # Rotation signals
        df['xlu_vs_xlk'] = xlu_return - xlk_return
        df['xlv_vs_xlk'] = xlv_return - xlk_return
        df['defensive_rotation'] = ((xlu_return > xlk_return) & (xlv_return > xlk_return)).astype(int)
        
        self.feature_names = ['xlu_vs_xlk', 'xlv_vs_xlk', 'defensive_rotation']
        return df
