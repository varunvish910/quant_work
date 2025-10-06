"""
Market Feature Engine

Orchestrates all market-level features (sectors, breadth, rotation).
"""

import pandas as pd
from typing import Dict, Optional, Any
from engines.base import BaseFeatureEngine
from features.market.sector_rotation import SectorRotationFeature
from features.market.rotation_indicators import RotationIndicatorFeature


class MarketFeatureEngine(BaseFeatureEngine):
    """Engine for market-level features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Market Features")
        self._initialize_features(config or {})
    
    def _initialize_features(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market features"""
        self.add_feature(SectorRotationFeature())
        self.add_feature(RotationIndicatorFeature())
