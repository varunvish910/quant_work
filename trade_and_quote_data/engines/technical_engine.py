"""
Technical Feature Engine

Orchestrates all technical indicator calculations.
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from engines.base import BaseFeatureEngine
from features.technicals.momentum import MomentumFeature, RSIFeature, MACDFeature
from features.technicals.volatility import VolatilityFeature, ATRFeature, BollingerBandsFeature
from features.technicals.moving_averages import SMAFeature, EMAFeature, MADistanceFeature
from features.technicals.volume import VolumeFeature
from features.technicals.trend import TrendFeature


class TechnicalFeatureEngine(BaseFeatureEngine):
    """Engine for all technical indicator features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Technical Features")
        self._initialize_features(config or {})
    
    def _initialize_features(self, config: Optional[Dict[str, Any]] = None):
        """Initialize all technical features"""
        
        # Momentum features
        self.add_feature(MomentumFeature(windows=[5, 10, 20, 50]))
        self.add_feature(RSIFeature(period=14))
        self.add_feature(MACDFeature())
        
        # Volatility features
        self.add_feature(VolatilityFeature(window=20))
        self.add_feature(ATRFeature(period=14))
        self.add_feature(BollingerBandsFeature())
        
        # Moving averages
        self.add_feature(SMAFeature(windows=[20, 50, 200]))
        self.add_feature(EMAFeature(windows=[12, 26, 50]))
        self.add_feature(MADistanceFeature(windows=[50, 200]))
        
        # Volume features
        self.add_feature(VolumeFeature())
        
        # Trend features
        self.add_feature(TrendFeature())
