"""
Volatility Index Feature Engine

Orchestrates all volatility index features (VIX, VVIX, etc.).
"""

import pandas as pd
from typing import Dict, Optional, Any
from engines.base import BaseFeatureEngine
from features.volatility_indices.vix import VIXFeature


class VolatilityFeatureEngine(BaseFeatureEngine):
    """Engine for volatility index features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Volatility Index Features")
        self._initialize_features(config or {})
    
    def _initialize_features(self, config: Optional[Dict[str, Any]] = None):
        """Initialize volatility features"""
        self.add_feature(VIXFeature())
