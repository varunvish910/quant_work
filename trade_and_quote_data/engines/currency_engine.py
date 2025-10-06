"""
Currency Feature Engine

Orchestrates all currency features.
"""

import pandas as pd
from typing import Dict, Optional, Any
from engines.base import BaseFeatureEngine
from features.currency.usdjpy import USDJPYFeature


class CurrencyFeatureEngine(BaseFeatureEngine):
    """Engine for currency features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Currency Features")
        self._initialize_features(config or {})
    
    def _initialize_features(self, config: Optional[Dict[str, Any]] = None):
        """Initialize currency features"""
        self.add_feature(USDJPYFeature())
