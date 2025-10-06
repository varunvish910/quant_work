"""
Unified Feature Engine V2

Master engine that orchestrates all feature calculation using the new modular engines.
This version uses the actual new engines instead of the compatibility layer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from engines.base import CompositeEngine
from engines.technical_engine import TechnicalFeatureEngine
from engines.market_engine import MarketFeatureEngine
from engines.currency_engine import CurrencyFeatureEngine
from engines.volatility_engine import VolatilityFeatureEngine
from engines.options_engine import OptionsFeatureEngine


class UnifiedFeatureEngineV2(CompositeEngine):
    """
    Unified engine using all new modular engines.
    
    This is the fully refactored version that uses:
    - TechnicalFeatureEngine
    - MarketFeatureEngine
    - CurrencyFeatureEngine
    - VolatilityFeatureEngine
    - OptionsFeatureEngine
    """
    
    def __init__(self, feature_sets: List[str] = None):
        super().__init__("Unified Feature Engine V2")
        
        if feature_sets is None:
            feature_sets = ['technicals']
        
        self.feature_sets = feature_sets
        self._initialize_engines()
        
        print(f"🔧 Initialized UnifiedFeatureEngineV2 with: {', '.join(feature_sets)}")
    
    def _initialize_engines(self):
        """Initialize requested engines"""
        
        if 'technicals' in self.feature_sets or 'all' in self.feature_sets:
            self.add_engine(TechnicalFeatureEngine())
        
        if 'market' in self.feature_sets or 'all' in self.feature_sets:
            self.add_engine(MarketFeatureEngine())
        
        if 'currency' in self.feature_sets or 'all' in self.feature_sets:
            self.add_engine(CurrencyFeatureEngine())
        
        if 'volatility' in self.feature_sets or 'all' in self.feature_sets:
            self.add_engine(VolatilityFeatureEngine())
        
        if 'options' in self.feature_sets or 'all' in self.feature_sets:
            self.add_engine(OptionsFeatureEngine())
    
    def calculate_all(self, 
                     spy_data: pd.DataFrame,
                     sector_data: Optional[Dict[str, pd.DataFrame]] = None,
                     currency_data: Optional[Dict[str, pd.DataFrame]] = None,
                     volatility_data: Optional[Dict[str, pd.DataFrame]] = None,
                     options_data: Optional[pd.DataFrame] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Calculate all features using new modular engines.
        
        Args:
            spy_data: SPY OHLCV data
            sector_data: Sector ETF data
            currency_data: Currency pair data
            volatility_data: Volatility index data
            options_data: Options chain data
            
        Returns:
            DataFrame with all features
        """
        print("=" * 80)
        print("🚀 UNIFIED FEATURE ENGINE V2 (Fully Refactored)")
        print("=" * 80)
        
        # Use the composite engine's calculate_all method
        df = super().calculate_all(
            spy_data,
            sector_data=sector_data,
            currency_data=currency_data,
            volatility_data=volatility_data,
            options_data=options_data
        )
        
        return df


if __name__ == "__main__":
    print("Testing UnifiedFeatureEngineV2...")
    
    import yfinance as yf
    
    # Test with sample data
    spy = yf.download('SPY', start='2024-01-01', end='2024-10-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # Test with just technicals
    engine = UnifiedFeatureEngineV2(feature_sets=['technicals'])
    result = engine.calculate_all(spy_data=spy)
    
    print(f"\n✅ Test complete: {len(result)} rows, {len(engine.get_feature_names())} features")
    print(f"Features: {engine.get_feature_names()[:10]}...")
