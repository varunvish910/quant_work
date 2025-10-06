"""
Unified Feature Engine

Master engine that orchestrates all feature calculation.
Currently acts as a bridge to the old system while migration completes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Import old system for now (compatibility layer)
sys.path.append(str(Path(__file__).parent.parent))
from core.features import FeatureEngine as OldFeatureEngine

from engines.base import CompositeEngine


class UnifiedFeatureEngine(CompositeEngine):
    """
    Unified engine that orchestrates all feature calculation.
    
    Currently wraps the old FeatureEngine for compatibility.
    As new features are migrated, they'll be added here.
    """
    
    def __init__(self, feature_sets: List[str] = None):
        super().__init__("Unified Feature Engine")
        
        if feature_sets is None:
            feature_sets = ['baseline']
        
        self.feature_sets = feature_sets
        
        # Use old engine for now (compatibility)
        self._old_engine = OldFeatureEngine(feature_sets=feature_sets)
        
        print(f"🔧 Initialized UnifiedFeatureEngine with: {', '.join(feature_sets)}")
    
    def calculate_all(self, 
                     spy_data: pd.DataFrame,
                     sector_data: Optional[Dict[str, pd.DataFrame]] = None,
                     currency_data: Optional[Dict[str, pd.DataFrame]] = None,
                     volatility_data: Optional[Dict[str, pd.DataFrame]] = None,
                     options_data: Optional[pd.DataFrame] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Calculate all features using compatibility layer.
        
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
        print("🚀 UNIFIED FEATURE ENGINE")
        print("=" * 80)
        print("⚠️  Using compatibility layer (old engine)")
        print("   New architecture features will be added incrementally")
        print("=" * 80)
        
        # Use old engine
        df = self._old_engine.calculate_features(
            spy_data=spy_data,
            sector_data=sector_data,
            currency_data=currency_data,
            volatility_data=volatility_data,
            options_data=options_data
        )
        
        self.feature_names = self._old_engine.get_feature_columns()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names"""
        return self._old_engine.get_feature_columns()


if __name__ == "__main__":
    print("Testing UnifiedFeatureEngine...")
    
    # Test with sample data
    import yfinance as yf
    
    spy = yf.download('SPY', start='2024-01-01', end='2024-10-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    engine = UnifiedFeatureEngine(feature_sets=['baseline'])
    result = engine.calculate_all(spy_data=spy)
    
    print(f"\n✅ Test complete: {len(result)} rows, {len(engine.get_feature_names())} features")
    print(f"Features: {engine.get_feature_names()[:5]}...")
