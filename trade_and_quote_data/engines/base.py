"""
Base Classes for Feature Engines

This module defines the abstract base classes for feature engines.
Engines orchestrate the calculation of multiple related features.

Design Principles:
- Engines manage groups of related features
- Engines handle data validation and error handling
- Engines provide clean interfaces for feature calculation
- Engines support flexible feature composition

Author: Refactored Architecture
Date: 2025-10-05
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from features.base import BaseFeature


class BaseFeatureEngine(ABC):
    """
    Abstract base class for all feature engines.
    
    An engine orchestrates the calculation of multiple related features.
    For example, TechnicalFeatureEngine manages all technical indicators.
    
    Attributes:
        name: Human-readable name of the engine
        features: List of features managed by this engine
        feature_names: List of all feature column names
    """
    
    def __init__(self, name: str, features: Optional[List[BaseFeature]] = None):
        """
        Initialize base feature engine.
        
        Args:
            name: Name of the engine (e.g., "Technical Features")
            features: Optional list of features to manage
        """
        self.name = name
        self.features: List[BaseFeature] = features or []
        self.feature_names: List[str] = []
        self._is_initialized = False
    
    def add_feature(self, feature: BaseFeature):
        """
        Add a feature to this engine.
        
        Args:
            feature: Feature instance to add
        """
        self.features.append(feature)
    
    def add_features(self, features: List[BaseFeature]):
        """
        Add multiple features to this engine.
        
        Args:
            features: List of feature instances to add
        """
        self.features.extend(features)
    
    @abstractmethod
    def _initialize_features(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize default features for this engine.
        
        This method should be implemented by subclasses to define
        which features are included by default.
        
        Args:
            config: Optional configuration dictionary
        """
        pass
    
    def calculate_all(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate all features managed by this engine.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional data sources (e.g., sector_data, currency_data)
            
        Returns:
            DataFrame with all feature columns added
        """
        print(f"🔧 {self.name}...")
        
        if not self._is_initialized:
            self._initialize_features()
            self._is_initialized = True
        
        df = data.copy()
        self.feature_names = []
        successful_features = 0
        failed_features = 0
        
        for feature in self.features:
            try:
                df = feature.calculate(df, **kwargs)
                self.feature_names.extend(feature.get_feature_names())
                successful_features += 1
            except Exception as e:
                print(f"   ⚠️  Failed to calculate {feature.name}: {e}")
                failed_features += 1
        
        print(f"   ✅ Calculated {successful_features} features ({len(self.feature_names)} columns)")
        if failed_features > 0:
            print(f"   ⚠️  {failed_features} features failed")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature column names.
        
        Returns:
            List of feature column names
        """
        return self.feature_names.copy()
    
    def get_features(self) -> List[BaseFeature]:
        """
        Get list of all feature instances.
        
        Returns:
            List of feature instances
        """
        return self.features.copy()
    
    def get_feature_count(self) -> int:
        """Get number of features managed by this engine."""
        return len(self.features)
    
    def get_feature_column_count(self) -> int:
        """Get number of feature columns created."""
        return len(self.feature_names)
    
    def __repr__(self) -> str:
        """String representation of engine."""
        return f"{self.__class__.__name__}(name='{self.name}', features={len(self.features)})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} ({len(self.features)} features, {len(self.feature_names)} columns)"


class CompositeEngine(BaseFeatureEngine):
    """
    An engine that composes multiple sub-engines.
    
    This is useful for creating hierarchical engine structures.
    For example, UnifiedFeatureEngine might compose TechnicalEngine,
    MarketEngine, CurrencyEngine, etc.
    """
    
    def __init__(self, name: str, engines: Optional[List[BaseFeatureEngine]] = None):
        """
        Initialize composite engine.
        
        Args:
            name: Name of the composite engine
            engines: Optional list of sub-engines
        """
        super().__init__(name)
        self.engines: List[BaseFeatureEngine] = engines or []
    
    def add_engine(self, engine: BaseFeatureEngine):
        """Add a sub-engine to this composite engine."""
        self.engines.append(engine)
    
    def add_engines(self, engines: List[BaseFeatureEngine]):
        """Add multiple sub-engines to this composite engine."""
        self.engines.extend(engines)
    
    def _initialize_features(self, config: Optional[Dict[str, Any]] = None):
        """Initialize features (handled by sub-engines)."""
        pass
    
    def calculate_all(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate features from all sub-engines.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional data sources
            
        Returns:
            DataFrame with all features from all engines
        """
        print(f"🚀 {self.name}...")
        print("=" * 80)
        
        df = data.copy()
        self.feature_names = []
        
        for engine in self.engines:
            try:
                df = engine.calculate_all(df, **kwargs)
                self.feature_names.extend(engine.get_feature_names())
            except Exception as e:
                print(f"   ❌ Engine {engine.name} failed: {e}")
        
        print("=" * 80)
        print(f"✅ {self.name} Complete: {len(self.feature_names)} total features")
        print("=" * 80)
        
        return df
    
    def get_engines(self) -> List[BaseFeatureEngine]:
        """Get list of all sub-engines."""
        return self.engines.copy()
    
    def get_engine_count(self) -> int:
        """Get number of sub-engines."""
        return len(self.engines)


if __name__ == "__main__":
    # Test base engine classes
    print("Testing base engine classes...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Create a simple test feature
    class TestFeature(BaseFeature):
        def __init__(self):
            super().__init__("Test Feature")
            self.required_columns = ['Close']
        
        def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
            df = data.copy()
            df['test_feature'] = df['Close'].pct_change()
            self.feature_names = ['test_feature']
            return df
    
    # Create a simple test engine
    class TestEngine(BaseFeatureEngine):
        def _initialize_features(self, config: Optional[Dict[str, Any]] = None):
            self.add_feature(TestFeature())
    
    print("✅ Testing feature engine...")
    engine = TestEngine("Test Engine")
    result = engine.calculate_all(sample_data)
    print(f"   Features created: {engine.get_feature_names()}")
    print(f"   Result shape: {result.shape}")
    
    print("\n✅ Base engine infrastructure tests passed!")
