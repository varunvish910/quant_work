"""
Base Classes for Feature Engineering

This module defines the abstract base classes that all feature implementations
must inherit from. This ensures a consistent interface across all features.

Design Principles:
- Each feature is self-contained and testable
- Features declare their dependencies (required columns)
- Features validate input data before calculation
- Features return clean, documented output

Author: Refactored Architecture
Date: 2025-10-05
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class BaseFeature(ABC):
    """
    Abstract base class for all features.
    
    All feature implementations must inherit from this class and implement
    the calculate() method.
    
    Attributes:
        name: Human-readable name of the feature
        params: Dictionary of parameters for the feature
        feature_names: List of column names this feature creates
        required_columns: List of columns required in input data
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base feature.
        
        Args:
            name: Name of the feature (e.g., "Momentum", "RSI")
            params: Optional parameters for feature calculation
        """
        self.name = name
        self.params = params or {}
        self.feature_names: List[str] = []
        self.required_columns: List[str] = []
        self._is_calculated = False
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate the feature and add columns to DataFrame.
        
        This method must be implemented by all subclasses.
        
        Args:
            data: Input DataFrame with required columns
            **kwargs: Additional data sources (e.g., sector_data, currency_data)
            
        Returns:
            DataFrame with new feature columns added
            
        Raises:
            ValueError: If required columns are missing
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data has required columns.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(
                f"{self.name} requires columns {missing_columns} which are missing from data. "
                f"Available columns: {list(data.columns)}"
            )
        
        return True
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names created by this feature.
        
        Returns:
            List of column names
        """
        return self.feature_names.copy()
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get feature parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.params.copy()
    
    def __repr__(self) -> str:
        """String representation of feature."""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} ({len(self.feature_names)} features)"


class FeatureGroup(BaseFeature):
    """
    A group of related features that are calculated together.
    
    This is useful for features that share intermediate calculations
    or need to be calculated in a specific order.
    """
    
    def __init__(self, name: str, features: Optional[List[BaseFeature]] = None):
        """
        Initialize feature group.
        
        Args:
            name: Name of the feature group
            features: List of features in this group
        """
        super().__init__(name)
        self.features = features or []
    
    def add_feature(self, feature: BaseFeature):
        """Add a feature to this group."""
        self.features.append(feature)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate all features in the group.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional data sources
            
        Returns:
            DataFrame with all group features added
        """
        df = data.copy()
        
        for feature in self.features:
            try:
                df = feature.calculate(df, **kwargs)
                self.feature_names.extend(feature.get_feature_names())
            except Exception as e:
                print(f"⚠️  Warning: Failed to calculate {feature.name}: {e}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names from all features in group."""
        all_names = []
        for feature in self.features:
            all_names.extend(feature.get_feature_names())
        return all_names


def validate_datetime_index(data: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has a datetime index.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If index is not datetime
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "Data must have a DatetimeIndex. "
            f"Current index type: {type(data.index)}"
        )
    
    return True


def validate_sorted_index(data: pd.DataFrame) -> bool:
    """
    Validate that DataFrame index is sorted chronologically.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If index is not sorted
    """
    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Data index must be sorted chronologically. "
            "Use data.sort_index() to fix."
        )
    
    return True


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result series with safe division
    """
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result


def calculate_returns(prices: pd.Series, periods: int = 1, 
                     method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        periods: Number of periods for return calculation
        method: 'simple' or 'log' returns
        
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change(periods)
    elif method == 'log':
        return np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unknown return method: {method}")


if __name__ == "__main__":
    # Test base classes
    print("Testing base feature classes...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Test validation functions
    print("✅ Testing datetime index validation...")
    validate_datetime_index(sample_data)
    
    print("✅ Testing sorted index validation...")
    validate_sorted_index(sample_data)
    
    print("✅ Testing safe divide...")
    result = safe_divide(sample_data['Close'], sample_data['Volume'])
    print(f"   Safe divide result: {len(result)} values, {result.isna().sum()} NaN")
    
    print("✅ Testing return calculation...")
    returns = calculate_returns(sample_data['Close'], periods=1)
    print(f"   Returns: mean={returns.mean():.4f}, std={returns.std():.4f}")
    
    print("\n✅ Base feature infrastructure tests passed!")
