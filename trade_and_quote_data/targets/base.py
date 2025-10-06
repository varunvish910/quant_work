"""
Base Classes for Target Creation

This module defines the abstract base classes for all target variables.
Targets represent what we're trying to predict (e.g., drawdowns, bounces, volatility spikes).

Design Principles:
- Each target is self-contained and testable
- Targets validate input data requirements
- Targets document their prediction horizon
- Targets handle edge cases (end of data, missing values)

Author: Refactored Architecture
Date: 2025-10-05
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class BaseTarget(ABC):
    """
    Abstract base class for all target variables.
    
    All target implementations must inherit from this class and implement
    the create() method.
    
    Attributes:
        name: Human-readable name of the target
        params: Dictionary of parameters for target creation
        target_column: Name of the target column created
        prediction_horizon: Number of days ahead this target predicts
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base target.
        
        Args:
            name: Name of the target (e.g., "early_warning", "mean_reversion")
            params: Optional parameters for target creation
        """
        self.name = name
        self.params = params or {}
        self.target_column = f"{name}_target"
        self.prediction_horizon = 0  # Set by subclasses
        self.required_columns = ['Close']  # Minimum required
        self._target_stats = {}
    
    @abstractmethod
    def create(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable and add to DataFrame.
        
        This method must be implemented by all subclasses.
        
        Args:
            data: Input DataFrame with OHLC data
            
        Returns:
            DataFrame with target column added
            
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
                f"{self.name} target requires columns {missing_columns} which are missing. "
                f"Available columns: {list(data.columns)}"
            )
        
        if len(data) < self.prediction_horizon + 50:
            raise ValueError(
                f"{self.name} target requires at least {self.prediction_horizon + 50} rows of data. "
                f"Provided: {len(data)} rows"
            )
        
        return True
    
    def get_target_column(self) -> str:
        """Get the name of the target column."""
        return self.target_column
    
    def get_prediction_horizon(self) -> int:
        """Get the prediction horizon in days."""
        return self.prediction_horizon
    
    def get_target_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the target variable.
        
        Returns:
            Dictionary with target statistics
        """
        return self._target_stats.copy()
    
    def _calculate_stats(self, target_series: pd.Series) -> Dict[str, Any]:
        """
        Calculate statistics for the target variable.
        
        Args:
            target_series: The target column
            
        Returns:
            Dictionary of statistics
        """
        total_samples = len(target_series)
        positive_samples = target_series.sum()
        positive_rate = positive_samples / total_samples if total_samples > 0 else 0
        
        stats = {
            'total_samples': int(total_samples),
            'positive_samples': int(positive_samples),
            'negative_samples': int(total_samples - positive_samples),
            'positive_rate': float(positive_rate),
            'class_balance': float(min(positive_rate, 1 - positive_rate) / max(positive_rate, 1 - positive_rate))
        }
        
        return stats
    
    def _print_target_summary(self, target_series: pd.Series):
        """
        Print a summary of the target variable.
        
        Args:
            target_series: The target column
        """
        stats = self._calculate_stats(target_series)
        self._target_stats = stats
        
        print(f"🎯 {self.name.upper()} TARGET CREATED")
        print(f"   Total samples: {stats['total_samples']:,}")
        print(f"   Positive samples: {stats['positive_samples']:,} ({stats['positive_rate']:.1%})")
        print(f"   Negative samples: {stats['negative_samples']:,} ({1-stats['positive_rate']:.1%})")
        print(f"   Class balance: {stats['class_balance']:.2f} (1.0 = perfect balance)")
        
        if stats['positive_rate'] < 0.05:
            print(f"   ⚠️  WARNING: Very imbalanced target ({stats['positive_rate']:.1%} positive)")
        elif stats['positive_rate'] > 0.95:
            print(f"   ⚠️  WARNING: Very imbalanced target ({stats['positive_rate']:.1%} positive)")
    
    def get_params(self) -> Dict[str, Any]:
        """Get target parameters."""
        return self.params.copy()
    
    def __repr__(self) -> str:
        """String representation of target."""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        horizon_str = f"{self.prediction_horizon}d horizon" if self.prediction_horizon > 0 else "variable horizon"
        return f"{self.name} ({horizon_str})"


class ForwardLookingTarget(BaseTarget):
    """
    Base class for targets that look forward in time.
    
    These targets predict events that will happen in the future.
    They require careful handling of the prediction horizon to avoid lookahead bias.
    """
    
    def __init__(self, name: str, min_lead_days: int, max_lead_days: int, 
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize forward-looking target.
        
        Args:
            name: Name of the target
            min_lead_days: Minimum days ahead to predict
            max_lead_days: Maximum days ahead to predict
            params: Optional parameters
        """
        super().__init__(name, params)
        self.min_lead_days = min_lead_days
        self.max_lead_days = max_lead_days
        self.prediction_horizon = max_lead_days
        
        if min_lead_days < 1:
            raise ValueError("min_lead_days must be at least 1 to avoid lookahead bias")
        
        if max_lead_days < min_lead_days:
            raise ValueError("max_lead_days must be >= min_lead_days")
    
    def _truncate_for_prediction_horizon(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows at the end that we can't predict for.
        
        Args:
            data: DataFrame with target column
            
        Returns:
            Truncated DataFrame
        """
        if len(data) > self.max_lead_days:
            return data.iloc[:-self.max_lead_days].copy()
        else:
            return data.copy()


class BackwardLookingTarget(BaseTarget):
    """
    Base class for targets that look backward in time.
    
    These targets identify patterns that have already occurred.
    Useful for mean reversion and pattern recognition strategies.
    """
    
    def __init__(self, name: str, lookback_days: int, 
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize backward-looking target.
        
        Args:
            name: Name of the target
            lookback_days: Number of days to look back
            params: Optional parameters
        """
        super().__init__(name, params)
        self.lookback_days = lookback_days
        self.prediction_horizon = 0  # No forward prediction
        
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")


def calculate_future_drawdown(prices: pd.Series, highs: pd.Series, lows: pd.Series,
                              lead_days: int, window: int = 10) -> pd.Series:
    """
    Calculate maximum drawdown in a future window.
    
    Args:
        prices: Current close prices
        highs: High prices
        lows: Low prices
        lead_days: Days ahead to start looking
        window: Size of window to check for drawdown
        
    Returns:
        Series of future drawdowns (negative values)
    """
    future_low = lows.shift(-lead_days).rolling(window, min_periods=1).min()
    drawdown = (future_low - prices) / prices
    return drawdown


def calculate_future_return(prices: pd.Series, highs: pd.Series,
                           lead_days: int, window: int = 10) -> pd.Series:
    """
    Calculate maximum return in a future window.
    
    Args:
        prices: Current close prices
        highs: High prices
        lead_days: Days ahead to start looking
        window: Size of window to check for return
        
    Returns:
        Series of future returns (positive values)
    """
    future_high = highs.shift(-lead_days).rolling(window, min_periods=1).max()
    return_pct = (future_high - prices) / prices
    return return_pct


if __name__ == "__main__":
    # Test base target classes
    print("Testing base target classes...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.randn(500).cumsum() + 100,
        'High': np.random.randn(500).cumsum() + 102,
        'Low': np.random.randn(500).cumsum() + 98,
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    print("✅ Testing future drawdown calculation...")
    drawdowns = calculate_future_drawdown(
        sample_data['Close'], 
        sample_data['High'], 
        sample_data['Low'],
        lead_days=3,
        window=10
    )
    print(f"   Drawdowns: min={drawdowns.min():.2%}, mean={drawdowns.mean():.2%}")
    
    print("✅ Testing future return calculation...")
    returns = calculate_future_return(
        sample_data['Close'],
        sample_data['High'],
        lead_days=3,
        window=10
    )
    print(f"   Returns: max={returns.max():.2%}, mean={returns.mean():.2%}")
    
    print("\n✅ Base target infrastructure tests passed!")
