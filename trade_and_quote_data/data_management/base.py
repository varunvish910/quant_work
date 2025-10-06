"""
Base Classes for Data Downloaders

This module defines abstract base classes for all data downloaders.
Each downloader handles a specific type of market data.

Design Principles:
- Each downloader is responsible for one data type
- Downloaders validate data integrity
- Downloaders handle caching and updates
- Downloaders use only approved data sources

Author: Refactored Architecture
Date: 2025-10-05
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BaseDataDownloader(ABC):
    """
    Abstract base class for all data downloaders.
    
    All downloader implementations must inherit from this class.
    
    Attributes:
        name: Human-readable name of the downloader
        data_source: Source of data (must be approved)
        cache_dir: Directory for caching downloaded data
    """
    
    def __init__(self, name: str, data_source: str, cache_dir: Optional[Path] = None):
        """
        Initialize base downloader.
        
        Args:
            name: Name of the downloader (e.g., "Equity Downloader")
            data_source: Data source (e.g., "yfinance", "polygon")
            cache_dir: Optional cache directory
        """
        self.name = name
        self.data_source = data_source
        self.cache_dir = cache_dir or Path("data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate data source
        self._validate_data_source()
    
    def _validate_data_source(self):
        """Validate that data source is approved"""
        approved_sources = ['yfinance', 'polygon', 'local']
        if self.data_source not in approved_sources:
            raise ValueError(
                f"Data source '{self.data_source}' not approved. "
                f"Approved sources: {approved_sources}"
            )
    
    @abstractmethod
    def download(self, 
                 symbols: List[str],
                 start_date: str,
                 end_date: str,
                 **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Download data for given symbols and date range.
        
        This method must be implemented by all subclasses.
        
        Args:
            symbols: List of symbols to download
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate downloaded data.
        
        Args:
            data: Downloaded data
            symbol: Symbol being validated
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        pass
    
    def save_to_cache(self, data: pd.DataFrame, symbol: str, data_type: str):
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            symbol: Symbol name
            data_type: Type of data (e.g., 'ohlc', 'options')
        """
        cache_path = self.cache_dir / data_type / f"{symbol}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path)
        print(f"   💾 Cached: {cache_path}")
    
    def load_from_cache(self, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.
        
        Args:
            symbol: Symbol name
            data_type: Type of data
            
        Returns:
            Cached data or None if not found
        """
        cache_path = self.cache_dir / data_type / f"{symbol}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name='{self.name}', source='{self.data_source}')"
    
    def __str__(self) -> str:
        """Human-readable string"""
        return f"{self.name} ({self.data_source})"


class YFinanceDownloader(BaseDataDownloader):
    """Base class for downloaders using yfinance"""
    
    def __init__(self, name: str, cache_dir: Optional[Path] = None):
        super().__init__(name, "yfinance", cache_dir)
    
    def _download_single(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download single symbol from yfinance"""
        import yfinance as yf
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data


if __name__ == "__main__":
    print("Testing BaseDataDownloader...")
    print("✅ Base classes created")
    print("   Subclasses will implement specific download logic")
