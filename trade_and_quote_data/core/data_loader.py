"""
Unified Data Loading Module

Centralized data loading with validation for all market data sources.
Consolidates logic from phase1, phase2, phase3 scripts.

CRITICAL RULES:
1. NEVER synthesize/fake data - Only use real data from approved sources
2. All data must pass DataIntegrityValidator checks
3. Maintain strict train/val/test temporal separation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.constants import (
    TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE,
    APPROVED_DATA_SOURCES, FORBIDDEN_PATTERNS,
    SPY_SYMBOL, SECTOR_ETFS, ROTATION_ETFS, CURRENCY_SYMBOLS, VOLATILITY_SYMBOLS,
    PRICE_VALIDATION, MIN_DATA_POINTS, MAX_MISSING_PCT
)


class DataIntegrityValidator:
    """Ensures all data is real and properly sourced"""
    
    @staticmethod
    def validate_data_source(data: pd.DataFrame, source_name: str, 
                           instrument_type: str = 'EQUITY') -> bool:
        """
        MANDATORY validation before any analysis
        
        Args:
            data: DataFrame with datetime index and OHLC columns
            source_name: Source of data (must be in APPROVED_DATA_SOURCES)
            instrument_type: Type for price validation (EQUITY, VIX, CURRENCY, etc.)
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        print(f"🔍 Validating data source: {source_name}")
        
        # Source validation
        if source_name not in APPROVED_DATA_SOURCES:
            raise ValueError(
                f"❌ FORBIDDEN: {source_name} not in approved sources {APPROVED_DATA_SOURCES}"
            )
        
        # Data quality checks
        if len(data) == 0:
            raise ValueError("❌ FORBIDDEN: Empty dataset")
        
        if len(data) < MIN_DATA_POINTS:
            raise ValueError(
                f"❌ FORBIDDEN: Insufficient data points ({len(data)} < {MIN_DATA_POINTS})"
            )
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("❌ FORBIDDEN: Data must have datetime index")
        
        if not data.index.is_monotonic_increasing:
            raise ValueError("❌ FORBIDDEN: Data must be chronologically ordered")
        
        # Price validation
        if 'Close' in data.columns:
            close_prices = data['Close'].dropna()
            close_min = float(close_prices.min())
            close_max = float(close_prices.max())
            
            # Get validation range for instrument type
            if instrument_type in PRICE_VALIDATION:
                val_range = PRICE_VALIDATION[instrument_type]
                if close_min < val_range['min'] or close_max > val_range['max']:
                    raise ValueError(
                        f"❌ FORBIDDEN: Unrealistic price range {close_min:.2f}-{close_max:.2f} "
                        f"for {instrument_type} (expected {val_range['min']}-{val_range['max']})"
                    )
            else:
                # Generic validation
                if close_min < 0.5 or close_max > 10000:
                    raise ValueError(
                        f"❌ FORBIDDEN: Unrealistic price range {close_min:.2f}-{close_max:.2f}"
                    )
            
            print(f"   📊 Price range: ${close_min:.2f} - ${close_max:.2f}")
        
        # Check for excessive missing data
        if 'Close' in data.columns:
            missing_pct = data['Close'].isna().sum() / len(data)
            if missing_pct > MAX_MISSING_PCT:
                raise ValueError(
                    f"❌ FORBIDDEN: Excessive missing data ({missing_pct:.1%} > {MAX_MISSING_PCT:.1%})"
                )
        
        print(
            f"✅ Data validation passed: {len(data)} records from "
            f"{data.index[0].date()} to {data.index[-1].date()}"
        )
        return True


class DataLoader:
    """Centralized data loading with validation"""
    
    def __init__(self, start_date: str = TRAIN_START_DATE, 
                 end_date: str = TEST_END_DATE):
        """
        Initialize DataLoader
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        """
        self.start_date = start_date
        self.end_date = end_date
        self.validator = DataIntegrityValidator()
    
    def load_spy_data(self) -> pd.DataFrame:
        """
        Load and validate SPY data from yfinance
        
        Returns:
            DataFrame with SPY OHLCV data
        """
        print(f"📥 Loading SPY data from {self.start_date} to {self.end_date}...")
        
        spy_data = yf.download(SPY_SYMBOL, start=self.start_date, 
                              end=self.end_date, progress=False)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)
        
        self.validator.validate_data_source(spy_data, 'yfinance', 'SPY')
        
        print(f"✅ SPY data loaded: {len(spy_data)} records")
        return spy_data
    
    def load_sector_data(self, sectors: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load sector ETF data from yfinance
        
        Args:
            sectors: List of sector symbols to load. If None, loads all SECTOR_ETFS
            
        Returns:
            Dictionary of sector symbol -> DataFrame
        """
        if sectors is None:
            sectors = list(SECTOR_ETFS.keys())
        
        print(f"📥 Loading {len(sectors)} sector ETFs...")
        sector_data = {}
        
        for sector in sectors:
            try:
                print(f"   Loading {sector} ({SECTOR_ETFS.get(sector, 'Unknown')})...")
                data = yf.download(sector, start=self.start_date, 
                                 end=self.end_date, progress=False)
                
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                self.validator.validate_data_source(data, 'yfinance', 'SECTOR_ETF')
                sector_data[sector] = data
                print(f"   ✅ {sector} loaded")
                
            except Exception as e:
                print(f"   ❌ Failed to load {sector}: {e}")
                print(f"   ⚠️  Continuing without {sector}")
        
        print(f"✅ Loaded {len(sector_data)}/{len(sectors)} sector ETFs")
        
        # Also load rotation indicators
        print(f"📥 Loading {len(ROTATION_ETFS)} rotation indicators...")
        for symbol, name in ROTATION_ETFS.items():
            try:
                print(f"   Loading {symbol} ({name})...")
                data = yf.download(symbol, start=self.start_date, 
                                 end=self.end_date, progress=False)
                
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if not data.empty:
                    self.validator.validate_data_source(data, 'yfinance', 'ROTATION_ETF')
                    sector_data[symbol] = data
                    print(f"   ✅ {symbol} loaded")
                else:
                    print(f"   ⚠️  {symbol}: No data available")
                
            except Exception as e:
                print(f"   ❌ Failed to load {symbol}: {e}")
                print(f"   ⚠️  Continuing without {symbol}")
        
        rotation_count = sum(1 for k in sector_data.keys() if k in ROTATION_ETFS)
        print(f"✅ Loaded {rotation_count}/{len(ROTATION_ETFS)} rotation indicators")
        
        return sector_data
    
    def load_currency_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load currency pair data from yfinance
        
        Returns:
            Dictionary of currency name -> DataFrame
        """
        print("💱 Loading currency data...")
        currency_data = {}
        
        for name, symbol in CURRENCY_SYMBOLS.items():
            try:
                print(f"   Loading {name} ({symbol})...")
                data = yf.download(symbol, start=self.start_date, 
                                 end=self.end_date, progress=False)
                
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                self.validator.validate_data_source(data, 'yfinance', 'CURRENCY')
                currency_data[name] = data
                print(f"   ✅ {name} loaded")
                
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
                if name == 'USDJPY':  # Critical for carry trade detection
                    print(f"   🚨 WARNING: USD/JPY is critical for July 2024 detection!")
                print(f"   ⚠️  Continuing without {name}")
        
        print(f"✅ Loaded {len(currency_data)}/{len(CURRENCY_SYMBOLS)} currency pairs")
        return currency_data
    
    def load_volatility_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load volatility index data from yfinance
        
        Returns:
            Dictionary of volatility index name -> DataFrame
        """
        print("📊 Loading volatility data...")
        volatility_data = {}
        
        for name, symbol in VOLATILITY_SYMBOLS.items():
            try:
                print(f"   Loading {name} ({symbol})...")
                data = yf.download(symbol, start=self.start_date, 
                                 end=self.end_date, progress=False)
                
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Use appropriate validation
                if name in ['VIX', 'VIX9D']:
                    self.validator.validate_data_source(data, 'yfinance', 'VIX')
                elif name == 'VVIX':
                    self.validator.validate_data_source(data, 'yfinance', 'VVIX')
                else:
                    self.validator.validate_data_source(data, 'yfinance', 'VIX')
                
                volatility_data[name] = data
                print(f"   ✅ {name} loaded")
                
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
                if name == 'VIX':  # VIX is critical
                    print(f"   🚨 CRITICAL: VIX required for volatility analysis!")
                    raise
                print(f"   ⚠️  Continuing without {name}")
        
        print(f"✅ Loaded {len(volatility_data)}/{len(VOLATILITY_SYMBOLS)} volatility indices")
        return volatility_data
    
    def load_all_data(self, include_sectors: bool = True,
                     include_currency: bool = True,
                     include_volatility: bool = True) -> Dict[str, any]:
        """
        Load all market data with validation
        
        Args:
            include_sectors: Whether to load sector ETFs
            include_currency: Whether to load currency data
            include_volatility: Whether to load volatility data
            
        Returns:
            Dictionary with keys: 'spy', 'sectors', 'currency', 'volatility'
        """
        print("=" * 80)
        print("📥 LOADING ALL MARKET DATA")
        print("=" * 80)
        
        data = {
            'spy': self.load_spy_data()
        }
        
        if include_sectors:
            data['sectors'] = self.load_sector_data()
        
        if include_currency:
            data['currency'] = self.load_currency_data()
        
        if include_volatility:
            data['volatility'] = self.load_volatility_data()
        
        print("=" * 80)
        print("✅ ALL DATA LOADED SUCCESSFULLY")
        print("=" * 80)
        
        return data
    
    def train_test_split(self, data: pd.DataFrame,
                        train_end: str = TRAIN_END_DATE,
                        val_end: str = VAL_END_DATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets with strict temporal separation
        
        Args:
            data: DataFrame with datetime index
            train_end: End date for training set
            val_end: End date for validation set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        print(f"📊 Splitting data: Train ≤ {train_end}, Val ≤ {val_end}, Test > {val_end}")
        
        train_data = data[data.index <= train_end].copy()
        val_data = data[(data.index > train_end) & (data.index <= val_end)].copy()
        test_data = data[data.index > val_end].copy()
        
        print(f"   Train: {len(train_data)} records ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"   Val:   {len(val_data)} records ({val_data.index[0].date()} to {val_data.index[-1].date()})")
        print(f"   Test:  {len(test_data)} records ({test_data.index[0].date()} to {test_data.index[-1].date()})")
        
        return train_data, val_data, test_data


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    data = loader.load_all_data()
    
    print("\n" + "=" * 80)
    print("DATA LOADING TEST COMPLETE")
    print("=" * 80)
    print(f"SPY records: {len(data['spy'])}")
    print(f"Sectors loaded: {len(data.get('sectors', {}))}")
    print(f"Currencies loaded: {len(data.get('currency', {}))}")
    print(f"Volatility indices loaded: {len(data.get('volatility', {}))}")

