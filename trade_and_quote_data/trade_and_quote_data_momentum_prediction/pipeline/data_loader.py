#!/usr/bin/env python3
"""
Data Loader for Momentum Prediction System

Unified data loading and preprocessing pipeline for any ticker.
Supports multiple data sources and formats with automatic ticker detection.

USAGE:
======
from pipeline.data_loader import DataLoader

# Basic usage
loader = DataLoader(ticker='SPY')
df = loader.load_historical(start_date='2020-01-01', end_date='2024-01-01')

# Auto-update with latest data
df_updated = loader.update_latest(df)

# Multiple data sources
loader = DataLoader(ticker='AAPL', data_source='yfinance')
df = loader.load_and_prepare(start_date='2023-01-01')
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Data source imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not available. Install with: pip install yfinance")

try:
    import pandas_datareader as pdr
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False
    print("⚠️  pandas_datareader not available. Install with: pip install pandas-datareader")

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for momentum prediction system.
    
    Supports multiple data sources:
    - yfinance (Yahoo Finance)
    - pandas_datareader (various sources)
    - CSV files
    - Parquet files
    """
    
    def __init__(self, ticker: str, data_source: str = 'yfinance'):
        """
        Initialize data loader.
        
        Args:
            ticker: Stock/ETF ticker symbol
            data_source: Data source ('yfinance', 'fred', 'csv', 'parquet')
        """
        self.ticker = ticker.upper()
        self.data_source = data_source
        self.data_cache = {}
        
        # Validate data source availability
        self._validate_data_source()
        
        logger.info(f"DataLoader initialized: ticker={self.ticker}, source={self.data_source}")
    
    def _validate_data_source(self):
        """Validate that requested data source is available."""
        if self.data_source == 'yfinance' and not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not available. Install with: pip install yfinance")
        
        if self.data_source in ['fred', 'stooq'] and not DATAREADER_AVAILABLE:
            raise ImportError("pandas_datareader not available. Install with: pip install pandas-datareader")
    
    def load_historical(self, start_date: str, end_date: str = None, 
                       interval: str = '1d') -> pd.DataFrame:
        """
        Load historical price data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD, defaults to today)
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n📥 Loading historical data for {self.ticker}...")
        print(f"   • Period: {start_date} to {end_date}")
        print(f"   • Source: {self.data_source}")
        
        if self.data_source == 'yfinance':
            df = self._load_yfinance_data(start_date, end_date, interval)
        elif self.data_source == 'fred':
            df = self._load_fred_data(start_date, end_date)
        elif self.data_source == 'csv':
            df = self._load_csv_data(start_date, end_date)
        elif self.data_source == 'parquet':
            df = self._load_parquet_data(start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
        
        # Standardize DataFrame
        df = self._standardize_dataframe(df)
        
        # Cache data
        cache_key = f"{self.ticker}_{start_date}_{end_date}_{interval}"
        self.data_cache[cache_key] = df.copy()
        
        print(f"   ✅ Loaded {len(df):,} records")
        return df
    
    def _load_yfinance_data(self, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Load data using yfinance."""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            df = ticker_obj.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load data from yfinance: {e}")
    
    def _load_fred_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data using FRED (pandas_datareader)."""
        try:
            df = pdr.get_data_fred(self.ticker, start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {self.ticker}")
            
            # FRED data is usually single column, create OHLC from it
            df = df.reset_index()
            price_col = df.columns[-1]  # Last column is the price
            
            df['Open'] = df[price_col]
            df['High'] = df[price_col]
            df['Low'] = df[price_col]
            df['Close'] = df[price_col]
            df['Volume'] = 0  # No volume data for FRED
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load data from FRED: {e}")
    
    def _load_csv_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from CSV file."""
        csv_path = f"data/raw/{self.ticker}.csv"
        
        try:
            df = pd.read_csv(csv_path)
            
            # Filter by date range if date column exists
            if 'Date' in df.columns or 'date' in df.columns:
                date_col = 'Date' if 'Date' in df.columns else 'date'
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV data: {e}")
    
    def _load_parquet_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from Parquet file."""
        parquet_path = f"data/raw/{self.ticker}.parquet"
        
        try:
            df = pd.read_parquet(parquet_path)
            
            # Filter by date range if date column exists
            date_cols = ['Date', 'date', 'timestamp', 'sip_timestamp']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load Parquet data: {e}")
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame column names and structure."""
        
        # Standardize date column
        date_cols = ['Date', 'date', 'timestamp', 'Datetime', 'datetime']
        for col in date_cols:
            if col in df.columns:
                df['date'] = pd.to_datetime(df[col])
                if col != 'date':
                    df = df.drop(columns=[col])
                break
        
        # Standardize OHLCV columns
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        if 'close' not in df.columns:
            # Try to find a price column
            price_candidates = ['price', 'Price', 'value', 'Value']
            for candidate in price_candidates:
                if candidate in df.columns:
                    df['close'] = df[candidate]
                    break
        
        if 'close' not in df.columns:
            raise ValueError("No price column found in data")
        
        # Add missing OHLC columns if only close is available
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Add adjusted close if missing
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # Add ticker column
        df['ticker'] = self.ticker
        
        return df
    
    def update_latest(self, df: pd.DataFrame, days_back: int = 7) -> pd.DataFrame:
        """
        Update DataFrame with latest data.
        
        Args:
            df: Existing DataFrame
            days_back: How many days back to fetch for overlap
            
        Returns:
            Updated DataFrame with latest data
        """
        if df.empty or 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column")
        
        # Get latest date in current data
        latest_date = df['date'].max()
        
        # Fetch new data starting a few days before latest date for overlap
        start_date = (latest_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n🔄 Updating {self.ticker} with latest data...")
        print(f"   • Update period: {start_date} to {end_date}")
        
        # Load new data
        new_df = self.load_historical(start_date, end_date)
        
        # Combine with existing data, removing duplicates
        if not new_df.empty:
            combined_df = pd.concat([df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            
            new_records = len(combined_df) - len(df)
            print(f"   ✅ Added {new_records} new records")
            
            return combined_df
        else:
            print(f"   ⚠️  No new data available")
            return df
    
    def load_and_prepare(self, start_date: str, end_date: str = None,
                        add_returns: bool = True,
                        add_basic_features: bool = True) -> pd.DataFrame:
        """
        Load data and add basic preprocessing.
        
        Args:
            start_date: Start date
            end_date: End date (optional)
            add_returns: Whether to calculate returns
            add_basic_features: Whether to add basic technical features
            
        Returns:
            Prepared DataFrame ready for feature engineering
        """
        # Load historical data
        df = self.load_historical(start_date, end_date)
        
        # Add returns
        if add_returns:
            df = self._add_returns(df)
        
        # Add basic features
        if add_basic_features:
            df = self._add_basic_features(df)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return calculations."""
        df['daily_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Multi-period returns
        for period in [5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical features."""
        # Price gaps
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_up'] = (df['gap'] > 0).astype(int)
        df['gap_down'] = (df['gap'] < 0).astype(int)
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Basic moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = ['date', 'close']
        for col in required_cols:
            if col not in df.columns:
                results['errors'].append(f"Missing required column: {col}")
                results['valid'] = False
        
        if not results['valid']:
            return results
        
        # Data quality checks
        total_rows = len(df)
        
        # Missing values
        for col in ['close', 'open', 'high', 'low']:
            if col in df.columns:
                missing = df[col].isnull().sum()
                missing_pct = missing / total_rows * 100
                
                if missing_pct > 5:
                    results['warnings'].append(f"{col} has {missing_pct:.1f}% missing values")
                elif missing_pct > 20:
                    results['errors'].append(f"{col} has too many missing values: {missing_pct:.1f}%")
                    results['valid'] = False
        
        # Price validation
        if 'close' in df.columns:
            closes = df['close'].dropna()
            if len(closes) > 0:
                if (closes <= 0).any():
                    results['errors'].append("Found non-positive prices")
                    results['valid'] = False
                
                # Check for extreme price movements
                returns = closes.pct_change().dropna()
                if len(returns) > 0:
                    extreme_moves = (abs(returns) > 0.5).sum()
                    if extreme_moves > len(returns) * 0.01:  # More than 1%
                        results['warnings'].append(f"Found {extreme_moves} extreme price movements (>50%)")
        
        # Date continuity
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date']).sort_values()
            if len(dates) > 1:
                # Check for large gaps
                date_diffs = dates.diff().dt.days.dropna()
                large_gaps = (date_diffs > 10).sum()  # More than 10 days
                if large_gaps > 0:
                    results['warnings'].append(f"Found {large_gaps} large date gaps (>10 days)")
        
        # Data statistics
        results['stats'] = {
            'total_rows': total_rows,
            'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A',
            'price_range': f"${df['close'].min():.2f} - ${df['close'].max():.2f}" if 'close' in df.columns else 'N/A',
            'columns': list(df.columns)
        }
        
        return results
    
    def get_supported_tickers(self) -> List[str]:
        """Get list of commonly supported tickers."""
        return [
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO',
            
            # Major Stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA',
            
            # Sector ETFs
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE'
        ]
    
    def __str__(self) -> str:
        return f"DataLoader(ticker={self.ticker}, source={self.data_source})"
    
    def __repr__(self) -> str:
        return f"DataLoader(ticker='{self.ticker}', data_source='{self.data_source}')"