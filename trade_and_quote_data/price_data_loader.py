#!/usr/bin/env python3
"""
Price Data Loader - Loads historical price data from various sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

class PriceDataLoader:
    """Loads historical price data for target creation"""
    
    def load_price_data(self, start_date: str, end_date: str, ticker: str = "SPY") -> pd.DataFrame:
        """Load historical price data from options data or fallback to yfinance"""
        try:
            # Try loading from options summary files first
            return self._load_from_options_data(start_date, end_date)
        except Exception as e:
            print(f"❌ Error loading from options data: {e}")
            # Fallback to yfinance
            return self._load_from_yfinance(start_date, end_date, ticker)
    
    def _load_from_options_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load from options summary files which contain underlying_price"""
        data_files = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Find all summary files in the date range
        for year in range(start_dt.year, end_dt.year + 1):
            year_dir = Path(f"data")
            for file_path in year_dir.glob(f"SPY_summary_{year}*.csv"):
                data_files.append(file_path)
        
        if not data_files:
            raise FileNotFoundError("No SPY summary files found in data directory")
        
        # Load and combine all data
        price_data = []
        for file_path in sorted(data_files):
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date range
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                
                if len(df) > 0:
                    price_data.append(df[['date', 'underlying_price']])
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not price_data:
            raise ValueError("No price data found in the specified date range")
        
        # Combine all data
        combined_df = pd.concat(price_data, ignore_index=True)
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        combined_df = combined_df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        print(f"✅ Loaded {len(combined_df)} days of price data from {combined_df['date'].min()} to {combined_df['date'].max()}")
        return combined_df
    
    def _load_from_yfinance(self, start_date: str, end_date: str, ticker: str) -> pd.DataFrame:
        """Fallback to yfinance for price data"""
        try:
            import yfinance as yf
            print("🔄 Falling back to yfinance...")
            spy = yf.Ticker(ticker)
            hist = spy.history(start=start_date, end=end_date)
            if hist.empty:
                raise ValueError("No data from yfinance")
            
            price_data = pd.DataFrame({
                'date': hist.index,
                'underlying_price': hist['Close']
            }).reset_index(drop=True)
            
            print(f"✅ Loaded {len(price_data)} days from yfinance")
            return price_data
        except Exception as e:
            raise Exception(f"Failed to load price data from yfinance: {e}")