"""
Equity Data Downloader

Downloads equity (stock) data from yfinance.
"""

import pandas as pd
from typing import List, Dict
from data_management.base import YFinanceDownloader


class EquityDownloader(YFinanceDownloader):
    """Download equity data"""
    
    def __init__(self, cache_dir=None):
        super().__init__("Equity Downloader", cache_dir)
    
    def download(self, symbols: List[str], start_date: str, end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download equity data for given symbols"""
        print(f"📥 Downloading {len(symbols)} equities...")
        
        data = {}
        for symbol in symbols:
            try:
                df = self._download_single(symbol, start_date, end_date)
                if self.validate_data(df, symbol):
                    data[symbol] = df
                    self.save_to_cache(df, symbol, 'equities')
            except Exception as e:
                print(f"   ❌ Failed to download {symbol}: {e}")
        
        print(f"✅ Downloaded {len(data)}/{len(symbols)} equities")
        return data
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate equity data"""
        if len(data) == 0:
            raise ValueError(f"No data for {symbol}")
        
        if 'Close' not in data.columns:
            raise ValueError(f"Missing Close column for {symbol}")
        
        return True
