"""
Sector ETF Data Downloader

Downloads sector ETF data from yfinance.
"""

import pandas as pd
from typing import List, Dict
from data_management.base import YFinanceDownloader


class SectorDownloader(YFinanceDownloader):
    """Download sector ETF data"""
    
    def __init__(self, cache_dir=None):
        super().__init__("Sector Downloader", cache_dir)
    
    def download(self, symbols: List[str], start_date: str, end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download sector ETF data"""
        print(f"📥 Downloading {len(symbols)} sector ETFs...")
        
        data = {}
        for symbol in symbols:
            try:
                df = self._download_single(symbol, start_date, end_date)
                if self.validate_data(df, symbol):
                    data[symbol] = df
                    self.save_to_cache(df, symbol, 'sectors')
            except Exception as e:
                print(f"   ❌ Failed to download {symbol}: {e}")
        
        print(f"✅ Downloaded {len(data)}/{len(symbols)} sector ETFs")
        return data
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate sector data"""
        if len(data) == 0:
            raise ValueError(f"No data for {symbol}")
        return True
