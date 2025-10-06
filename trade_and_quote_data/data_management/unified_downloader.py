"""
Unified Data Downloader

Orchestrates all data downloaders.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from data_management.downloaders.equity import EquityDownloader
from data_management.downloaders.sector import SectorDownloader


class UnifiedDataDownloader:
    """Master downloader that orchestrates all data sources"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data")
        
        # Initialize downloaders
        self.equity_downloader = EquityDownloader(cache_dir)
        self.sector_downloader = SectorDownloader(cache_dir)
    
    def download_all(self, 
                     equities: List[str] = None,
                     sectors: List[str] = None,
                     start_date: str = "2020-01-01",
                     end_date: str = "2024-12-31") -> Dict:
        """Download all requested data"""
        
        print("=" * 80)
        print("📥 UNIFIED DATA DOWNLOADER")
        print("=" * 80)
        
        data = {}
        
        # Download equities
        if equities:
            data['equities'] = self.equity_downloader.download(equities, start_date, end_date)
        
        # Download sectors
        if sectors:
            data['sectors'] = self.sector_downloader.download(sectors, start_date, end_date)
        
        print("=" * 80)
        print("✅ DOWNLOAD COMPLETE")
        print("=" * 80)
        
        return data
