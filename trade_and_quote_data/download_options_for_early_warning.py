#!/usr/bin/env python3
"""
Download Options Data for Early Warning Model

Strategy:
- 2016-2024: Polygon options chains (REAL historical data with IV, volume, OI)
- NO PROXIES - Only real options data

Yahoo Finance only provides CURRENT options chains, not historical.
So we use Polygon for all historical options data (2016+).
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from data_management.unified_options_downloader import OptimizedOptionsDownloader
from config.api_config import get_polygon_api_key
import pandas as pd
import yfinance as yf

print("=" * 80)
print("📊 DOWNLOADING OPTIONS DATA FOR EARLY WARNING MODEL")
print("=" * 80)
print()

# ============================================================================
# PART 1: Download Polygon Options Chains (2016-2024)
# ============================================================================
print("📥 PART 1: Polygon Options Chains (2016-2024)")
print("-" * 80)

api_key = get_polygon_api_key()
downloader = OptimizedOptionsDownloader(api_key=api_key, data_dir='data/options_chains')

print("Downloading SPY options chains from Polygon...")
print("Date range: 2016-01-01 to 2024-12-31")
print()

try:
    # Download in yearly chunks for better performance
    years = range(2016, 2025)
    all_snapshots = {}
    
    for year in years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31" if year < 2024 else "2024-12-31"
        
        print(f"\n📅 Downloading {year}...")
        snapshots = downloader.download_date_range('SPY', start_date, end_date)
        
        if snapshots:
            all_snapshots.update(snapshots)
            print(f"✅ {year}: {len(snapshots)} days downloaded")
        else:
            print(f"⚠️  {year}: No data")
    
    # Save snapshots
    if all_snapshots:
        print(f"\n💾 Saving {len(all_snapshots)} daily snapshots...")
        output_dir = Path('data/options_chains/polygon')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for each year
        for date_str, df in all_snapshots.items():
            year = date_str[:4]
            year_dir = output_dir / f"year={year}"
            year_dir.mkdir(exist_ok=True)
            
            filepath = year_dir / f"spy_options_{date_str}.parquet"
            df.to_parquet(filepath, index=False)
        
        print(f"✅ Saved to: {output_dir}")
        print(f"   Total days: {len(all_snapshots)}")
        print(f"   Date range: {min(all_snapshots.keys())} to {max(all_snapshots.keys())}")
    else:
        print("❌ No Polygon data downloaded")

except Exception as e:
    print(f"❌ Error downloading Polygon data: {e}")
    import traceback
    traceback.print_exc()

# Note: We're NOT using VIX/VVIX as proxies
# VIX data is already in our volatility features from data_loader.py
# We only want REAL options chain data from Polygon

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ DOWNLOAD COMPLETE")
print("=" * 80)
print()
print("Data Coverage:")
print("  📊 Polygon Options Chains: 2016-2024")
print("     - Real historical IV for each strike")
print("     - Volume and Open Interest")
print("     - Put/Call ratios")
print("     - IV skew calculations")
print()
print("Next Steps:")
print("  1. Calculate daily aggregated options features")
print("  2. Create options feature class")
print("  3. Integrate into FeatureEngine")
print("  4. Train early warning model (2016-2024)")
print()
