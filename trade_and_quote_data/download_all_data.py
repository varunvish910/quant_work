#!/usr/bin/env python3
"""
Download All Market Data

Downloads all required data for the SPY Early Warning System:
- SPY OHLCV data
- Sector ETF data (10 sectors)
- Currency data (USD/JPY, EUR/USD)
- Volatility indices (VIX, VIX9D, VVIX)
- Options data (from Polygon API)

Usage:
    python3 download_all_data.py
    python3 download_all_data.py --start-date 2020-01-01 --end-date 2024-12-31
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import time

# Data directories
DATA_DIR = Path('data')
OHLC_DIR = DATA_DIR / 'ohlc'
SECTORS_DIR = DATA_DIR / 'sectors'
CURRENCY_DIR = DATA_DIR / 'currency'
VOLATILITY_DIR = DATA_DIR / 'volatility'
OPTIONS_DIR = DATA_DIR / 'options_chains'

# Create directories
for dir_path in [OHLC_DIR, SECTORS_DIR, CURRENCY_DIR, VOLATILITY_DIR, OPTIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def download_spy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download SPY OHLCV data from yfinance
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with SPY data
    """
    print("📥 Downloading SPY data...")
    
    try:
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        # Validate data
        if len(spy) == 0:
            raise ValueError("No SPY data downloaded")
        
        # Save to parquet
        output_file = OHLC_DIR / 'SPY.parquet'
        spy.to_parquet(output_file)
        
        print(f"✅ SPY: {len(spy)} records ({spy.index.min().date()} to {spy.index.max().date()})")
        print(f"   Saved to: {output_file}")
        
        return spy
        
    except Exception as e:
        print(f"❌ Failed to download SPY: {e}")
        return None


def download_sector_etfs(start_date: str, end_date: str) -> dict:
    """
    Download sector ETF data from yfinance
    
    Standard Sectors (SPDR):
    - XLK: Technology
    - XLF: Financials
    - XLV: Healthcare
    - XLU: Utilities
    - XLE: Energy
    - XLI: Industrials
    - XLY: Consumer Discretionary
    - XLP: Consumer Staples
    - XLRE: Real Estate
    - XLB: Materials
    
    Rotation Indicators:
    - MAGS: Magnificent 7 (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
    - RSP: Equal-weight S&P 500 (breadth indicator)
    - QQQ: Nasdaq 100 (tech-heavy)
    - QQQE: Equal-weight Nasdaq 100 (tech breadth)
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dictionary of sector DataFrames
    """
    print("\n📥 Downloading sector ETF data...")
    
    sectors = {
        # Standard SPDR Sectors
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLU': 'Utilities',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLRE': 'Real Estate',
        'XLB': 'Materials',
        # Rotation Indicators
        'MAGS': 'Magnificent 7',
        'RSP': 'Equal-Weight S&P 500',
        'QQQ': 'Nasdaq 100',
        'QQQE': 'Equal-Weight Nasdaq 100'
    }
    
    sector_data = {}
    
    for symbol, name in sectors.items():
        try:
            print(f"   Downloading {symbol} ({name})...")
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) > 0:
                # Save to parquet
                output_file = SECTORS_DIR / f'{symbol}.parquet'
                data.to_parquet(output_file)
                
                sector_data[symbol] = data
                print(f"   ✅ {symbol}: {len(data)} records")
            else:
                print(f"   ⚠️  {symbol}: No data")
                
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
    
    print(f"\n✅ Downloaded {len(sector_data)}/14 sector ETFs + rotation indicators")
    return sector_data


def download_currency_data(start_date: str, end_date: str) -> dict:
    """
    Download currency pair data from yfinance
    
    Pairs:
    - JPY=X: USD/JPY (critical for carry trade detection)
    - EURUSD=X: EUR/USD
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dictionary of currency DataFrames
    """
    print("\n📥 Downloading currency data...")
    
    currencies = {
        'JPY=X': 'USD/JPY',
        'EURUSD=X': 'EUR/USD'
    }
    
    currency_data = {}
    
    for symbol, name in currencies.items():
        try:
            print(f"   Downloading {name}...")
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) > 0:
                # Save to parquet
                clean_symbol = symbol.replace('=', '').replace('X', '')
                output_file = CURRENCY_DIR / f'{clean_symbol}.parquet'
                data.to_parquet(output_file)
                
                currency_data[clean_symbol] = data
                print(f"   ✅ {name}: {len(data)} records")
            else:
                print(f"   ⚠️  {name}: No data")
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    print(f"\n✅ Downloaded {len(currency_data)}/2 currency pairs")
    return currency_data


def download_volatility_indices(start_date: str, end_date: str) -> dict:
    """
    Download volatility index data from yfinance
    
    Indices:
    - ^VIX: CBOE Volatility Index
    - ^VIX9D: CBOE 9-Day Volatility Index
    - ^VVIX: CBOE VIX of VIX
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dictionary of volatility DataFrames
    """
    print("\n📥 Downloading volatility indices...")
    
    indices = {
        '^VIX': 'VIX',
        '^VIX9D': 'VIX9D',
        '^VVIX': 'VVIX'
    }
    
    volatility_data = {}
    
    for symbol, name in indices.items():
        try:
            print(f"   Downloading {name}...")
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) > 0:
                # Save to parquet
                clean_symbol = symbol.replace('^', '')
                output_file = VOLATILITY_DIR / f'{clean_symbol}.parquet'
                data.to_parquet(output_file)
                
                volatility_data[clean_symbol] = data
                print(f"   ✅ {name}: {len(data)} records")
            else:
                print(f"   ⚠️  {name}: No data")
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    print(f"\n✅ Downloaded {len(volatility_data)}/3 volatility indices")
    return volatility_data


def download_options_data_sample():
    """
    Options data download - SKIPPED FOR NOW
    
    Note: Options features will be added in Phase 3.5+
    This is a separate phase before backtesting
    """
    print("\n📥 Options data...")
    print("   ℹ️  Skipped - Options features are Phase 3.5+ (before backtesting)")
    print("   ℹ️  Will be implemented after model improvements")


def validate_data():
    """Validate all downloaded data"""
    print("\n" + "=" * 80)
    print("🔍 VALIDATING DOWNLOADED DATA")
    print("=" * 80)
    
    validation_results = {
        'SPY': False,
        'Sectors': 0,
        'Currency': 0,
        'Volatility': 0
    }
    
    # Check SPY
    spy_file = OHLC_DIR / 'SPY.parquet'
    if spy_file.exists():
        spy = pd.read_parquet(spy_file)
        validation_results['SPY'] = len(spy) > 1000  # At least 1000 records
        print(f"✅ SPY: {len(spy)} records")
    else:
        print(f"❌ SPY: File not found")
    
    # Check sectors
    sector_files = list(SECTORS_DIR.glob('*.parquet'))
    validation_results['Sectors'] = len(sector_files)
    print(f"✅ Sectors: {len(sector_files)}/14 files (10 sectors + 4 rotation indicators)")
    
    # Check currency
    currency_files = list(CURRENCY_DIR.glob('*.parquet'))
    validation_results['Currency'] = len(currency_files)
    print(f"✅ Currency: {len(currency_files)}/2 files")
    
    # Check volatility
    volatility_files = list(VOLATILITY_DIR.glob('*.parquet'))
    validation_results['Volatility'] = len(volatility_files)
    print(f"✅ Volatility: {len(volatility_files)}/3 files")
    
    # Overall status
    print("\n" + "=" * 80)
    all_good = (
        validation_results['SPY'] and
        validation_results['Sectors'] >= 12 and  # At least 12/14 (sectors + rotation)
        validation_results['Currency'] >= 1 and  # At least 1 currency pair
        validation_results['Volatility'] >= 2    # At least 2 volatility indices
    )
    
    if all_good:
        print("✅ DATA VALIDATION PASSED")
        print("   All required data downloaded successfully!")
    else:
        print("⚠️  DATA VALIDATION WARNING")
        print("   Some data may be missing - check errors above")
    
    print("=" * 80)
    
    return validation_results


def main():
    """Main download function"""
    parser = argparse.ArgumentParser(description='Download all market data')
    parser.add_argument('--start-date', type=str, default='2000-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("📥 DOWNLOADING ALL MARKET DATA")
    print("=" * 80)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output directory: {DATA_DIR.absolute()}")
    print("=" * 80)
    
    # Download all data
    spy_data = download_spy_data(args.start_date, args.end_date)
    sector_data = download_sector_etfs(args.start_date, args.end_date)
    currency_data = download_currency_data(args.start_date, args.end_date)
    volatility_data = download_volatility_indices(args.start_date, args.end_date)
    download_options_data_sample()
    
    # Validate
    validation_results = validate_data()
    
    print("\n" + "=" * 80)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python3 test_new_architecture.py")
    print("  2. Run: python3 daily_usage_example.py")
    print("  3. Train models: python3 train_all_targets.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
