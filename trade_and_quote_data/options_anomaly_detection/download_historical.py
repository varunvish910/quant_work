#!/usr/bin/env python3
"""
Smart Historical Options Chain Downloader

Downloads options chain data month by month, automatically skipping months
that already have data and continuing to the next missing month.

Usage:
    python download_historical.py --ticker SPY --start-month 2016-01
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import calendar

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the optimized options chain downloader
try:
    from data_management.optimized_options_downloader import OptimizedOptionsDownloader
    DOWNLOADER_AVAILABLE = True
except ImportError as e:
    DOWNLOADER_AVAILABLE = False
    print(f"❌ Could not import OptimizedOptionsDownloader: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('options_chain_download_historical.log')
    ]
)
logger = logging.getLogger(__name__)


def check_month_exists(ticker: str, year: int, month: int, data_dir: str = "data") -> bool:
    """
    Check if data already exists for a given month
    
    Returns True if the month directory exists and has parquet files
    """
    month_dir = Path(data_dir) / "options_chains" / ticker / f"{year:04d}" / f"{month:02d}"
    
    if not month_dir.exists():
        return False
    
    # Check if there are any parquet files in the directory
    parquet_files = list(month_dir.glob("*.parquet"))
    
    if len(parquet_files) > 0:
        logger.info(f"✓ Found {len(parquet_files)} files for {ticker} {year:04d}-{month:02d}")
        return True
    
    return False


def get_next_missing_month(ticker: str, start_year: int, start_month: int, 
                          end_year: int, end_month: int, data_dir: str = "data") -> tuple:
    """
    Find the next missing month in the range
    
    Returns (year, month) tuple or (None, None) if all months exist
    """
    current_year = start_year
    current_month = start_month
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        if not check_month_exists(ticker, current_year, current_month, data_dir):
            logger.info(f"📍 Next missing month: {current_year:04d}-{current_month:02d}")
            return (current_year, current_month)
        
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    logger.info("✅ All months in range already have data!")
    return (None, None)


def download_month(downloader, ticker: str, year: int, month: int, data_dir: str = "data"):
    """
    Download data for a specific month
    """
    # Get first and last day of the month
    first_day = f"{year:04d}-{month:02d}-01"
    last_day_num = calendar.monthrange(year, month)[1]
    last_day = f"{year:04d}-{month:02d}-{last_day_num:02d}"
    
    logger.info(f"📥 Downloading {ticker} for {year:04d}-{month:02d}")
    logger.info(f"   Date range: {first_day} to {last_day}")
    
    try:
        # Download the month
        daily_snapshots = downloader.download_date_range(ticker, first_day, last_day)
        
        if not daily_snapshots:
            logger.warning(f"⚠️  No data downloaded for {year:04d}-{month:02d}")
            return False
        
        # Save snapshots
        saved_files = downloader.save_daily_snapshots(daily_snapshots, ticker)
        
        # Create summary report
        summary_df = downloader.create_summary_report(daily_snapshots)
        
        # Save summary
        summary_file = Path(data_dir) / f"{ticker}_summary_{year:04d}{month:02d}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"✅ {year:04d}-{month:02d}: {len(daily_snapshots)} days, {len(saved_files)} files saved")
        logger.info(f"📊 Summary: {summary_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error downloading {year:04d}-{month:02d}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Historical Options Chain Downloader')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start-month', type=str, required=True, help='Start month (YYYY-MM)')
    parser.add_argument('--end-month', type=str, help='End month (YYYY-MM), defaults to start-month')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--auto-continue', action='store_true', 
                       help='Automatically download next missing month if start-month exists')
    
    args = parser.parse_args()
    
    # Parse start month
    try:
        start_parts = args.start_month.split('-')
        start_year = int(start_parts[0])
        start_month = int(start_parts[1])
    except:
        print(f"❌ Invalid start-month format: {args.start_month}. Use YYYY-MM")
        return
    
    # Parse end month (default to start month)
    if args.end_month:
        try:
            end_parts = args.end_month.split('-')
            end_year = int(end_parts[0])
            end_month = int(end_parts[1])
        except:
            print(f"❌ Invalid end-month format: {args.end_month}. Use YYYY-MM")
            return
    else:
        end_year = start_year
        end_month = start_month
    
    print(f"🚀 SMART OPTIONS CHAIN DOWNLOADER")
    print(f"📊 Ticker: {args.ticker}")
    print(f"📅 Range: {start_year:04d}-{start_month:02d} to {end_year:04d}-{end_month:02d}")
    print(f"🔍 Auto-continue: {args.auto_continue}")
    print("=" * 60)
    
    # Initialize downloader
    downloader = OptimizedOptionsDownloader(data_dir=args.data_dir)
    
    if not downloader.api_key:
        print("❌ POLYGON_API_KEY environment variable required")
        return
    
    # Check if start month exists
    if check_month_exists(args.ticker, start_year, start_month, args.data_dir):
        print(f"✓ Data already exists for {start_year:04d}-{start_month:02d}")
        
        if args.auto_continue:
            print("🔄 Looking for next missing month...")
            year, month = get_next_missing_month(
                args.ticker, start_year, start_month, end_year, end_month, args.data_dir
            )
            
            if year is None:
                print("✅ All months in range already have data!")
                return
            
            # Download the next missing month
            success = download_month(downloader, args.ticker, year, month, args.data_dir)
            
            if success:
                print(f"\n🎉 Successfully downloaded {year:04d}-{month:02d}")
            else:
                print(f"\n❌ Failed to download {year:04d}-{month:02d}")
        else:
            print("💡 Use --auto-continue to download next missing month")
    else:
        # Download the start month
        print(f"📥 Downloading {start_year:04d}-{start_month:02d}...")
        success = download_month(downloader, args.ticker, start_year, start_month, args.data_dir)
        
        if success:
            print(f"\n🎉 Successfully downloaded {start_year:04d}-{start_month:02d}")
        else:
            print(f"\n❌ Failed to download {start_year:04d}-{start_month:02d}")


if __name__ == "__main__":
    main()
