#!/usr/bin/env python3
"""
Download Historical SPY Options Trade Data from Polygon
Downloads 30-40 trading days of data for behavioral analysis
"""

import sys
sys.path.append('trade_and_quote_data/dealer_positioning')

from spy_trades_downloader import SPYTradesDownloader
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import yfinance as yf

print("="*80)
print("SPY OPTIONS HISTORICAL DATA DOWNLOADER")
print("="*80)

# Configuration
API_KEY = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
START_DATE = "2025-09-01"  # Start of September
END_DATE = "2025-10-06"    # Today

# Get current SPY price
print("\n📊 Getting current SPY price...")
spy = yf.Ticker('SPY')
current_price = spy.history(period="1d")["Close"].iloc[-1]
print(f"   Current SPY: ${current_price:.2f}")

# Initialize downloader
print(f"\n🚀 Initializing downloader...")
output_dir = "trade_and_quote_data/data_management"
downloader = SPYTradesDownloader(api_key=API_KEY, output_dir=output_dir)

# Generate trading days (exclude weekends)
print(f"\n📅 Generating trading days from {START_DATE} to {END_DATE}...")
start = datetime.strptime(START_DATE, '%Y-%m-%d')
end = datetime.strptime(END_DATE, '%Y-%m-%d')

trading_days = []
current = start
while current <= end:
    # Skip weekends
    if current.weekday() < 5:  # Monday=0, Friday=4
        trading_days.append(current.strftime('%Y-%m-%d'))
    current += timedelta(days=1)

print(f"   Found {len(trading_days)} potential trading days")
print(f"   First: {trading_days[0]}")
print(f"   Last: {trading_days[-1]}")

# Download data for each day
print(f"\n📥 Downloading SPY options trade data...")
print(f"   This will take approximately {len(trading_days) * 2} minutes")
print(f"   (downloading Polygon flat files ~2 min per day)")

all_trades = []
successful_days = []
failed_days = []

for i, date in enumerate(trading_days, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(trading_days)}] Processing {date}")
    print(f"{'='*80}")

    try:
        # Download flat file
        flat_file = downloader.download_flat_file(date, "trades")

        if flat_file is None:
            print(f"   ⚠️  No data available for {date} (possibly holiday)")
            failed_days.append((date, "No flat file"))
            continue

        # Parse SPY options
        spy_trades = downloader.parse_spy_options_from_flat_file(flat_file)

        if spy_trades is None or len(spy_trades) == 0:
            print(f"   ⚠️  No SPY options found for {date}")
            failed_days.append((date, "No SPY options"))
            continue

        # Add date column
        spy_trades['date'] = date

        # Save individual day file
        day_file = Path(output_dir) / f"trades_data_{i}_{date}.csv.gz"
        spy_trades.to_csv(day_file, index=False, compression='gzip')
        print(f"   ✅ Saved {len(spy_trades):,} trades to {day_file.name}")

        # Accumulate for combined file
        all_trades.append(spy_trades)
        successful_days.append(date)

    except Exception as e:
        print(f"   ❌ Error processing {date}: {e}")
        failed_days.append((date, str(e)))
        continue

# Combine all data
print(f"\n{'='*80}")
print(f"DOWNLOAD SUMMARY")
print(f"{'='*80}")

if all_trades:
    print(f"\n✅ Successfully downloaded {len(successful_days)} days of data")
    print(f"   Total trades: {sum(len(df) for df in all_trades):,}")

    # Save combined file
    print(f"\n📦 Combining all data into single file...")
    combined = pd.concat(all_trades, ignore_index=True)

    combined_file = Path(output_dir) / f"spy_options_trades_{START_DATE}_to_{END_DATE}.csv.gz"
    combined.to_csv(combined_file, index=False, compression='gzip')

    file_size_mb = combined_file.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved combined file: {combined_file.name}")
    print(f"   📊 Size: {file_size_mb:.1f} MB")
    print(f"   📊 Total trades: {len(combined):,}")

    # Summary stats
    print(f"\n📈 Data Summary:")
    print(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"   Calls: {(combined['option_type']=='c').sum():,}")
    print(f"   Puts:  {(combined['option_type']=='p').sum():,}")
    print(f"   Put/Call Ratio: {(combined['option_type']=='p').sum() / max((combined['option_type']=='c').sum(), 1):.2f}")

else:
    print(f"\n❌ No data downloaded successfully")

if failed_days:
    print(f"\n⚠️  Failed to download {len(failed_days)} days:")
    for date, reason in failed_days[:10]:  # Show first 10
        print(f"   - {date}: {reason}")
    if len(failed_days) > 10:
        print(f"   ... and {len(failed_days) - 10} more")

print(f"\n{'='*80}")
print(f"Download complete!")
print(f"{'='*80}")
