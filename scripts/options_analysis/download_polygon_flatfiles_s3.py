#!/usr/bin/env python3
"""
Download Polygon Flat Files using S3 (boto3)
Downloads historical SPY options trade data from Sept-Oct 2025
"""

import boto3
from botocore.config import Config
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import gzip

print("="*80)
print("POLYGON FLAT FILES DOWNLOADER - S3 METHOD")
print("="*80)

# AWS credentials for Polygon S3
AWS_ACCESS_KEY = '86959ae1-29bc-4433-be13-1a41b935d9d1'
AWS_SECRET_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

# Initialize S3 session
print("\n🔐 Initializing S3 session...")
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

s3 = session.client(
    's3',
    endpoint_url='https://files.polygon.io',
    config=Config(signature_version='s3v4'),
)

print("   ✅ S3 session initialized")

# Configuration
BUCKET_NAME = 'flatfiles'
PREFIX = 'us_options_opra/trades_v1'  # Options trades
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Date range
START_DATE = '2025-09-01'
END_DATE = '2025-10-06'

# Generate trading days
print(f"\n📅 Date range: {START_DATE} to {END_DATE}")
start = datetime.strptime(START_DATE, '%Y-%m-%d')
end = datetime.strptime(END_DATE, '%Y-%m-%d')

trading_days = []
current = start
while current <= end:
    if current.weekday() < 5:  # Skip weekends
        trading_days.append(current)
    current += timedelta(days=1)

print(f"   {len(trading_days)} potential trading days")

# Download flat files
successful_downloads = []
failed_downloads = []

for i, date in enumerate(trading_days, 1):
    date_str = date.strftime('%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"[{i}/{len(trading_days)}] Downloading {date_str}")
    print(f"{'='*80}")

    # S3 object key format: us_options_opra/trades_v1/YYYY/MM/YYYY-MM-DD.csv.gz
    year = date.strftime('%Y')
    month = date.strftime('%m')
    object_key = f"{PREFIX}/{year}/{month}/{date_str}.csv.gz"

    # Local file path
    local_file = OUTPUT_DIR / f"{date_str}.csv.gz"

    # Check if already downloaded
    if local_file.exists():
        file_size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Already exists ({file_size_mb:.1f} MB) - skipping")
        successful_downloads.append(date_str)
        continue

    try:
        print(f"   📥 Downloading from S3: {object_key}")

        # Download the file
        s3.download_file(BUCKET_NAME, object_key, str(local_file))

        # Check file size
        file_size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Downloaded successfully ({file_size_mb:.1f} MB)")

        successful_downloads.append(date_str)

    except s3.exceptions.NoSuchKey:
        print(f"   ⚠️  No data available (possibly holiday/weekend)")
        failed_downloads.append((date_str, "No data"))
    except Exception as e:
        print(f"   ❌ Error: {e}")
        failed_downloads.append((date_str, str(e)))

# Summary
print(f"\n{'='*80}")
print("DOWNLOAD SUMMARY")
print(f"{'='*80}")

print(f"\n✅ Successfully downloaded: {len(successful_downloads)} files")
if successful_downloads:
    total_size = sum((OUTPUT_DIR / f"{d}.csv.gz").stat().st_size
                     for d in successful_downloads if (OUTPUT_DIR / f"{d}.csv.gz").exists())
    total_size_gb = total_size / (1024 ** 3)
    print(f"   Total size: {total_size_gb:.2f} GB")
    print(f"   Files: {', '.join(successful_downloads[:5])}{'...' if len(successful_downloads) > 5 else ''}")

if failed_downloads:
    print(f"\n⚠️  Failed to download: {len(failed_downloads)} files")
    for date_str, reason in failed_downloads[:5]:
        print(f"   - {date_str}: {reason}")
    if len(failed_downloads) > 5:
        print(f"   ... and {len(failed_downloads) - 5} more")

# Parse SPY options from downloaded files
print(f"\n{'='*80}")
print("PARSING SPY OPTIONS DATA")
print(f"{'='*80}")

all_spy_trades = []

for date_str in successful_downloads:  # Process all days
    file_path = OUTPUT_DIR / f"{date_str}.csv.gz"

    if not file_path.exists():
        continue

    print(f"\n📊 Parsing {date_str}...")

    try:
        # Read compressed file in chunks
        spy_trades_day = []
        chunk_num = 0

        with gzip.open(file_path, 'rt') as f:
            for chunk in pd.read_csv(f, chunksize=100000):
                chunk_num += 1

                # Filter for SPY options
                spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]

                if len(spy_chunk) > 0:
                    spy_trades_day.append(spy_chunk)

                if chunk_num % 10 == 0:
                    print(f"   Processed {chunk_num} chunks...")

        if spy_trades_day:
            day_df = pd.concat(spy_trades_day, ignore_index=True)
            day_df['date'] = date_str

            print(f"   ✅ Found {len(day_df):,} SPY options trades")

            # Parse option details
            def parse_ticker(ticker):
                try:
                    if not ticker.startswith('O:SPY'):
                        return None, None
                    parts = ticker[5:]  # Remove O:SPY
                    if len(parts) >= 15:
                        option_type = parts[6].lower()
                        return option_type, ticker
                    return None, None
                except:
                    return None, None

            day_df[['option_type', 'ticker_clean']] = day_df['ticker'].apply(
                lambda x: pd.Series(parse_ticker(x))
            )

            calls = day_df[day_df['option_type'] == 'c']
            puts = day_df[day_df['option_type'] == 'p']

            call_volume = calls['size'].sum() if len(calls) > 0 else 0
            put_volume = puts['size'].sum() if len(puts) > 0 else 0
            pc_ratio = put_volume / max(call_volume, 1)

            print(f"   Calls: {len(calls):,} trades, {call_volume:,.0f} volume")
            print(f"   Puts: {len(puts):,} trades, {put_volume:,.0f} volume")
            print(f"   P/C Ratio: {pc_ratio:.2f}")

            all_spy_trades.append({
                'date': date_str,
                'total_trades': len(day_df),
                'call_trades': len(calls),
                'put_trades': len(puts),
                'call_volume': call_volume,
                'put_volume': put_volume,
                'pc_ratio': pc_ratio
            })

    except Exception as e:
        print(f"   ❌ Error parsing: {e}")

# Save summary
if all_spy_trades:
    summary_df = pd.DataFrame(all_spy_trades)
    summary_file = OUTPUT_DIR / 'spy_options_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print(f"\n{'='*80}")
    print("SPY OPTIONS SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    print(f"\n✅ Saved summary to: {summary_file}")

print(f"\n{'='*80}")
print("Complete!")
print(f"{'='*80}")
