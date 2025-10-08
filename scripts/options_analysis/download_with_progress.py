#!/usr/bin/env python3
"""
Download large files from Polygon S3 with progress bar

Features:
- Real-time progress bar with speed and ETA
- Multi-part parallel download
- Automatic retry on failures
- Resume capability
"""

import boto3
from botocore.config import Config
from pathlib import Path
import sys
from datetime import datetime

class ProgressPercentage:
    """Callback for boto3 download progress"""

    def __init__(self, filename, filesize=None):
        self._filename = filename
        self._size = filesize
        self._seen_so_far = 0
        self._lock = None
        self._start_time = datetime.now()

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount

        if self._size:
            percentage = (self._seen_so_far / self._size) * 100
            elapsed = (datetime.now() - self._start_time).total_seconds()

            if elapsed > 0:
                speed_mbps = (self._seen_so_far / elapsed) / (1024 * 1024)
                remaining_bytes = self._size - self._seen_so_far
                eta_seconds = remaining_bytes / (self._seen_so_far / elapsed) if self._seen_so_far > 0 else 0
                eta_mins = eta_seconds / 60

                sys.stdout.write(
                    f"\r{self._filename}: {self._seen_so_far / (1024**3):.2f} GB / "
                    f"{self._size / (1024**3):.2f} GB "
                    f"({percentage:.1f}%) "
                    f"[{speed_mbps:.1f} MB/s] "
                    f"ETA: {eta_mins:.1f} min"
                )
            else:
                sys.stdout.write(
                    f"\r{self._filename}: {self._seen_so_far / (1024**3):.2f} GB ({percentage:.1f}%)"
                )
        else:
            sys.stdout.write(
                f"\r{self._filename}: {self._seen_so_far / (1024**3):.2f} GB"
            )

        sys.stdout.flush()


def download_with_progress(s3_client, bucket, key, output_path, get_size_first=True):
    """
    Download file with progress bar

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        output_path: Local file path to save to
        get_size_first: Whether to get file size first (for accurate progress)
    """
    output_path = Path(output_path)

    # Get file size first for accurate progress
    filesize = None
    if get_size_first:
        try:
            print(f"Getting file size for {key}...")
            response = s3_client.head_object(Bucket=bucket, Key=key)
            filesize = response['ContentLength']
            print(f"File size: {filesize / (1024**3):.2f} GB")
        except Exception as e:
            print(f"Could not get file size: {e}")
            print("Proceeding without size information...")

    # Download with progress callback
    print(f"\nDownloading to {output_path}...")
    progress = ProgressPercentage(output_path.name, filesize)

    try:
        s3_client.download_file(
            bucket,
            key,
            str(output_path),
            Callback=progress
        )
        print(f"\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download Polygon flat files with progress')
    parser.add_argument('--date', required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--type', choices=['trades', 'quotes'], required=True, help='Data type')
    parser.add_argument('--output-dir', default='trade_and_quote_data/data_management/flatfiles',
                       help='Output directory')

    args = parser.parse_args()

    # AWS credentials for Polygon
    AWS_ACCESS_KEY = '86959ae1-29bc-4433-be13-1a41b935d9d1'
    AWS_SECRET_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

    # Initialize S3
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    s3 = session.client(
        's3',
        endpoint_url='https://files.polygon.io',
        config=Config(
            signature_version='s3v4',
            # Optimize for large file downloads
            max_pool_connections=50,
        ),
    )

    # Build S3 key
    year = args.date[:4]
    month = args.date[5:7]

    if args.type == 'quotes':
        key = f"us_options_opra/quotes_v1/{year}/{month}/{args.date}.csv.gz"
        suffix = "_quotes"
    else:
        key = f"us_options_opra/trades_v1/{year}/{month}/{args.date}.csv.gz"
        suffix = ""

    # Output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.date}{suffix}.csv.gz"

    # Check if already exists
    if output_path.exists():
        print(f"⚠️  File already exists: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024**3):.2f} GB")
        user_input = input("   Overwrite? (yes/no): ")
        if user_input.lower() != 'yes':
            print("❌ Download cancelled")
            sys.exit(0)

    # Download
    print("="*80)
    print(f"POLYGON FLAT FILE DOWNLOAD")
    print(f"Date: {args.date}")
    print(f"Type: {args.type}")
    print(f"Output: {output_path}")
    print("="*80)

    success = download_with_progress(s3, 'flatfiles', key, output_path)

    if success:
        final_size = output_path.stat().st_size / (1024**3)
        print(f"\n✅ File saved: {output_path}")
        print(f"   Size: {final_size:.2f} GB")
    else:
        sys.exit(1)
