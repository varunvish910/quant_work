#!/usr/bin/env python3
"""
List available flat files in Polygon S3 to see what dates we actually have
"""

import boto3
from botocore.config import Config
from datetime import datetime

# AWS credentials
AWS_ACCESS_KEY = '86959ae1-29bc-4433-be13-1a41b935d9d1'
AWS_SECRET_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

print("Listing available Polygon flat files...")

# Initialize S3 session
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

s3 = session.client(
    's3',
    endpoint_url='https://files.polygon.io',
    config=Config(signature_version='s3v4'),
)

# List files
paginator = s3.get_paginator('list_objects_v2')
prefix = 'us_options_opra/trades_v1'

print(f"\n📂 Listing files in: {prefix}")
print("="*80)

file_count = 0
recent_files = []

for page in paginator.paginate(Bucket='flatfiles', Prefix=prefix, MaxKeys=100):
    if 'Contents' in page:
        for obj in page['Contents']:
            file_count += 1
            key = obj['Key']
            size_mb = obj['Size'] / (1024 * 1024)
            modified = obj['LastModified']

            recent_files.append({
                'key': key,
                'size_mb': size_mb,
                'modified': modified
            })

            if file_count <= 20:  # Show first 20
                print(f"{key:<60} {size_mb:>8.1f} MB  {modified}")

print(f"\n{'='*80}")
print(f"Total files found: {file_count}")

if recent_files:
    # Sort by modification date
    recent_files.sort(key=lambda x: x['modified'], reverse=True)

    print(f"\n📅 Most recent 10 files:")
    for f in recent_files[:10]:
        print(f"   {f['key']:<60} {f['modified']}")

    # Extract dates from filenames
    dates = []
    for f in recent_files:
        parts = f['key'].split('/')
        if len(parts) >= 3 and parts[-1].endswith('.csv.gz'):
            date_str = parts[-1].replace('.csv.gz', '')
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_obj)
            except:
                pass

    if dates:
        dates.sort()
        print(f"\n📊 Date range available:")
        print(f"   Earliest: {dates[0].strftime('%Y-%m-%d')}")
        print(f"   Latest: {dates[-1].strftime('%Y-%m-%d')}")
        print(f"   Total days: {len(dates)}")

print("\nDone!")
