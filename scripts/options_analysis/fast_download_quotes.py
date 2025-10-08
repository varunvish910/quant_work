#!/usr/bin/env python3
"""
Fast download using presigned URL + aria2c (multi-connection)
"""
import boto3
from botocore.config import Config
from pathlib import Path
import subprocess

AWS_ACCESS_KEY = '86959ae1-29bc-4433-be13-1a41b935d9d1'
AWS_SECRET_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

s3 = session.client(
    's3',
    endpoint_url='https://files.polygon.io',
    config=Config(signature_version='s3v4'),
)

DATE = '2025-10-06'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUOTES_FILE = OUTPUT_DIR / f"{DATE}_quotes.csv.gz"

year = DATE[:4]
month = DATE[5:7]
quotes_key = f"us_options_opra/quotes_v1/{year}/{month}/{DATE}.csv.gz"

print("Generating presigned URL...")
presigned_url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'flatfiles', 'Key': quotes_key},
    ExpiresIn=7200  # 2 hours
)

print(f"Presigned URL: {presigned_url[:100]}...")
print(f"\nTo download with aria2c (multi-connection, faster):")
print(f"\naria2c -x 16 -s 16 -d {OUTPUT_DIR} -o {QUOTES_FILE.name} '{presigned_url}'")
print(f"\nOr with curl (single connection, with resume):")
print(f"\ncurl -C - -o {QUOTES_FILE} '{presigned_url}'")
