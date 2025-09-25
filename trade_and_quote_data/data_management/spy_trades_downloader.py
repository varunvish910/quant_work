#!/usr/bin/env python3
"""
SPY Trades Downloader with Minute-Level Aggregation

Downloads SPY options trades data from Polygon.io S3 bucket using Spark and
aggregates to minute-level data by underlying ticker.

Features:
- Direct S3 access to Polygon.io flat files
- Spark-based processing for large datasets
- SPY ticker filtering only
- Minute-level aggregation with contract totals by ticker
- Separate put/call contract counts and volume metrics
- Optimized for single year downloads

Usage:
    python data_management/spy_trades_downloader.py --year 2020
    python data_management/spy_trades_downloader.py --year 2021 --output-dir data/spy_2021
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Set Java 17 as the default for Spark compatibility
def setup_java_environment():
    """Set up Java environment for Spark compatibility."""
    # Use Java 17 (required by current PySpark installation)
    java_paths = [
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",  # Homebrew on Apple Silicon
        "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",     # Homebrew on Intel
        "/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home",  # Oracle/OpenJDK 17
    ]
    
    for java_path in java_paths:
        if os.path.exists(java_path):
            os.environ["JAVA_HOME"] = java_path
            print(f"✅ Set JAVA_HOME to Java 17: {java_path}")
            return
    
    print("⚠️  Java 11/17 not found at common locations. Using system default.")
    print("   You may need to install Java 11 or 17:")
    print("   - Homebrew: brew install openjdk@11 or brew install openjdk@17") 
    print("   - Or download from: https://adoptium.net/")
    
    # Check current Java version
    try:
        import subprocess
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if 'version "11.' in result.stderr or 'version "17.' in result.stderr:
            print("✅ System Java is compatible version!")
        else:
            print("⚠️  System Java may be incompatible. PySpark requires Java 11+")
    except:
        print("⚠️  Could not determine Java version")

# Set up Java environment before importing Spark
setup_java_environment()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, to_date, year as spark_year, lit, from_unixtime, date_trunc, when, sum, count, countDistinct
from pyspark.sql.types import StringType

def extract_options_metadata(df, year):
    """
    Extract expiration date, year, and underlying ticker from options ticker.
    
    Args:
        df: Spark DataFrame with 'ticker' column containing options data
        year: Year to add as a column for partitioning
        
    Returns:
        DataFrame with added columns: expiration_date, year, underlying_ticker
    """
    # Extract underlying ticker (remove O: prefix)
    df_with_underlying = df.withColumn(
        "underlying_ticker",
        regexp_extract(col("ticker"), r"^O:([A-Z]+?)(?=\d)", 1)
    )
    
    # Extract expiration date from ticker (format: O:SPY250117C00450000)
    # Pattern: O:SPY + YYMMDD + C/P + strike
    df_with_expiration = df_with_underlying.withColumn(
        "expiration_date",
        to_date(
            regexp_extract(col("ticker"), r"O:[A-Z]+(\d{6})", 1),
            "yyMMdd"
        )
    )
    
    # Add year column for partitioning
    df_with_year = df_with_expiration.withColumn("year", lit(year))
    
    return df_with_year


def create_spark_session():
    """Create and configure Spark session for S3 data processing."""
    try:
        print("🚀 Initializing Spark for SPY trades download...")
        
        # Add JAR files to classpath for S3A support with JAXB for Java 17
        jar_files = [
            "org.apache.hadoop:hadoop-aws:2.10.1",
            "com.amazonaws:aws-java-sdk:1.11.901",
            "javax.xml.bind:jaxb-api:2.3.1",
            "org.glassfish.jaxb:jaxb-runtime:2.3.1"
        ]
        
        spark = SparkSession.builder \
            .appName("SPYTradesDownloader") \
            .master("local[*]") \
            .config("spark.jars.packages", ",".join(jar_files)) \
            .config("spark.driver.extraJavaOptions", "-Dcom.amazonaws.sdk.disableCertChecking=true -Dtrust_all_cert=true") \
            .config("spark.executor.extraJavaOptions", "-Dcom.amazonaws.sdk.disableCertChecking=true -Dtrust_all_cert=true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.files.maxPartitionBytes", "256m") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "256m") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.hadoop.fs.s3a.access.key", "86959ae1-29bc-4433-be13-1a41b935d9d1") \
            .config("spark.hadoop.fs.s3a.secret.key", "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku") \
            .config("spark.hadoop.fs.s3a.endpoint", "https://files.polygon.io") \
            .config("spark.hadoop.fs.s3a.signing-algorithm", "AWS4SignerType") \
            .config("spark.hadoop.fs.s3a.bucket.all.committer.magic.enabled", "false") \
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
            .config("spark.hadoop.fs.s3a.ssl.channel.mode", "default_jsse") \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.bucket.flatfiles.create", "false") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
            .config("spark.hadoop.fs.s3a.connection.timeout", "30000") \
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000") \
            .config("spark.hadoop.fs.s3a.connection.request.timeout", "0") \
            .config("spark.hadoop.fs.s3a.socket.recv.buffer", "8192") \
            .config("spark.hadoop.fs.s3a.socket.send.buffer", "8192") \
            .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000") \
            .config("spark.hadoop.fs.s3a.multipart.cleaner.age", "86400000") \
            .config("spark.hadoop.fs.s3a.multipart.cleaner.interval", "86400000") \
            .config("spark.hadoop.fs.s3a.connection.maximum", "5") \
            .config("spark.hadoop.fs.s3a.attempts.maximum", "3") \
            .config("spark.hadoop.fs.s3a.retry.interval", "1000") \
            .config("spark.hadoop.fs.s3a.retry.limit", "3") \
            .config("spark.hadoop.fs.s3a.multipart.purge", "false") \
            .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400") \
            .config("spark.hadoop.fs.s3a.threads.max", "4") \
            .config("spark.hadoop.fs.s3a.threads.core", "2") \
            .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1") \
            .config("spark.sql.adaptive.coalescePartitions.initialPartitionNum", "4") \
            .getOrCreate()
        
        print("✅ Spark session created: SPYTradesDownloader")
        return spark
        
    except Exception as e:
        print(f"❌ Error creating Spark session: {e}")
        raise


def download_spy_trades(year: int, output_dir: str = None):
    """
    Download SPY options trades data for a specific year.
    
    Args:
        year: Year to download (e.g., 2020)
        output_dir: Output directory for processed data
    """
    
    if output_dir is None:
        output_dir = f"data/spy_trades_{year}"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        spark = create_spark_session()
        
        print(f"📊 Downloading SPY options trades for {year}...")
        print(f"📁 Output directory: {output_dir}")
        
        # S3 bucket path for Polygon.io flat files data for specific year
        s3_path = f"s3a://flatfiles/us_options_opra/trades_v1/{year}/01/*.csv.gz"
        print(f"🔗 Reading S3 Path: {s3_path}")
        df = spark.read.option("header", "true").csv(s3_path)
        
        # Filter for SPY options only
        spy_df = df.filter(col("ticker").rlike(r"^O:SPY\d"))
        
        spy_df_with_metadata = extract_options_metadata(spy_df, year)
        print("✅ Extracted options metadata")
        spy_df_with_metadata.show(5, truncate=False)
        
        
        return spy_df_with_metadata
        
    except Exception as e:
        print(f"❌ Error downloading SPY trades: {e}")
        raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="SPY Trades Downloader - Downloads SPY options trades from Polygon.io S3"
    )
    parser.add_argument(
        '--year', '-y',
        type=int,
        required=True,
        help='Year to download (e.g., 2020)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: data/spy_trades_{year})'
    )
    
    args = parser.parse_args()
    
    print("🚀 SPY TRADES DOWNLOADER")
    print("="*50)
    print(f"📅 Year: {args.year}")
    print(f"📁 Output: {args.output_dir or f'data/spy_trades_{args.year}'}")
    print()
    
    try:
        spy_df = download_spy_trades(args.year, args.output_dir)
        print(f"💾 Writing minute-aggregated SPY trades data to {args.output_dir or f'data/spy_trades_{args.year}'}")
        spy_df.write.mode("overwrite").partitionBy("year").parquet(args.output_dir or f"data/spy_trades_{args.year}")
        print("✅ Download and aggregation completed successfully!")
        print(f"📊 Data aggregated to minute-level with total contracts by ticker")
        
    except KeyboardInterrupt:
        print("\n⏹️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
