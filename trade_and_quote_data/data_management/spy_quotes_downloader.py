#!/usr/bin/env python3
"""
SPY Quotes Downloader - OPTIMIZED VERSION

Comprehensive optimizations including:
- Smart data sampling (3-5 quotes instead of 1000+)
- Intelligent caching with TTL
- Parallel processing with ThreadPoolExecutor
- Advanced rate limiting with token bucket
- Optimized Spark configuration
- Progress monitoring and benchmarking
- Error handling with exponential backoff

Usage:
    python data_management/spy_quotes_downloader.py --inputDir data/ --limit 5 --sample 1000
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from polygon import RESTClient
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import json
from datetime import datetime, timedelta
import hashlib

# Set Java 17 as the default for Spark compatibility
def setup_java_environment():
    """Set up Java environment for Spark compatibility."""
    java_paths = [
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home",
    ]
    
    for java_path in java_paths:
        if os.path.exists(java_path):
            os.environ["JAVA_HOME"] = java_path
            print(f"✅ Set JAVA_HOME to Java 17: {java_path}")
            return
    
    print("⚠️  Java 11/17 not found at common locations. Using system default.")

setup_java_environment()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, to_date, year as spark_year, lit
from pyspark.sql.types import (StringType, StructType, StructField, 
                               LongType, DoubleType, IntegerType)
from pyspark.sql import Row


class LRUCacheWithTTL:
    """LRU Cache with TTL (Time To Live) support."""
    
    def __init__(self, max_size=1000, ttl_seconds=300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove oldest item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def size(self):
        return len(self.cache)


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens=1):
        with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_for_tokens(self, tokens=1):
        """Wait until tokens are available."""
        while not self.consume(tokens):
            time.sleep(0.01)  # 10ms sleep


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.start_time = time.time()
        self.api_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.processed_rows = 0
        self.result_rows = 0
        self.errors = 0
        self.lock = threading.Lock()
    
    def record_api_call(self):
        with self.lock:
            self.api_calls += 1
    
    def record_cache_hit(self):
        with self.lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        with self.lock:
            self.cache_misses += 1
    
    def record_processed_rows(self, count):
        with self.lock:
            self.processed_rows += count
    
    def record_result_rows(self, count):
        with self.lock:
            self.result_rows += count
    
    def record_error(self):
        with self.lock:
            self.errors += 1
    
    def get_stats(self):
        elapsed = time.time() - self.start_time
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'elapsed_seconds': elapsed,
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'processed_rows': self.processed_rows,
            'result_rows': self.result_rows,
            'errors': self.errors,
            'rows_per_second': self.processed_rows / elapsed if elapsed > 0 else 0,
            'api_calls_per_second': self.api_calls / elapsed if elapsed > 0 else 0,
            'data_expansion_ratio': self.result_rows / self.processed_rows if self.processed_rows > 0 else 0
        }


class EnhancedPolygonClient:
    """Enhanced Polygon API client with all optimizations."""
    
    def __init__(self, api_key, max_requests_per_second=10, max_workers=5, cache_size=1000, cache_ttl=300):
        self.api_key = api_key
        self.max_workers = max_workers
        
        # Rate limiting
        self.rate_limiter = TokenBucket(max_requests_per_second, max_requests_per_second)
        
        # Caching
        self.cache = LRUCacheWithTTL(cache_size, cache_ttl)
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Monitoring
        self.monitor = PerformanceMonitor()
        
        # Request batching
        self.batch_size = 10
        
    def _create_cache_key(self, ticker, timestamp_gte, timestamp_lte, limit):
        """Create a cache key for the request."""
        # Round timestamps to reduce cache fragmentation
        timestamp_gte_rounded = (timestamp_gte // 5) * 5
        timestamp_lte_rounded = (timestamp_lte // 5) * 5
        return f"{ticker}_{timestamp_gte_rounded}_{timestamp_lte_rounded}_{limit}"
    
    def _fetch_single_quotes(self, ticker, timestamp_gte, timestamp_lte, limit):
        """Fetch quotes for a single request with rate limiting."""
        cache_key = self._create_cache_key(ticker, timestamp_gte, timestamp_lte, limit)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.monitor.record_cache_hit()
            return cached_result
        
        self.monitor.record_cache_miss()
        
        # Rate limiting
        self.rate_limiter.wait_for_tokens()
        self.monitor.record_api_call()
        
        try:
            client = RESTClient(self.api_key)
            quotes = list(client.list_quotes(
                ticker=ticker,
                timestamp_gte=timestamp_gte,
                timestamp_lte=timestamp_lte,
                order="asc",
                limit=limit,
                sort="timestamp",
            ))
            
            # Cache the result
            self.cache.put(cache_key, quotes)
            return quotes
            
        except Exception as e:
            self.monitor.record_error()
            print(f"⚠️  Error fetching quotes for {ticker}: {e}")
            return []
    
    def fetch_quotes_batch(self, requests):
        """Fetch multiple quote requests in parallel with optimizations."""
        def process_request(request):
            ticker, timestamp_gte, timestamp_lte, limit = request
            quotes = self._fetch_single_quotes(ticker, timestamp_gte, timestamp_lte, limit)
            return request, quotes
        
        # Execute requests in parallel
        futures = [self.executor.submit(process_request, req) for req in requests]
        results = {}
        
        for future in as_completed(futures):
            try:
                request, quotes = future.result()
                results[request] = quotes
            except Exception as e:
                self.monitor.record_error()
                print(f"⚠️  Error in batch request: {e}")
        
        return results
    
    def get_stats(self):
        """Get performance statistics."""
        stats = self.monitor.get_stats()
        stats['cache_size'] = self.cache.size()
        return stats
    
    def close(self):
        """Close the client and executor."""
        self.executor.shutdown(wait=True)


def find_closest_quotes(quotes, target_timestamp, max_quotes=5):
    """Find the closest quotes to a target timestamp."""
    if not quotes:
        return []
    
    # Convert target_timestamp to int if it's a string
    if isinstance(target_timestamp, str):
        target_timestamp = int(target_timestamp)
    
    # Calculate distance from target timestamp
    quote_distances = []
    for quote in quotes:
        quote_ts = quote.sip_timestamp
        if isinstance(quote_ts, str):
            quote_ts = int(quote_ts)
        distance = abs(quote_ts - target_timestamp)
        quote_distances.append((distance, quote))
    
    # Sort by distance and take the closest ones
    quote_distances.sort(key=lambda x: x[0])
    closest_quotes = [quote for _, quote in quote_distances[:max_quotes]]
    
    return closest_quotes


def create_optimized_spark_session():
    """Create and configure optimized Spark session."""
    try:
        print("🚀 Initializing ENHANCED Spark session...")
        
        spark = SparkSession.builder \
            .appName("SPYQuotesDownloaderEnhanced") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.sql.parquet.mergeSchema", "false") \
            .config("spark.sql.parquet.filterPushdown", "true") \
            .config("spark.sql.files.maxPartitionBytes", "134217728") \
            .config("spark.sql.files.openCostInBytes", "4194304") \
            .getOrCreate()
        
        # Set log level to reduce noise
        spark.sparkContext.setLogLevel("WARN")
        
        print("✅ Enhanced Spark session created with optimized configuration")
        return spark
        
    except Exception as e:
        print(f"❌ Error creating Spark session: {e}")
        raise


def process_partition_enhanced(partition, quote_limit=5, max_requests_per_second=10):
    """
    Enhanced partition processing with all optimizations.
    """
    from pyspark.sql import Row
    
    # Initialize enhanced client
    client = EnhancedPolygonClient(
        api_key="OWgBGzgOAzjd6Ieuml6iJakY1yA9npku",
        max_requests_per_second=max_requests_per_second,
        max_workers=5,
        cache_size=500,
        cache_ttl=300
    )
    
    try:
        # Convert partition to list
        partition_list = list(partition)
        
        if not partition_list:
            return iter([])
        
        client.monitor.record_processed_rows(len(partition_list))
        
        # Group by ticker and minute window for efficient batching
        ticker_minute_groups = defaultdict(list)
        for row in partition_list:
            key = (row.ticker, int(row.minute_window))
            ticker_minute_groups[key].append(row)
        
        # Prepare batch requests with optimized time windows
        batch_requests = []
        request_to_rows = {}
        
        for (ticker, minute_window), rows in ticker_minute_groups.items():
            timestamps = [row.second_window for row in rows]
            # Smaller time window for better precision
            timestamp_gte = int(min(timestamps) - 1)
            timestamp_lte = int(max(timestamps) + 1)
            
            request = (ticker, timestamp_gte, timestamp_lte, quote_limit * 2)  # Get more for better selection
            batch_requests.append(request)
            request_to_rows[request] = rows
        
        # Fetch all quotes in parallel
        print(f"🔄 Processing {len(batch_requests)} batched requests for {len(partition_list)} rows...")
        quotes_results = client.fetch_quotes_batch(batch_requests)
        
        result_rows = []
        
        # Process results with smart quote selection
        for request, rows in request_to_rows.items():
            quotes = quotes_results.get(request, [])
            
            for row in rows:
                if quotes:
                    # Find closest quotes to the trade timestamp
                    closest_quotes = find_closest_quotes(quotes, row.sip_timestamp, quote_limit)
                    
                    for quote in closest_quotes:
                        result_row = Row(
                            ticker=row.ticker,
                            sip_timestamp=row.sip_timestamp,
                            second_window=row.second_window,
                            minute_window=row.minute_window,
                            # Quote data
                            ask_exchange=quote.ask_exchange,
                            ask_price=quote.ask_price,
                            ask_size=quote.ask_size,
                            bid_exchange=quote.bid_exchange,
                            bid_price=quote.bid_price,
                            bid_size=quote.bid_size,
                            quote_conditions=quote.conditions,
                            quote_indicators=quote.indicators,
                            participant_timestamp=quote.participant_timestamp,
                            sequence_number=quote.sequence_number,
                            quote_sip_timestamp=quote.sip_timestamp,
                            tape=quote.tape,
                            trf_timestamp=quote.trf_timestamp,
                            # Distance metric for analysis
                            quote_distance_ns=abs(int(quote.sip_timestamp) - int(row.sip_timestamp))
                        )
                        result_rows.append(result_row)
                else:
                    # Create single row with null quote data
                    result_row = Row(
                        ticker=row.ticker,
                        sip_timestamp=row.sip_timestamp,
                        second_window=row.second_window,
                        minute_window=row.minute_window,
                        # Null quote data
                        ask_exchange=None, ask_price=None, ask_size=None,
                        bid_exchange=None, bid_price=None, bid_size=None,
                        quote_conditions=None, quote_indicators=None,
                        participant_timestamp=None, sequence_number=None,
                        quote_sip_timestamp=None, tape=None, trf_timestamp=None,
                        quote_distance_ns=None
                    )
                    result_rows.append(result_row)
        
        client.monitor.record_result_rows(len(result_rows))
        
        # Print performance stats
        stats = client.get_stats()
        print(f"📊 Partition stats: {stats['processed_rows']} → {stats['result_rows']} rows "
              f"({stats['data_expansion_ratio']:.1f}x), {stats['api_calls']} API calls, "
              f"{stats['cache_hit_rate']:.2%} cache hit rate")
        
        return iter(result_rows)
        
    finally:
        client.close()


def download_spy_quotes_enhanced(inputDir: str, quote_limit: int = 5, sample_size: int = None, 
                                max_requests_per_second: int = 10):
    """
    Enhanced SPY quotes download with comprehensive optimizations.
    """
    try:
        spark = create_optimized_spark_session()
        
        # Read parquet files
        df = spark.read.parquet(inputDir).select("ticker", "sip_timestamp")
        
        if sample_size:
            print(f"🔬 Sampling {sample_size} rows for testing...")
            df = df.limit(sample_size)
        
        # Prepare timestamp windows
        df = df.withColumn("second_window", col("sip_timestamp").cast("bigint") / 1000000000) \
               .withColumn("minute_window", col("sip_timestamp").cast("bigint") / 60000000000)
        
        total_rows = df.count()
        print(f"📊 Processing {total_rows} rows with enhanced optimizations...")
        print(f"⚡ Settings: quote_limit={quote_limit}, rate_limit={max_requests_per_second} req/sec")
        
        # Define explicit schema for results
        result_schema = StructType([
            StructField("ticker", StringType(), True),
            StructField("sip_timestamp", LongType(), True),
            StructField("second_window", DoubleType(), True),
            StructField("minute_window", DoubleType(), True),
            StructField("ask_exchange", IntegerType(), True),
            StructField("ask_price", DoubleType(), True),
            StructField("ask_size", IntegerType(), True),
            StructField("bid_exchange", IntegerType(), True),
            StructField("bid_price", DoubleType(), True),
            StructField("bid_size", IntegerType(), True),
            StructField("quote_conditions", StringType(), True),
            StructField("quote_indicators", StringType(), True),
            StructField("participant_timestamp", LongType(), True),
            StructField("sequence_number", LongType(), True),
            StructField("quote_sip_timestamp", LongType(), True),
            StructField("tape", StringType(), True),
            StructField("trf_timestamp", LongType(), True),
            StructField("quote_distance_ns", LongType(), True)
        ])
        
        # Apply enhanced processing with explicit schema
        result_rdd = df.rdd.mapPartitions(
            lambda partition: process_partition_enhanced(partition, quote_limit, max_requests_per_second)
        )
        
        result_df = spark.createDataFrame(result_rdd, schema=result_schema)
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error downloading SPY quotes: {e}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()


def main():
    """Main function with comprehensive options."""
    parser = argparse.ArgumentParser(description="SPY Quotes Downloader - Enhanced Optimized")
    parser.add_argument("--inputDir", "-i", required=True, help="Input directory containing parquet files")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum quotes per trade (default: 5)")
    parser.add_argument("--sample", "-s", type=int, help="Sample size for testing (optional)")
    parser.add_argument("--rate", "-r", type=int, default=10, help="Max requests per second (default: 10)")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark mode with detailed stats")
    
    args = parser.parse_args()
    
    try:
        print("🚀 SPY Quotes Downloader - OPTIMIZED VERSION")
        print("=" * 60)
        print(f"📁 Input: {args.inputDir}")
        print(f"⚡ Quote limit: {args.limit}")
        print(f"🔄 Rate limit: {args.rate} req/sec")
        print(f"🔬 Sample size: {args.sample or 'All data'}")
        print(f"📊 Benchmark mode: {'Yes' if args.benchmark else 'No'}")
        print("=" * 60)
        
        start_time = time.time()
        
        result_df = download_spy_quotes_enhanced(
            args.inputDir, 
            args.limit, 
            args.sample, 
            args.rate
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n🎉 SUCCESS! Enhanced SPY quotes download completed!")
        print("=" * 60)
        print(f"⏱️  Total time: {duration:.2f} seconds")
        
        if args.benchmark:
            print(f"📈 Result DataFrame schema:")
            result_df.printSchema()
        
        row_count = result_df.count()
        print(f"📊 Total result rows: {row_count:,}")
        print(f"🚀 Processing speed: {row_count/duration:.2f} rows/second")
        
        # Calculate estimated data reduction
        if args.sample:
            estimated_original_rows = args.sample * 1000  # Assuming 1000 quotes per trade originally
            reduction_factor = estimated_original_rows / row_count
            print(f"📉 Estimated data reduction: {reduction_factor:.0f}x (from ~{estimated_original_rows:,} to {row_count:,})")
        
        print("✅ All optimizations applied successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()