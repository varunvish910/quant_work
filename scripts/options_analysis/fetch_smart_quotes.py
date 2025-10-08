#!/usr/bin/env python3
"""
Smart quote fetcher for trade direction analysis
Prioritizes high-volume tickers and uses time bucketing
Estimated runtime: 30-45 minutes for full day
"""

import pandas as pd
import numpy as np
import gzip
import requests
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Configuration
POLYGON_API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
DATA_DIR = Path('trade_and_quote_data/data_management/flatfiles')
CACHE_DIR = Path('trade_and_quote_data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class SmartQuoteFetcher:
    """Smart quote fetching with prioritization and bucketing"""

    def __init__(self, date, test_mode=False):
        self.date = date
        self.api_key = POLYGON_API_KEY
        self.test_mode = test_mode
        self.api_calls = 0
        self.quote_cache = {}
        self.errors = []
        self.checkpoint_counter = 0

        # Cache files
        self.cache_file = CACHE_DIR / f"smart_quotes_{date}.json"
        self.checkpoint_file = CACHE_DIR / f"smart_quotes_{date}_checkpoint.json"

        # Load existing cache if available
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.quote_cache = json.load(f)
                print(f"   📂 Loaded {len(self.quote_cache)} cached quotes")

    def fetch_quote_at_time(self, ticker, timestamp):
        """Fetch quote at specific timestamp"""

        # Check cache first
        cache_key = f"{ticker}_{timestamp}"
        if cache_key in self.quote_cache:
            return self.quote_cache[cache_key]

        # Convert timestamp to nanoseconds for API
        ts = pd.Timestamp(timestamp)
        ts_ns = int(ts.value)

        # API endpoint - get quotes around this timestamp
        url = f"https://api.polygon.io/v3/quotes/{ticker}"

        params = {
            'timestamp.gte': ts_ns - 1000000000,  # 1 second before
            'timestamp.lte': ts_ns + 1000000000,  # 1 second after
            'limit': 1,
            'sort': 'timestamp',
            'order': 'desc',
            'apiKey': self.api_key
        }

        try:
            response = requests.get(url, params=params)
            self.api_calls += 1

            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    quote = data['results'][0]
                    result = {
                        'bid': quote.get('bid_price', 0),
                        'ask': quote.get('ask_price', 0),
                        'bid_size': quote.get('bid_size', 0),
                        'ask_size': quote.get('ask_size', 0),
                        'timestamp': quote.get('participant_timestamp', ts_ns)
                    }

                    # Cache the result
                    self.quote_cache[cache_key] = result
                    return result
            else:
                self.errors.append(f"API error {response.status_code} for {ticker}")

        except Exception as e:
            self.errors.append(f"Error fetching {ticker}: {str(e)}")

        return {'bid': 0, 'ask': 0, 'bid_size': 0, 'ask_size': 0}

    def save_checkpoint(self):
        """Save progress checkpoint"""
        checkpoint_data = {
            'cache': self.quote_cache,
            'api_calls': self.api_calls,
            'errors': self.errors
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

    def save_final(self):
        """Save final cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.quote_cache, f)
        print(f"   💾 Saved {len(self.quote_cache)} quotes to cache")


def classify_trade(trade_price, bid, ask):
    """Classify trade direction based on price vs bid/ask"""
    if bid <= 0 or ask <= 0:
        return 'UNKNOWN'

    # Allow 1% tolerance for rounding
    if trade_price >= ask * 0.99:
        return 'BTO'  # Buy to open (customer buying)
    elif trade_price <= bid * 1.01:
        return 'STO'  # Sell to open (customer selling)
    else:
        # Mid-market trade
        mid = (bid + ask) / 2
        if trade_price > mid:
            return 'BTO_MID'  # Likely buy
        else:
            return 'STO_MID'  # Likely sell


def main(date='2025-10-06', test_mode=False):
    """Main function for smart quote fetching"""

    print("="*80)
    print("SMART QUOTE FETCHER FOR TRADE DIRECTION")
    print("="*80)
    print(f"Date: {date}")
    print(f"Mode: {'TEST (100 quotes max)' if test_mode else 'FULL'}")

    # Load trades data
    trade_file = DATA_DIR / f"{date}.csv.gz"

    if not trade_file.exists():
        print(f"❌ Trade file not found: {trade_file}")
        return

    print(f"\n📊 Loading SPY trades from {trade_file.name}...")

    # Load SPY trades only
    spy_trades = []
    with gzip.open(trade_file, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=100000):
            spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY')]
            if len(spy_chunk) > 0:
                spy_trades.append(spy_chunk)

    if not spy_trades:
        print("   ❌ No SPY options found")
        return

    df = pd.concat(spy_trades, ignore_index=True)
    print(f"   ✅ Loaded {len(df):,} SPY option trades")

    # Parse timestamps
    df['sip_timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')

    # Initialize fetcher
    fetcher = SmartQuoteFetcher(date, test_mode)

    # =============================================================================
    # PRIORITIZATION STRATEGY
    # =============================================================================

    print(f"\n📊 Analyzing trade distribution...")

    # Count trades per ticker
    ticker_counts = df['ticker'].value_counts()

    # Categorize tickers
    high_volume = ticker_counts[ticker_counts > 1000]
    medium_volume = ticker_counts[(ticker_counts > 100) & (ticker_counts <= 1000)]
    low_volume = ticker_counts[ticker_counts <= 100]

    print(f"   High volume (>1000 trades): {len(high_volume)} tickers")
    print(f"   Medium volume (100-1000): {len(medium_volume)} tickers")
    print(f"   Low volume (<100): {len(low_volume)} tickers")

    # =============================================================================
    # BUILD QUOTE REQUEST LIST
    # =============================================================================

    print(f"\n🎯 Building smart quote request list...")

    quote_requests = []

    # Strategy 1: High-volume tickers - bucket by 30 seconds
    for ticker in high_volume.index:
        ticker_trades = df[df['ticker'] == ticker].copy()
        ticker_trades['time_bucket'] = ticker_trades['sip_timestamp'].dt.floor('30s')

        # Get unique 30-second buckets
        for bucket_time in ticker_trades['time_bucket'].unique():
            quote_requests.append({
                'ticker': ticker,
                'timestamp': bucket_time,
                'priority': 1,
                'trade_count': len(ticker_trades[ticker_trades['time_bucket'] == bucket_time])
            })

    # Strategy 2: Medium-volume tickers - bucket by minute
    for ticker in medium_volume.index[:50]:  # Top 50 medium volume
        ticker_trades = df[df['ticker'] == ticker].copy()
        ticker_trades['time_bucket'] = ticker_trades['sip_timestamp'].dt.floor('1min')

        for bucket_time in ticker_trades['time_bucket'].unique():
            quote_requests.append({
                'ticker': ticker,
                'timestamp': bucket_time,
                'priority': 2,
                'trade_count': len(ticker_trades[ticker_trades['time_bucket'] == bucket_time])
            })

    # Strategy 3: Sample of low-volume - one quote per ticker
    for ticker in low_volume.index[:20]:  # Sample 20
        ticker_trades = df[df['ticker'] == ticker]
        mid_time = ticker_trades['sip_timestamp'].iloc[len(ticker_trades)//2]

        quote_requests.append({
            'ticker': ticker,
            'timestamp': mid_time,
            'priority': 3,
            'trade_count': len(ticker_trades)
        })

    # Sort by priority
    quote_requests = sorted(quote_requests, key=lambda x: x['priority'])

    print(f"   Total quote requests: {len(quote_requests)}")
    print(f"   Estimated API calls: {len(quote_requests)}")
    print(f"   Estimated time: {len(quote_requests)/2/60:.1f} minutes at 2 calls/sec")

    if test_mode:
        quote_requests = quote_requests[:100]
        print(f"\n   🧪 TEST MODE: Processing only {len(quote_requests)} quotes")

    # =============================================================================
    # FETCH QUOTES
    # =============================================================================

    print(f"\n📥 Fetching quotes with smart prioritization...")

    start_time = time.time()

    for i, request in enumerate(quote_requests, 1):
        # Fetch quote
        quote = fetcher.fetch_quote_at_time(
            request['ticker'],
            request['timestamp']
        )

        # Progress update
        if i % 20 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(quote_requests) - i) / rate if rate > 0 else 0
            print(f"   Progress: {i}/{len(quote_requests)} ({i/len(quote_requests)*100:.1f}%) "
                  f"- Rate: {rate:.1f}/sec - ETA: {eta/60:.1f} min")

        # Checkpoint every 100 quotes
        if i % 100 == 0:
            fetcher.save_checkpoint()
            print(f"   💾 Checkpoint saved at {i} quotes")

        # Rate limiting
        time.sleep(0.5)  # 2 calls per second

    # Save final cache
    fetcher.save_final()

    # =============================================================================
    # ANALYZE TRADE DIRECTIONS
    # =============================================================================

    print(f"\n📊 Analyzing trade directions with fetched quotes...")

    # Apply quotes to classify trades
    trades_with_direction = []
    cache_hits = 0
    cache_misses = 0

    for _, trade in df.iterrows():
        ticker = trade['ticker']

        # Find appropriate time bucket
        timestamp = trade['sip_timestamp']
        bucket_30s = timestamp.floor('30s')
        bucket_1m = timestamp.floor('1min')

        # Try to find quote (check different bucket sizes)
        quote = None
        for ts in [timestamp, bucket_30s, bucket_1m]:
            cache_key = f"{ticker}_{ts}"
            if cache_key in fetcher.quote_cache:
                quote = fetcher.quote_cache[cache_key]
                cache_hits += 1
                break

        if not quote:
            cache_misses += 1
            continue

        # Classify trade
        direction = classify_trade(trade['price'], quote['bid'], quote['ask'])

        trades_with_direction.append({
            'ticker': ticker,
            'price': trade['price'],
            'size': trade['size'],
            'bid': quote['bid'],
            'ask': quote['ask'],
            'direction': direction
        })

    # Summarize results
    if trades_with_direction:
        direction_df = pd.DataFrame(trades_with_direction)

        print(f"\n   Quote coverage: {cache_hits}/{cache_hits+cache_misses} "
              f"({cache_hits/(cache_hits+cache_misses)*100:.1f}%)")

        # Direction breakdown
        direction_counts = direction_df['direction'].value_counts()
        print(f"\n   Trade Direction Classification:")
        for direction, count in direction_counts.items():
            pct = count / len(direction_df) * 100
            print(f"      {direction}: {count:,} ({pct:.1f}%)")

        # Save results
        results_file = CACHE_DIR / f"trade_directions_{date}.csv"
        direction_df.to_csv(results_file, index=False)
        print(f"\n   💾 Trade directions saved to: {results_file}")

    # Summary
    elapsed_time = time.time() - start_time

    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"   Total API calls: {fetcher.api_calls}")
    print(f"   Cached quotes: {len(fetcher.quote_cache)}")
    print(f"   Errors: {len(fetcher.errors)}")
    print(f"   Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"   Average rate: {fetcher.api_calls/elapsed_time:.2f} calls/sec")

    print(f"\n✅ Smart quotes saved to: {fetcher.cache_file}")

    return fetcher.cache_file


if __name__ == '__main__':
    import sys

    # Parse arguments
    date = sys.argv[1] if len(sys.argv) > 1 else '2025-10-06'
    test_mode = '--test' in sys.argv

    main(date, test_mode)