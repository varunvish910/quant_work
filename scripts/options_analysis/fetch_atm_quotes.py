#!/usr/bin/env python3
"""
Fetch ATM quotes only - Fast approach for VIX term structure
Only ~600 API calls instead of 769,207
Runtime: ~5 minutes
"""

import pandas as pd
import numpy as np
import gzip
import requests
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
POLYGON_API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
DATA_DIR = Path('trade_and_quote_data/data_management/flatfiles')
CACHE_DIR = Path('trade_and_quote_data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Current approximate prices
SPY_PRICE = 669
VIX_PRICE = 16.6


class ATMQuoteFetcher:
    """Fetch quotes only for ATM options"""

    def __init__(self, date, test_mode=False):
        self.date = date
        self.api_key = POLYGON_API_KEY
        self.test_mode = test_mode
        self.api_calls = 0
        self.quote_cache = {}
        self.errors = []

        # Cache file
        self.cache_file = CACHE_DIR / f"atm_quotes_{date}.json"

        # Load existing cache if available
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.quote_cache = json.load(f)
                print(f"   📂 Loaded {len(self.quote_cache)} cached quotes")

    def is_atm(self, ticker, underlying_price):
        """Check if an option is ATM (within ±2% of spot)"""
        try:
            # Parse strike from ticker
            if ticker.startswith('O:SPY'):
                # Format: O:SPY251006C00672000
                parts = ticker.replace('O:SPY', '')
                # Skip date (6 chars), get type (1 char), then strike
                strike_str = parts[7:]
                strike = int(strike_str) / 1000

                # Check if within ±2% of spot
                return abs(strike / underlying_price - 1) <= 0.02

            elif ticker.startswith('O:VIX'):
                # Format: O:VIX251022C00010000
                parts = ticker.replace('O:VIX', '')
                strike_str = parts[7:]
                strike = int(strike_str) / 1000

                # For VIX, use ±20% since it's more volatile
                return abs(strike / underlying_price - 1) <= 0.20

        except Exception as e:
            return False

        return False

    def fetch_quote(self, ticker, timestamp):
        """Fetch a single quote from Polygon API"""

        # Check cache first
        cache_key = f"{ticker}_{timestamp}"
        if cache_key in self.quote_cache:
            return self.quote_cache[cache_key]

        # Format timestamp for API
        ts_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d')

        # API endpoint for quotes
        url = f"https://api.polygon.io/v3/quotes/{ticker}"

        params = {
            'timestamp': ts_str,
            'limit': 1,
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
                        'ask_size': quote.get('ask_size', 0)
                    }

                    # Cache the result
                    self.quote_cache[cache_key] = result
                    return result

            # Log error but continue
            self.errors.append(f"Failed to fetch {ticker}: {response.status_code}")

        except Exception as e:
            self.errors.append(f"Error fetching {ticker}: {str(e)}")

        return {'bid': 0, 'ask': 0, 'bid_size': 0, 'ask_size': 0}

    def save_cache(self):
        """Save quote cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.quote_cache, f)
        print(f"   💾 Saved {len(self.quote_cache)} quotes to cache")


def main(date='2025-10-06', test_mode=False):
    """Main function to fetch ATM quotes"""

    print("="*80)
    print("ATM QUOTE FETCHER")
    print("="*80)
    print(f"Date: {date}")
    print(f"Mode: {'TEST (10 quotes max)' if test_mode else 'FULL'}")

    # Load trades data
    trade_file = DATA_DIR / f"{date}.csv.gz"

    if not trade_file.exists():
        print(f"❌ Trade file not found: {trade_file}")
        return

    print(f"\n📊 Loading trades from {trade_file.name}...")

    # Load SPY and VIX trades
    all_trades = []
    with gzip.open(trade_file, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=100000):
            # Filter for SPY and VIX options
            options = chunk[(chunk['ticker'].str.startswith('O:SPY')) |
                          (chunk['ticker'].str.startswith('O:VIX'))]
            if len(options) > 0:
                all_trades.append(options)

    if not all_trades:
        print("   ❌ No SPY/VIX options found")
        return

    df = pd.concat(all_trades, ignore_index=True)
    print(f"   ✅ Loaded {len(df):,} SPY/VIX option trades")

    # Initialize fetcher
    fetcher = ATMQuoteFetcher(date, test_mode)

    # Filter for ATM options only
    print(f"\n🎯 Filtering for ATM options...")

    spy_trades = df[df['ticker'].str.startswith('O:SPY')]
    vix_trades = df[df['ticker'].str.startswith('O:VIX')]

    # Find unique ATM tickers
    atm_tickers = set()

    # SPY ATM tickers
    for ticker in spy_trades['ticker'].unique():
        if fetcher.is_atm(ticker, SPY_PRICE):
            atm_tickers.add(ticker)

    # VIX ATM tickers
    for ticker in vix_trades['ticker'].unique():
        if fetcher.is_atm(ticker, VIX_PRICE):
            atm_tickers.add(ticker)

    print(f"   ✅ Found {len(atm_tickers)} unique ATM tickers")

    # Get unique SPY and VIX ATM counts
    spy_atm = [t for t in atm_tickers if t.startswith('O:SPY')]
    vix_atm = [t for t in atm_tickers if t.startswith('O:VIX')]
    print(f"      SPY ATM: {len(spy_atm)} tickers")
    print(f"      VIX ATM: {len(vix_atm)} tickers")

    if test_mode:
        # In test mode, only process first 10
        atm_tickers = list(atm_tickers)[:10]
        print(f"\n   🧪 TEST MODE: Processing only {len(atm_tickers)} tickers")

    # Fetch quotes for ATM tickers
    print(f"\n📥 Fetching quotes for ATM options...")
    print(f"   Estimated API calls: {len(atm_tickers)}")
    print(f"   Estimated time: {len(atm_tickers)/2:.1f} seconds at 2 calls/sec")

    start_time = time.time()

    # Use a representative timestamp (mid-day)
    quote_timestamp = f"{date} 13:00:00"

    for i, ticker in enumerate(atm_tickers, 1):
        # Fetch quote
        quote = fetcher.fetch_quote(ticker, quote_timestamp)

        # Progress update
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(atm_tickers) - i) / rate if rate > 0 else 0
            print(f"   Progress: {i}/{len(atm_tickers)} ({i/len(atm_tickers)*100:.1f}%) "
                  f"- Rate: {rate:.1f}/sec - ETA: {eta:.0f}s")

        # Rate limiting (2 calls per second)
        time.sleep(0.5)

    # Save cache
    fetcher.save_cache()

    # Summary
    elapsed_time = time.time() - start_time

    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"   Total API calls: {fetcher.api_calls}")
    print(f"   Cached quotes: {len(fetcher.quote_cache)}")
    print(f"   Errors: {len(fetcher.errors)}")
    print(f"   Time elapsed: {elapsed_time:.1f} seconds")
    print(f"   Average rate: {fetcher.api_calls/elapsed_time:.2f} calls/sec")

    if fetcher.errors and len(fetcher.errors) <= 10:
        print(f"\n   ⚠️ Errors encountered:")
        for error in fetcher.errors[:10]:
            print(f"      {error}")

    print(f"\n✅ ATM quotes saved to: {fetcher.cache_file}")

    return fetcher.cache_file


if __name__ == '__main__':
    import sys

    # Parse arguments
    date = sys.argv[1] if len(sys.argv) > 1 else '2025-10-06'
    test_mode = '--test' in sys.argv

    main(date, test_mode)