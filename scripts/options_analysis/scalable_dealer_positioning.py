#!/usr/bin/env python3
"""
SCALABLE dealer positioning analysis using smart quote caching

Key optimization:
- Download trades flat file (60 MB, fast)
- Fetch quotes via API ONCE per unique ticker (not per trade)
- Cache quotes in memory
- 769k trades → 4,818 unique tickers → 4,818 API calls
- Time: ~16 minutes per day (vs 106 min flat file download)

For 1 year: 252 days × 16 min = 67 hours = 3 days (vs 27 days!)
"""

import pandas as pd
import gzip
from pathlib import Path
import requests
from collections import defaultdict
import time
from datetime import datetime
import json

# Configuration
POLYGON_API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
CACHE_DIR = Path('trade_and_quote_data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class QuoteCache:
    """
    Smart quote cache to minimize API calls

    Strategy:
    - Group trades by ticker
    - Fetch quote ONCE per ticker (use mid-day timestamp)
    - Apply same quote to all trades of that ticker
    - Assumption: Bid/ask doesn't change dramatically within a day for classification purposes
    """

    def __init__(self, date, api_key):
        self.date = date
        self.api_key = api_key
        self.cache = {}  # {ticker: {'bid': X, 'ask': Y}}
        self.api_calls = 0
        self.cache_hits = 0

        # Load from disk cache if exists
        self.cache_file = CACHE_DIR / f"quotes_cache_{date}.json"
        if self.cache_file.exists():
            print(f"Loading quote cache from {self.cache_file}...")
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
            print(f"Loaded {len(self.cache)} cached quotes")

    def get_quote(self, ticker, timestamp_ns):
        """
        Get quote for ticker at timestamp
        Uses cache if available, otherwise fetches from API
        """
        # Check cache first
        if ticker in self.cache:
            self.cache_hits += 1
            return self.cache[ticker]

        # Fetch from API
        self.api_calls += 1

        # Rate limiting: 5 req/sec
        if self.api_calls % 5 == 0:
            time.sleep(1)

        # Convert nanosecond timestamp to milliseconds
        timestamp_ms = int(timestamp_ns / 1_000_000)

        url = f"https://api.polygon.io/v3/quotes/{ticker}"
        params = {
            'timestamp': timestamp_ms,
            'limit': 1,
            'order': 'desc',
            'apiKey': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                quote = {
                    'bid': result.get('bid_price', 0),
                    'ask': result.get('ask_price', 0),
                    'bid_size': result.get('bid_size', 0),
                    'ask_size': result.get('ask_size', 0)
                }

                # Cache it
                self.cache[ticker] = quote

                # Print progress
                if self.api_calls % 100 == 0:
                    print(f"   API calls: {self.api_calls}, Cache hits: {self.cache_hits}, "
                          f"Hit rate: {self.cache_hits/(self.api_calls+self.cache_hits)*100:.1f}%")

                return quote
            else:
                return {'bid': 0, 'ask': 0, 'bid_size': 0, 'ask_size': 0}

        except Exception as e:
            print(f"   Error fetching quote for {ticker}: {e}")
            return {'bid': 0, 'ask': 0, 'bid_size': 0, 'ask_size': 0}

    def save_cache(self):
        """Save cache to disk for reuse"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print(f"\n💾 Saved {len(self.cache)} quotes to cache: {self.cache_file}")


def classify_trade_direction(price, bid, ask):
    """Classify trade as BUY or SELL based on price vs bid/ask"""
    if bid <= 0 or ask <= 0:
        return 'UNKNOWN'

    spread = ask - bid
    mid = (bid + ask) / 2

    # Tolerance based on spread width
    if spread > mid * 0.1:
        tolerance = 0.05
    else:
        tolerance = 0.01

    # Classify
    if price >= ask * (1 - tolerance):
        return 'BUY'
    elif price <= bid * (1 + tolerance):
        return 'SELL'
    elif price > mid:
        return 'BUY'
    else:
        return 'SELL'


def parse_option_type(ticker):
    """Extract option type (c/p) from ticker"""
    try:
        if not ticker.startswith('O:SPY'):
            return None
        parts = ticker[5:]
        if len(parts) >= 15:
            return parts[6].lower()
    except:
        pass
    return None


def analyze_dealer_positioning(date):
    """
    Analyze dealer positioning for a single day using smart quote caching

    Args:
        date: Date string (YYYY-MM-DD)

    Returns:
        dict: Summary metrics
    """
    print("="*80)
    print(f"SCALABLE DEALER POSITIONING ANALYSIS - {date}")
    print("Using smart quote caching (API-based)")
    print("="*80)

    trades_file = OUTPUT_DIR / f"{date}.csv.gz"

    # Step 1: Load trades
    print(f"\n📊 Step 1: Loading SPY trades from {trades_file}...")

    spy_trades = []
    with gzip.open(trades_file, 'rt') as f:
        chunk_num = 0
        for chunk in pd.read_csv(f, chunksize=100000):
            chunk_num += 1
            spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
            if len(spy_chunk) > 0:
                spy_trades.append(spy_chunk)

    trades = pd.concat(spy_trades, ignore_index=True)
    print(f"   ✅ Loaded {len(trades):,} SPY trades")

    # Step 2: Get unique tickers
    unique_tickers = trades['ticker'].unique()
    print(f"\n🎯 Step 2: Found {len(unique_tickers):,} unique SPY option tickers")
    print(f"   Strategy: Fetch 1 quote per ticker (not per trade)")
    print(f"   Expected API calls: ~{len(unique_tickers):,}")
    print(f"   Estimated time: ~{len(unique_tickers) / 5 / 60:.1f} minutes")

    # Step 3: Initialize quote cache
    print(f"\n🗄️  Step 3: Initializing quote cache...")
    quote_cache = QuoteCache(date, POLYGON_API_KEY)

    # Step 4: Fetch quotes for each unique ticker (use first trade timestamp as reference)
    print(f"\n📡 Step 4: Fetching quotes (with caching)...")

    ticker_quotes = {}
    for idx, ticker in enumerate(unique_tickers):
        # Get first trade timestamp for this ticker
        first_trade = trades[trades['ticker'] == ticker].iloc[0]
        timestamp_ns = first_trade['sip_timestamp']

        # Get quote (cached or API)
        quote = quote_cache.get_quote(ticker, timestamp_ns)
        ticker_quotes[ticker] = quote

        # Progress
        if (idx + 1) % 500 == 0:
            print(f"   Progress: {idx+1:,} / {len(unique_tickers):,} tickers "
                  f"({(idx+1)/len(unique_tickers)*100:.1f}%)")

    # Save cache
    quote_cache.save_cache()

    print(f"\n   ✅ Quote fetching complete!")
    print(f"      API calls made: {quote_cache.api_calls:,}")
    print(f"      Cache hits: {quote_cache.cache_hits:,}")
    print(f"      Hit rate: {quote_cache.cache_hits/(quote_cache.api_calls+quote_cache.cache_hits)*100:.1f}%")

    # Step 5: Match quotes to trades and classify
    print(f"\n🎯 Step 5: Classifying trade direction...")

    trades['bid'] = trades['ticker'].map(lambda t: ticker_quotes.get(t, {}).get('bid', 0))
    trades['ask'] = trades['ticker'].map(lambda t: ticker_quotes.get(t, {}).get('ask', 0))
    trades['direction'] = trades.apply(
        lambda row: classify_trade_direction(row['price'], row['bid'], row['ask']),
        axis=1
    )
    trades['option_type'] = trades['ticker'].apply(parse_option_type)

    # Direction distribution
    direction_counts = trades['direction'].value_counts()
    print(f"\n   Trade Direction Distribution:")
    for direction, count in direction_counts.items():
        pct = count / len(trades) * 100
        print(f"      {direction:8s}: {count:>10,} ({pct:>5.1f}%)")

    # Step 6: Calculate dealer positioning
    print(f"\n⚡ Step 6: Calculating dealer positioning...")

    calls = trades[trades['option_type'] == 'c']
    puts = trades[trades['option_type'] == 'p']

    call_buy_vol = calls[calls['direction'] == 'BUY']['size'].sum()
    call_sell_vol = calls[calls['direction'] == 'SELL']['size'].sum()
    put_buy_vol = puts[puts['direction'] == 'BUY']['size'].sum()
    put_sell_vol = puts[puts['direction'] == 'SELL']['size'].sum()

    net_call_buying = call_buy_vol - call_sell_vol
    net_put_buying = put_buy_vol - put_sell_vol

    print(f"\n   Customer Flow:")
    print(f"      Call BUY:  {call_buy_vol:>15,.0f}")
    print(f"      Call SELL: {call_sell_vol:>15,.0f}")
    print(f"      Net call buying: {net_call_buying:>12,.0f}")
    print(f"")
    print(f"      Put BUY:   {put_buy_vol:>15,.0f}")
    print(f"      Put SELL:  {put_sell_vol:>15,.0f}")
    print(f"      Net put buying:  {net_put_buying:>12,.0f}")

    # Determine gamma status
    if net_call_buying > 0 and net_put_buying > 0:
        gamma_status = "NET SHORT GAMMA"
        risk_level = "HIGH"
    elif net_call_buying < 0 and net_put_buying < 0:
        gamma_status = "NET LONG GAMMA"
        risk_level = "LOW"
    else:
        gamma_status = "MIXED GAMMA"
        risk_level = "MODERATE"

    print(f"\n   Dealer Positioning:")
    print(f"      Dealer SHORT calls: {net_call_buying:>15,.0f} contracts")
    print(f"      Dealer SHORT puts:  {net_put_buying:>15,.0f} contracts")
    print(f"")
    print(f"      Gamma Status: {gamma_status}")
    print(f"      Risk Level: 🔴 {risk_level}" if risk_level == "HIGH" else f"      Risk Level: 🟢 {risk_level}")

    # Step 7: Save summary
    summary = {
        'date': date,
        'total_trades': len(trades),
        'unique_tickers': len(unique_tickers),
        'api_calls': quote_cache.api_calls,
        'cache_hits': quote_cache.cache_hits,
        'call_buy_volume': int(call_buy_vol),
        'call_sell_volume': int(call_sell_vol),
        'put_buy_volume': int(put_buy_vol),
        'put_sell_volume': int(put_sell_vol),
        'net_call_buying': int(net_call_buying),
        'net_put_buying': int(net_put_buying),
        'gamma_status': gamma_status,
        'risk_level': risk_level
    }

    summary_file = OUTPUT_DIR / f"{date}_dealer_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved summary to: {summary_file}")

    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print(f"{'='*80}")

    return summary


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scalable_dealer_positioning.py YYYY-MM-DD")
        sys.exit(1)

    date = sys.argv[1]
    summary = analyze_dealer_positioning(date)
