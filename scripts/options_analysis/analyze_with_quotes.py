#!/usr/bin/env python3
"""
Analyze options data using cached quotes
Calculate VIX term structure, dealer positioning, and trade directions
"""

import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path('trade_and_quote_data/data_management/flatfiles')
CACHE_DIR = Path('trade_and_quote_data/cache')

# Current prices
SPY_PRICE = 669
VIX_PRICE = 16.6


def load_cached_quotes(date):
    """Load cached quotes from ATM and smart fetchers"""
    quotes = {}

    # Load ATM quotes
    atm_file = CACHE_DIR / f"atm_quotes_{date}.json"
    if atm_file.exists():
        with open(atm_file, 'r') as f:
            atm_quotes = json.load(f)
            quotes.update(atm_quotes)
            print(f"   ✅ Loaded {len(atm_quotes)} ATM quotes")

    # Load smart quotes
    smart_file = CACHE_DIR / f"smart_quotes_{date}.json"
    if smart_file.exists():
        with open(smart_file, 'r') as f:
            smart_quotes = json.load(f)
            quotes.update(smart_quotes)
            print(f"   ✅ Loaded {len(smart_quotes)} smart quotes")

    return quotes


def calculate_vix_term_structure(trades_df, quotes):
    """Calculate VIX term structure from ATM options"""

    print("\n" + "="*80)
    print("VIX TERM STRUCTURE ANALYSIS")
    print("="*80)

    # Filter VIX options
    vix_trades = trades_df[trades_df['ticker'].str.startswith('O:VIX')].copy()

    if len(vix_trades) == 0:
        print("   ❌ No VIX options found")
        return None

    # Parse VIX options
    def parse_vix_ticker(ticker):
        try:
            parts = ticker.replace('O:VIX', '')
            year = int('20' + parts[0:2])
            month = int(parts[2:4])
            day = int(parts[4:6])
            expiry = datetime(year, month, day)
            option_type = parts[6].lower()
            strike = int(parts[7:15]) / 1000.0
            return pd.Series({'expiry': expiry, 'type': option_type, 'strike': strike})
        except:
            return pd.Series({'expiry': None, 'type': None, 'strike': None})

    vix_details = vix_trades['ticker'].apply(parse_vix_ticker)
    vix_trades = pd.concat([vix_trades, vix_details], axis=1)
    vix_trades = vix_trades.dropna(subset=['expiry', 'type', 'strike'])

    # Group by expiry and find ATM
    term_structure = []

    for expiry in sorted(vix_trades['expiry'].unique()):
        exp_data = vix_trades[vix_trades['expiry'] == expiry]
        calls = exp_data[exp_data['type'] == 'c']

        if len(calls) == 0:
            continue

        # Find ATM calls (within 20% of spot)
        atm_calls = calls[abs(calls['strike'] / VIX_PRICE - 1) <= 0.20]

        if len(atm_calls) > 0:
            # Look for quotes
            forward_vix = None

            for _, trade in atm_calls.iterrows():
                # Try to find quote for this trade
                cache_key = f"{trade['ticker']}_2025-10-06 13:00:00"
                if cache_key in quotes:
                    quote = quotes[cache_key]
                    mid_price = (quote['bid'] + quote['ask']) / 2
                    if mid_price > 0:
                        # Use strike as forward VIX approximation
                        forward_vix = trade['strike']
                        break

            if forward_vix:
                dte = (expiry - datetime(2025, 10, 6)).days
                term_structure.append({
                    'expiry': expiry,
                    'dte': dte,
                    'forward_vix': forward_vix,
                    'change_pct': (forward_vix / VIX_PRICE - 1) * 100
                })

    if term_structure:
        ts_df = pd.DataFrame(term_structure).sort_values('dte')

        print(f"\n📊 VIX TERM STRUCTURE (from ATM options):")
        print(f"   Current VIX: {VIX_PRICE:.1f}\n")
        print(f"   {'DTE':>4} | {'Expiry':^12} | {'Forward VIX':>11} | {'vs Spot':>8}")
        print("   " + "-"*50)

        for _, row in ts_df.head(10).iterrows():
            print(f"   {row['dte']:4.0f} | {row['expiry'].strftime('%Y-%m-%d'):^12} | "
                  f"{row['forward_vix']:11.1f} | {row['change_pct']:+7.1f}%")

        # Calculate slope
        if len(ts_df) >= 2:
            near_term = ts_df[ts_df['dte'] <= 30]['forward_vix'].mean()
            medium_term = ts_df[(ts_df['dte'] > 30) & (ts_df['dte'] <= 90)]['forward_vix'].mean()

            if not pd.isna(near_term) and not pd.isna(medium_term):
                slope = (medium_term - near_term) / near_term * 100
                print(f"\n   Term Structure Slope: {slope:+.1f}%")

                if slope > 10:
                    print("   → CONTANGO: Expecting higher volatility")
                elif slope < -10:
                    print("   → BACKWARDATION: Near-term stress")
                else:
                    print("   → FLAT: No strong directional bias")

        return ts_df
    else:
        print("   ⚠️ Insufficient quote data for term structure")
        return None


def analyze_trade_directions(trades_df, quotes):
    """Analyze trade directions using cached quotes"""

    print("\n" + "="*80)
    print("TRADE DIRECTION ANALYSIS")
    print("="*80)

    # Filter SPY options
    spy_trades = trades_df[trades_df['ticker'].str.startswith('O:SPY')].copy()

    directions = []
    matched = 0
    unmatched = 0

    for _, trade in spy_trades.iterrows():
        # Try different time buckets
        ts = pd.Timestamp(trade['sip_timestamp'])
        found_quote = False

        # Try exact time, 30s bucket, 1min bucket, fixed time
        for timestamp in [ts, ts.floor('30s'), ts.floor('1min'),
                         f"2025-10-06 13:00:00"]:
            cache_key = f"{trade['ticker']}_{timestamp}"
            if cache_key in quotes:
                quote = quotes[cache_key]
                bid = quote['bid']
                ask = quote['ask']

                if bid > 0 and ask > 0:
                    # Classify trade
                    if trade['price'] >= ask * 0.99:
                        direction = 'BTO'
                    elif trade['price'] <= bid * 1.01:
                        direction = 'STO'
                    else:
                        mid = (bid + ask) / 2
                        direction = 'MID_BUY' if trade['price'] > mid else 'MID_SELL'

                    directions.append({
                        'ticker': trade['ticker'],
                        'size': trade['size'],
                        'price': trade['price'],
                        'bid': bid,
                        'ask': ask,
                        'direction': direction
                    })

                    matched += 1
                    found_quote = True
                    break

        if not found_quote:
            unmatched += 1

    print(f"\n📊 Quote Coverage:")
    print(f"   Matched: {matched:,} trades")
    print(f"   Unmatched: {unmatched:,} trades")
    print(f"   Coverage: {matched/(matched+unmatched)*100:.1f}%")

    if directions:
        dir_df = pd.DataFrame(directions)

        # Summarize by direction
        print(f"\n📊 Trade Direction Breakdown:")
        for direction in ['BTO', 'STO', 'MID_BUY', 'MID_SELL']:
            dir_trades = dir_df[dir_df['direction'] == direction]
            if len(dir_trades) > 0:
                volume = dir_trades['size'].sum()
                pct = len(dir_trades) / len(dir_df) * 100
                print(f"   {direction:8s}: {len(dir_trades):8,} trades, "
                      f"{volume:10,.0f} contracts ({pct:5.1f}%)")

        # Calculate dealer positioning
        customer_buy = dir_df[dir_df['direction'].isin(['BTO', 'MID_BUY'])]['size'].sum()
        customer_sell = dir_df[dir_df['direction'].isin(['STO', 'MID_SELL'])]['size'].sum()

        print(f"\n📊 Net Customer Flow:")
        print(f"   Customer Buying: {customer_buy:,.0f} contracts")
        print(f"   Customer Selling: {customer_sell:,.0f} contracts")
        print(f"   Net: {customer_buy - customer_sell:+,.0f} contracts")

        if customer_buy > customer_sell:
            print("   → Dealers SHORT (negative gamma)")
        else:
            print("   → Dealers LONG (positive gamma)")

        return dir_df
    else:
        print("   ⚠️ No trades matched with quotes")
        return None


def main(date='2025-10-06'):
    """Main analysis function"""

    print("="*80)
    print("OPTIONS ANALYSIS WITH CACHED QUOTES")
    print("="*80)
    print(f"Date: {date}")

    # Load cached quotes
    print("\n📂 Loading cached quotes...")
    quotes = load_cached_quotes(date)

    if not quotes:
        print("   ❌ No cached quotes found. Run fetch scripts first.")
        return

    print(f"   Total cached quotes: {len(quotes)}")

    # Load trades data
    trade_file = DATA_DIR / f"{date}.csv.gz"
    print(f"\n📊 Loading trades from {trade_file.name}...")

    trades_list = []
    with gzip.open(trade_file, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=100000):
            options = chunk[(chunk['ticker'].str.startswith('O:SPY')) |
                          (chunk['ticker'].str.startswith('O:VIX'))]
            if len(options) > 0:
                trades_list.append(options)

    if not trades_list:
        print("   ❌ No options trades found")
        return

    trades_df = pd.concat(trades_list, ignore_index=True)
    trades_df['sip_timestamp'] = pd.to_datetime(trades_df['sip_timestamp'], unit='ns')
    print(f"   ✅ Loaded {len(trades_df):,} option trades")

    # Analyze VIX term structure
    vix_ts = calculate_vix_term_structure(trades_df, quotes)

    # Analyze trade directions
    directions = analyze_trade_directions(trades_df, quotes)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    if vix_ts is not None and len(vix_ts) > 0:
        avg_forward = vix_ts['forward_vix'].mean()
        print(f"\n📊 VIX Term Structure:")
        print(f"   Current VIX: {VIX_PRICE:.1f}")
        print(f"   Average Forward VIX: {avg_forward:.1f}")
        print(f"   Implied Change: {(avg_forward/VIX_PRICE - 1)*100:+.1f}%")

    if directions is not None:
        print(f"\n📊 Trade Direction Summary:")
        bto_pct = len(directions[directions['direction'] == 'BTO']) / len(directions) * 100
        sto_pct = len(directions[directions['direction'] == 'STO']) / len(directions) * 100
        print(f"   Clear Buys (BTO): {bto_pct:.1f}%")
        print(f"   Clear Sells (STO): {sto_pct:.1f}%")

    print("\n✅ Analysis complete")


if __name__ == '__main__':
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else '2025-10-06'
    main(date)