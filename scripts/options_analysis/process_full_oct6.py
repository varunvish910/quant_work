#!/usr/bin/env python3
"""
Process FULL Oct 6 dataset - all 769k SPY trades
Using ±30 second quote window for 100% coverage
Estimated time: 87.8 hours (~3.7 days)
"""

from polygon import RESTClient
import pandas as pd
import gzip
from pathlib import Path
import time
import json
from datetime import datetime

# Configuration
API_KEY = 'OWgBGzgOAzjd6Ieuml6iJakY1yA9npku'
DATE = '2025-10-06'
OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
TRADES_FILE = OUTPUT_DIR / f"{DATE}.csv.gz"
PROGRESS_FILE = OUTPUT_DIR / f"{DATE}_progress.json"
RESULTS_FILE = OUTPUT_DIR / f"{DATE}_classified_trades.csv"
SUMMARY_FILE = OUTPUT_DIR / f"{DATE}_dealer_summary.json"

print("="*80)
print(f"PROCESSING FULL OCT 6 DATASET - ALL SPY TRADES")
print(f"Date: {DATE}")
print(f"Estimated time: ~87.8 hours (~3.7 days)")
print("="*80)

# Load trades
print(f"\n📊 Step 1: Loading ALL SPY trades...")
spy_trades = []
total_rows = 0

with gzip.open(TRADES_FILE, 'rt') as f:
    for chunk_num, chunk in enumerate(pd.read_csv(f, chunksize=100000), 1):
        total_rows += len(chunk)
        spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)

        if chunk_num % 10 == 0:
            current_spy = sum(len(df) for df in spy_trades)
            print(f"   Processed {total_rows:,} total rows, found {current_spy:,} SPY trades...")

trades = pd.concat(spy_trades, ignore_index=True)
print(f"\n   ✅ Loaded {len(trades):,} SPY trades from {total_rows:,} total rows")

# Parse option type
def parse_option_type(ticker):
    try:
        if not ticker.startswith('O:SPY'):
            return None
        parts = ticker[5:]
        if len(parts) >= 15:
            return parts[6].lower()
    except:
        pass
    return None

trades['option_type'] = trades['ticker'].apply(parse_option_type)

# Check for existing progress
start_idx = 0
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
        start_idx = progress.get('last_completed_idx', 0) + 1
        print(f"\n   📝 Resuming from trade {start_idx:,} (found checkpoint)")

# Initialize client
client = RESTClient(API_KEY)

# Process trades
print(f"\n📡 Step 2: Processing trades {start_idx:,} to {len(trades):,}...")
print(f"   Using ±30 second time window")
print(f"   Progress will be saved every 1000 trades")
print(f"   You can stop/restart anytime - progress will resume\n")

results = []
api_calls = 0
no_quotes = 0
start_time = time.time()
last_save_time = time.time()

# Load existing results if resuming
if start_idx > 0 and RESULTS_FILE.exists():
    print(f"   📂 Loading existing results from {RESULTS_FILE}...")
    existing_results = pd.read_csv(RESULTS_FILE)
    results = existing_results.to_dict('records')
    print(f"   ✅ Loaded {len(results):,} existing results\n")

for idx in range(start_idx, len(trades)):
    trade = trades.iloc[idx]

    ticker = trade['ticker']
    price = trade['price']
    timestamp_ns = trade['sip_timestamp']
    size = trade['size']
    option_type = trade['option_type']

    # Time window: ±30 seconds
    window_start = timestamp_ns - 30_000_000_000
    window_end = timestamp_ns + 30_000_000_000

    try:
        # Fetch quotes
        quotes = []
        for quote in client.list_quotes(
            ticker=ticker,
            timestamp_gte=window_start,
            timestamp_lte=window_end,
            order="asc",
            limit=50,
            sort="timestamp",
        ):
            quotes.append(quote)

        api_calls += 1

        # Find closest quote before trade
        pre_trade_quotes = [q for q in quotes if q.sip_timestamp <= timestamp_ns]

        if pre_trade_quotes:
            closest_quote = pre_trade_quotes[-1]
            bid = closest_quote.bid_price
            ask = closest_quote.ask_price
            mid = (bid + ask) / 2

            # Classify
            if price >= ask * 0.99:
                direction = 'BUY'
            elif price <= bid * 1.01:
                direction = 'SELL'
            elif price > mid:
                direction = 'BUY'
            else:
                direction = 'SELL'

            results.append({
                'ticker': ticker,
                'price': price,
                'size': size,
                'option_type': option_type,
                'bid': bid,
                'ask': ask,
                'direction': direction,
                'timestamp_ns': timestamp_ns
            })
        else:
            no_quotes += 1
            results.append({
                'ticker': ticker,
                'price': price,
                'size': size,
                'option_type': option_type,
                'bid': 0,
                'ask': 0,
                'direction': 'UNKNOWN',
                'timestamp_ns': timestamp_ns
            })

    except Exception as e:
        no_quotes += 1
        results.append({
            'ticker': ticker,
            'price': price,
            'size': size,
            'option_type': option_type,
            'bid': 0,
            'ask': 0,
            'direction': 'UNKNOWN',
            'timestamp_ns': timestamp_ns
        })
        if idx < 10:
            print(f"   ⚠️  Error on trade {idx+1}: {e}")

    # Progress updates every 1000 trades
    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - start_time
        total_processed = idx + 1 - start_idx
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining_trades = len(trades) - (idx + 1)
        eta_seconds = remaining_trades / rate if rate > 0 else 0
        eta_hours = eta_seconds / 3600

        completed_pct = ((idx + 1) / len(trades)) * 100

        print(f"   [{datetime.now().strftime('%H:%M:%S')}] Progress: {idx+1:,}/{len(trades):,} ({completed_pct:.1f}%)")
        print(f"      Rate: {rate:.1f} trades/sec | API calls: {api_calls:,} | No quotes: {no_quotes}")
        print(f"      ETA: {eta_hours:.1f} hours remaining\n")

    # Save progress every 1000 trades
    if (idx + 1) % 1000 == 0:
        # Save results
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

        # Save progress checkpoint
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                'last_completed_idx': idx,
                'total_processed': idx + 1,
                'api_calls': api_calls,
                'no_quotes': no_quotes,
                'timestamp': datetime.now().isoformat()
            }, f)

        last_save_time = time.time()

    # Rate limiting disabled for max speed
    # if api_calls % 5 == 0:
    #     time.sleep(1)

# Final save
elapsed_total = time.time() - start_time

print(f"\n   ✅ Processing complete!")
print(f"      Total time: {elapsed_total/3600:.1f} hours")
print(f"      Total processed: {len(trades):,} trades")
print(f"      Rate: {len(trades)/elapsed_total:.1f} trades/sec")
print(f"      API calls: {api_calls:,}")
print(f"      No quotes: {no_quotes} ({no_quotes/len(trades)*100:.2f}%)")

# Save final results
print(f"\n📊 Step 3: Analyzing and saving results...")
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False)
print(f"   ✅ Saved classified trades to: {RESULTS_FILE}")

# Analyze by option type
calls = results_df[results_df['option_type'] == 'c']
puts = results_df[results_df['option_type'] == 'p']

call_buy_vol = calls[calls['direction'] == 'BUY']['size'].sum()
call_sell_vol = calls[calls['direction'] == 'SELL']['size'].sum()
put_buy_vol = puts[puts['direction'] == 'BUY']['size'].sum()
put_sell_vol = puts[puts['direction'] == 'SELL']['size'].sum()

net_call_buying = call_buy_vol - call_sell_vol
net_put_buying = put_buy_vol - put_sell_vol

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

# Create summary
summary = {
    'date': DATE,
    'total_trades': len(trades),
    'api_calls': api_calls,
    'processing_time_hours': elapsed_total / 3600,
    'call_buy_volume': int(call_buy_vol),
    'call_sell_volume': int(call_sell_vol),
    'put_buy_volume': int(put_buy_vol),
    'put_sell_volume': int(put_sell_vol),
    'net_call_buying': int(net_call_buying),
    'net_put_buying': int(net_put_buying),
    'dealer_short_calls': int(net_call_buying),
    'dealer_short_puts': int(net_put_buying),
    'gamma_status': gamma_status,
    'risk_level': risk_level,
    'no_quote_rate': no_quotes / len(trades) * 100
}

with open(SUMMARY_FILE, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"   ✅ Saved summary to: {SUMMARY_FILE}")

# Print summary
print(f"\n📈 DEALER POSITIONING SUMMARY")
print(f"   {'='*60}")
print(f"\n   Call Flow:")
print(f"      Customer BUY:  {call_buy_vol:>15,.0f} contracts")
print(f"      Customer SELL: {call_sell_vol:>15,.0f} contracts")
print(f"      Net buying:    {net_call_buying:>15,.0f} contracts")
print(f"\n   Put Flow:")
print(f"      Customer BUY:  {put_buy_vol:>15,.0f} contracts")
print(f"      Customer SELL: {put_sell_vol:>15,.0f} contracts")
print(f"      Net buying:    {net_put_buying:>15,.0f} contracts")
print(f"\n   Dealer Positioning:")
print(f"      Dealer SHORT calls: {net_call_buying:>15,.0f} contracts")
print(f"      Dealer SHORT puts:  {net_put_buying:>15,.0f} contracts")
print(f"\n   Status: {gamma_status}")
print(f"   Risk Level: {risk_level}")

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*80}")

# Cleanup progress file
if PROGRESS_FILE.exists():
    PROGRESS_FILE.unlink()
    print(f"\n🗑️  Cleaned up progress checkpoint file")
