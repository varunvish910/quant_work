#!/usr/bin/env python3
"""
Analyze SPY trades to understand quote requirements
Goal: Find ways to reduce API calls
"""

import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime

print("="*80)
print("SPY OPTIONS TRADE & QUOTE STATISTICS")
print("="*80)

# Load October 6 data
data_file = Path('trade_and_quote_data/data_management/flatfiles/2025-10-06.csv.gz')

print(f"\n📊 Loading SPY options trades from {data_file.name}...")

spy_trades = []
with gzip.open(data_file, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY')]
        if len(spy_chunk) > 0:
            spy_trades.append(spy_chunk)

df = pd.concat(spy_trades, ignore_index=True)
print(f"   ✅ Loaded {len(df):,} SPY options trades")

# Convert timestamp
df['sip_timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')

# =============================================================================
# BASIC STATISTICS
# =============================================================================

print("\n" + "="*80)
print("TRADE STATISTICS")
print("="*80)

total_trades = len(df)
unique_tickers = df['ticker'].nunique()

print(f"\n📊 OCTOBER 6, 2025 SUMMARY:")
print(f"   Total SPY option trades: {total_trades:,}")
print(f"   Unique tickers traded: {unique_tickers:,}")
print(f"   Average trades per ticker: {total_trades/unique_tickers:.1f}")

# =============================================================================
# TIME DISTRIBUTION
# =============================================================================

print("\n📈 TIME DISTRIBUTION:")

# Add hour column
df['hour'] = df['sip_timestamp'].dt.hour
df['minute'] = df['sip_timestamp'].dt.minute
df['second'] = df['sip_timestamp'].dt.second

# Trades per hour
hourly_trades = df.groupby('hour').size()
print(f"\n   Trades by Hour:")
for hour in range(9, 17):
    if hour in hourly_trades.index:
        count = hourly_trades[hour]
        pct = count/total_trades*100
        print(f"   {hour:02d}:00 - {hour:02d}:59  {count:8,} ({pct:5.1f}%)")

# Peak minute
df['hour_minute'] = df['hour'] * 60 + df['minute']
minute_counts = df.groupby('hour_minute').size()
peak_minute = minute_counts.idxmax()
peak_hour = peak_minute // 60
peak_min = peak_minute % 60
print(f"\n   Peak trading minute: {peak_hour:02d}:{peak_min:02d} ({minute_counts.max():,} trades)")

# =============================================================================
# TICKER CONCENTRATION
# =============================================================================

print("\n" + "="*80)
print("TICKER CONCENTRATION")
print("="*80)

ticker_counts = df['ticker'].value_counts()

print(f"\n📊 TICKER DISTRIBUTION:")
print(f"   Top 10% of tickers account for: {ticker_counts.head(int(unique_tickers*0.1)).sum()/total_trades*100:.1f}% of trades")
print(f"   Top 20% of tickers account for: {ticker_counts.head(int(unique_tickers*0.2)).sum()/total_trades*100:.1f}% of trades")
print(f"   Top 50% of tickers account for: {ticker_counts.head(int(unique_tickers*0.5)).sum()/total_trades*100:.1f}% of trades")

print(f"\n🎯 TOP 10 MOST TRADED TICKERS:")
for ticker, count in ticker_counts.head(10).items():
    pct = count/total_trades*100
    print(f"   {ticker}: {count:6,} trades ({pct:4.1f}%)")

# =============================================================================
# QUOTE REQUIREMENTS ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("QUOTE REQUIREMENTS ANALYSIS")
print("="*80)

print(f"\n📊 IF WE NEED ONE QUOTE PER TRADE:")
print(f"   Total quotes needed: {total_trades:,}")
print(f"   At 1 call/sec: {total_trades/3600:.1f} hours")
print(f"   At 2 calls/sec: {total_trades/7200:.1f} hours")
print(f"   At 5 calls/sec: {total_trades/18000:.1f} hours")

print(f"\n📊 IF WE USE ONE QUOTE PER TICKER:")
print(f"   Total quotes needed: {unique_tickers:,}")
print(f"   At 1 call/sec: {unique_tickers/60:.1f} minutes")
print(f"   At 2 calls/sec: {unique_tickers/120:.1f} minutes")
print(f"   At 5 calls/sec: {unique_tickers/300:.1f} minutes")

# =============================================================================
# TIME BUCKETING ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("TIME BUCKETING POTENTIAL")
print("="*80)

# Group by ticker and 1-second buckets
df['timestamp_second'] = df['sip_timestamp'].dt.floor('S')
ticker_second_groups = df.groupby(['ticker', 'timestamp_second']).size()

print(f"\n📊 IF WE BUCKET BY SECOND:")
print(f"   Unique (ticker, second) combinations: {len(ticker_second_groups):,}")
print(f"   Reduction from per-trade: {(1 - len(ticker_second_groups)/total_trades)*100:.1f}%")
print(f"   At 1 call/sec: {len(ticker_second_groups)/3600:.1f} hours")

# Group by ticker and 10-second buckets
df['timestamp_10s'] = df['sip_timestamp'].dt.floor('10S')
ticker_10s_groups = df.groupby(['ticker', 'timestamp_10s']).size()

print(f"\n📊 IF WE BUCKET BY 10 SECONDS:")
print(f"   Unique (ticker, 10s) combinations: {len(ticker_10s_groups):,}")
print(f"   Reduction from per-trade: {(1 - len(ticker_10s_groups)/total_trades)*100:.1f}%")
print(f"   At 1 call/sec: {len(ticker_10s_groups)/3600:.1f} hours")

# Group by ticker and minute
df['timestamp_minute'] = df['sip_timestamp'].dt.floor('T')
ticker_minute_groups = df.groupby(['ticker', 'timestamp_minute']).size()

print(f"\n📊 IF WE BUCKET BY MINUTE:")
print(f"   Unique (ticker, minute) combinations: {len(ticker_minute_groups):,}")
print(f"   Reduction from per-trade: {(1 - len(ticker_minute_groups)/total_trades)*100:.1f}%")
print(f"   At 1 call/sec: {len(ticker_minute_groups)/3600:.1f} hours")

# =============================================================================
# SMART SAMPLING STRATEGIES
# =============================================================================

print("\n" + "="*80)
print("SMART SAMPLING STRATEGIES")
print("="*80)

# High volume tickers (>1000 trades)
high_volume_tickers = ticker_counts[ticker_counts > 1000]
high_volume_trades = high_volume_tickers.sum()

print(f"\n📊 HIGH-VOLUME TICKER STRATEGY:")
print(f"   Tickers with >1000 trades: {len(high_volume_tickers):,}")
print(f"   Total trades in these: {high_volume_trades:,} ({high_volume_trades/total_trades*100:.1f}%)")
print(f"   If we sample these hourly: {len(high_volume_tickers) * 8:,} quotes")
print(f"   If we use one quote per ticker: {unique_tickers - len(high_volume_tickers):,} quotes")
print(f"   Total quotes needed: {len(high_volume_tickers) * 8 + (unique_tickers - len(high_volume_tickers)):,}")

# ATM-only strategy
# Parse strikes from tickers
def get_strike_from_ticker(ticker):
    try:
        # Format: O:SPY241011C00670000
        parts = ticker.split('C')
        if len(parts) == 1:
            parts = ticker.split('P')
        strike = int(parts[1]) / 1000
        return strike
    except:
        return None

df['strike'] = df['ticker'].apply(get_strike_from_ticker)
df = df.dropna(subset=['strike'])

# Assume SPY is around 669
spy_price = 669
atm_range = (spy_price * 0.98, spy_price * 1.02)
atm_trades = df[(df['strike'] >= atm_range[0]) & (df['strike'] <= atm_range[1])]

print(f"\n📊 ATM-ONLY STRATEGY (±2% of spot):")
print(f"   ATM trades: {len(atm_trades):,} ({len(atm_trades)/total_trades*100:.1f}%)")
print(f"   Unique ATM tickers: {atm_trades['ticker'].nunique():,}")
print(f"   If we only quote ATM: {atm_trades['ticker'].nunique():,} API calls")

# =============================================================================
# RECOMMENDATIONS
# =============================================================================

print("\n" + "="*80)
print("OPTIMAL STRATEGIES")
print("="*80)

print("\n💡 RECOMMENDED APPROACHES:")

print("\n1. **ONE QUOTE PER TICKER** (Fastest)")
print(f"   • API calls: {unique_tickers:,}")
print(f"   • Time at 2/sec: {unique_tickers/120:.1f} minutes")
print(f"   • Coverage: Good for P/C ratios, volume analysis")
print(f"   • Limitation: Can't determine intraday changes")

print("\n2. **BUCKET BY MINUTE** (Balance)")
print(f"   • API calls: {len(ticker_minute_groups):,}")
print(f"   • Time at 2/sec: {len(ticker_minute_groups)/120:.1f} minutes")
print(f"   • Coverage: Better for trade direction")
print(f"   • Reduction: {(1 - len(ticker_minute_groups)/total_trades)*100:.1f}% fewer calls")

print("\n3. **ATM + HIGH VOLUME HYBRID**")
atm_plus_hv = atm_trades['ticker'].nunique() + len(high_volume_tickers) * 4
print(f"   • API calls: ~{atm_plus_hv:,}")
print(f"   • Time at 2/sec: {atm_plus_hv/120:.1f} minutes")
print(f"   • Coverage: Focus on important strikes")
print(f"   • Best for: VIX term structure, dealer positioning")

print("\n4. **SMART SAMPLING** (Most Practical)")
print(f"   • Sample 100% of top 20% tickers by minute")
print(f"   • Sample 10% of remaining tickers")
print(f"   • Estimated API calls: ~{int(len(ticker_minute_groups)*0.3):,}")
print(f"   • Time at 2/sec: {len(ticker_minute_groups)*0.3/120:.1f} minutes")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)