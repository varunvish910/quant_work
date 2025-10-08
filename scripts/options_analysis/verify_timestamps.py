#!/usr/bin/env python3
"""
Verify timestamp format in trade file
"""

import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path('trade_and_quote_data/data_management/flatfiles')
TRADES_FILE = OUTPUT_DIR / "2025-10-06.csv.gz"

print("="*80)
print("VERIFYING TIMESTAMP FORMAT")
print("="*80)

# Load first SPY trade
with gzip.open(TRADES_FILE, 'rt') as f:
    for chunk in pd.read_csv(f, chunksize=100000):
        spy_chunk = chunk[chunk['ticker'].str.match(r'^O:SPY\d', na=False)]
        if len(spy_chunk) > 0:
            trade = spy_chunk.iloc[0]
            break

print(f"\n📊 First SPY Trade:")
print(f"   Ticker: {trade['ticker']}")
print(f"   Price: ${trade['price']:.2f}")
print(f"   sip_timestamp: {trade['sip_timestamp']}")

# Try different timestamp interpretations
timestamp_ns = trade['sip_timestamp']

print(f"\n🕐 Timestamp Interpretations:")

# 1. Nanoseconds since Unix epoch (what Polygon should use)
dt_ns = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
print(f"\n   As nanoseconds since epoch (UTC):")
print(f"      {dt_ns}")
print(f"      Date: {dt_ns.strftime('%Y-%m-%d')}")
print(f"      Time: {dt_ns.strftime('%H:%M:%S %Z')}")

# 2. Check if this is actually Oct 6, 2025
oct_6_2025_start = datetime(2025, 10, 6, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1_000_000_000
oct_6_2025_end = datetime(2025, 10, 7, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1_000_000_000

print(f"\n   Expected Oct 6, 2025 range:")
print(f"      Start: {oct_6_2025_start}")
print(f"      End:   {oct_6_2025_end}")
print(f"      Trade: {timestamp_ns}")

if oct_6_2025_start <= timestamp_ns < oct_6_2025_end:
    print(f"      ✅ Trade timestamp IS within Oct 6, 2025")
else:
    print(f"      ❌ Trade timestamp is NOT Oct 6, 2025!")

    # Calculate actual date
    actual_dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
    print(f"      Actual date: {actual_dt.strftime('%Y-%m-%d')}")

# 3. Check market hours (9:30 AM - 4:00 PM ET)
# ET is UTC-4 during EDT (summer) or UTC-5 during EST (winter)
# Oct 6 is still EDT (before Nov DST change)
et_offset = -4  # EDT
market_open_utc = datetime(2025, 10, 6, 9+4, 30, 0, tzinfo=timezone.utc)  # 9:30 ET = 13:30 UTC
market_close_utc = datetime(2025, 10, 6, 16+4, 0, 0, tzinfo=timezone.utc)  # 4:00 ET = 20:00 UTC

market_open_ns = market_open_utc.timestamp() * 1_000_000_000
market_close_ns = market_close_utc.timestamp() * 1_000_000_000

print(f"\n   Market hours (9:30 AM - 4:00 PM ET):")
print(f"      Open (UTC):  {market_open_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"      Close (UTC): {market_close_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"      Open (ns):   {market_open_ns}")
print(f"      Close (ns):  {market_close_ns}")
print(f"      Trade (ns):  {timestamp_ns}")

if market_open_ns <= timestamp_ns < market_close_ns:
    print(f"      ✅ Trade during regular market hours")
else:
    print(f"      ❌ Trade OUTSIDE regular market hours")

    if timestamp_ns < market_open_ns:
        diff_sec = (market_open_ns - timestamp_ns) / 1_000_000_000
        print(f"      Trade was {diff_sec/60:.1f} minutes BEFORE market open")
    else:
        diff_sec = (timestamp_ns - market_close_ns) / 1_000_000_000
        print(f"      Trade was {diff_sec/60:.1f} minutes AFTER market close")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
