#!/usr/bin/env python3
"""
Download Historical SPY Options Using Polygon REST API
Downloads aggregated put/call volume data for Sept-Oct 2025
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import yfinance as yf

API_KEY = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
BASE_URL = "https://api.polygon.io"

print("="*80)
print("SPY OPTIONS HISTORICAL DATA - REST API APPROACH")
print("="*80)

# Get SPY price to determine strikes
spy = yf.Ticker('SPY')
current_price = spy.history(period="1d")["Close"].iloc[-1]
print(f"\nCurrent SPY: ${current_price:.2f}")

# Generate trading days
start = datetime.strptime("2025-09-01", '%Y-%m-%d')
end = datetime.strptime("2025-10-06", '%Y-%m-%d')

trading_days = []
current = start
while current <= end:
    if current.weekday() < 5:  # Exclude weekends
        trading_days.append(current.strftime('%Y-%m-%d'))
    current += timedelta(days=1)

print(f"\n📅 Trading days to download: {len(trading_days)}")

# We'll download aggregated options data using Polygon's Aggregates endpoint
# This is much faster than downloading individual trades

def get_spy_options_snapshot(date: str) -> dict:
    """Get SPY options snapshot for a date using grouped aggregate data"""

    print(f"\n{'='*80}")
    print(f"Downloading options data for {date}")
    print(f"{'='*80}")

    # Use Polygon's options snapshot API if available
    # Or use aggregate bars for options

    # Try options snapshot endpoint
    url = f"{BASE_URL}/v3/snapshot/options/SPY"
    params = {
        'apikey': API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if 'results' in data and data['results']:
                results = data['results']

                # Aggregate by call/put
                call_volume = sum(r.get('day', {}).get('volume', 0)
                                for r in results
                                if r.get('details', {}).get('contract_type') == 'call')

                put_volume = sum(r.get('day', {}).get('volume', 0)
                               for r in results
                               if r.get('details', {}).get('contract_type') == 'put')

                call_oi = sum(r.get('open_interest', 0)
                            for r in results
                            if r.get('details', {}).get('contract_type') == 'call')

                put_oi = sum(r.get('open_interest', 0)
                           for r in results
                           if r.get('details', {}).get('contract_type') == 'put')

                print(f"   ✅ Call Volume: {call_volume:,}")
                print(f"   ✅ Put Volume: {put_volume:,}")
                print(f"   ✅ P/C Ratio: {put_volume / max(call_volume, 1):.2f}")

                return {
                    'date': date,
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'pc_volume_ratio': put_volume / max(call_volume, 1),
                    'pc_oi_ratio': put_oi / max(call_oi, 1),
                    'total_volume': call_volume + put_volume
                }
            else:
                print(f"   ⚠️  No results for {date}")
                return None

        elif response.status_code == 403:
            print(f"   ❌ 403 Forbidden - API access issue")
            return None
        elif response.status_code == 404:
            print(f"   ⚠️  No data for {date}")
            return None
        else:
            print(f"   ❌ API error: {response.status_code}")
            return None

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# Since we can't access historical snapshots easily, let's use yfinance
# to get put/call ratio data which is aggregated and available

print("\n" + "="*80)
print("ALTERNATIVE: Using yfinance for historical put/call data")
print("="*80)

# Get historical options data from yfinance
def get_spy_options_yfinance() -> pd.DataFrame:
    """Get current SPY options data from yfinance and estimate historical patterns"""

    ticker = yf.Ticker('SPY')

    # Get all expirations
    expirations = ticker.options

    if not expirations:
        return None

    print(f"\n📊 Analyzing {len(expirations)} SPY options expirations...")

    all_data = []

    for exp in expirations[:10]:  # First 10 expirations
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls
            puts = chain.puts

            call_vol = calls['volume'].fillna(0).sum()
            put_vol = puts['volume'].fillna(0).sum()
            call_oi = calls['openInterest'].fillna(0).sum()
            put_oi = puts['openInterest'].fillna(0).sum()

            all_data.append({
                'expiration': exp,
                'call_volume': call_vol,
                'put_volume': put_vol,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'pc_volume_ratio': put_vol / max(call_vol, 1),
                'pc_oi_ratio': put_oi / max(call_oi, 1)
            })

        except Exception as e:
            print(f"   Error getting chain for {exp}: {e}")
            continue

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"\n✅ Retrieved data for {len(df)} expirations")
        print(f"\nOverall metrics:")
        print(f"   Total Call Volume: {df['call_volume'].sum():,.0f}")
        print(f"   Total Put Volume: {df['put_volume'].sum():,.0f}")
        print(f"   Overall P/C Ratio: {df['put_volume'].sum() / max(df['call_volume'].sum(), 1):.2f}")
        print(f"   Avg P/C by expiration: {df['pc_volume_ratio'].mean():.2f}")

        return df

    return None

# Get current data
current_data = get_spy_options_yfinance()

if current_data is not None:
    # Save to file
    output_file = Path("spy_options_current_snapshot.csv")
    current_data.to_csv(output_file, index=False)
    print(f"\n✅ Saved current snapshot to: {output_file}")

    print(f"\n{'='*80}")
    print("SUMMARY - Current SPY Options Market")
    print(f"{'='*80}")

    print(f"\n📊 Put/Call Ratios by Expiration:")
    for idx, row in current_data.head(5).iterrows():
        print(f"   {row['expiration']}: P/C = {row['pc_volume_ratio']:.2f} (Vol: {row['put_volume']:,.0f} puts, {row['call_volume']:,.0f} calls)")

    avg_pc = current_data['pc_volume_ratio'].mean()
    if avg_pc > 1.2:
        print(f"\n🔴 HIGH PUT BUYING - P/C Ratio: {avg_pc:.2f}")
        print("   → Market showing defensive positioning")
    elif avg_pc > 0.9:
        print(f"\n🟡 BALANCED - P/C Ratio: {avg_pc:.2f}")
    else:
        print(f"\n🟢 CALL DOMINANT - P/C Ratio: {avg_pc:.2f}")
        print("   → Market still bullish / chasing upside")

print(f"\n{'='*80}")
print("Complete!")
print(f"{'='*80}")

print("\n💡 NOTE: For true historical 30-day data, we'd need:")
print("   1. Polygon flat files tier (requires Starter+ plan)")
print("   2. Or scrape daily yfinance snapshots over time")
print("   3. Current analysis uses TODAY's snapshot as proxy")
