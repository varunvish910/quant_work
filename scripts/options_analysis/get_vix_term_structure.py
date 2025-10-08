#!/usr/bin/env python3
"""
Get VIX Term Structure from available sources
Since we can't get quotes efficiently, use alternative approaches
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("VIX TERM STRUCTURE ANALYSIS")
print("="*80)

# Method 1: Use VIX futures if available
print("\n📊 METHOD 1: Attempting to fetch VIX futures data...")

try:
    # VIX futures symbols
    # Note: These may not work on yfinance, but worth trying
    futures_symbols = {
        'Spot': '^VIX',
        'Oct 2025': 'VXV25',  # Oct 2025
        'Nov 2025': 'VXX25',  # Nov 2025
        'Dec 2025': 'VXZ25',  # Dec 2025
    }

    futures_data = {}
    for name, symbol in futures_symbols.items():
        try:
            data = yf.Ticker(symbol)
            info = data.info
            if 'regularMarketPrice' in info:
                futures_data[name] = info['regularMarketPrice']
                print(f"   ✅ {name}: {info['regularMarketPrice']:.2f}")
        except:
            # Try alternative symbol format
            alt_symbol = symbol.replace('VX', 'VIX/')
            try:
                data = yf.download(alt_symbol, period='1d', progress=False)
                if len(data) > 0:
                    futures_data[name] = data['Close'].iloc[-1]
                    print(f"   ✅ {name}: {data['Close'].iloc[-1]:.2f}")
            except:
                pass

    if len(futures_data) > 1:
        print("\n   Successfully retrieved VIX futures!")
    else:
        print("   ⚠️ Could not retrieve VIX futures from yfinance")

except Exception as e:
    print(f"   ❌ Error fetching futures: {e}")

# Method 2: Use VIX9D and other VIX variants as proxy
print("\n📊 METHOD 2: Using VIX variants as proxy...")

vix_variants = {
    'VIX (30-day)': '^VIX',
    'VIX9D (9-day)': '^VIX9D',
    'VIX3M (3-month)': '^VIX3M',
    'VIX6M (6-month)': '^VIX6M',
}

vix_data = {}
for name, symbol in vix_variants.items():
    try:
        data = yf.download(symbol, period='5d', progress=False)
        if len(data) > 0:
            latest = data['Close'].iloc[-1]
            vix_data[name] = latest
            print(f"   ✅ {name}: {latest:.2f}")
    except Exception as e:
        print(f"   ⚠️ Could not fetch {name}")

# Calculate term structure from variants
if 'VIX (30-day)' in vix_data and len(vix_data) > 0:
    vix_spot = float(vix_data['VIX (30-day)'])

    print("\n📈 TERM STRUCTURE FROM VIX VARIANTS:")
    print(f"   Current VIX: {vix_spot:.2f}")
    print("\n   Period     | Level  | vs Spot")
    print("   " + "-"*35)

    for name, value in vix_data.items():
        if name != 'VIX (30-day)':
            diff = (value / vix_spot - 1) * 100
            print(f"   {name:10s} | {value:6.2f} | {diff:+6.1f}%")

    # Interpret
    if 'VIX9D (9-day)' in vix_data:
        vix9d = vix_data['VIX9D (9-day)']
        short_term_diff = (vix9d / vix_spot - 1) * 100

        print(f"\n   SHORT-TERM STRUCTURE (9D vs 30D):")
        print(f"   Difference: {short_term_diff:+.1f}%")

        # IMPORTANT: This is NOT true term structure!
        print("\n   ⚠️ NOTE: VIX9D/VIX is NOT forward term structure")
        print("   It's historical realized vol at different windows")
        print("   For true term structure, we need VIX futures")

# Method 3: Use ETF/ETN products as proxy
print("\n📊 METHOD 3: Using VIX ETF/ETN products...")

vix_products = {
    'VXX (Short-term futures)': 'VXX',
    'VXZ (Mid-term futures)': 'VXZ',
    'VIXY (Short-term)': 'VIXY',
}

etf_data = {}
for name, symbol in vix_products.items():
    try:
        data = yf.download(symbol, period='5d', progress=False)
        if len(data) > 0:
            latest = data['Close'].iloc[-1]
            etf_data[name] = latest
            print(f"   ✅ {name}: ${latest:.2f}")
    except:
        print(f"   ⚠️ Could not fetch {name}")

# Method 4: Calculate from SPY options if we had quotes
print("\n📊 METHOD 4: SPY Options Implied Vol (needs quotes)...")
print("   ❌ Cannot calculate without bid/ask quotes")
print("   Would need:")
print("   1. ATM SPY option prices at different expirations")
print("   2. Black-Scholes inverse to get implied volatility")
print("   3. Convert to VIX-equivalent scale")

# Method 5: Use public data sources
print("\n📊 METHOD 5: Alternative data sources...")
print("   • CBOE website has delayed VIX futures quotes")
print("   • Bloomberg/Reuters terminals have real-time data")
print("   • Some brokers provide VIX futures via API")

# Summary
print("\n" + "="*80)
print("VIX TERM STRUCTURE SUMMARY")
print("="*80)

if len(vix_data) > 0:
    print("\n📊 AVAILABLE DATA:")
    for name, value in vix_data.items():
        print(f"   {name}: {value:.2f}")

    print("\n⚠️ LIMITATIONS:")
    print("   • VIX9D/VIX3M/VIX6M are BACKWARD-looking volatility")
    print("   • They measure realized vol over past N days")
    print("   • NOT forward-looking like VIX futures")

    print("\n💡 INTERPRETATION:")
    if 'VIX9D (9-day)' in vix_data and 'VIX (30-day)' in vix_data:
        if vix_data['VIX9D (9-day)'] > vix_data['VIX (30-day)']:
            print("   • VIX9D > VIX: Recent volatility spike")
            print("   • Market experienced stress in past 9 days")
        else:
            print("   • VIX9D < VIX: Recent calm")
            print("   • Volatility was higher 10-30 days ago")
else:
    print("\n❌ Could not retrieve VIX data")

print("\n📌 CONCLUSION:")
print("   Without VIX futures or option quotes, we cannot calculate")
print("   true forward-looking term structure. The VIX9D/VIX ratio")
print("   shows HISTORICAL volatility patterns, not future expectations.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")