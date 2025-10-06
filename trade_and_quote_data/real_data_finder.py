#!/usr/bin/env python3
"""
Real Data Finder - Check what actual SPY options data is available
"""

import pandas as pd
from datetime import datetime, timedelta
from spy_trades_downloader import SPYTradesDownloader
import yfinance as yf


def test_date_ranges():
    """Test different date ranges to find available data"""
    
    api_key = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
    downloader = SPYTradesDownloader(api_key)
    
    # Test different date ranges
    test_dates = [
        # Recent dates
        "2024-09-30", "2024-09-27", "2024-09-26", "2024-09-25", "2024-09-24", "2024-09-23",
        "2024-09-20", "2024-09-19", "2024-09-18", "2024-09-17", "2024-09-16", "2024-09-13",
        # Earlier dates
        "2024-09-12", "2024-09-11", "2024-09-10", "2024-09-09", "2024-09-06", "2024-09-05",
        "2024-09-04", "2024-09-03", "2024-08-30", "2024-08-29", "2024-08-28", "2024-08-27",
        # Even earlier
        "2024-08-26", "2024-08-23", "2024-08-22", "2024-08-21", "2024-08-20", "2024-08-19"
    ]
    
    available_dates = []
    
    for date in test_dates:
        print(f"\n🔍 Testing {date}...")
        
        try:
            # Try to download flat file
            flat_file = downloader.download_flat_file(date, "trades")
            
            if flat_file and flat_file.exists():
                print(f"   ✅ Found flat file for {date}")
                
                # Try to parse SPY options
                spy_data = downloader.parse_spy_options_from_flat_file(flat_file)
                
                if spy_data is not None and len(spy_data) > 0:
                    available_dates.append({
                        'date': date,
                        'trades_count': len(spy_data),
                        'unique_strikes': spy_data['strike'].nunique(),
                        'expiries': spy_data['expiry'].nunique()
                    })
                    print(f"   ✅ SPY data: {len(spy_data):,} trades, {spy_data['strike'].nunique()} strikes")
                else:
                    print(f"   ⚠️  No SPY options found in flat file")
            else:
                print(f"   ❌ No flat file available for {date}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"AVAILABLE DATA SUMMARY")
    print(f"{'='*60}")
    
    if available_dates:
        for data in available_dates:
            print(f"📅 {data['date']}: {data['trades_count']:,} trades, {data['unique_strikes']} strikes, {data['expiries']} expiries")
        
        # Return the most recent dates with good data
        return [d['date'] for d in available_dates[:10]]  # Top 10 dates
    else:
        print("❌ No available data found")
        return []


def download_real_historical_data():
    """Download real historical data for available dates"""
    
    print("🚀 Finding real SPY options data...")
    
    # Find available dates
    available_dates = test_date_ranges()
    
    if not available_dates:
        print("❌ No real data available")
        return None
    
    print(f"\n📊 Found {len(available_dates)} dates with real data")
    print(f"Date range: {min(available_dates)} to {max(available_dates)}")
    
    # Target expiry dates around the available data period
    # Look for October 2024 expiries since we have Sep 2024 data
    expiry_dates = [
        "2024-10-04",  # Friday
        "2024-10-07",  # Monday
        "2024-10-08",  # Tuesday
        "2024-10-09",  # Wednesday
        "2024-10-10",  # Thursday
        "2024-10-11"   # Friday
    ]
    
    print(f"Target expiries: {', '.join(expiry_dates)}")
    
    # Download and process the real data
    api_key = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
    downloader = SPYTradesDownloader(api_key, "data/real_spy_historical")
    
    # Get current SPY price
    try:
        spy = yf.Ticker('SPY')
        current_price = spy.history(period="1d")["Close"].iloc[-1]
        print(f"✓ Current SPY price: ${current_price:.2f}")
    except:
        current_price = 569.21
        print(f"Using fallback SPY price: ${current_price:.2f}")
    
    historical_data = {}
    
    for date in available_dates[:5]:  # Process first 5 available dates
        print(f"\n📥 Processing real data for {date}...")
        
        try:
            # Download trades
            trades_df = downloader.download_trades(date, expiry_dates, current_price)
            
            if len(trades_df) > 0:
                print(f"   ✅ Downloaded {len(trades_df):,} trades")
                
                # Enrich with quotes
                enriched_df = downloader.enrich_and_save(trades_df, date)
                
                historical_data[date] = {
                    'trades_count': len(enriched_df),
                    'unique_strikes': enriched_df['strike'].nunique(),
                    'expiries_found': enriched_df['expiry'].nunique()
                }
                
                print(f"   ✅ Processed: {len(enriched_df):,} trades, {enriched_df['strike'].nunique()} strikes")
            else:
                print(f"   ⚠️  No trades found for target expiries")
                
        except Exception as e:
            print(f"   ❌ Error processing {date}: {e}")
    
    print(f"\n🎉 Real historical data collection complete!")
    print(f"📊 Successfully processed {len(historical_data)} dates")
    
    return historical_data, expiry_dates


if __name__ == "__main__":
    download_real_historical_data()