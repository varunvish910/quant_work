#!/usr/bin/env python3
"""
Realistic SPY Options Downloader
Downloads SPY options data from actual historical dates that should have data available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
from spy_trades_downloader import SPYTradesDownloader
import yfinance as yf


class RealisticSPYDownloader:
    """Downloads SPY options data from realistic historical dates"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/realistic_spy"):
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "daily_trades").mkdir(exist_ok=True)
        (self.output_dir / "enriched").mkdir(exist_ok=True)
        (self.output_dir / "summary").mkdir(exist_ok=True)
        
        self.downloader = SPYTradesDownloader(self.api_key, str(self.output_dir))
        
    def get_realistic_trading_dates(self) -> list:
        """Get realistic trading dates from 2024 that should have data"""
        # Use dates from September/October 2024 - these should have data
        base_dates = [
            "2024-09-30", "2024-09-27", "2024-09-26", "2024-09-25", "2024-09-24", "2024-09-23",
            "2024-09-20", "2024-09-19", "2024-09-18", "2024-09-17", "2024-09-16", "2024-09-13",
            "2024-09-12", "2024-09-11", "2024-09-10", "2024-09-09", "2024-09-06", "2024-09-05",
            "2024-09-04", "2024-09-03", "2024-08-30", "2024-08-29", "2024-08-28", "2024-08-27",
            "2024-08-26", "2024-08-23", "2024-08-22", "2024-08-21", "2024-08-20", "2024-08-19"
        ]
        
        # Filter to only weekdays
        trading_dates = []
        for date_str in base_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            if date_obj.weekday() < 5:  # Monday to Friday
                trading_dates.append(date_str)
        
        return trading_dates
    
    def get_realistic_expiry_dates(self) -> list:
        """Get realistic expiry dates for 2024 data"""
        # October 2024 expiries that should exist
        return [
            "2024-10-04",  # Friday
            "2024-10-07",  # Monday
            "2024-10-08",  # Tuesday
            "2024-10-09",  # Wednesday
            "2024-10-10",  # Thursday
            "2024-10-11",  # Friday
            "2024-10-14",  # Monday
            "2024-10-15",  # Tuesday
            "2024-10-16",  # Wednesday
            "2024-10-17",  # Thursday
            "2024-10-18"   # Friday
        ]
    
    def get_current_spy_price(self) -> float:
        """Get current SPY price"""
        try:
            spy = yf.Ticker('SPY')
            current_price = spy.history(period="1d")["Close"].iloc[-1]
            print(f"✓ Current SPY price: ${current_price:.2f}")
            return current_price
        except Exception as e:
            print(f"Warning: Could not fetch SPY price: {e}")
            fallback_price = 569.21  # More realistic for Sept 2024
            print(f"Using fallback SPY price: ${fallback_price:.2f}")
            return fallback_price
    
    def download_realistic_data(self) -> dict:
        """Download SPY options data from realistic historical dates"""
        
        print(f"\n{'='*60}")
        print(f"DOWNLOADING REALISTIC SPY OPTIONS DATA")
        print(f"Historical dates: September/October 2024")
        print(f"Multiple expiry dates")
        print(f"{'='*60}")
        
        # Get realistic dates and expiries
        trading_dates = self.get_realistic_trading_dates()
        expiry_dates = self.get_realistic_expiry_dates()
        current_price = self.get_current_spy_price()
        
        results = {
            'trading_dates_attempted': trading_dates,
            'expiry_dates': expiry_dates,
            'dates_successful': [],
            'dates_failed': [],
            'total_trades': 0,
            'daily_summaries': {}
        }
        
        print(f"\n📅 Attempting to download data:")
        print(f"   Trading dates: {len(trading_dates)} dates from {trading_dates[-1]} to {trading_dates[0]}")
        print(f"   Expiry dates: {len(expiry_dates)} expiries from {expiry_dates[0]} to {expiry_dates[-1]}")
        
        successful_count = 0
        max_attempts = 10  # Limit attempts to avoid too much output
        
        for i, date in enumerate(trading_dates[:max_attempts], 1):
            print(f"\n[{i}/{max_attempts}] Processing {date}...")
            
            try:
                # Try to download trades for this date with multiple expiries
                trades_df = self.downloader.download_trades(
                    date, expiry_dates, current_price
                )
                
                if len(trades_df) > 0:
                    print(f"   ✅ Downloaded {len(trades_df):,} trades for {date}")
                    
                    # Enrich with quotes
                    enriched_df = self.downloader.enrich_and_save(trades_df, date)
                    
                    # Save daily summary
                    daily_summary = {
                        'date': date,
                        'total_trades': len(enriched_df),
                        'unique_strikes': enriched_df['strike'].nunique(),
                        'unique_expiries': enriched_df['expiry'].nunique(),
                        'price_range': [float(enriched_df['strike'].min()), float(enriched_df['strike'].max())],
                        'volume': int(enriched_df['size'].sum()),
                        'avg_price': float(enriched_df['price'].mean())
                    }
                    
                    results['daily_summaries'][date] = daily_summary
                    results['dates_successful'].append(date)
                    results['total_trades'] += len(enriched_df)
                    successful_count += 1
                    
                    print(f"   📊 Strikes: {enriched_df['strike'].nunique()}, Expiries: {enriched_df['expiry'].nunique()}, Volume: {enriched_df['size'].sum():,}")
                    
                    # If we get some successful data, we can continue or stop here
                    if successful_count >= 3:  # Stop after 3 successful downloads
                        print(f"\n✅ Got {successful_count} successful downloads, stopping early")
                        break
                    
                else:
                    print(f"   ⚠️  No trades found for any expiry on {date}")
                    results['dates_failed'].append(date)
                    
            except Exception as e:
                print(f"   ❌ Failed to download {date}: {e}")
                results['dates_failed'].append(date)
                continue
        
        # Save summary
        self._save_download_summary(results)
        
        print(f"\n{'='*60}")
        print(f"REALISTIC DATA DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(results['dates_successful'])} dates")
        print(f"❌ Failed: {len(results['dates_failed'])} dates")
        print(f"📊 Total trades: {results['total_trades']:,}")
        
        if results['dates_successful']:
            print(f"\n📈 Successful dates:")
            for date in results['dates_successful']:
                summary = results['daily_summaries'][date]
                print(f"   {date}: {summary['total_trades']:,} trades, {summary['unique_strikes']} strikes, {summary['unique_expiries']} expiries")
        
        return results
    
    def _save_download_summary(self, results: dict):
        """Save download summary"""
        summary_file = self.output_dir / "summary" / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save as CSV for easy viewing
        if results['daily_summaries']:
            summary_df = pd.DataFrame(list(results['daily_summaries'].values()))
            summary_df.to_csv(self.output_dir / "summary" / "daily_summaries.csv", index=False)
        
        print(f"✅ Summary saved to {summary_file}")


def main():
    """Main execution"""
    
    print("🚀 Realistic SPY Options Data Download")
    print("=" * 40)
    
    downloader = RealisticSPYDownloader()
    
    try:
        # Download realistic historical data
        results = downloader.download_realistic_data()
        
        if results['dates_successful']:
            print(f"\n🎉 Realistic data download complete!")
            print(f"📁 Data saved to: {downloader.output_dir}")
            print(f"🎯 Ready for analysis and visualization")
            
            # Show next steps
            print(f"\n📋 Next steps:")
            print(f"   1. Analyze the downloaded historical data")
            print(f"   2. Create interactive visualizations")
            print(f"   3. Generate positioning reports")
            print(f"   4. Show how positioning evolved over the downloaded dates")
            
        else:
            print(f"\n⚠️  No data was successfully downloaded")
            print(f"💡 This indicates that Polygon's flat file API may not have")
            print(f"   historical data readily available for these date ranges.")
            print(f"   You may need to use a different data source or API approach.")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()