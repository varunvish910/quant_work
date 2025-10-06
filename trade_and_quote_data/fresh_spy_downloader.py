#!/usr/bin/env python3
"""
Fresh SPY Options Downloader
Downloads SPY options data for the last 30 days with specific expiry dates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
from spy_trades_downloader import SPYTradesDownloader
import yfinance as yf


class FreshSPYDownloader:
    """Downloads fresh SPY options data with specific expiry focus"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/fresh_spy"):
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "daily_trades").mkdir(exist_ok=True)
        (self.output_dir / "enriched").mkdir(exist_ok=True)
        (self.output_dir / "summary").mkdir(exist_ok=True)
        
        self.downloader = SPYTradesDownloader(self.api_key, str(self.output_dir))
        
    def get_trading_dates_last_30_days(self) -> list:
        """Get trading dates for the last 30 calendar days (excluding weekends)"""
        dates = []
        current_date = datetime.now().date()
        
        for i in range(30):
            check_date = current_date - timedelta(days=i)
            # Only include weekdays (Monday=0, Sunday=6)
            if check_date.weekday() < 5:  # Monday to Friday
                dates.append(check_date.strftime('%Y-%m-%d'))
        
        return list(reversed(dates))  # Return in chronological order
    
    def get_current_spy_price(self) -> float:
        """Get current SPY price"""
        try:
            spy = yf.Ticker('SPY')
            current_price = spy.history(period="1d")["Close"].iloc[-1]
            print(f"✓ Current SPY price: ${current_price:.2f}")
            return current_price
        except Exception as e:
            print(f"Warning: Could not fetch SPY price: {e}")
            fallback_price = 569.21
            print(f"Using fallback SPY price: ${fallback_price:.2f}")
            return fallback_price
    
    def download_fresh_data(self, target_expiry: str = "2025-10-06") -> dict:
        """Download fresh SPY options data for the last 30 days"""
        
        print(f"\n{'='*60}")
        print(f"DOWNLOADING FRESH SPY OPTIONS DATA")
        print(f"Target expiry: {target_expiry}")
        print(f"Last 30 trading days")
        print(f"{'='*60}")
        
        # Get trading dates
        trading_dates = self.get_trading_dates_last_30_days()
        current_price = self.get_current_spy_price()
        
        results = {
            'target_expiry': target_expiry,
            'dates_attempted': [],
            'dates_successful': [],
            'dates_failed': [],
            'total_trades': 0,
            'daily_summaries': {}
        }
        
        print(f"\n📅 Attempting to download data for {len(trading_dates)} trading days:")
        print(f"   Date range: {trading_dates[0]} to {trading_dates[-1]}")
        
        for i, date in enumerate(trading_dates, 1):
            print(f"\n[{i}/{len(trading_dates)}] Processing {date}...")
            results['dates_attempted'].append(date)
            
            try:
                # Try to download trades for this date
                trades_df = self.downloader.download_trades(
                    date, [target_expiry], current_price
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
                        'price_range': [float(enriched_df['strike'].min()), float(enriched_df['strike'].max())],
                        'volume': int(enriched_df['size'].sum()),
                        'avg_price': float(enriched_df['price'].mean())
                    }
                    
                    results['daily_summaries'][date] = daily_summary
                    results['dates_successful'].append(date)
                    results['total_trades'] += len(enriched_df)
                    
                    print(f"   📊 Strikes: {enriched_df['strike'].nunique()}, Volume: {enriched_df['size'].sum():,}")
                    
                else:
                    print(f"   ⚠️  No trades found for {target_expiry} on {date}")
                    results['dates_failed'].append(date)
                    
            except Exception as e:
                print(f"   ❌ Failed to download {date}: {e}")
                results['dates_failed'].append(date)
                continue
        
        # Save summary
        self._save_download_summary(results)
        
        print(f"\n{'='*60}")
        print(f"DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(results['dates_successful'])} dates")
        print(f"❌ Failed: {len(results['dates_failed'])} dates")
        print(f"📊 Total trades: {results['total_trades']:,}")
        
        if results['dates_successful']:
            print(f"\n📈 Successful dates:")
            for date in results['dates_successful'][-5:]:  # Show last 5
                summary = results['daily_summaries'][date]
                print(f"   {date}: {summary['total_trades']:,} trades, {summary['unique_strikes']} strikes")
        
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
    
    print("🚀 Fresh SPY Options Data Download")
    print("=" * 40)
    
    # Target expiry date
    target_expiry = "2025-10-06"  # October 6, 2025
    
    downloader = FreshSPYDownloader()
    
    try:
        # Download fresh data
        results = downloader.download_fresh_data(target_expiry)
        
        if results['dates_successful']:
            print(f"\n🎉 Fresh data download complete!")
            print(f"📁 Data saved to: {downloader.output_dir}")
            print(f"🎯 Ready for analysis and visualization")
            
            # Show next steps
            print(f"\n📋 Next steps:")
            print(f"   1. Run analysis on the downloaded data")
            print(f"   2. Create interactive visualizations")
            print(f"   3. Generate positioning reports")
            
        else:
            print(f"\n⚠️  No data was successfully downloaded")
            print(f"💡 This might be because:")
            print(f"   • The expiry date {target_expiry} doesn't exist yet")
            print(f"   • No trading activity for this expiry on recent dates")
            print(f"   • API limitations or temporary issues")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()