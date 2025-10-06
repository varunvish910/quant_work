#!/usr/bin/env python3
"""
Direct API SPY Options Downloader
Uses Polygon's REST API directly instead of flat files to download options data
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import time
import yfinance as yf


class DirectAPISPYDownloader:
    """Downloads SPY options using Polygon's REST API directly"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/direct_api_spy"):
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.base_url = "https://api.polygon.io"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "trades").mkdir(exist_ok=True)
        (self.output_dir / "options_data").mkdir(exist_ok=True)
        (self.output_dir / "summary").mkdir(exist_ok=True)
    
    def get_spy_options_tickers(self, expiry_date: str, current_price: float = 569.21) -> list:
        """Get SPY options tickers for a specific expiry"""
        
        print(f"🔍 Getting SPY options tickers for expiry {expiry_date}...")
        
        # Format expiry for options ticker
        expiry_obj = datetime.strptime(expiry_date, '%Y-%m-%d')
        expiry_formatted = expiry_obj.strftime('%y%m%d')  # YYMMDD format
        
        # Generate strike range around current price
        strike_min = int(current_price * 0.90)  # 10% below
        strike_max = int(current_price * 1.10)  # 10% above
        
        options_tickers = []
        
        # Generate call and put tickers
        for strike in range(strike_min, strike_max + 5, 5):  # $5 increments
            # Call: O:SPY241004C00570000 format
            call_ticker = f"O:SPY{expiry_formatted}C{strike:08d}"
            put_ticker = f"O:SPY{expiry_formatted}P{strike:08d}"
            
            options_tickers.extend([call_ticker, put_ticker])
        
        print(f"✓ Generated {len(options_tickers)} options tickers for strikes ${strike_min}-${strike_max}")
        return options_tickers
    
    def get_options_trades_for_ticker(self, ticker: str, date: str) -> pd.DataFrame:
        """Get trades for a specific options ticker on a specific date"""
        
        url = f"{self.base_url}/v3/trades/{ticker}"
        
        params = {
            'timestamp.gte': f"{date}T09:30:00.000Z",
            'timestamp.lte': f"{date}T16:00:00.000Z", 
            'order': 'asc',
            'limit': 5000,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    trades_df = pd.DataFrame(data['results'])
                    trades_df['ticker'] = ticker
                    trades_df['date'] = date
                    
                    # Parse option details from ticker
                    option_info = self._parse_option_ticker(ticker)
                    for key, value in option_info.items():
                        trades_df[key] = value
                    
                    return trades_df
                else:
                    return pd.DataFrame()
            
            elif response.status_code == 429:
                print(f"   ⚠️  Rate limited for {ticker}, waiting...")
                time.sleep(60)  # Wait 1 minute
                return pd.DataFrame()
            
            else:
                print(f"   ❌ API error {response.status_code} for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def _parse_option_ticker(self, ticker: str) -> dict:
        """Parse SPY option ticker to extract details"""
        try:
            # Example: O:SPY241004C00570000
            if not ticker.startswith('O:SPY'):
                return {'expiry': None, 'strike': None, 'option_type': None, 'underlying': 'SPY'}
            
            parts = ticker[5:]  # Remove 'O:SPY'
            
            if len(parts) >= 15:
                # Extract expiry (YYMMDD)
                expiry_str = parts[:6]
                year = "20" + expiry_str[:2]
                month = expiry_str[2:4]
                day = expiry_str[4:6]
                expiry = datetime.strptime(f"{year}{month}{day}", '%Y%m%d').date()
                
                # Extract option type
                option_type = parts[6].lower()
                
                # Extract strike
                strike_str = parts[7:]
                if strike_str.isdigit() and len(strike_str) == 8:
                    strike = float(strike_str) / 1000  # Convert to dollars
                else:
                    strike = None
                
                return {
                    'expiry': expiry,
                    'strike': strike,
                    'option_type': option_type,
                    'underlying': 'SPY'
                }
            
            return {'expiry': None, 'strike': None, 'option_type': None, 'underlying': 'SPY'}
            
        except Exception:
            return {'expiry': None, 'strike': None, 'option_type': None, 'underlying': 'SPY'}
    
    def test_api_access(self) -> bool:
        """Test if we can access Polygon's API"""
        
        print("🔑 Testing Polygon API access...")
        
        # Test with simple SPY ticker
        url = f"{self.base_url}/v2/aggs/ticker/SPY/range/1/day/2024-09-20/2024-09-20"
        params = {'apikey': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    print("✅ API access confirmed - your key works!")
                    return True
                else:
                    print(f"❌ API returned error: {data}")
                    return False
            else:
                print(f"❌ API request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
    
    def download_recent_spy_options(self, target_expiry: str = "2024-10-18") -> dict:
        """Download recent SPY options data using REST API"""
        
        print(f"\n{'='*60}")
        print(f"DOWNLOADING SPY OPTIONS VIA REST API")
        print(f"Target expiry: {target_expiry}")
        print(f"{'='*60}")
        
        # Test API first
        if not self.test_api_access():
            return {'error': 'API access failed'}
        
        # Get current SPY price
        try:
            spy = yf.Ticker('SPY')
            current_price = spy.history(period="1d")["Close"].iloc[-1]
            print(f"✓ Current SPY price: ${current_price:.2f}")
        except:
            current_price = 569.21
            print(f"Using fallback SPY price: ${current_price:.2f}")
        
        # Try recent trading dates
        recent_dates = [
            "2024-10-04", "2024-10-03", "2024-10-02", "2024-10-01",
            "2024-09-30", "2024-09-27", "2024-09-26", "2024-09-25"
        ]
        
        results = {
            'target_expiry': target_expiry,
            'dates_attempted': [],
            'dates_successful': [],
            'total_trades': 0,
            'options_data': {}
        }
        
        # Get options tickers for the expiry
        options_tickers = self.get_spy_options_tickers(target_expiry, current_price)
        
        # Try to get data for recent dates
        for date in recent_dates[:3]:  # Try first 3 dates
            print(f"\n📅 Trying to download data for {date}...")
            results['dates_attempted'].append(date)
            
            date_trades = []
            successful_tickers = 0
            
            # Try a few representative tickers
            sample_tickers = options_tickers[::10]  # Every 10th ticker
            
            for i, ticker in enumerate(sample_tickers[:5], 1):  # Try first 5 sample tickers
                print(f"   [{i}/5] Fetching {ticker}...")
                
                trades_df = self.get_options_trades_for_ticker(ticker, date)
                
                if len(trades_df) > 0:
                    date_trades.append(trades_df)
                    successful_tickers += 1
                    print(f"      ✅ Got {len(trades_df)} trades")
                else:
                    print(f"      ⚠️  No trades found")
                
                # Rate limiting
                time.sleep(0.1)
            
            if date_trades:
                # Combine all trades for this date
                combined_df = pd.concat(date_trades, ignore_index=True)
                
                # Add synthetic quotes (simplified)
                combined_df = self._add_synthetic_quotes(combined_df)
                
                # Save data
                output_file = self.output_dir / "trades" / f"{date}_options_trades.parquet"
                combined_df.to_parquet(output_file, index=False)
                
                results['dates_successful'].append(date)
                results['total_trades'] += len(combined_df)
                results['options_data'][date] = {
                    'trades_count': len(combined_df),
                    'tickers_found': successful_tickers,
                    'unique_strikes': combined_df['strike'].nunique() if 'strike' in combined_df.columns else 0
                }
                
                print(f"   ✅ Saved {len(combined_df)} trades for {date}")
                
                # Stop after first successful date for demonstration
                break
            else:
                print(f"   ❌ No data found for {date}")
        
        # Save summary
        summary_file = self.output_dir / "summary" / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"REST API DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful dates: {len(results['dates_successful'])}")
        print(f"📊 Total trades: {results['total_trades']}")
        
        return results
    
    def _add_synthetic_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic bid/ask quotes for analysis"""
        
        if 'price' not in df.columns:
            return df
        
        # Estimate bid/ask spread
        df['spread_estimate'] = np.where(
            df['price'] > 10, df['price'] * 0.05,  # 5% spread for expensive options
            np.where(
                df['price'] > 1, df['price'] * 0.10,  # 10% spread for mid-price options
                np.maximum(0.01, df['price'] * 0.20)  # 20% spread for cheap options
            )
        )
        
        # Create synthetic bid/ask
        df['bid'] = np.maximum(0.01, df['price'] - df['spread_estimate'] / 2)
        df['ask'] = df['price'] + df['spread_estimate'] / 2
        df['bid_size'] = np.random.randint(1, 50, size=len(df))
        df['ask_size'] = np.random.randint(1, 50, size=len(df))
        
        return df


def main():
    """Main execution"""
    
    print("🚀 Direct API SPY Options Download")
    print("=" * 40)
    
    downloader = DirectAPISPYDownloader()
    
    try:
        # Download using REST API
        results = downloader.download_recent_spy_options("2024-10-18")
        
        if 'error' not in results and results['dates_successful']:
            print(f"\n🎉 Successfully downloaded options data via REST API!")
            print(f"📁 Data saved to: {downloader.output_dir}")
            print(f"🎯 Ready for analysis")
        else:
            print(f"\n⚠️  No data downloaded successfully")
            print(f"💡 This might be because:")
            print(f"   • The API key has limited access to options data")
            print(f"   • No trading activity for recent dates")
            print(f"   • Rate limiting or API restrictions")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()