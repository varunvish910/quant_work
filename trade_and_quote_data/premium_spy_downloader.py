#!/usr/bin/env python3
"""
Premium SPY Options Downloader
Uses premium Polygon API access to download comprehensive SPY options data
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


class PremiumSPYDownloader:
    """Downloads SPY options using premium Polygon API access"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/premium_spy"):
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.base_url = "https://api.polygon.io"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "trades").mkdir(exist_ok=True)
        (self.output_dir / "enriched").mkdir(exist_ok=True)
        (self.output_dir / "summary").mkdir(exist_ok=True)
    
    def get_available_spy_options(self, date: str = "2024-10-04") -> list:
        """Get list of available SPY options for a given date using premium API"""
        
        print(f"🔍 Getting available SPY options for {date}...")
        
        url = f"{self.base_url}/v3/reference/options/contracts"
        
        params = {
            'underlying_ticker': 'SPY',
            'contract_type': 'option',
            'expired': 'false',
            'limit': 1000,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    options_list = []
                    for option in data['results']:
                        if option.get('ticker', '').startswith('O:SPY'):
                            options_list.append({
                                'ticker': option['ticker'],
                                'expiry': option.get('expiration_date'),
                                'strike': option.get('strike_price'),
                                'option_type': option.get('contract_type')
                            })
                    
                    print(f"✅ Found {len(options_list)} available SPY options")
                    return options_list
                else:
                    print("⚠️  No options data in response")
                    return []
            else:
                print(f"❌ API error {response.status_code}: {response.text[:200]}")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching options list: {e}")
            return []
    
    def get_recent_trading_dates(self, days_back: int = 7) -> list:
        """Get recent trading dates"""
        dates = []
        current_date = datetime.now().date()
        
        i = 0
        while len(dates) < days_back and i < 14:  # Look back up to 14 days
            check_date = current_date - timedelta(days=i)
            if check_date.weekday() < 5:  # Monday to Friday
                dates.append(check_date.strftime('%Y-%m-%d'))
            i += 1
        
        return dates
    
    def download_spy_options_trades(self, ticker: str, date: str) -> pd.DataFrame:
        """Download trades for specific SPY option ticker"""
        
        url = f"{self.base_url}/v3/trades/{ticker}"
        
        params = {
            'timestamp.gte': f"{date}T09:30:00.000Z",
            'timestamp.lte': f"{date}T16:00:00.000Z",
            'order': 'asc',
            'limit': 50000,  # Higher limit with premium access
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    trades_df = pd.DataFrame(data['results'])
                    
                    # Clean and standardize column names
                    if 'sip_timestamp' in trades_df.columns:
                        trades_df['timestamp'] = trades_df['sip_timestamp']
                    if 'size' not in trades_df.columns and 'quantity' in trades_df.columns:
                        trades_df['size'] = trades_df['quantity']
                    
                    trades_df['ticker'] = ticker
                    trades_df['date'] = date
                    
                    # Parse option details
                    option_info = self._parse_option_ticker(ticker)
                    for key, value in option_info.items():
                        trades_df[key] = value
                    
                    return trades_df
                else:
                    return pd.DataFrame()
            
            elif response.status_code == 429:
                print(f"      ⚠️  Rate limited, waiting...")
                time.sleep(12)  # Wait 12 seconds for premium tier
                return pd.DataFrame()
            
            else:
                return pd.DataFrame()
                
        except Exception as e:
            return pd.DataFrame()
    
    def _parse_option_ticker(self, ticker: str) -> dict:
        """Parse SPY option ticker to extract details"""
        try:
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
    
    def download_comprehensive_spy_data(self, target_expiry: str = None) -> dict:
        """Download comprehensive SPY options data using premium access"""
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE SPY OPTIONS DOWNLOAD (PREMIUM)")
        print(f"{'='*60}")
        
        # Get available options first
        available_options = self.get_available_spy_options()
        
        if not available_options:
            return {'error': 'No available options found'}
        
        # If no target expiry specified, use the nearest Friday
        if not target_expiry:
            # Find the nearest Friday expiry
            expiries = [opt['expiry'] for opt in available_options if opt['expiry']]
            expiries = sorted(list(set(expiries)))
            
            # Use the first expiry that's a Friday and in the future
            for expiry in expiries:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                if expiry_date.weekday() == 4 and expiry_date.date() >= datetime.now().date():  # Friday
                    target_expiry = expiry
                    break
            
            if not target_expiry and expiries:
                target_expiry = expiries[0]  # Fallback to first available
        
        print(f"🎯 Target expiry: {target_expiry}")
        
        # Filter available options for target expiry
        target_options = [opt for opt in available_options if opt['expiry'] == target_expiry]
        
        if not target_options:
            print(f"❌ No options found for expiry {target_expiry}")
            return {'error': f'No options for expiry {target_expiry}'}
        
        print(f"✅ Found {len(target_options)} options for {target_expiry}")
        
        # Get recent trading dates
        recent_dates = self.get_recent_trading_dates(5)
        print(f"📅 Will try dates: {recent_dates}")
        
        results = {
            'target_expiry': target_expiry,
            'available_options': len(target_options),
            'dates_attempted': [],
            'dates_successful': [],
            'total_trades': 0,
            'daily_summaries': {}
        }
        
        # Try to download data for recent dates
        for date in recent_dates:
            print(f"\n📅 Processing {date}...")
            results['dates_attempted'].append(date)
            
            daily_trades = []
            successful_tickers = 0
            
            # Try a sample of options (not all to avoid rate limits)
            sample_options = target_options[::5][:10]  # Every 5th option, max 10
            
            for i, option in enumerate(sample_options, 1):
                ticker = option['ticker']
                print(f"   [{i}/{len(sample_options)}] {ticker} (${option['strike']} {option['option_type']})...")
                
                trades_df = self.download_spy_options_trades(ticker, date)
                
                if len(trades_df) > 0:
                    daily_trades.append(trades_df)
                    successful_tickers += 1
                    print(f"      ✅ {len(trades_df)} trades")
                
                # Rate limiting for premium tier
                time.sleep(0.05)
            
            if daily_trades:
                # Combine all trades for this date
                combined_df = pd.concat(daily_trades, ignore_index=True)
                
                # Add synthetic bid/ask
                combined_df = self._add_synthetic_quotes(combined_df)
                
                # Save enriched data
                output_file = self.output_dir / "enriched" / f"{date}_spy_options.parquet"
                combined_df.to_parquet(output_file, index=False)
                
                # Daily summary
                daily_summary = {
                    'date': date,
                    'total_trades': len(combined_df),
                    'unique_strikes': combined_df['strike'].nunique(),
                    'successful_tickers': successful_tickers,
                    'volume': int(combined_df['size'].sum()),
                    'price_range': [float(combined_df['strike'].min()), float(combined_df['strike'].max())],
                    'avg_option_price': float(combined_df['price'].mean())
                }
                
                results['daily_summaries'][date] = daily_summary
                results['dates_successful'].append(date)
                results['total_trades'] += len(combined_df)
                
                print(f"   ✅ Saved {len(combined_df)} trades, {successful_tickers} active tickers")
                
                # Success! Continue with more dates or stop here
                if len(combined_df) > 100:  # If we got good data, we can continue
                    print(f"   🎉 Good data volume, continuing...")
                
            else:
                print(f"   ❌ No trades found for {date}")
        
        # Save results summary
        summary_file = self.output_dir / "summary" / "premium_download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create CSV summary for easy viewing
        if results['daily_summaries']:
            summary_df = pd.DataFrame(list(results['daily_summaries'].values()))
            summary_df.to_csv(self.output_dir / "summary" / "daily_summaries.csv", index=False)
        
        print(f"\n{'='*60}")
        print(f"PREMIUM DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"🎯 Target expiry: {target_expiry}")
        print(f"✅ Successful dates: {len(results['dates_successful'])}")
        print(f"📊 Total trades: {results['total_trades']:,}")
        
        if results['dates_successful']:
            print(f"\n📈 Daily breakdown:")
            for date in results['dates_successful']:
                summary = results['daily_summaries'][date]
                print(f"   {date}: {summary['total_trades']:,} trades, {summary['unique_strikes']} strikes, {summary['successful_tickers']} tickers")
        
        return results
    
    def _add_synthetic_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic bid/ask quotes"""
        
        if 'price' not in df.columns:
            return df
        
        # Better spread estimation for SPY options
        df['spread_estimate'] = np.where(
            df['price'] > 20, df['price'] * 0.02,  # 2% for expensive options
            np.where(
                df['price'] > 5, df['price'] * 0.05,   # 5% for mid-priced
                np.where(
                    df['price'] > 1, df['price'] * 0.10,  # 10% for cheap options
                    np.maximum(0.01, df['price'] * 0.25)  # 25% for very cheap, min $0.01
                )
            )
        )
        
        # Create bid/ask
        df['bid'] = np.maximum(0.01, df['price'] - df['spread_estimate'] / 2)
        df['ask'] = df['price'] + df['spread_estimate'] / 2
        df['bid_size'] = np.random.randint(1, 100, size=len(df))  # SPY has high volume
        df['ask_size'] = np.random.randint(1, 100, size=len(df))
        df['quote_timestamp'] = df['timestamp']
        df['time_diff_seconds'] = 0.0
        
        return df


def main():
    """Main execution"""
    
    print("🚀 Premium SPY Options Download")
    print("=" * 40)
    
    downloader = PremiumSPYDownloader()
    
    try:
        # Download comprehensive data
        results = downloader.download_comprehensive_spy_data()
        
        if 'error' not in results and results['dates_successful']:
            print(f"\n🎉 Premium download successful!")
            print(f"📁 Data saved to: {downloader.output_dir}")
            print(f"🎯 Ready for dealer positioning analysis")
            
            print(f"\n📋 Next steps:")
            print(f"   1. Analyze the downloaded options data")
            print(f"   2. Calculate dealer positioning and Greeks")
            print(f"   3. Create interactive visualizations")
            print(f"   4. Generate comprehensive reports")
            
        else:
            print(f"\n⚠️  Download not fully successful")
            if 'error' in results:
                print(f"Error: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error during premium download: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()