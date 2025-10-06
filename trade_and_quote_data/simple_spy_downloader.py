#!/usr/bin/env python3
"""
Simple SPY Options Downloader
Direct approach to get any available SPY options data
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import yfinance as yf


class SimpleSPYDownloader:
    """Simple approach to download SPY options data"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/simple_spy"):
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.base_url = "https://api.polygon.io"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
    
    def test_spy_options_directly(self) -> dict:
        """Test direct SPY options tickers that should exist"""
        
        print(f"🧪 Testing direct SPY options access...")
        
        # Get current SPY price for realistic strikes
        try:
            spy = yf.Ticker('SPY')
            current_price = spy.history(period="1d")["Close"].iloc[-1]
            print(f"✓ Current SPY price: ${current_price:.2f}")
        except:
            current_price = 569.0
            print(f"Using fallback SPY price: ${current_price:.2f}")
        
        # Try today's date
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Generate realistic option tickers for current expiry cycle
        # Most recent Friday for weekly options
        next_friday = datetime.now()
        while next_friday.weekday() != 4:  # Find next Friday
            next_friday += timedelta(days=1)
        
        expiry_date = next_friday.strftime('%y%m%d')  # YYMMDD format
        
        print(f"📅 Testing expiry: {next_friday.strftime('%Y-%m-%d')} (formatted: {expiry_date})")
        
        # Test common SPY strikes around current price
        atm_strike = int(current_price)
        test_strikes = [atm_strike - 10, atm_strike - 5, atm_strike, atm_strike + 5, atm_strike + 10]
        
        results = {
            'test_date': today,
            'expiry_tested': next_friday.strftime('%Y-%m-%d'),
            'spy_price': current_price,
            'strikes_tested': test_strikes,
            'successful_tickers': [],
            'failed_tickers': [],
            'total_trades_found': 0
        }
        
        print(f"🎯 Testing strikes around ${atm_strike}: {test_strikes}")
        
        for strike in test_strikes:
            # Test both calls and puts
            for option_type in ['C', 'P']:
                ticker = f"O:SPY{expiry_date}{option_type}{strike:08d}"
                print(f"   Testing {ticker}...")
                
                trades_data = self._test_option_ticker(ticker, today)
                
                if trades_data is not None and len(trades_data) > 0:
                    results['successful_tickers'].append({
                        'ticker': ticker,
                        'strike': strike,
                        'option_type': option_type,
                        'trades_count': len(trades_data)
                    })
                    results['total_trades_found'] += len(trades_data)
                    print(f"      ✅ Found {len(trades_data)} trades!")
                else:
                    results['failed_tickers'].append(ticker)
                    print(f"      ❌ No trades")
                
                time.sleep(0.1)  # Small delay
        
        # Save test results
        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Test Results:")
        print(f"   ✅ Successful tickers: {len(results['successful_tickers'])}")
        print(f"   ❌ Failed tickers: {len(results['failed_tickers'])}")
        print(f"   📈 Total trades found: {results['total_trades_found']}")
        
        return results
    
    def _test_option_ticker(self, ticker: str, date: str) -> pd.DataFrame:
        """Test a specific option ticker for trades"""
        
        url = f"{self.base_url}/v3/trades/{ticker}"
        
        params = {
            'timestamp.gte': f"{date}T09:30:00.000Z",
            'timestamp.lte': f"{date}T16:00:00.000Z",
            'order': 'asc',
            'limit': 1000,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    return pd.DataFrame(data['results'])
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            return None
    
    def download_working_options(self) -> dict:
        """Download data for options that we know work"""
        
        print(f"\n{'='*50}")
        print(f"DOWNLOADING WORKING SPY OPTIONS")
        print(f"{'='*50}")
        
        # First, test to find working tickers
        test_results = self.test_spy_options_directly()
        
        if not test_results['successful_tickers']:
            print("❌ No working option tickers found")
            return test_results
        
        print(f"\n📥 Downloading full data for {len(test_results['successful_tickers'])} working tickers...")
        
        all_trades = []
        successful_downloads = 0
        
        for ticker_info in test_results['successful_tickers']:
            ticker = ticker_info['ticker']
            print(f"   Downloading {ticker}...")
            
            # Try multiple recent dates
            recent_dates = []
            current_date = datetime.now()
            for i in range(5):  # Last 5 trading days
                check_date = current_date - timedelta(days=i)
                if check_date.weekday() < 5:  # Weekday
                    recent_dates.append(check_date.strftime('%Y-%m-%d'))
            
            ticker_trades = []
            
            for date in recent_dates:
                trades_df = self._download_full_option_data(ticker, date)
                if trades_df is not None and len(trades_df) > 0:
                    trades_df['download_date'] = date
                    ticker_trades.append(trades_df)
            
            if ticker_trades:
                combined_ticker_df = pd.concat(ticker_trades, ignore_index=True)
                all_trades.append(combined_ticker_df)
                successful_downloads += 1
                print(f"      ✅ {len(combined_ticker_df)} trades across {len(ticker_trades)} dates")
        
        if all_trades:
            # Combine all data
            final_df = pd.concat(all_trades, ignore_index=True)
            
            # Add option details
            final_df = self._add_option_details(final_df)
            
            # Add synthetic quotes
            final_df = self._add_synthetic_quotes(final_df)
            
            # Save processed data
            output_file = self.output_dir / "processed" / "spy_options_data.parquet"
            final_df.to_parquet(output_file, index=False)
            
            print(f"\n🎉 Successfully downloaded SPY options data!")
            print(f"📊 Total trades: {len(final_df):,}")
            print(f"📁 Saved to: {output_file}")
            
            # Create summary
            summary = {
                'total_trades': len(final_df),
                'unique_tickers': final_df['ticker'].nunique(),
                'unique_strikes': final_df['strike'].nunique() if 'strike' in final_df.columns else 0,
                'date_range': [final_df['download_date'].min(), final_df['download_date'].max()],
                'successful_downloads': successful_downloads,
                'file_path': str(output_file)
            }
            
            with open(self.output_dir / "download_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return summary
        else:
            print("❌ No data successfully downloaded")
            return {'error': 'No data downloaded'}
    
    def _download_full_option_data(self, ticker: str, date: str) -> pd.DataFrame:
        """Download full data for a specific option ticker"""
        
        url = f"{self.base_url}/v3/trades/{ticker}"
        
        params = {
            'timestamp.gte': f"{date}T09:30:00.000Z",
            'timestamp.lte': f"{date}T16:00:00.000Z",
            'order': 'asc',
            'limit': 50000,  # Max trades
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['ticker'] = ticker
                    return df
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            return None
    
    def _add_option_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add option details by parsing tickers"""
        
        def parse_ticker(ticker):
            try:
                if not ticker.startswith('O:SPY'):
                    return {'strike': None, 'option_type': None, 'expiry': None}
                
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
                    if strike_str.isdigit():
                        strike = float(strike_str) / 1000
                    else:
                        strike = None
                    
                    return {
                        'strike': strike,
                        'option_type': option_type,
                        'expiry': expiry,
                        'underlying': 'SPY'
                    }
                
                return {'strike': None, 'option_type': None, 'expiry': None, 'underlying': 'SPY'}
                
            except Exception:
                return {'strike': None, 'option_type': None, 'expiry': None, 'underlying': 'SPY'}
        
        # Parse all tickers
        parsed_data = df['ticker'].apply(parse_ticker)
        parsed_df = pd.DataFrame(parsed_data.tolist())
        
        # Add to original dataframe
        for col in parsed_df.columns:
            df[col] = parsed_df[col]
        
        return df
    
    def _add_synthetic_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic bid/ask quotes"""
        
        if 'price' not in df.columns:
            return df
        
        # Estimate spread
        df['spread_estimate'] = np.where(
            df['price'] > 10, df['price'] * 0.05,
            np.where(
                df['price'] > 1, df['price'] * 0.10,
                np.maximum(0.01, df['price'] * 0.20)
            )
        )
        
        df['bid'] = np.maximum(0.01, df['price'] - df['spread_estimate'] / 2)
        df['ask'] = df['price'] + df['spread_estimate'] / 2
        df['bid_size'] = np.random.randint(1, 50, size=len(df))
        df['ask_size'] = np.random.randint(1, 50, size=len(df))
        
        # Use existing timestamp or create one
        if 'sip_timestamp' in df.columns:
            df['quote_timestamp'] = df['sip_timestamp']
        elif 'timestamp' in df.columns:
            df['quote_timestamp'] = df['timestamp']
        
        df['time_diff_seconds'] = 0.0
        
        return df


def main():
    """Main execution"""
    
    print("🚀 Simple SPY Options Download")
    print("=" * 30)
    
    downloader = SimpleSPYDownloader()
    
    try:
        # Download working options data
        results = downloader.download_working_options()
        
        if 'error' not in results and 'total_trades' in results:
            print(f"\n🎉 SPY options download successful!")
            print(f"📊 Downloaded: {results['total_trades']:,} trades")
            print(f"🎯 Tickers: {results['successful_downloads']}")
            print(f"📁 Data ready for analysis!")
            
        else:
            print(f"\n⚠️  Download not successful")
            if 'error' in results:
                print(f"Error: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()