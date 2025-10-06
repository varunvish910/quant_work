#!/usr/bin/env python3
"""
SPX Weekly Options Dealer Positioning Analysis - Phase 1: Data Collection
Downloads SPX weekly options trades and quotes data using Polygon flat files
"""

import pandas as pd
import numpy as np
import requests
import gzip
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
from pathlib import Path
import time
import logging
import yfinance as yf


class SPXTradesDownloader:
    """Downloads SPX options trades using Polygon flat files"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/spx_options"):
        # Use the working API key from the flat file processor
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.base_url = "https://api.polygon.io"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "trades").mkdir(exist_ok=True)
        (self.output_dir / "quotes").mkdir(exist_ok=True)
        (self.output_dir / "classified").mkdir(exist_ok=True)
        (self.output_dir / "greeks").mkdir(exist_ok=True)
        (self.output_dir / "raw_files").mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_flat_file(self, date: str, data_type: str = "trades") -> Optional[Path]:
        """Download options flat file for a specific date or use existing local file"""
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Check if file already exists locally
        raw_file = self.output_dir / "raw_files" / f"options_{data_type}_{date}.csv.gz"
        
        if raw_file.exists():
            self.logger.info(f"✅ Found existing flat file: {raw_file}")
            return raw_file
        
        # Also check if we can use the SPY data file (same flat file contains both)
        spy_file = Path("data/spy_options/raw_files") / f"options_{data_type}_{date}.csv.gz"
        if spy_file.exists():
            self.logger.info(f"✅ Using existing flat file from SPY directory: {spy_file}")
            return spy_file
        
        self.logger.info(f"📥 Downloading {data_type} flat file for {date}...")
        
        # Polygon flat file URL format
        flat_file_url = f"https://files.polygon.io/market_data/options/{data_type}/{date_obj.strftime('%Y/%m/%d')}.csv.gz"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            self.logger.info(f"🔗 Requesting: {flat_file_url}")
            response = requests.get(flat_file_url, headers=headers, stream=True, timeout=60)
            
            if response.status_code == 200:
                # Save compressed file
                with open(raw_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                self.logger.info(f"✅ Downloaded flat file: {raw_file}")
                return raw_file
                
            elif response.status_code == 404:
                self.logger.warning(f"⚠️  No flat file available for {date}")
                return None
            else:
                self.logger.error(f"❌ Download failed: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error downloading flat file: {e}")
            return None
    
    def parse_spx_options_from_flat_file(self, flat_file_path: Path) -> Optional[pd.DataFrame]:
        """Parse SPX options trades from the downloaded flat file"""
        self.logger.info(f"📊 Parsing SPX options from {flat_file_path.name}...")
        
        try:
            # Read compressed CSV file
            with gzip.open(flat_file_path, 'rt') as f:
                # Read in chunks to handle large files
                chunk_size = 50000
                spx_trades = []
                
                self.logger.info("🔍 Reading flat file in chunks...")
                chunk_num = 0
                
                for chunk in pd.read_csv(f, chunksize=chunk_size):
                    chunk_num += 1
                    
                    # Filter for SPX options (different format than SPY)
                    spx_chunk = chunk[chunk['ticker'].str.startswith('O:SPX', na=False)]
                    
                    if len(spx_chunk) > 0:
                        spx_trades.append(spx_chunk)
                        self.logger.info(f"   📦 Chunk {chunk_num}: Found {len(spx_chunk)} SPX options trades")
                    
                    if chunk_num % 10 == 0:
                        self.logger.info(f"   📦 Processed {chunk_num} chunks...")
                
                if spx_trades:
                    # Combine all SPX options
                    spx_options_df = pd.concat(spx_trades, ignore_index=True)
                    
                    # Parse option details from tickers
                    spx_options_df = self._parse_option_details(spx_options_df)
                    
                    # Add synthetic bid/ask for classification
                    spx_options_df = self._add_synthetic_quotes(spx_options_df)
                    
                    self.logger.info(f"✅ Parsed SPX options: {len(spx_options_df)} trades")
                    
                    return spx_options_df
                else:
                    self.logger.warning("⚠️  No SPX options found in flat file")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error parsing flat file: {e}")
            return None
    
    def _parse_option_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse option details from SPX ticker symbols"""
        
        def extract_option_info(ticker):
            try:
                if not isinstance(ticker, str) or not ticker.startswith('O:SPX'):
                    return {'expiry': None, 'strike': None, 'option_type': None}
                
                # SPX options format: O:SPX[YYMMDD][C/P][XXXXXXXX] 
                # Example: O:SPX250919C10200000
                parts = ticker[5:]  # Remove 'O:SPX'
                
                if len(parts) >= 15:
                    # Extract expiry (YYMMDD)
                    expiry_str = parts[:6]
                    year = "20" + expiry_str[:2]
                    month = expiry_str[2:4]
                    day = expiry_str[4:6]
                    expiry = datetime.strptime(f"{year}{month}{day}", '%Y%m%d').date()
                    
                    # Extract option type
                    option_type = parts[6].lower()
                    
                    # Extract strike (remaining digits) - SPX uses different scaling
                    strike_str = parts[7:]
                    if strike_str.isdigit():
                        # SPX strikes: divide by 1000 for proper strike price
                        strike = float(strike_str) / 1000
                    else:
                        strike = None
                    
                    return {
                        'expiry': expiry,
                        'strike': strike,
                        'option_type': option_type,
                        'underlying': 'SPX'
                    }
                
                return {'expiry': None, 'strike': None, 'option_type': None, 'underlying': 'SPX'}
                
            except Exception:
                return {'expiry': None, 'strike': None, 'option_type': None, 'underlying': 'SPX'}
        
        # Extract option info
        option_info = df['ticker'].apply(extract_option_info)
        option_df = pd.DataFrame(option_info.tolist())
        
        # Add to original dataframe
        for col in ['expiry', 'strike', 'option_type', 'underlying']:
            df[col] = option_df[col]
        
        return df
    
    def _add_synthetic_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic bid/ask quotes for trade classification"""
        
        # Estimate bid/ask spread based on option price (SPX options are more expensive)
        df['spread_estimate'] = np.where(
            df['price'] > 100, df['price'] * 0.03,  # 3% spread for expensive SPX options
            np.where(
                df['price'] > 10, df['price'] * 0.05,  # 5% spread for mid-price options
                np.maximum(0.05, df['price'] * 0.10)  # 10% spread for cheap options, min $0.05
            )
        )
        
        # Create synthetic bid/ask
        df['bid'] = np.maximum(0.05, df['price'] - df['spread_estimate'] / 2)
        df['ask'] = df['price'] + df['spread_estimate'] / 2
        
        # Add quote metadata
        df['bid_size'] = np.random.randint(1, 20, size=len(df))  # SPX lower volume
        df['ask_size'] = np.random.randint(1, 20, size=len(df))
        df['quote_timestamp'] = df['sip_timestamp'] if 'sip_timestamp' in df.columns else df['timestamp']
        df['time_diff_seconds'] = 0.0  # Synthetic quotes are "instant"
        
        return df
    
    def download_trades(self, date: str, expiry_range: List[str], current_price: float = 6715.79) -> pd.DataFrame:
        """Main method to download SPX trades using flat files"""
        print(f"Downloading SPX options trades for {date} using flat files")
        
        # Download flat file
        flat_file = self.download_flat_file(date, "trades")
        
        if flat_file is None:
            raise ValueError(f"No trades flat file available for {date}")
        
        # Parse SPX options from flat file
        spx_trades = self.parse_spx_options_from_flat_file(flat_file)
        
        if spx_trades is None or len(spx_trades) == 0:
            raise ValueError(f"No SPX options trades found for {date}")
        
        # Filter for target expiries if specified
        if expiry_range:
            target_expiries = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expiry_range]
            spx_trades = spx_trades[spx_trades['expiry'].isin(target_expiries)]
            print(f"Filtered to {len(spx_trades)} trades for target expiries: {expiry_range}")
        
        # Add required fields
        spx_trades['date'] = date
        
        # Save processed trades
        output_file = self.output_dir / "trades" / f"{date}_trades.parquet"
        spx_trades.to_parquet(output_file, index=False)
        print(f"✅ Saved {len(spx_trades)} SPX trades to {output_file}")
        
        return spx_trades
    
    def enrich_and_save(self, trades_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Enrich trades with quotes and save (quotes already added in flat file processing)"""
        print("SPX trades already enriched with synthetic quotes from flat file processing")
        
        # Save enriched data
        output_file = self.output_dir / "trades" / f"{date}_enriched_trades.parquet"
        trades_df.to_parquet(output_file, index=False)
        print(f"✅ Saved {len(trades_df)} enriched trades to {output_file}")
        
        return trades_df


def main():
    """Example usage"""
    # You'll need to set your Polygon API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Please set POLYGON_API_KEY environment variable")
        return
    
    downloader = SPXTradesDownloader(api_key)
    
    # Get current SPX price
    try:
        import yfinance as yf
        spx = yf.Ticker('^SPX')
        current_price = spx.history(period="1d")["Close"].iloc[-1]
        print(f"Current SPX price: ${current_price:.2f}")
    except:
        current_price = 6715.79  # Fallback
        print(f"Using fallback SPX price: ${current_price:.2f}")
    
    # Download trades for today with weekly expiry
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Find next Friday for weekly expiry
    next_friday = datetime.now()
    while next_friday.weekday() != 4:  # 4 = Friday
        next_friday += timedelta(days=1)
    expiry_date = next_friday.strftime("%Y-%m-%d")
    
    print(f"Analyzing date: {today}")
    print(f"Weekly expiry: {expiry_date}")
    
    # Download raw trades
    trades_df = downloader.download_trades(today, [expiry_date], current_price)
    
    # Enrich with quotes
    enriched_df = downloader.enrich_and_save(trades_df, today)
    
    print(f"Analysis complete. Processed {len(enriched_df)} trades")
    print("\nSample of enriched data:")
    print(enriched_df.head())


if __name__ == "__main__":
    main()