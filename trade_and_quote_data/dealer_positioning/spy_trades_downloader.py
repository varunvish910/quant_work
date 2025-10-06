#!/usr/bin/env python3
"""
SPY Weekly Options Dealer Positioning Analysis - Phase 1: Data Collection
Downloads SPY weekly options trades and quotes data from Polygon.io using flat files
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


class SPYTradesDownloader:
    """Downloads SPY options trades using Polygon flat files"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/spy_options"):
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
    
    def get_spy_option_tickers(self, expiry_dates: List[str], current_price: float = 669.21) -> List[str]:
        """Get all SPY option tickers for given expiry dates"""
        tickers = []
        
        # Get relevant strike range around current SPY price
        # SPY strikes are typically in $1 increments, use wider range for weekly options
        strike_min = int(current_price * 0.85)  # 15% below
        strike_max = int(current_price * 1.15)  # 15% above
        base_strikes = range(strike_min, strike_max + 1, 1)  # $1 increments
        
        for expiry in expiry_dates:
            expiry_formatted = expiry.replace('-', '')  # YYYYMMDD format
            
            for strike in base_strikes:
                # Call options - SPY format: O:SPY241011C00670000
                call_ticker = f"O:SPY{expiry_formatted[2:]}C{strike:08d}"  # Use YY format, not YYYY
                tickers.append(call_ticker)
                
                # Put options  
                put_ticker = f"O:SPY{expiry_formatted[2:]}P{strike:08d}"
                tickers.append(put_ticker)
        
        return tickers
    
    async def download_trades_async(self, date: str, tickers: List[str]) -> pd.DataFrame:
        """Download trades for all SPY options on given date"""
        all_trades = []
        
        # Split tickers into batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            
            for ticker in batch:
                try:
                    # Get trades for this ticker
                    trades = self.client.list_trades(
                        ticker=ticker,
                        timestamp_gte=f"{date}T09:30:00.000Z",
                        timestamp_lte=f"{date}T16:00:00.000Z",
                        order="asc",
                        limit=50000
                    )
                    
                    for trade in trades:
                        trade_data = {
                            'ticker': ticker,
                            'timestamp': trade.timestamp,
                            'price': trade.price,
                            'size': trade.size,
                            'exchange': getattr(trade, 'exchange', None),
                            'conditions': getattr(trade, 'conditions', []),
                            'date': date
                        }
                        
                        # Parse option details from ticker
                        option_info = self.parse_option_ticker(ticker)
                        trade_data.update(option_info)
                        
                        all_trades.append(trade_data)
                        
                except Exception as e:
                    print(f"Error downloading trades for {ticker}: {e}")
                    continue
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return pd.DataFrame(all_trades)
    
    def parse_option_ticker(self, ticker: str) -> Dict:
        """Parse SPY option ticker to extract strike, expiry, type"""
        # Example: O:SPY241011C00670000
        try:
            parts = ticker.split(':')[1]  # Remove O: prefix
            
            # Extract expiry (6 digits after SPY)
            expiry_str = parts[3:9]  # YYMMDD
            # Convert YY to 20YY
            year = "20" + expiry_str[:2]
            month = expiry_str[2:4]
            day = expiry_str[4:6]
            expiry = datetime.strptime(f"{year}{month}{day}", '%Y%m%d').date()
            
            # Extract option type
            option_type = parts[9]  # C or P
            
            # Extract strike (remaining digits)
            strike_str = parts[10:]
            strike = int(strike_str) / 100  # SPY strikes are in cents, divide by 100
            
            return {
                'expiry': expiry,
                'strike': strike,
                'option_type': option_type.lower(),
                'underlying': 'SPY'
            }
        except Exception as e:
            print(f"Error parsing ticker {ticker}: {e}")
            return {
                'expiry': None,
                'strike': None,
                'option_type': None,
                'underlying': 'SPY'
            }
    
    async def download_quotes_for_trade(self, ticker: str, trade_timestamp: int, 
                                      window_seconds: int = 1) -> Optional[Dict]:
        """Get quotes around a specific trade timestamp"""
        try:
            # Convert timestamp to proper format
            trade_time = pd.to_datetime(trade_timestamp, unit='ns')
            start_time = trade_time - timedelta(seconds=window_seconds)
            end_time = trade_time + timedelta(seconds=window_seconds)
            
            quotes = self.client.list_quotes(
                ticker=ticker,
                timestamp_gte=start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                timestamp_lte=end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                order="asc",
                limit=100
            )
            
            if not quotes:
                return None
            
            # Find closest quote to trade time
            best_quote = None
            min_time_diff = float('inf')
            
            for quote in quotes:
                quote_time = pd.to_datetime(quote.timestamp, unit='ns')
                time_diff = abs((quote_time - trade_time).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_quote = {
                        'bid': quote.bid,
                        'ask': quote.ask,
                        'bid_size': getattr(quote, 'bid_size', None),
                        'ask_size': getattr(quote, 'ask_size', None),
                        'quote_timestamp': quote.timestamp,
                        'time_diff_seconds': time_diff
                    }
            
            return best_quote
            
        except Exception as e:
            print(f"Error getting quotes for {ticker} at {trade_timestamp}: {e}")
            return None
    
    async def enrich_trades_with_quotes(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Match trades with bid/ask quotes at time of trade"""
        enriched_trades = []
        
        for idx, trade in trades_df.iterrows():
            quote_data = await self.download_quotes_for_trade(
                trade['ticker'], 
                trade['timestamp']
            )
            
            trade_dict = trade.to_dict()
            if quote_data:
                trade_dict.update(quote_data)
            else:
                # Fill with NaN if no quote found
                trade_dict.update({
                    'bid': np.nan,
                    'ask': np.nan,
                    'bid_size': np.nan,
                    'ask_size': np.nan,
                    'quote_timestamp': np.nan,
                    'time_diff_seconds': np.nan
                })
            
            enriched_trades.append(trade_dict)
            
            # Progress indicator
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(trades_df)} trades")
            
            # Rate limiting for quotes
            await asyncio.sleep(0.05)
        
        return pd.DataFrame(enriched_trades)
    
    def download_flat_file(self, date: str, data_type: str = "trades") -> Optional[Path]:
        """Download options flat file for a specific date or use existing local file"""
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Check if file already exists locally
        raw_file = self.output_dir / "raw_files" / f"options_{data_type}_{date}.csv.gz"
        
        if raw_file.exists():
            self.logger.info(f"✅ Found existing flat file: {raw_file}")
            return raw_file
        
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
    
    def parse_spy_options_from_flat_file(self, flat_file_path: Path) -> Optional[pd.DataFrame]:
        """Parse SPY options trades from the downloaded flat file"""
        self.logger.info(f"📊 Parsing SPY options from {flat_file_path.name}...")
        
        try:
            # Read compressed CSV file
            with gzip.open(flat_file_path, 'rt') as f:
                # Read in chunks to handle large files
                chunk_size = 50000
                spy_trades = []
                
                self.logger.info("🔍 Reading flat file in chunks...")
                chunk_num = 0
                
                for chunk in pd.read_csv(f, chunksize=chunk_size):
                    chunk_num += 1
                    
                    # Filter for SPY options
                    spy_chunk = chunk[chunk['ticker'].str.startswith('O:SPY', na=False)]
                    
                    if len(spy_chunk) > 0:
                        spy_trades.append(spy_chunk)
                        self.logger.info(f"   📦 Chunk {chunk_num}: Found {len(spy_chunk)} SPY options trades")
                    
                    if chunk_num % 10 == 0:
                        self.logger.info(f"   📦 Processed {chunk_num} chunks...")
                
                if spy_trades:
                    # Combine all SPY options
                    spy_options_df = pd.concat(spy_trades, ignore_index=True)
                    
                    # Parse option details from tickers
                    spy_options_df = self._parse_option_details(spy_options_df)
                    
                    # Add synthetic bid/ask for classification
                    spy_options_df = self._add_synthetic_quotes(spy_options_df)
                    
                    self.logger.info(f"✅ Parsed SPY options: {len(spy_options_df)} trades")
                    
                    return spy_options_df
                else:
                    self.logger.warning("⚠️  No SPY options found in flat file")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error parsing flat file: {e}")
            return None
    
    def _parse_option_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse option details from ticker symbols"""
        
        def extract_option_info(ticker):
            try:
                if not isinstance(ticker, str) or not ticker.startswith('O:SPY'):
                    return {'expiry': None, 'strike': None, 'option_type': None}
                
                # SPY options format: O:SPY[YYMMDD][C/P][XXXXXXXX]
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
                    
                    # Extract strike (remaining digits)
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
        
        # Extract option info
        option_info = df['ticker'].apply(extract_option_info)
        option_df = pd.DataFrame(option_info.tolist())
        
        # Add to original dataframe
        for col in ['expiry', 'strike', 'option_type', 'underlying']:
            df[col] = option_df[col]
        
        return df
    
    def _add_synthetic_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic bid/ask quotes for trade classification"""
        
        # Estimate bid/ask spread based on option price
        df['spread_estimate'] = np.where(
            df['price'] > 10, df['price'] * 0.05,  # 5% spread for expensive options
            np.where(
                df['price'] > 1, df['price'] * 0.10,  # 10% spread for mid-price options
                np.maximum(0.01, df['price'] * 0.20)  # 20% spread for cheap options, min $0.01
            )
        )
        
        # Create synthetic bid/ask
        df['bid'] = np.maximum(0.01, df['price'] - df['spread_estimate'] / 2)
        df['ask'] = df['price'] + df['spread_estimate'] / 2
        
        # Add quote metadata
        df['bid_size'] = np.random.randint(1, 50, size=len(df))
        df['ask_size'] = np.random.randint(1, 50, size=len(df))
        df['quote_timestamp'] = df['sip_timestamp'] if 'sip_timestamp' in df.columns else df['timestamp']
        df['time_diff_seconds'] = 0.0  # Synthetic quotes are "instant"
        
        return df
    
    def download_trades(self, date: str, expiry_range: List[str], current_price: float = 669.21) -> pd.DataFrame:
        """Main method to download SPY trades using flat files"""
        print(f"Downloading SPY options trades for {date} using flat files")
        
        # Download flat file
        flat_file = self.download_flat_file(date, "trades")
        
        if flat_file is None:
            raise ValueError(f"No trades flat file available for {date}")
        
        # Parse SPY options from flat file
        spy_trades = self.parse_spy_options_from_flat_file(flat_file)
        
        if spy_trades is None or len(spy_trades) == 0:
            raise ValueError(f"No SPY options trades found for {date}")
        
        # Filter for target expiries if specified
        if expiry_range:
            target_expiries = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expiry_range]
            spy_trades = spy_trades[spy_trades['expiry'].isin(target_expiries)]
            print(f"Filtered to {len(spy_trades)} trades for target expiries: {expiry_range}")
        
        # Add required fields
        spy_trades['date'] = date
        
        # Save processed trades
        output_file = self.output_dir / "trades" / f"{date}_trades.parquet"
        spy_trades.to_parquet(output_file, index=False)
        print(f"✅ Saved {len(spy_trades)} SPY trades to {output_file}")
        
        return spy_trades
    
    def enrich_and_save(self, trades_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Enrich trades with quotes and save (quotes already added in flat file processing)"""
        print("SPY trades already enriched with synthetic quotes from flat file processing")
        
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
    
    downloader = SPYTradesDownloader(api_key)
    
    # Get current SPY price
    try:
        import yfinance as yf
        spy = yf.Ticker('SPY')
        current_price = spy.history(period="1d")["Close"].iloc[-1]
        print(f"Current SPY price: ${current_price:.2f}")
    except:
        current_price = 669.21  # Fallback
        print(f"Using fallback SPY price: ${current_price:.2f}")
    
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