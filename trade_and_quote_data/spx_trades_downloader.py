#!/usr/bin/env python3
"""
SPX Weekly Options Dealer Positioning Analysis - Phase 1: Data Collection
Downloads SPX weekly options trades and quotes data from Polygon.io
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
from pathlib import Path
import time
from polygon import RESTClient


class SPXTradesDownloader:
    """Downloads SPX options trades and enriches with quote data"""
    
    def __init__(self, api_key: str, output_dir: str = "data/spx_options"):
        self.api_key = api_key
        self.client = RESTClient(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "trades").mkdir(exist_ok=True)
        (self.output_dir / "quotes").mkdir(exist_ok=True)
        (self.output_dir / "classified").mkdir(exist_ok=True)
        (self.output_dir / "greeks").mkdir(exist_ok=True)
    
    def get_spx_option_tickers(self, expiry_dates: List[str]) -> List[str]:
        """Get all SPX option tickers for given expiry dates"""
        tickers = []
        
        # Get current SPX price to determine relevant strike range
        # For now, use a wide range around typical SPX levels
        base_strikes = range(5000, 6500, 25)  # Adjust based on current SPX level
        
        for expiry in expiry_dates:
            expiry_formatted = expiry.replace('-', '')  # YYYYMMDD format
            
            for strike in base_strikes:
                # Call options
                call_ticker = f"O:SPX{expiry_formatted}C{strike:08d}000"
                tickers.append(call_ticker)
                
                # Put options  
                put_ticker = f"O:SPX{expiry_formatted}P{strike:08d}000"
                tickers.append(put_ticker)
        
        return tickers
    
    async def download_trades_async(self, date: str, tickers: List[str]) -> pd.DataFrame:
        """Download trades for all SPX options on given date"""
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
        """Parse option ticker to extract strike, expiry, type"""
        # Example: O:SPX20251005C00055000000
        try:
            parts = ticker.split(':')[1]  # Remove O: prefix
            
            # Extract expiry (8 digits after SPX)
            expiry_str = parts[3:11]  # YYYYMMDD
            expiry = datetime.strptime(expiry_str, '%Y%m%d').date()
            
            # Extract option type
            option_type = parts[11]  # C or P
            
            # Extract strike (remaining digits, divide by 1000)
            strike_str = parts[12:]
            strike = int(strike_str) / 1000
            
            return {
                'expiry': expiry,
                'strike': strike,
                'option_type': option_type.lower(),
                'underlying': 'SPX'
            }
        except Exception as e:
            print(f"Error parsing ticker {ticker}: {e}")
            return {
                'expiry': None,
                'strike': None,
                'option_type': None,
                'underlying': 'SPX'
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
    
    def download_trades(self, date: str, expiry_range: List[str]) -> pd.DataFrame:
        """Main method to download all SPX trades for specific expiries"""
        print(f"Downloading SPX options trades for {date}")
        
        # Get all relevant option tickers
        tickers = self.get_spx_option_tickers(expiry_range)
        print(f"Found {len(tickers)} option tickers to download")
        
        # Download trades asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trades_df = loop.run_until_complete(
            self.download_trades_async(date, tickers)
        )
        loop.close()
        
        print(f"Downloaded {len(trades_df)} trades")
        
        # Save raw trades
        output_file = self.output_dir / "trades" / f"{date}_trades.parquet"
        trades_df.to_parquet(output_file, index=False)
        print(f"Saved trades to {output_file}")
        
        return trades_df
    
    def enrich_and_save(self, trades_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Enrich trades with quotes and save"""
        print("Enriching trades with quote data...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        enriched_df = loop.run_until_complete(
            self.enrich_trades_with_quotes(trades_df)
        )
        loop.close()
        
        # Save enriched data
        output_file = self.output_dir / "trades" / f"{date}_enriched_trades.parquet"
        enriched_df.to_parquet(output_file, index=False)
        print(f"Saved enriched trades to {output_file}")
        
        return enriched_df


def main():
    """Example usage"""
    # You'll need to set your Polygon API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Please set POLYGON_API_KEY environment variable")
        return
    
    downloader = SPXTradesDownloader(api_key)
    
    # Download trades for October 5, 2025 with weekly expiry
    date = "2025-10-05"
    expiry_range = ["2025-10-11"]  # Weekly expiry
    
    # Download raw trades
    trades_df = downloader.download_trades(date, expiry_range)
    
    # Enrich with quotes
    enriched_df = downloader.enrich_and_save(trades_df, date)
    
    print(f"Analysis complete. Processed {len(enriched_df)} trades")
    print("\nSample of enriched data:")
    print(enriched_df.head())


if __name__ == "__main__":
    main()