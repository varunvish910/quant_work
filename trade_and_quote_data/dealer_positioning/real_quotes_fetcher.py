#!/usr/bin/env python3
"""
Real Quotes Fetcher for SPY Options
Fetches real bid/ask quotes for each trade using individual Polygon API calls
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealQuotesFetcher:
    """Fetches real quotes for SPY options trades using individual API calls"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit_delay = 0.1  # 100ms between requests to avoid rate limits
        
    def fetch_quotes_for_trades(self, trades_df: pd.DataFrame, 
                              max_workers: int = 10, 
                              window_seconds: int = 1) -> pd.DataFrame:
        """
        Fetch real quotes for all trades, matching within 1 second of sip_timestamp
        
        Args:
            trades_df: DataFrame with SPY trades
            max_workers: Number of concurrent API requests
            window_seconds: Time window for quote matching (default 1 second)
        
        Returns:
            DataFrame with real bid/ask quotes added
        """
        logger.info(f"Fetching real quotes for {len(trades_df):,} SPY trades")
        
        # Group trades by ticker for efficient batching
        ticker_groups = trades_df.groupby('ticker')
        total_tickers = len(ticker_groups)
        
        logger.info(f"Processing {total_tickers} unique SPY option tickers")
        
        all_enriched_trades = []
        
        # Process tickers in batches to manage rate limits
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for ticker, trades_group in ticker_groups:
                future = executor.submit(
                    self._fetch_quotes_for_ticker_group, 
                    ticker, 
                    trades_group, 
                    window_seconds
                )
                futures.append(future)
            
            # Process completed futures with progress bar
            with tqdm(total=len(futures), desc="Fetching quotes") as pbar:
                for future in as_completed(futures):
                    try:
                        enriched_group = future.result()
                        if enriched_group is not None:
                            all_enriched_trades.append(enriched_group)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing ticker group: {e}")
                        pbar.update(1)
        
        if all_enriched_trades:
            result_df = pd.concat(all_enriched_trades, ignore_index=True)
            logger.info(f"✅ Successfully enriched {len(result_df):,} trades with real quotes")
            return result_df
        else:
            logger.error("❌ No trades were enriched with quotes")
            return trades_df
    
    def _fetch_quotes_for_ticker_group(self, ticker: str, trades_group: pd.DataFrame, 
                                     window_seconds: int) -> Optional[pd.DataFrame]:
        """Fetch quotes for all trades of a specific ticker"""
        
        try:
            logger.debug(f"Processing {len(trades_group)} trades for {ticker}")
            
            # Get time range for this ticker's trades
            min_ts = trades_group['sip_timestamp'].min()
            max_ts = trades_group['sip_timestamp'].max()
            
            # Convert to datetime and add buffer
            start_time = pd.to_datetime(min_ts, unit='ns') - pd.Timedelta(seconds=window_seconds)
            end_time = pd.to_datetime(max_ts, unit='ns') + pd.Timedelta(seconds=window_seconds)
            
            # Fetch quotes for this time range
            quotes_df = self._fetch_quotes_for_time_range(ticker, start_time, end_time)
            
            if quotes_df is None or len(quotes_df) == 0:
                logger.warning(f"No quotes found for {ticker}, using synthetic quotes")
                return self._add_synthetic_quotes(trades_group)
            
            # Match trades with quotes using timestamp proximity
            enriched_trades = self._match_trades_with_quotes(trades_group, quotes_df, window_seconds)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return enriched_trades
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return self._add_synthetic_quotes(trades_group)
    
    def _fetch_quotes_for_time_range(self, ticker: str, start_time: datetime, 
                                   end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch quotes for a specific ticker and time range"""
        
        try:
            # Format timestamps for API
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            # Polygon quotes API endpoint
            quotes_url = f"{self.base_url}/v3/quotes/{ticker}"
            params = {
                'timestamp.gte': start_str,
                'timestamp.lte': end_str,
                'order': 'asc',
                'limit': 50000,
                'apikey': self.api_key
            }
            
            response = requests.get(quotes_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    quotes_df = pd.DataFrame(data['results'])
                    
                    # Standardize column names based on actual Polygon API v3 format
                    column_mapping = {
                        'sip_timestamp': 'quote_timestamp',
                        'bid_price': 'bid',
                        'ask_price': 'ask', 
                        'bid_size': 'bid_size',
                        'ask_size': 'ask_size'
                    }
                    
                    # Rename columns that exist
                    for old_col, new_col in column_mapping.items():
                        if old_col in quotes_df.columns:
                            quotes_df = quotes_df.rename(columns={old_col: new_col})
                    
                    # Ensure we have required columns
                    required_cols = ['quote_timestamp', 'bid', 'ask']
                    if all(col in quotes_df.columns for col in required_cols):
                        # Convert timestamp to datetime
                        quotes_df['quote_datetime'] = pd.to_datetime(quotes_df['quote_timestamp'], unit='ns')
                        return quotes_df
                    else:
                        logger.warning(f"Missing required columns in quotes for {ticker}")
                        return None
                else:
                    logger.debug(f"No quotes in API response for {ticker}")
                    return None
            else:
                logger.warning(f"API error for {ticker}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching quotes for {ticker}: {e}")
            return None
    
    def _match_trades_with_quotes(self, trades_group: pd.DataFrame, quotes_df: pd.DataFrame,
                                window_seconds: int) -> pd.DataFrame:
        """Match each trade with the closest quote within the time window"""
        
        try:
            trades_copy = trades_group.copy()
            
            # Convert trade timestamps to datetime
            trades_copy['trade_datetime'] = pd.to_datetime(trades_copy['sip_timestamp'], unit='ns')
            
            # Sort both by timestamp
            trades_copy = trades_copy.sort_values('trade_datetime')
            quotes_df = quotes_df.sort_values('quote_datetime')
            
            # For each trade, find the closest quote within the window
            bid_values = []
            ask_values = []
            bid_sizes = []
            ask_sizes = []
            quote_timestamps = []
            time_diffs = []
            
            for _, trade in trades_copy.iterrows():
                trade_time = trade['trade_datetime']
                
                # Find quotes within the time window
                window_start = trade_time - pd.Timedelta(seconds=window_seconds)
                window_end = trade_time + pd.Timedelta(seconds=window_seconds)
                
                window_quotes = quotes_df[
                    (quotes_df['quote_datetime'] >= window_start) & 
                    (quotes_df['quote_datetime'] <= window_end)
                ]
                
                if len(window_quotes) > 0:
                    # Find the closest quote to trade time
                    time_diffs_window = np.abs((window_quotes['quote_datetime'] - trade_time).dt.total_seconds())
                    closest_idx = time_diffs_window.idxmin()
                    closest_quote = window_quotes.loc[closest_idx]
                    
                    bid_values.append(closest_quote['bid'])
                    ask_values.append(closest_quote['ask'])
                    bid_sizes.append(closest_quote.get('bid_size', 10))
                    ask_sizes.append(closest_quote.get('ask_size', 10))
                    quote_timestamps.append(closest_quote['quote_timestamp'])
                    time_diffs.append(time_diffs_window.loc[closest_idx])
                else:
                    # No quotes in window, use synthetic
                    synthetic_bid, synthetic_ask = self._estimate_bid_ask(trade['price'])
                    bid_values.append(synthetic_bid)
                    ask_values.append(synthetic_ask)
                    bid_sizes.append(10)
                    ask_sizes.append(10)
                    quote_timestamps.append(trade['sip_timestamp'])
                    time_diffs.append(999.0)  # Flag as synthetic
            
            # Add quote data to trades
            trades_copy['bid'] = bid_values
            trades_copy['ask'] = ask_values
            trades_copy['bid_size'] = bid_sizes
            trades_copy['ask_size'] = ask_sizes
            trades_copy['quote_timestamp'] = quote_timestamps
            trades_copy['time_diff_seconds'] = time_diffs
            
            # Remove temporary datetime column
            trades_copy = trades_copy.drop('trade_datetime', axis=1)
            
            return trades_copy
            
        except Exception as e:
            logger.error(f"Error matching trades with quotes: {e}")
            return self._add_synthetic_quotes(trades_group)
    
    def _estimate_bid_ask(self, price: float) -> Tuple[float, float]:
        """Estimate realistic bid/ask spread as fallback"""
        if price < 0.5:
            spread = 0.01
        elif price < 2:
            spread = 0.05
        elif price < 10:
            spread = price * 0.02
        else:
            spread = price * 0.015
        
        bid = max(0.01, price - spread/2)
        ask = price + spread/2
        return bid, ask
    
    def _add_synthetic_quotes(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic quotes as fallback when real quotes unavailable"""
        
        trades_copy = trades_df.copy()
        
        # Estimate bid/ask for each trade
        bid_ask_data = trades_copy['price'].apply(lambda p: pd.Series(self._estimate_bid_ask(p)))
        trades_copy['bid'] = bid_ask_data[0]
        trades_copy['ask'] = bid_ask_data[1]
        
        # Add synthetic quote metadata
        trades_copy['bid_size'] = np.random.randint(5, 50, len(trades_copy))
        trades_copy['ask_size'] = np.random.randint(5, 50, len(trades_copy))
        trades_copy['quote_timestamp'] = trades_copy['sip_timestamp']
        trades_copy['time_diff_seconds'] = 999.0  # Flag as synthetic
        
        return trades_copy


def main():
    """Example usage"""
    
    # Load real SPY trades
    import os
    trades_file = "../data_management/real_spy_trades_sample.parquet"
    
    if os.path.exists(trades_file):
        trades_df = pd.read_parquet(trades_file)
        print(f"Loaded {len(trades_df)} trades")
        
        # Initialize quotes fetcher
        api_key = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        fetcher = RealQuotesFetcher(api_key)
        
        # Fetch real quotes
        enriched_df = fetcher.fetch_quotes_for_trades(trades_df.head(100))  # Test with first 100
        
        print(f"Enriched {len(enriched_df)} trades with real quotes")
        print("\nReal vs Synthetic quotes:")
        real_quotes = enriched_df[enriched_df['time_diff_seconds'] < 999]
        synthetic_quotes = enriched_df[enriched_df['time_diff_seconds'] >= 999]
        print(f"Real quotes: {len(real_quotes)} ({len(real_quotes)/len(enriched_df)*100:.1f}%)")
        print(f"Synthetic fallback: {len(synthetic_quotes)} ({len(synthetic_quotes)/len(enriched_df)*100:.1f}%)")
    else:
        print(f"Test file not found: {trades_file}")


if __name__ == "__main__":
    main()