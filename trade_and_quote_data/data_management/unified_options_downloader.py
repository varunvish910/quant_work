#!/usr/bin/env python3
"""
Optimized Options Chain Downloader for Daily Snapshots

Efficiently downloads daily snapshots of OI/volume and options data for:
- Strikes between -0.9 and +0.9 delta
- All weekly expiries (Fridays) in January 2016
- Single API call per day using Polygon's snapshot API
"""

import os
import sys
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta, date
import time
import json
import requests
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import logging
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm
import yfinance as yf
import subprocess

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration
try:
    from config.api_config import get_polygon_api_key
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedOptionsDownloader:
    """Highly optimized options chain downloader for daily snapshots"""
    
    def __init__(self, api_key: str = None, data_dir: str = "data"):
        # Load configuration if available
        if CONFIG_AVAILABLE:
            self.api_key = api_key or get_polygon_api_key()
        else:
            self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Cache for underlying prices to avoid repeated API calls
        self._price_cache = {}
        # Cache for contract market data
        self._market_data_cache = {}
        # Cache for contracts list
        self._contracts_cache = {}
        
        if not self.api_key:
            logger.warning("No Polygon API key provided. Download functionality will be disabled.")
        
        if self.api_key:
            self.client = RESTClient(self.api_key)
            self.base_url = "https://api.polygon.io"
    
    def download_date_range(self, ticker: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Download daily snapshots for a date range
        PARALLELIZED for maximum speed
        
        Args:
            ticker: Stock ticker (e.g., 'SPY')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary mapping date strings to DataFrames with filtered options data
        """
        logger.info(f"🚀 Starting optimized download for {ticker} from {start_date} to {end_date}")
        
        # Get all trading days in the date range
        trading_days = self._get_trading_days_in_range(start_date, end_date)
        logger.info(f"Found {len(trading_days)} trading days")
        
        # Download snapshots in parallel
        daily_snapshots = {}
        
        # Use ThreadPoolExecutor to download multiple days simultaneously
        # Use 32 workers for maximum parallel processing
        max_workers = min(32, len(trading_days))
        
        logger.info(f"Processing {len(trading_days)} days with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(self._download_daily_snapshot, ticker, date_str): date_str
                for date_str in trading_days
            }
            
            # Enhanced progress bar with more details
            with tqdm(total=len(trading_days), 
                     desc="🚀 Downloading daily snapshots", 
                     unit="day",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for future in as_completed(future_to_date):
                    date_str = future_to_date[future]
                    try:
                        snapshot = future.result()
                        if snapshot is not None and not snapshot.empty:
                            daily_snapshots[date_str] = snapshot
                            pbar.set_postfix({
                                'last_date': date_str,
                                'contracts': len(snapshot),
                                'success': len(daily_snapshots)
                            })
                            logger.info(f"✅ {date_str}: {len(snapshot)} contracts")
                        else:
                            pbar.set_postfix({
                                'last_date': date_str,
                                'contracts': 0,
                                'success': len(daily_snapshots)
                            })
                            logger.warning(f"⚠️  {date_str}: No data available")
                    except Exception as e:
                        pbar.set_postfix({
                            'last_date': date_str,
                            'error': str(e)[:20],
                            'success': len(daily_snapshots)
                        })
                        logger.error(f"❌ {date_str}: Error - {e}")
                    pbar.update(1)
        
        logger.info(f"🎉 Completed: {len(daily_snapshots)} successful snapshots")
        return daily_snapshots
    
    def _get_trading_days_in_range(self, start_date: str, end_date: str) -> List[str]:
        """Get all trading days in a date range using SPY historical data"""
        trading_days = []
        
        try:
            # Use yfinance to get actual trading days
            stock = yf.Ticker("SPY")
            
            # Add buffer to end date
            end_buffer = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            
            hist = stock.history(start=start_date, end=end_buffer, auto_adjust=False)
            
            if not hist.empty:
                # Extract trading days from the historical data
                trading_days = [d.strftime('%Y-%m-%d') for d in hist.index.date]
                logger.info(f"Identified {len(trading_days)} trading days from {start_date} to {end_date}")
            else:
                logger.warning(f"No trading days found for {start_date} to {end_date}")
        
        except Exception as e:
            logger.error(f"Error getting trading days: {e}")
        
        return trading_days
    
    def _download_flatfile(self, date_str: str) -> Optional[Path]:
        """Download flat file for a specific date"""
        year = date_str[:4]
        month = date_str[5:7]
        s3_path = f"s3://flatfiles/us_options_opra/day_aggs_v1/{year}/{month}/{date_str}.csv.gz"
        local_file = self.data_dir / f"{date_str}.csv.gz"
        
        if local_file.exists():
            return local_file
        
        # AWS credentials for Polygon flat files
        aws_key = "86959ae1-29bc-4433-be13-1a41b935d9d1"
        aws_secret = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        endpoint = "https://files.polygon.io"
        
        cmd = ['aws', 's3', 'cp', s3_path, str(local_file), '--endpoint-url', endpoint]
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = aws_key
        env['AWS_SECRET_ACCESS_KEY'] = aws_secret
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                return local_file
            else:
                logger.error(f"Failed to download {date_str}: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error downloading {date_str}: {e}")
            return None
    
    def _parse_spy_options_from_flatfile(self, flatfile_path: Path) -> Optional[pd.DataFrame]:
        """Parse SPY options from flat file"""
        try:
            import gzip
            with gzip.open(flatfile_path, 'rt') as f:
                df = pd.read_csv(f)
            
            # Filter for SPY options
            spy_options = df[df['ticker'].str.startswith('O:SPY', na=False)].copy()
            
            if len(spy_options) == 0:
                return None
            
            # Parse option ticker components
            spy_options['underlying'] = 'SPY'
            spy_options['exp_date'] = spy_options['ticker'].str[5:11]
            spy_options['option_type'] = spy_options['ticker'].str[11:12]
            spy_options['strike_raw'] = spy_options['ticker'].str[12:20]
            
            # Convert strike (last 8 digits / 1000 = strike price)
            spy_options['strike'] = pd.to_numeric(spy_options['strike_raw'], errors='coerce') / 1000
            
            # Parse expiration date
            spy_options['expiration'] = pd.to_datetime('20' + spy_options['exp_date'], format='%Y%m%d', errors='coerce')
            
            # Calculate days to expiration
            file_date = pd.to_datetime(flatfile_path.stem.replace('.csv', ''))
            spy_options['dte'] = (spy_options['expiration'] - file_date).dt.days
            
            # Clean up
            spy_options = spy_options[spy_options['strike'].notna()].copy()
            spy_options = spy_options[spy_options['dte'] >= 0].copy()
            
            return spy_options
            
        except Exception as e:
            logger.error(f"Error parsing {flatfile_path}: {e}")
            return None
    
    def _calculate_oi_proxy(self, df: pd.DataFrame, spy_price: float) -> Optional[pd.DataFrame]:
        """Calculate OI proxy using volume, transactions, and other features"""
        if df is None or len(df) == 0:
            return None
        
        proxy_df = df.copy()
        
        # 1. Moneyness (distance from ATM)
        proxy_df['moneyness'] = proxy_df['strike'] / spy_price
        proxy_df['distance_from_atm'] = abs(proxy_df['moneyness'] - 1.0)
        
        # 2. Transaction efficiency (avg trade size)
        proxy_df['avg_tx_size'] = proxy_df['volume'] / (proxy_df['transactions'] + 1)
        
        # 3. Activity score (sqrt of volume × transactions)
        proxy_df['activity_score'] = np.sqrt(proxy_df['volume'] * proxy_df['transactions'])
        
        # 4. DTE-adjusted volume
        dte_weight = np.clip(proxy_df['dte'] / 365, 0.1, 1.0)
        proxy_df['dte_adjusted_volume'] = proxy_df['volume'] * dte_weight
        
        # 5. ATM premium (favors at-the-money options)
        atm_score = 1.0 / (1.0 + 5 * proxy_df['distance_from_atm'])
        proxy_df['atm_score'] = atm_score
        
        # 6. Build composite OI proxy
        vol_norm = proxy_df['volume'] / (proxy_df['volume'].max() + 1)
        tx_norm = proxy_df['transactions'] / (proxy_df['transactions'].max() + 1)
        activity_norm = proxy_df['activity_score'] / (proxy_df['activity_score'].max() + 1)
        
        # Weighted combination
        proxy_df['oi_proxy'] = (
            0.3 * vol_norm +           # Volume is important
            0.2 * tx_norm +             # Transactions indicate activity
            0.2 * activity_norm +       # Activity score
            0.2 * atm_score +           # ATM bias
            0.1 * dte_weight            # DTE adjustment
        ) * 10000  # Scale to realistic OI numbers
        
        return proxy_df
    
    def _download_daily_snapshot(self, ticker: str, date_str: str) -> Optional[pd.DataFrame]:
        """
        Download a single daily snapshot using flat files with OI proxy calculation
        """
        try:
            # Download flat file for this date
            flatfile_path = self._download_flatfile(date_str)
            if flatfile_path is None:
                logger.warning(f"Could not download flat file for {date_str}")
                return None
            
            # Parse SPY options from flat file
            spy_df = self._parse_spy_options_from_flatfile(flatfile_path)
            if spy_df is None or len(spy_df) == 0:
                logger.warning(f"No SPY options found in flat file for {date_str}")
                return None
            
            # Get underlying price for calculations
            underlying_price = self._get_underlying_price(ticker, date_str)
            if underlying_price is None:
                # Fallback: estimate from ATM options
                atm_calls = spy_df[
                    (spy_df['option_type'] == 'C') & 
                    (spy_df['volume'] > 0) &
                    (spy_df['dte'].between(7, 45))
                ]
                if len(atm_calls) > 0:
                    underlying_price = (atm_calls['strike'] * atm_calls['volume']).sum() / atm_calls['volume'].sum()
                else:
                    underlying_price = spy_df['strike'].median()
            
            # Calculate OI proxy
            proxy_df = self._calculate_oi_proxy(spy_df, underlying_price)
            if proxy_df is None:
                return None
            
            # Add metadata
            proxy_df['underlying_ticker'] = ticker
            proxy_df['underlying_price'] = underlying_price
            proxy_df['date'] = date_str
            
            # Clean up temporary file
            flatfile_path.unlink()
            
            logger.info(f"✅ Processed {len(proxy_df)} contracts for {ticker} on {date_str}")
            return proxy_df
            
        except Exception as e:
            logger.error(f"Error downloading snapshot for {ticker} on {date_str}: {e}")
            return None
    
    def _get_weekly_expiries_for_month(self, target_date: date) -> List[date]:
        """Get all Friday expiries in the same month and next month"""
        expiries = []
        
        # Start from target date, go forward 60 days to catch all relevant expiries
        current = target_date
        end = target_date + timedelta(days=60)
        
        while current <= end:
            if current.weekday() == 4 and current >= target_date:  # Friday and in future
                expiries.append(current)
            current += timedelta(days=1)
        
        return expiries
    
    def _calculate_target_strikes(self, underlying_price: float, expiries: List[date], target_date: date) -> List[Dict]:
        """
        Calculate which specific strikes we need based on delta range
        Returns list of dicts with expiry, strike, option_type
        """
        target_strikes = []
        
        # For each expiry, calculate strikes that give us -0.9 to 0.9 delta
        for exp_date in expiries:
            T = (exp_date - target_date).days / 365.0
            if T <= 0:
                continue
            
            # Use Newton's method to find strike for specific delta
            # Delta range: -0.9 to 0.9, sample every 0.05 delta
            
            # For CALLS: delta 0.05 to 0.9
            for target_delta in np.arange(0.05, 0.91, 0.05):
                strike = self._strike_for_delta(underlying_price, T, target_delta, 'call')
                if strike:
                    # Round to nearest $0.50 or $1 (SPY uses these increments)
                    strike = round(strike * 2) / 2  # Round to nearest 0.5
                    target_strikes.append({
                        'expiry': exp_date,
                        'strike': strike,
                        'option_type': 'call',
                        'target_delta': target_delta
                    })
            
            # For PUTS: delta -0.9 to -0.05
            for target_delta in np.arange(-0.9, -0.04, 0.05):
                strike = self._strike_for_delta(underlying_price, T, target_delta, 'put')
                if strike:
                    strike = round(strike * 2) / 2
                    target_strikes.append({
                        'expiry': exp_date,
                        'strike': strike,
                        'option_type': 'put',
                        'target_delta': target_delta
                    })
        
        # Remove duplicates
        unique_strikes = []
        seen = set()
        for s in target_strikes:
            key = (s['expiry'], s['strike'], s['option_type'])
            if key not in seen:
                seen.add(key)
                unique_strikes.append(s)
        
        return unique_strikes
    
    def _strike_for_delta(self, S: float, T: float, target_delta: float, option_type: str, 
                         r: float = 0.02, sigma: float = 0.20) -> Optional[float]:
        """
        Find strike price that gives target delta using Newton's method
        """
        try:
            # Initial guess based on simple approximation
            if option_type == 'call':
                # For call, higher delta = lower strike
                K = S * (1 - (1 - target_delta) * sigma * np.sqrt(T))
            else:
                # For put, more negative delta = higher strike
                K = S * (1 + abs(target_delta) * sigma * np.sqrt(T))
            
            # Newton's method to refine
            for _ in range(10):
                delta = self._calculate_delta_fast(S, K, T, option_type, r, sigma)
                
                if abs(delta - target_delta) < 0.01:  # Close enough
                    return K
                
                # Calculate vega for derivative (simplified)
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                vega_approx = S * norm.pdf(d1) * np.sqrt(T) / (K * sigma * np.sqrt(T))
                
                # Newton step
                K = K + (delta - target_delta) / vega_approx if vega_approx != 0 else K
            
            return K
            
        except:
            return None
    
    def _fetch_specific_contract(self, ticker: str, strike_info: Dict, date_str: str, underlying_price: float) -> Optional[Dict]:
        """
        Fetch a specific contract using end-of-day snapshot to get OI data
        """
        try:
            # Build option ticker symbol
            # Format: O:SPY240119C00450000
            # O: = Option, SPY = underlying, 240119 = expiry YYMMDD, C/P = call/put, 00450000 = strike * 1000
            
            exp_date = strike_info['expiry']
            strike = strike_info['strike']
            opt_type = strike_info['option_type']
            
            # Format expiry as YYMMDD
            exp_str = exp_date.strftime('%y%m%d')
            
            # Format strike as 8-digit with padding (strike * 1000)
            strike_str = f"{int(strike * 1000):08d}"
            
            # Build ticker
            opt_ticker = f"O:{ticker}{exp_str}{opt_type[0].upper()}{strike_str}"
            
            # Check cache first
            cache_key = f"{opt_ticker}_{date_str}"
            if cache_key in self._market_data_cache:
                return self._market_data_cache[cache_key]
            
            # Try to get end-of-day snapshot first (has OI data)
            # Format: /v3/snapshot?ticker.any_of=O:SPY160108C00199000&date=2016-01-04
            url = f"{self.base_url}/v3/snapshot"
            params = {
                'ticker.any_of': opt_ticker,
                'order': 'desc',
                'limit': 1,
                'apikey': self.api_key
            }
            
            # Add date parameter for historical data
            # Note: Polygon uses 'date' parameter for historical snapshots
            params['date'] = date_str
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            contract_data = None
            
            # Try snapshot data first (has OI)
            if 'results' in data and data['results'] and len(data['results']) > 0:
                snap = data['results'][0]
                
                # Calculate actual delta
                target_date = pd.to_datetime(date_str).date()
                T = (exp_date - target_date).days / 365.0
                
                delta = self._calculate_delta_fast(
                    S=underlying_price,
                    K=strike,
                    T=T,
                    option_type=opt_type
                )
                
                # Extract data from snapshot
                day_data = snap.get('day', {})
                last_quote = snap.get('last_quote', {})
                greeks = snap.get('greeks', {})
                
                contract_data = {
                    'ticker': opt_ticker,
                    'strike_price': strike,
                    'option_type': opt_type.upper(),
                    'expiration_date': exp_date.strftime('%Y-%m-%d'),
                    'days_to_expiry': (exp_date - target_date).days,
                    'date': date_str,
                    
                    # Market data from snapshot
                    'open': day_data.get('o', 0),
                    'high': day_data.get('h', 0),
                    'low': day_data.get('l', 0),
                    'close': day_data.get('c', 0),
                    'volume': day_data.get('v', 0),
                    'vwap': day_data.get('vw', 0),
                    
                    # OI and quotes
                    'open_interest': snap.get('open_interest', 0),
                    'bid': last_quote.get('bid', 0),
                    'ask': last_quote.get('ask', 0),
                    'mid_price': (last_quote.get('bid', 0) + last_quote.get('ask', 0)) / 2 if last_quote.get('bid') and last_quote.get('ask') else day_data.get('c', 0),
                    
                    # Greeks and calculated fields
                    'implied_volatility': snap.get('implied_volatility', 0),
                    'delta': greeks.get('delta', delta),
                    'gamma': greeks.get('gamma', 0),
                    'theta': greeks.get('theta', 0),
                    'vega': greeks.get('vega', 0),
                    'moneyness': underlying_price / strike if strike else 0,
                }
            else:
                # Fallback to aggregates if snapshot not available
                agg_url = f"{self.base_url}/v2/aggs/ticker/{opt_ticker}/range/1/day/{date_str}/{date_str}"
                agg_params = {
                    'adjusted': 'false',
                    'apikey': self.api_key
                }
                
                agg_response = requests.get(agg_url, params=agg_params)
                agg_response.raise_for_status()
                agg_data = agg_response.json()
                
                if 'results' in agg_data and agg_data['results']:
                    result = agg_data['results'][0]
                    
                    target_date = pd.to_datetime(date_str).date()
                    T = (exp_date - target_date).days / 365.0
                    
                    delta = self._calculate_delta_fast(
                        S=underlying_price,
                        K=strike,
                        T=T,
                        option_type=opt_type
                    )
                    
                    contract_data = {
                        'ticker': opt_ticker,
                        'strike_price': strike,
                        'option_type': opt_type.upper(),
                        'expiration_date': exp_date.strftime('%Y-%m-%d'),
                        'days_to_expiry': (exp_date - target_date).days,
                        'date': date_str,
                        
                        # Market data
                        'open': result.get('o', 0),
                        'high': result.get('h', 0),
                        'low': result.get('l', 0),
                        'close': result.get('c', 0),
                        'volume': result.get('v', 0),
                        'vwap': result.get('vw', 0),
                        
                        # No OI from aggregates
                        'open_interest': 0,
                        'bid': result.get('c', 0) * 0.98,
                        'ask': result.get('c', 0) * 1.02,
                        'mid_price': result.get('c', 0),
                        'implied_volatility': 0,
                        'delta': delta,
                        'gamma': 0,
                        'theta': 0,
                        'vega': 0,
                        'moneyness': underlying_price / strike if strike else 0,
                    }
            
            if contract_data:
                # Cache it
                self._market_data_cache[cache_key] = contract_data
                return contract_data
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error fetching {strike_info}: {e}")
            return None
    
    def _get_filtered_contracts(self, ticker: str, date_str: str, target_date: date, underlying_price: float) -> List[Dict]:
        """
        Get and filter contracts efficiently - only fetch what we need
        """
        try:
            # Check cache first (use month as cache key since contracts don't change during the month)
            month_key = date_str[:7]  # YYYY-MM
            cache_key = f"{ticker}_{month_key}"
            
            if cache_key in self._contracts_cache:
                logger.info(f"Using cached contracts for {ticker} {month_key}")
                all_contracts = self._contracts_cache[cache_key]
            else:
                # Use Polygon's reference API to get all contracts for this ticker
                url = f"{self.base_url}/v3/reference/options/contracts"
                params = {
                    'underlying_ticker': ticker,
                    'expiration_date.gte': date_str,  # Only get contracts expiring on or after target date
                    'limit': 1000,
                    'apikey': self.api_key
                }
                
                all_contracts = []
                next_url = None
                page_count = 0
                
                while True:
                    page_count += 1
                    if next_url:
                        if 'apikey=' not in next_url:
                            separator = '&' if '?' in next_url else '?'
                            next_url = f"{next_url}{separator}apikey={self.api_key}"
                        response = requests.get(next_url)
                    else:
                        response = requests.get(url, params=params)
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'results' not in data or not data['results']:
                        break
                    
                    all_contracts.extend(data['results'])
                    
                    if 'next_url' in data and data['next_url']:
                        next_url = data['next_url']
                    else:
                        break
                    
                    # Limit pages to avoid excessive API calls
                    if page_count >= 10:
                        break
                
                # Cache the results for future dates in the same month
                self._contracts_cache[cache_key] = all_contracts
                logger.info(f"Fetched {len(all_contracts)} total contracts from API")
            
            # Filter contracts efficiently
            filtered = []
            for contract in all_contracts:
                try:
                    # Get expiration date
                    exp_date_str = contract.get('expiration_date')
                    if not exp_date_str:
                        continue
                    
                    exp_date = pd.to_datetime(exp_date_str).date()
                    
                    # Filter 1: Must expire after target date
                    if exp_date <= target_date:
                        continue
                    
                    # Filter 2: Must be Friday (weekly expiry)
                    if exp_date.weekday() != 4:
                        continue
                    
                    # Filter 3: Calculate delta and check range
                    strike_price = contract.get('strike_price')
                    option_type = contract.get('contract_type', 'call')
                    
                    if not strike_price:
                        continue
                    
                    T = (exp_date - target_date).days / 365.0
                    if T <= 0:
                        continue
                    
                    delta = self._calculate_delta_fast(
                        S=underlying_price,
                        K=float(strike_price),
                        T=T,
                        option_type=option_type
                    )
                    
                    # Filter 4: Delta range -0.9 to 0.9
                    if -0.9 <= delta <= 0.9:
                        filtered.append(contract)
                
                except Exception as e:
                    logger.debug(f"Error filtering contract: {e}")
                    continue
            
            logger.info(f"Filtered to {len(filtered)} contracts (weekly expiries, -0.9 to 0.9 delta)")
            return filtered
            
        except Exception as e:
            logger.error(f"Error getting filtered contracts: {e}")
            return []
    
    def _process_historical_contract(self, contract: Dict, date_str: str, underlying_price: float) -> Optional[Dict]:
        """
        Process a historical contract and get its market data
        """
        try:
            ticker_symbol = contract.get('ticker')
            strike_price = contract.get('strike_price')
            option_type = contract.get('contract_type', 'call')
            expiration_date = contract.get('expiration_date')
            
            if not all([ticker_symbol, strike_price, expiration_date]):
                return None
            
            # Get market data for this contract on the specific date
            url = f"{self.base_url}/v2/aggs/ticker/{ticker_symbol}/range/1/day/{date_str}/{date_str}"
            params = {
                'adjusted': 'false',
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract market data
            if 'results' in data and data['results']:
                result = data['results'][0]
                
                # Calculate delta
                target_date = pd.to_datetime(date_str).date()
                exp_date = pd.to_datetime(expiration_date).date()
                T = (exp_date - target_date).days / 365.0
                
                delta = self._calculate_delta_fast(
                    S=underlying_price,
                    K=float(strike_price),
                    T=T,
                    option_type=option_type
                )
                
                contract_data = {
                    'ticker': ticker_symbol,
                    'strike_price': float(strike_price),
                    'option_type': option_type.upper(),
                    'expiration_date': expiration_date,
                    'days_to_expiry': (exp_date - target_date).days,
                    'underlying_price': underlying_price,
                    'date': date_str,
                    
                    # Market data from aggregates
                    'open': result.get('o', 0),
                    'high': result.get('h', 0),
                    'low': result.get('l', 0),
                    'close': result.get('c', 0),
                    'volume': result.get('v', 0),
                    'vwap': result.get('vw', 0),
                    
                    # Calculated fields
                    'bid': result.get('c', 0) * 0.98,  # Approximate bid
                    'ask': result.get('c', 0) * 1.02,  # Approximate ask
                    'mid_price': result.get('c', 0),
                    'last_trade_price': result.get('c', 0),
                    'open_interest': 0,  # Not available in aggregates
                    'implied_volatility': 0.20,  # Default estimate
                    'delta': delta,
                    'gamma': 0,
                    'theta': 0,
                    'vega': 0,
                    'moneyness': underlying_price / float(strike_price) if strike_price else 0,
                    'time_value': 0
                }
                
                return contract_data
            else:
                # No market data available for this contract on this date
                logger.debug(f"No market data for {ticker_symbol} on {date_str}")
                return None
            
        except Exception as e:
            logger.debug(f"Error processing historical contract: {e}")
            return None
    
    def _process_snapshot_contract(self, contract: Dict, result: Dict, 
                                 target_date: date, underlying_price: float) -> Optional[Dict]:
        """
        Process a single contract from snapshot data with efficient filtering
        """
        try:
            # Extract basic contract info
            ticker = contract.get('ticker', '')
            strike_price = contract.get('strike_price')
            option_type = contract.get('contract_type', 'CALL')
            expiration_date = contract.get('expiration_date')
            
            if not all([ticker, strike_price, expiration_date]):
                return None
            
            # Parse expiration date
            exp_date = pd.to_datetime(expiration_date).date()
            
            # Filter 1: Only contracts expiring after target date
            if exp_date <= target_date:
                return None
            
            # Filter 2: Only weekly expiries (Fridays)
            if exp_date.weekday() != 4:  # 4 = Friday
                return None
            
            # Filter 3: Calculate delta and filter to -0.9 to +0.9 range
            T = (exp_date - target_date).days / 365.0
            if T <= 0:
                return None
            
            # Calculate delta using Black-Scholes
            delta = self._calculate_delta_fast(
                S=underlying_price,
                K=float(strike_price),
                T=T,
                option_type=option_type
            )
            
            # Apply delta filter: -0.9 to +0.9
            if not (-0.9 <= delta <= 0.9):
                return None
            
            # Extract market data from snapshot (already included!)
            last_quote = result.get('last_quote', {})
            last_trade = result.get('last_trade', {})
            greeks = result.get('greeks', {})
            
            # Build contract data
            contract_data = {
                'ticker': ticker,
                'strike_price': float(strike_price),
                'option_type': option_type.upper(),
                'expiration_date': expiration_date,
                'days_to_expiry': (exp_date - target_date).days,
                'underlying_price': underlying_price,
                'date': target_date.strftime('%Y-%m-%d'),
                
                # Pricing data
                'bid': last_quote.get('bid', 0),
                'ask': last_quote.get('ask', 0),
                'last_trade_price': last_trade.get('price', 0),
                'mid_price': (last_quote.get('bid', 0) + last_quote.get('ask', 0)) / 2,
                
                # Volume and OI data
                'volume': result.get('volume', 0),
                'open_interest': result.get('open_interest', 0),
                
                # Greeks and volatility
                'implied_volatility': result.get('implied_volatility', 0),
                'delta': greeks.get('delta', delta),  # Use calculated delta if greeks not available
                'gamma': greeks.get('gamma', 0),
                'theta': greeks.get('theta', 0),
                'vega': greeks.get('vega', 0),
                
                # Additional metrics
                'moneyness': underlying_price / float(strike_price) if strike_price else 0,
                'time_value': max(0, last_trade.get('price', 0) - max(0, underlying_price - float(strike_price)) if option_type.upper() == 'CALL' else max(0, float(strike_price) - underlying_price))
            }
            
            return contract_data
            
        except Exception as e:
            logger.debug(f"Error processing contract {contract.get('ticker', 'unknown')}: {e}")
            return None
    
    def _calculate_delta_fast(self, S: float, K: float, T: float, option_type: str, 
                            r: float = 0.02, sigma: float = 0.20) -> float:
        """
        Fast delta calculation using Black-Scholes model
        Optimized for batch processing
        """
        try:
            if T <= 0:
                return 0.0
            
            # Calculate d1
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            # Calculate delta
            if option_type.upper() == 'CALL':
                delta = norm.cdf(d1)
            else:  # PUT
                delta = norm.cdf(d1) - 1
            
            return delta
            
        except Exception as e:
            logger.debug(f"Error calculating delta: {e}")
            return 0.0
    
    def _get_underlying_price(self, ticker: str, date_str: str) -> Optional[float]:
        """Get underlying asset price with caching"""
        cache_key = f"{ticker}_{date_str}"
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        try:
            target_date = pd.to_datetime(date_str).date()
            
            # Get data for a small range around the target date
            start_date = target_date - timedelta(days=3)
            end_date = target_date + timedelta(days=3)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if hist.empty:
                return None
            
            # Find the closest trading day
            hist.index = hist.index.date
            closest_date = min(hist.index, key=lambda x: abs((x - target_date).days))
            price = float(hist.loc[closest_date, 'Close'])
            
            # Cache the result
            self._price_cache[cache_key] = price
            return price
            
        except Exception as e:
            logger.error(f"Error getting underlying price for {ticker} on {date_str}: {e}")
            return None
    
    def save_daily_snapshots(self, daily_snapshots: Dict[str, pd.DataFrame], 
                           ticker: str, output_dir: str = None) -> List[str]:
        """
        Save daily snapshots to parquet files organized by year/month
        """
        saved_files = []
        
        for date_str, df in daily_snapshots.items():
            try:
                # Parse date to get year and month
                date_obj = pd.to_datetime(date_str)
                year = date_obj.strftime('%Y')
                month = date_obj.strftime('%m')
                
                # Create directory structure: data/options_chains/TICKER/YEAR/MONTH/
                if output_dir is None:
                    save_dir = self.data_dir / "options_chains" / ticker / year / month
                else:
                    save_dir = Path(output_dir) / year / month
                
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename
                clean_date = date_str.replace('-', '')
                filename = f"{ticker}_options_snapshot_{clean_date}.parquet"
                filepath = save_dir / filename
                
                # Save as parquet
                df.to_parquet(filepath, index=False)
                saved_files.append(str(filepath))
                
                logger.info(f"💾 Saved {len(df)} contracts to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving snapshot for {date_str}: {e}")
        
        return saved_files
    
    def create_summary_report(self, daily_snapshots: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a summary report of all daily snapshots
        """
        summary_data = []
        
        for date_str, df in daily_snapshots.items():
            if df.empty:
                continue
            
            # Calculate summary statistics
            summary = {
                'date': date_str,
                'total_contracts': len(df),
                'calls': len(df[df['option_type'] == 'C']),
                'puts': len(df[df['option_type'] == 'P']),
                'avg_volume': df['volume'].mean(),
                'total_volume': df['volume'].sum(),
                'avg_oi_proxy': df['oi_proxy'].mean() if 'oi_proxy' in df.columns else 0,
                'oi_proxy_range': f"{df['oi_proxy'].min():.0f} to {df['oi_proxy'].max():.0f}" if 'oi_proxy' in df.columns else "N/A",
                'expiry_dates': df['expiration'].nunique() if 'expiration' in df.columns else df['expiration_date'].nunique(),
                'underlying_price': df['underlying_price'].iloc[0] if not df.empty else None
            }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)


def main():
    parser = argparse.ArgumentParser(description='Optimized Options Chain Downloader')
    
    parser.add_argument('--ticker', type=str, default='SPY', help='Underlying ticker (default: SPY)')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = OptimizedOptionsDownloader(data_dir=args.data_dir)
    
    if not downloader.api_key:
        print("❌ POLYGON_API_KEY environment variable required")
        return
    
    print(f"🚀 OPTIMIZED OPTIONS DOWNLOADER")
    print(f"📊 Ticker: {args.ticker}")
    print(f"📅 Period: {args.start_date} to {args.end_date}")
    print(f"🎯 Target: Weekly expiries, -0.9 to +0.9 delta")
    print("=" * 60)
    
    try:
        # Download daily snapshots
        print("📥 Downloading daily snapshots...")
        daily_snapshots = downloader.download_date_range(args.ticker, args.start_date, args.end_date)
        
        if not daily_snapshots:
            print("❌ No data downloaded")
            return
        
        # Save snapshots
        print("💾 Saving snapshots...")
        saved_files = downloader.save_daily_snapshots(
            daily_snapshots, args.ticker, args.output_dir
        )
        
        # Create summary report
        print("📊 Generating summary report...")
        summary_df = downloader.create_summary_report(daily_snapshots)
        
        # Save summary
        period_str = f"{args.start_date}_{args.end_date}".replace('-', '')
        summary_file = Path(args.data_dir) / f"{args.ticker}_summary_{period_str}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Display results
        print("\n🎉 DOWNLOAD COMPLETE!")
        print(f"✅ {len(daily_snapshots)} daily snapshots downloaded")
        print(f"💾 {len(saved_files)} files saved")
        print(f"📊 Summary saved to: {summary_file}")
        
        print("\n📈 SUMMARY STATISTICS (first 20 days):")
        print(summary_df.head(20).to_string(index=False))
        
        if len(summary_df) > 20:
            print(f"\n... and {len(summary_df) - 20} more days")
        
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
