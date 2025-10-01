#!/usr/bin/env python3
"""
Complete Options Chain Data System

A comprehensive system for downloading, storing, and analyzing historical options chain data
using Polygon API. Provides both data collection and analysis capabilities in a single module.

Key Features:
- Download complete historical options chains for any date
- Reconstruct point-in-time options chains from stored data  
- Advanced analytics: IV surfaces, Greeks calculation, skew analysis
- Backtest-ready data export with efficient storage
- High performance parallel processing with intelligent rate limiting

Usage:
    # Download data
    python options_chain_downloader.py --download --ticker SPY --date 2025-01-01
    python options_chain_downloader.py --download --ticker SPY --start-date 2025-01-01 --end-date 2025-01-31
    
    # Analyze data
    python options_chain_downloader.py --analyze --ticker SPY --date 2025-01-01
    python options_chain_downloader.py --export --ticker SPY --start-date 2025-01-01 --end-date 2025-01-31

API Usage:
    from options_chain_downloader import OptionsChainSystem
    
    system = OptionsChainSystem(api_key="your_key")
    chain = system.download_chain("SPY", "2025-01-01")
    summary = system.analyze_chain(chain)
    iv_surface = system.build_iv_surface(chain)
"""

import os
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta, date
import time
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import argparse
import requests
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any, Union
import math
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChainFilter:
    """Configuration for filtering options chains"""
    min_volume: int = 0
    min_open_interest: int = 0
    min_days_to_expiration: int = 0
    max_days_to_expiration: int = 365
    min_moneyness: float = 0.5
    max_moneyness: float = 2.0
    contract_types: List[str] = None
    min_data_quality: float = 0.0
    
    def __post_init__(self):
        if self.contract_types is None:
            self.contract_types = ['call', 'put']

class BlackScholesCalculator:
    """Black-Scholes options pricing and Greeks calculator"""
    
    @staticmethod
    def calculate_d1_d2(S, K, T, r, sigma):
        """Calculate d1 and d2 for Black-Scholes formula"""
        if T <= 0 or sigma <= 0:
            return 0, 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def calculate_price(S, K, T, r, sigma, option_type='call'):
        """Calculate theoretical option price using Black-Scholes"""
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """Calculate all Greeks for an option"""
        if T <= 0:
            return {
                'delta': 1.0 if (option_type == 'call' and S > K) else 0.0,
                'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0
            }
        
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_part1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            theta = (theta_part1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (theta_part1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def calculate_implied_volatility(market_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Brent's method"""
        if T <= 0 or market_price <= 0:
            return 0.0
        
        def objective(sigma):
            return BlackScholesCalculator.calculate_price(S, K, T, r, sigma, option_type) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
            return iv
        except (ValueError, RuntimeError):
            return 0.0


class OptionsChainSystem:
    """Complete options chain data system - download, store, and analyze"""
    
    def __init__(self, api_key: str = None, risk_free_rate: float = 0.05, data_dir: str = "data"):
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            logger.warning("No Polygon API key provided. Download functionality will be disabled.")
        
        self.risk_free_rate = risk_free_rate
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        if self.api_key:
            self.client = RESTClient(self.api_key)
            self.base_url = "https://api.polygon.io"
        
        self.bs_calculator = BlackScholesCalculator()
        self.cache = {}
        self.cache_size_limit = 100
        
        # Rate limiting
        self.request_delay = 0.12  # For 5000 req/min limit
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Simple rate limiting"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get all trading days between start and end date"""
        business_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Filter out major holidays
        holidays = [
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
            '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
            '2026-01-01', '2026-01-19', '2026-02-16', '2026-05-25', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25'
        ]
        
        return [day.strftime('%Y-%m-%d') for day in business_days 
                if day.strftime('%Y-%m-%d') not in holidays]
    
    # Download functionality
    def download_chain(self, ticker: str, date_str: str) -> Optional[pd.DataFrame]:
        """Download complete options chain for a specific date"""
        if not self.api_key:
            raise ValueError("API key required for downloading data")
        
        logger.info(f"Downloading complete chain for {ticker} on {date_str}")
        
        # Get underlying price
        underlying_price = self._get_underlying_price(ticker, date_str)
        if underlying_price is None:
            logger.error(f"Could not get underlying price for {ticker} on {date_str}")
            return None
        
        # Get all contracts
        contracts = self._get_options_contracts(ticker, date_str)
        if not contracts:
            logger.warning(f"No contracts found for {ticker} on {date_str}")
            return None
        
        # Get market data for each contract in parallel
        chain_data = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_contract = {}
            
            for contract in contracts:
                contract_ticker = contract.get('ticker', '')
                if contract_ticker:
                    future = executor.submit(self._process_contract, contract, date_str, underlying_price)
                    future_to_contract[future] = contract
            
            with tqdm(total=len(future_to_contract), desc=f"Processing {ticker} contracts") as pbar:
                for future in as_completed(future_to_contract):
                    try:
                        contract_data = future.result()
                        if contract_data:
                            chain_data.append(contract_data)
                    except Exception as e:
                        logger.warning(f"Error processing contract: {e}")
                    
                    pbar.update(1)
        
        if not chain_data:
            return None
        
        # Convert to DataFrame and add metadata
        df = pd.DataFrame(chain_data)
        df['underlying_ticker'] = ticker
        df['underlying_price'] = underlying_price
        df['date'] = date_str
        df['download_timestamp'] = datetime.now().isoformat()
        df['data_source'] = 'polygon'
        df['data_quality_score'] = self._calculate_quality_score(df)
        
        logger.info(f"Downloaded complete chain: {len(df)} contracts for {ticker} on {date_str}")
        return df
    
    def download_date_range(self, ticker: str, start_date: str, end_date: str):
        """Download options chains for a range of dates"""
        if not self.api_key:
            raise ValueError("API key required for downloading data")
        
        trading_days = self.get_trading_days(start_date, end_date)
        logger.info(f"Downloading {ticker} options chains for {len(trading_days)} trading days")
        
        with tqdm(total=len(trading_days), desc=f"Downloading {ticker} chains") as pbar:
            for date_str in trading_days:
                # Check if file already exists
                if self._chain_file_exists(ticker, date_str):
                    logger.info(f"Skipping {date_str} - file already exists")
                    pbar.update(1)
                    continue
                
                # Download and save
                chain_df = self.download_chain(ticker, date_str)
                if chain_df is not None:
                    self.save_chain(chain_df, ticker, date_str)
                else:
                    logger.warning(f"Failed to download data for {date_str}")
                
                pbar.update(1)
    
    def save_chain(self, df: pd.DataFrame, ticker: str, date_str: str):
        """Save options chain data to parquet file with partitioning"""
        if df is None or len(df) == 0:
            logger.warning(f"No data to save for {ticker} on {date_str}")
            return
        
        # Create directory structure
        year = date_str[:4]
        month = date_str[5:7]
        
        output_path = self.data_dir / ticker.lower() / "options_chains" / f"year={year}" / f"month={month}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        filename = f"chains_{date_str}.parquet"
        filepath = output_path / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} contracts to {filepath}")
    
    # Analysis functionality
    def load_chain(self, ticker: str, date_str: str, filter_config: Optional[ChainFilter] = None) -> Optional[pd.DataFrame]:
        """Load options chain from stored data"""
        cache_key = f"{ticker}_{date_str}"
        
        # Check cache first
        if cache_key in self.cache:
            chain = self.cache[cache_key].copy()
        else:
            # Load from file
            chain = self._load_chain_from_file(ticker, date_str)
            if chain is None:
                return None
            
            # Cache the result
            self._add_to_cache(cache_key, chain)
        
        # Apply filters if provided
        if filter_config:
            chain = self._apply_filters(chain, filter_config)
        
        return chain
    
    def load_chains_for_period(self, ticker: str, start_date: str, end_date: str,
                              filter_config: Optional[ChainFilter] = None) -> Dict[str, pd.DataFrame]:
        """Load options chains for a period of dates"""
        trading_days = self.get_trading_days(start_date, end_date)
        chains = {}
        
        for date_str in trading_days:
            chain = self.load_chain(ticker, date_str, filter_config)
            if chain is not None:
                chains[date_str] = chain
        
        logger.info(f"Loaded {len(chains)} chains for {ticker} from {start_date} to {end_date}")
        return chains
    
    def analyze_chain(self, chain: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive chain summary statistics"""
        if len(chain) == 0:
            return {}
        
        summary = {
            'date': chain['date'].iloc[0],
            'underlying_ticker': chain['underlying_ticker'].iloc[0],
            'underlying_price': chain['underlying_price'].iloc[0],
            'total_contracts': len(chain),
            'call_contracts': len(chain[chain['contract_type'] == 'call']),
            'put_contracts': len(chain[chain['contract_type'] == 'put']),
            'total_volume': chain['volume'].sum(),
            'total_open_interest': chain['open_interest'].sum(),
            'avg_volume': chain['volume'].mean(),
            'avg_open_interest': chain['open_interest'].mean(),
            'min_strike': chain['strike_price'].min(),
            'max_strike': chain['strike_price'].max(),
            'atm_strike': self._find_atm_strike(chain),
        }
        
        # Expiration analysis
        expirations = chain['expiration_date'].unique()
        summary.update({
            'num_expirations': len(expirations),
            'nearest_expiration': min(expirations),
            'furthest_expiration': max(expirations),
            'avg_days_to_expiration': chain['days_to_expiration'].mean(),
        })
        
        # IV analysis
        valid_iv = chain[chain['implied_volatility'] > 0]['implied_volatility']
        if len(valid_iv) > 0:
            summary.update({
                'avg_iv': valid_iv.mean(),
                'median_iv': valid_iv.median(),
                'iv_std': valid_iv.std(),
                'min_iv': valid_iv.min(),
                'max_iv': valid_iv.max(),
            })
        
        # Put-call ratios
        call_volume = chain[chain['contract_type'] == 'call']['volume'].sum()
        put_volume = chain[chain['contract_type'] == 'put']['volume'].sum()
        summary['put_call_volume_ratio'] = put_volume / call_volume if call_volume > 0 else 0
        
        call_oi = chain[chain['contract_type'] == 'call']['open_interest'].sum()
        put_oi = chain[chain['contract_type'] == 'put']['open_interest'].sum()
        summary['put_call_oi_ratio'] = put_oi / call_oi if call_oi > 0 else 0
        
        return summary
    
    def build_iv_surface(self, chain: pd.DataFrame) -> Dict[str, Any]:
        """Build implied volatility surface from options chain"""
        calls = chain[chain['contract_type'] == 'call'].copy()
        puts = chain[chain['contract_type'] == 'put'].copy()
        
        surfaces = {}
        
        for option_type, data in [('call', calls), ('put', puts)]:
            if len(data) == 0:
                continue
            
            # Filter out zero or invalid IV
            valid_iv = data[
                (data['implied_volatility'] > 0) & 
                (data['implied_volatility'] < 5) &
                (data['days_to_expiration'] > 0)
            ].copy()
            
            if len(valid_iv) < 3:
                continue
            
            # Create surface
            surface = self._interpolate_iv_surface(valid_iv)
            if surface:
                surfaces[option_type] = surface
        
        return surfaces
    
    def get_atm_contracts(self, chain: pd.DataFrame, num_strikes: int = 5) -> pd.DataFrame:
        """Get at-the-money contracts"""
        if len(chain) == 0:
            return pd.DataFrame()
        
        underlying_price = chain['underlying_price'].iloc[0]
        
        chain_copy = chain.copy()
        chain_copy['strike_distance'] = abs(chain_copy['strike_price'] - underlying_price)
        
        unique_strikes = chain_copy.nsmallest(num_strikes * 2, 'strike_distance')['strike_price'].unique()
        atm_strikes = sorted(unique_strikes)[:num_strikes]
        
        atm_contracts = chain_copy[chain_copy['strike_price'].isin(atm_strikes)].copy()
        return atm_contracts.drop('strike_distance', axis=1)
    
    def export_for_backtesting(self, chains: Dict[str, pd.DataFrame], output_file: str = None) -> pd.DataFrame:
        """Export chain data in format optimized for backtesting"""
        all_chains = []
        
        for date_str, chain in chains.items():
            chain_copy = chain.copy()
            chain_copy['date_index'] = pd.to_datetime(date_str)
            all_chains.append(chain_copy)
        
        combined = pd.concat(all_chains, ignore_index=True)
        combined = combined.sort_values(['date_index', 'contract_symbol'])
        
        # Add additional derived columns
        combined['log_moneyness'] = np.log(combined['moneyness'])
        combined['annualized_dte'] = combined['days_to_expiration'] / 365.0
        
        if output_file:
            combined.to_parquet(output_file, index=False)
            logger.info(f"Exported {len(combined)} records to {output_file}")
        
        return combined
    
    # Private helper methods
    def _get_underlying_price(self, ticker: str, date_str: str) -> Optional[float]:
        """Get underlying asset price for a specific date"""
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{date_str}/{date_str}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    return results[0].get('c')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting underlying price: {e}")
            return None
    
    def _get_options_contracts(self, ticker: str, date_str: str) -> List[Dict]:
        """Get all options contracts available for a ticker on a specific date"""
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/v3/reference/options/contracts"
            params = {
                'underlying_ticker': ticker,
                'as_of': date_str,
                'limit': 1000,
                'apikey': self.api_key
            }
            
            all_contracts = []
            
            while True:
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    break
                
                data = response.json()
                contracts = data.get('results', [])
                all_contracts.extend(contracts)
                
                if data.get('next_url'):
                    params['cursor'] = data.get('next_url').split('cursor=')[1]
                else:
                    break
            
            return all_contracts
            
        except Exception as e:
            logger.error(f"Error getting contracts: {e}")
            return []
    
    def _get_contract_aggregates(self, contract_ticker: str, date_str: str) -> Optional[Dict]:
        """Get daily aggregate data for a specific options contract"""
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/v2/aggs/ticker/{contract_ticker}/range/1/day/{date_str}/{date_str}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    return results[0]
            
            return None
            
        except Exception as e:
            return None
    
    def _process_contract(self, contract: Dict, date_str: str, underlying_price: float) -> Optional[Dict]:
        """Process a single contract to get market data and calculate metrics"""
        try:
            contract_ticker = contract.get('ticker', '')
            aggregate_data = self._get_contract_aggregates(contract_ticker, date_str)
            
            # Build contract data record
            contract_data = {
                'contract_symbol': contract_ticker,
                'strike_price': contract.get('strike_price', 0),
                'expiration_date': contract.get('expiration_date', ''),
                'contract_type': contract.get('contract_type', 'call'),
                'date': date_str,
                'open': aggregate_data.get('o', 0) if aggregate_data else 0,
                'high': aggregate_data.get('h', 0) if aggregate_data else 0,
                'low': aggregate_data.get('l', 0) if aggregate_data else 0,
                'close': aggregate_data.get('c', 0) if aggregate_data else 0,
                'volume': aggregate_data.get('v', 0) if aggregate_data else 0,
                'vwap': aggregate_data.get('vw', 0) if aggregate_data else 0,
                'bid': 0, 'ask': 0, 'bid_size': 0, 'ask_size': 0,
                'spread': 0, 'mid_price': 0, 'open_interest': 0
            }
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(contract_data, underlying_price)
            contract_data.update(additional_metrics)
            
            return contract_data
            
        except Exception as e:
            return None
    
    def _calculate_additional_metrics(self, contract_data: Dict, underlying_price: float) -> Dict:
        """Calculate additional options metrics"""
        strike = contract_data.get('strike_price', 0)
        expiration_str = contract_data.get('expiration_date', '')
        contract_type = contract_data.get('contract_type', 'call')
        market_price = contract_data.get('close', 0)
        
        # Calculate days to expiration
        try:
            expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d').date()
            current_date = datetime.strptime(contract_data.get('date', ''), '%Y-%m-%d').date()
            days_to_expiration = (expiration_date - current_date).days
        except:
            days_to_expiration = 0
        
        T = max(days_to_expiration / 365.0, 1/365)
        moneyness = underlying_price / strike if strike > 0 else 0
        
        # Calculate intrinsic and time value
        if contract_type == 'call':
            intrinsic_value = max(underlying_price - strike, 0)
            break_even = strike + market_price
        else:
            intrinsic_value = max(strike - underlying_price, 0)
            break_even = strike - market_price
        
        time_value = max(market_price - intrinsic_value, 0)
        
        # Calculate implied volatility and Greeks
        implied_vol = 0.0
        greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        if market_price > 0 and T > 0:
            implied_vol = self.bs_calculator.calculate_implied_volatility(
                market_price, underlying_price, strike, T, self.risk_free_rate, contract_type
            )
            
            if implied_vol > 0:
                greeks = self.bs_calculator.calculate_greeks(
                    underlying_price, strike, T, self.risk_free_rate, implied_vol, contract_type
                )
        
        return {
            'days_to_expiration': days_to_expiration,
            'moneyness': moneyness,
            'intrinsic_value': intrinsic_value,
            'time_value': time_value,
            'break_even': break_even,
            'implied_volatility': implied_vol,
            **greeks
        }
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each contract"""
        scores = []
        
        for _, row in df.iterrows():
            score = 1.0
            
            if row['volume'] == 0:
                score *= 0.7
            if row['close'] == 0:
                score *= 0.5
            if row['implied_volatility'] <= 0 or row['implied_volatility'] > 5:
                score *= 0.8
            if 0.5 <= row['moneyness'] <= 2.0:
                score *= 1.1
            
            scores.append(min(score, 1.0))
        
        return pd.Series(scores)
    
    def _load_chain_from_file(self, ticker: str, date_str: str) -> Optional[pd.DataFrame]:
        """Load options chain from parquet file"""
        try:
            year = date_str[:4]
            month = date_str[5:7]
            
            filepath = (self.data_dir / ticker.lower() / "options_chains" / 
                       f"year={year}" / f"month={month}" / f"chains_{date_str}.parquet")
            
            if not filepath.exists():
                return None
            
            return pd.read_parquet(filepath)
            
        except Exception as e:
            logger.error(f"Error loading chain: {e}")
            return None
    
    def _chain_file_exists(self, ticker: str, date_str: str) -> bool:
        """Check if chain file already exists"""
        year = date_str[:4]
        month = date_str[5:7]
        filepath = (self.data_dir / ticker.lower() / "options_chains" / 
                   f"year={year}" / f"month={month}" / f"chains_{date_str}.parquet")
        return filepath.exists()
    
    def _apply_filters(self, chain: pd.DataFrame, filter_config: ChainFilter) -> pd.DataFrame:
        """Apply filtering criteria to options chain"""
        filtered = chain.copy()
        
        if filter_config.min_volume > 0:
            filtered = filtered[filtered['volume'] >= filter_config.min_volume]
        
        if filter_config.min_open_interest > 0:
            filtered = filtered[filtered['open_interest'] >= filter_config.min_open_interest]
        
        filtered = filtered[
            (filtered['days_to_expiration'] >= filter_config.min_days_to_expiration) &
            (filtered['days_to_expiration'] <= filter_config.max_days_to_expiration) &
            (filtered['moneyness'] >= filter_config.min_moneyness) &
            (filtered['moneyness'] <= filter_config.max_moneyness)
        ]
        
        if filter_config.contract_types:
            filtered = filtered[filtered['contract_type'].isin(filter_config.contract_types)]
        
        if filter_config.min_data_quality > 0:
            filtered = filtered[filtered['data_quality_score'] >= filter_config.min_data_quality]
        
        return filtered
    
    def _add_to_cache(self, key: str, data: pd.DataFrame):
        """Add data to cache with size management"""
        if len(self.cache) >= self.cache_size_limit:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = data.copy()
    
    def _find_atm_strike(self, chain: pd.DataFrame) -> float:
        """Find the at-the-money strike price"""
        underlying_price = chain['underlying_price'].iloc[0]
        strikes = chain['strike_price'].unique()
        return min(strikes, key=lambda x: abs(x - underlying_price))
    
    def _interpolate_iv_surface(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create interpolated IV surface"""
        try:
            moneyness = data['moneyness'].values
            days = data['days_to_expiration'].values
            iv = data['implied_volatility'].values
            
            # Create grids
            moneyness_grid = np.linspace(0.7, 1.3, 31)
            days_grid = np.array([7, 14, 21, 30, 45, 60, 90, 120, 180, 365])
            
            days_mesh, moneyness_mesh = np.meshgrid(days_grid, moneyness_grid)
            
            iv_surface = griddata(
                (days, moneyness), iv, 
                (days_mesh, moneyness_mesh), 
                method='linear',
                fill_value=np.nan
            )
            
            return {
                'moneyness_grid': moneyness_grid,
                'days_grid': days_grid,
                'iv_surface': iv_surface,
                'raw_data': data[['moneyness', 'days_to_expiration', 'implied_volatility']].copy()
            }
            
        except Exception as e:
            logger.error(f"Error interpolating IV surface: {e}")
            return {}


# Utility functions
def quick_chain_summary(ticker: str, date: str, data_dir: str = "data") -> Dict[str, Any]:
    """Quick summary of options chain for a date"""
    system = OptionsChainSystem(data_dir=data_dir)
    chain = system.load_chain(ticker, date)
    
    if chain is None:
        return {}
    
    return system.analyze_chain(chain)


def main():
    parser = argparse.ArgumentParser(description='Complete Options Chain Data System')
    
    # Action arguments
    parser.add_argument('--download', action='store_true', help='Download options chain data')
    parser.add_argument('--analyze', action='store_true', help='Analyze stored options chain data')
    parser.add_argument('--export', action='store_true', help='Export data for backtesting')
    
    # Data specification
    parser.add_argument('--ticker', type=str, required=True, help='Underlying ticker (e.g., SPY)')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='Start date for range (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for range (YYYY-MM-DD)')
    
    # Configuration
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--risk-free-rate', type=float, default=0.05, help='Risk-free rate')
    
    args = parser.parse_args()
    
    if not any([args.download, args.analyze, args.export]):
        print("Please specify an action: --download, --analyze, or --export")
        return
    
    # Initialize system
    system = OptionsChainSystem(data_dir=args.data_dir, risk_free_rate=args.risk_free_rate)
    
    print(f"🚀 OPTIONS CHAIN DATA SYSTEM")
    print(f"📊 Ticker: {args.ticker}")
    print("=" * 50)
    
    try:
        if args.download:
            if not system.api_key:
                print("❌ POLYGON_API_KEY environment variable required for downloading")
                return
            
            if args.date:
                print(f"📥 Downloading single date: {args.date}")
                chain = system.download_chain(args.ticker, args.date)
                if chain is not None:
                    system.save_chain(chain, args.ticker, args.date)
                    print(f"✅ Downloaded {len(chain)} contracts")
                else:
                    print("❌ Download failed")
            
            elif args.start_date and args.end_date:
                print(f"📥 Downloading date range: {args.start_date} to {args.end_date}")
                system.download_date_range(args.ticker, args.start_date, args.end_date)
                print("✅ Range download completed")
            
            else:
                print("❌ Please specify --date or both --start-date and --end-date")
        
        elif args.analyze:
            date = args.date or args.start_date
            if not date:
                print("❌ Please specify --date for analysis")
                return
            
            print(f"📊 Analyzing chain for {args.ticker} on {date}")
            chain = system.load_chain(args.ticker, date)
            
            if chain is None:
                print("❌ No data found. Download data first.")
                return
            
            summary = system.analyze_chain(chain)
            print(f"\n📈 Chain Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            
            # Get ATM contracts
            atm_contracts = system.get_atm_contracts(chain)
            print(f"\n🎯 ATM Contracts: {len(atm_contracts)}")
            for _, contract in atm_contracts.head(5).iterrows():
                print(f"  ${contract['strike_price']:.0f} {contract['contract_type']}: "
                      f"${contract['close']:.2f}, IV: {contract['implied_volatility']:.2f}")
        
        elif args.export:
            if not (args.start_date and args.end_date):
                print("❌ Please specify both --start-date and --end-date for export")
                return
            
            print(f"📤 Exporting data for {args.ticker} from {args.start_date} to {args.end_date}")
            chains = system.load_chains_for_period(args.ticker, args.start_date, args.end_date)
            
            if not chains:
                print("❌ No data found for export")
                return
            
            output_file = f"{args.ticker}_backtest_data.parquet"
            backtest_df = system.export_for_backtesting(chains, output_file)
            print(f"✅ Exported {len(backtest_df)} records to {output_file}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Operation failed: {e}")


if __name__ == '__main__':
    main()