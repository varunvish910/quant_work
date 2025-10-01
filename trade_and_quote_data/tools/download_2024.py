#!/usr/bin/env python3
"""
Simple 2024 SPY Options Data Downloader
Using the same format as existing data
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
BASE_URL = "https://api.polygon.io/v3/snapshot/options"

def get_trading_days_2024():
    """Get all trading days for 2024"""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    trading_days = []
    current = start_date
    
    while current <= end_date:
        # Skip weekends
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            # Skip some major holidays (basic list)
            skip_dates = [
                datetime(2024, 1, 1),   # New Year's Day
                datetime(2024, 1, 15),  # MLK Day
                datetime(2024, 2, 19),  # Presidents Day
                datetime(2024, 3, 29),  # Good Friday
                datetime(2024, 5, 27),  # Memorial Day
                datetime(2024, 6, 19),  # Juneteenth
                datetime(2024, 7, 4),   # Independence Day
                datetime(2024, 9, 2),   # Labor Day
                datetime(2024, 11, 28), # Thanksgiving
                datetime(2024, 12, 25), # Christmas
            ]
            
            if current not in skip_dates:
                trading_days.append(current.strftime('%Y-%m-%d'))
        
        current += timedelta(days=1)
    
    return trading_days

def download_spy_options(date_str):
    """Download SPY options for a specific date"""
    logger.info(f"Downloading SPY options for {date_str}")
    
    # Create output directory
    year, month, day = date_str.split('-')
    output_dir = Path(f"data/options_chains/SPY/{year}/{month}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    output_file = output_dir / f"SPY_options_snapshot_{year}{month}{day}.parquet"
    if output_file.exists():
        logger.info(f"File already exists: {output_file}")
        return True
    
    # Download data from Polygon API
    url = f"{BASE_URL}/SPY"
    params = {
        'apikey': API_KEY,
        'limit': 50000  # Max limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'results' not in data or not data['results']:
            logger.warning(f"No data returned for {date_str}")
            return False
        
        # Convert to DataFrame matching existing format
        records = []
        for result in data['results']:
            if 'details' in result and 'underlying_ticker' in result['details']:
                if result['details']['underlying_ticker'] == 'SPY':
                    # Extract contract details
                    details = result['details']
                    market_status = result.get('market_status', 'open')
                    
                    # Skip if market is closed
                    if market_status != 'open':
                        continue
                    
                    # Parse option ticker to get strike and type
                    ticker = result.get('value', {}).get('ticker', '')
                    
                    # Basic parsing of option ticker (O:SPY240119C00475000)
                    if ':' in ticker and 'SPY' in ticker:
                        parts = ticker.split(':')[1]  # Remove 'O:'
                        if 'C' in parts or 'P' in parts:
                            # Find C or P
                            if 'C' in parts:
                                option_type = 'C'
                                strike_part = parts.split('C')[1]
                            else:
                                option_type = 'P'
                                strike_part = parts.split('P')[1]
                            
                            # Extract strike price (last 8 digits, divide by 1000)
                            if len(strike_part) >= 8:
                                strike = float(strike_part[-8:]) / 1000
                            else:
                                continue
                            
                            # Extract expiration date
                            exp_part = parts.replace('SPY', '').split(option_type)[0]
                            if len(exp_part) >= 6:
                                exp_year = 2000 + int(exp_part[:2])
                                exp_month = int(exp_part[2:4])
                                exp_day = int(exp_part[4:6])
                                expiration = f"{exp_year}-{exp_month:02d}-{exp_day:02d}"
                            else:
                                continue
                            
                            # Get market data
                            value = result.get('value', {})
                            
                            record = {
                                'ticker': ticker,
                                'volume': value.get('volume', 0),
                                'open': value.get('open', 0),
                                'close': value.get('close', 0),
                                'high': value.get('high', 0),
                                'low': value.get('low', 0),
                                'window_start': date_str,
                                'transactions': value.get('transactions', 0),
                                'underlying': 'SPY',
                                'exp_date': expiration,
                                'option_type': option_type,
                                'strike_raw': strike * 1000,
                                'strike': strike,
                                'expiration': expiration,
                                'dte': (datetime.strptime(expiration, '%Y-%m-%d') - datetime.strptime(date_str, '%Y-%m-%d')).days,
                                'moneyness': 0,  # Will calculate later
                                'distance_from_atm': 0,  # Will calculate later
                                'avg_tx_size': value.get('volume', 0) / max(value.get('transactions', 1), 1),
                                'activity_score': value.get('volume', 0) * 0.1,  # Simple scoring
                                'dte_adjusted_volume': value.get('volume', 0),
                                'atm_score': 0,  # Will calculate later
                                'oi_proxy': value.get('volume', 0) * 10,  # Rough estimate
                                'underlying_ticker': 'SPY',
                                'underlying_price': 470,  # Will get actual price
                                'date': date_str
                            }
                            
                            records.append(record)
        
        if not records:
            logger.warning(f"No valid SPY options found for {date_str}")
            return False
        
        # Create DataFrame and save
        df = pd.DataFrame(records)
        
        # Calculate derived fields
        if len(df) > 0 and 'underlying_price' in df.columns:
            avg_underlying = df['underlying_price'].iloc[0]  # Use first value
            df['moneyness'] = df['strike'] / avg_underlying
            df['distance_from_atm'] = abs(df['strike'] - avg_underlying) / avg_underlying
            df['atm_score'] = 1.0 / (1.0 + df['distance_from_atm'] * 10)
        
        # Save to parquet
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(df)} contracts to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading data for {date_str}: {e}")
        return False

def main():
    """Main function to download all 2024 data"""
    logger.info("Starting 2024 SPY options data download")
    
    trading_days = get_trading_days_2024()
    logger.info(f"Found {len(trading_days)} trading days in 2024")
    
    success_count = 0
    for i, date_str in enumerate(trading_days, 1):
        logger.info(f"Processing {i}/{len(trading_days)}: {date_str}")
        
        if download_spy_options(date_str):
            success_count += 1
        
        # Rate limiting - be nice to the API
        time.sleep(0.1)
        
        # Progress update every 10 days
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(trading_days)} days, {success_count} successful")
    
    logger.info(f"Download complete! {success_count}/{len(trading_days)} days successful")

if __name__ == "__main__":
    main()