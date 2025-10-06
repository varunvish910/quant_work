#!/usr/bin/env python3
"""
SPY Options Downloader with OI Proxy
Simple command-line tool to download SPY options data from Polygon flat files
with Open Interest proxy calculation from volume and transaction data.
"""

import pandas as pd
import numpy as np
import gzip
import os
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPYOptionsDownloader:
    """Download SPY options with OI proxy from Polygon flat files"""
    
    def __init__(self, output_dir='data/spy_options'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # AWS credentials for Polygon flat files
        self.aws_key = "86959ae1-29bc-4433-be13-1a41b935d9d1"
        self.aws_secret = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.endpoint = "https://files.polygon.io"
    
    def get_trading_days(self, start_date, end_date):
        """Get trading days between start and end date"""
        business_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Major holidays (simplified list)
        holidays = {
            '2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05', '2016-11-24', '2016-12-26',
            '2017-01-02', '2017-01-16', '2017-02-20', '2017-05-29', '2017-07-04', '2017-09-04', '2017-11-23', '2017-12-25',
            '2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28', '2018-07-04', '2018-09-03', '2018-11-22', '2018-12-25',
            '2019-01-01', '2019-01-21', '2019-02-18', '2019-05-27', '2019-07-04', '2019-09-02', '2019-11-28', '2019-12-25',
            '2020-01-01', '2020-01-20', '2020-02-17', '2020-05-25', '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
            '2021-01-01', '2021-01-18', '2021-02-15', '2021-05-31', '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24',
            '2022-01-17', '2022-02-21', '2022-05-30', '2022-06-20', '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
            '2023-01-02', '2023-01-16', '2023-02-20', '2023-05-29', '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
            '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
        }
        
        return [day.strftime('%Y-%m-%d') for day in business_days 
                if day.strftime('%Y-%m-%d') not in holidays]
    
    def download_flatfile(self, date_str):
        """Download flat file for a specific date"""
        year = date_str[:4]
        month = date_str[5:7]
        s3_path = f"s3://flatfiles/us_options_opra/day_aggs_v1/{year}/{month}/{date_str}.csv.gz"
        local_file = self.output_dir / f"{date_str}.csv.gz"
        
        if local_file.exists():
            return local_file
        
        cmd = ['aws', 's3', 'cp', s3_path, str(local_file), '--endpoint-url', self.endpoint]
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = self.aws_key
        env['AWS_SECRET_ACCESS_KEY'] = self.aws_secret
        
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
    
    def parse_spy_options(self, flatfile_path):
        """Parse SPY options from flat file"""
        try:
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
    
    def calculate_oi_proxy(self, df, spy_price=None):
        """Calculate OI proxy using volume, transactions, and other features"""
        if df is None or len(df) == 0:
            return None
        
        proxy_df = df.copy()
        
        # Get SPY price from ATM options if not provided
        if spy_price is None:
            atm_calls = proxy_df[
                (proxy_df['option_type'] == 'C') & 
                (proxy_df['volume'] > 0) &
                (proxy_df['dte'].between(7, 45))
            ]
            
            if len(atm_calls) > 0:
                spy_price = (atm_calls['strike'] * atm_calls['volume']).sum() / atm_calls['volume'].sum()
            else:
                spy_price = proxy_df['strike'].median()
        
        # 1. Moneyness (distance from ATM)
        proxy_df['moneyness'] = proxy_df['strike'] / spy_price
        proxy_df['distance_from_atm'] = abs(proxy_df['moneyness'] - 1.0)
        
        # 2. Transaction efficiency
        proxy_df['avg_tx_size'] = proxy_df['volume'] / (proxy_df['transactions'] + 1)
        
        # 3. Liquidity score
        proxy_df['liquidity_score'] = np.sqrt(proxy_df['volume'] * proxy_df['transactions'])
        
        # 4. DTE-adjusted volume
        dte_weight = np.clip(proxy_df['dte'] / 365, 0.1, 1.0)
        proxy_df['dte_adjusted_volume'] = proxy_df['volume'] * dte_weight
        
        # 5. ATM premium
        atm_score = 1.0 / (1.0 + 5 * proxy_df['distance_from_atm'])
        proxy_df['atm_score'] = atm_score
        
        # 6. Build composite OI proxy
        vol_norm = proxy_df['volume'] / (proxy_df['volume'].max() + 1)
        tx_norm = proxy_df['transactions'] / (proxy_df['transactions'].max() + 1)
        liq_norm = proxy_df['liquidity_score'] / (proxy_df['liquidity_score'].max() + 1)
        
        # Weighted combination
        proxy_df['oi_proxy'] = (
            0.3 * vol_norm +           # Volume is important
            0.2 * tx_norm +             # Transactions indicate activity
            0.2 * liq_norm +            # Liquidity score
            0.2 * atm_score +           # ATM bias
            0.1 * dte_weight            # DTE adjustment
        ) * 10000  # Scale to realistic OI numbers
        
        return proxy_df
    
    def process_date(self, date_str):
        """Process a single date"""
        logger.info(f"Processing {date_str}")
        
        # Download flat file
        flatfile = self.download_flatfile(date_str)
        if flatfile is None:
            return None
        
        # Parse SPY options
        spy_df = self.parse_spy_options(flatfile)
        if spy_df is None or len(spy_df) == 0:
            return None
        
        # Calculate OI proxy
        proxy_df = self.calculate_oi_proxy(spy_df)
        if proxy_df is None:
            return None
        
        # Add date column
        proxy_df['date'] = date_str
        
        return proxy_df
    
    def save_data(self, df, date_str, individual_files=False):
        """Save data with new organization structure: data/ticker=spy/year={year}/month={month}"""
        if df is None or len(df) == 0:
            return

        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month

        # New directory structure: data/ticker=spy/year={year}/month={month}
        save_dir = self.output_dir / f"ticker=SPY" / f"year={year}" / f"month={month:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if individual_files:
            filename = f"spy_options_{date_str}.parquet"
            filepath = save_dir / filename
            df.to_parquet(filepath, index=False)
            logger.info(f"💾 Saved individual file: {filepath}")
        else:
            # For combined file, save to the main output_dir
            # This logic needs to be adjusted if combined file should also follow partitioning
            pass # Combined file logic is handled in the main run method

    def download_range(self, start_date, end_date, save_individual=False):
        """Download SPY options data for a date range"""
        logger.info(f"🚀 DOWNLOADING SPY OPTIONS: {start_date} to {end_date}")
        
        trading_days = self.get_trading_days(start_date, end_date)
        logger.info(f"Trading days: {len(trading_days)}")
        
        all_data = []
        successful_dates = []
        
        for date_str in trading_days:
            try:
                df = self.process_date(date_str)
                if df is not None:
                    all_data.append(df)
                    successful_dates.append(date_str)
                    logger.info(f"✅ {date_str}: {len(df):,} contracts")
                    
                    # Save individual files with new structure
                    self.save_data(df, date_str, save_individual)
                else:
                    logger.warning(f"⚠️  {date_str}: No data")
            except Exception as e:
                logger.error(f"❌ {date_str}: Error - {e}")
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save combined file
            combined_file = self.output_dir / f"spy_options_{start_date}_to_{end_date}.parquet"
            combined_df.to_parquet(combined_file, index=False)
            
            # Summary
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 DOWNLOAD SUMMARY")
            logger.info(f"{'='*70}")
            logger.info(f"Successful dates: {len(successful_dates)}/{len(trading_days)}")
            logger.info(f"Total contracts: {len(combined_df):,}")
            logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            logger.info(f"Strike range: ${combined_df['strike'].min():.0f} to ${combined_df['strike'].max():.0f}")
            logger.info(f"DTE range: {combined_df['dte'].min():.0f} to {combined_df['dte'].max():.0f} days")
            logger.info(f"OI Proxy range: {combined_df['oi_proxy'].min():.0f} to {combined_df['oi_proxy'].max():.0f}")
            logger.info(f"💾 Saved to: {combined_file}")
            
            return combined_df
        else:
            logger.error("❌ No data downloaded")
            return None

def main():
    parser = argparse.ArgumentParser(description='Download SPY options data with OI proxy')
    parser.add_argument('start_date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('end_date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', default='data/spy_options', help='Output directory')
    parser.add_argument('--individual', '-i', action='store_true', help='Save individual date files')
    
    args = parser.parse_args()
    
    downloader = SPYOptionsDownloader(args.output)
    result = downloader.download_range(args.start_date, args.end_date, args.individual)
    
    if result is not None:
        print(f"\n✅ SUCCESS: Downloaded {len(result):,} SPY options contracts")
        print(f"   With OI proxy calculated from volume and transaction data")
    else:
        print(f"\n❌ FAILED: No data downloaded")

if __name__ == '__main__':
    main()
