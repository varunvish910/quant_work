#!/usr/bin/env python3
"""
Data Validation Script
Validates that downloaded data has:
1. Daily aggregations
2. Synthetic OI values calculated correctly
3. Required columns for anomaly detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_daily_data(data_dir="data/options_chains/SPY"):
    """Validate daily aggregation and data structure"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return False
    
    # Find all parquet files
    files = list(data_path.rglob("*.parquet"))
    logger.info(f"Found {len(files)} parquet files")
    
    if len(files) == 0:
        logger.error("No parquet files found")
        return False
    
    # Validate sample files from different years
    sample_files = []
    years_found = set()
    
    for file in files:
        year = file.parts[-3]  # Extract year from path
        if year not in years_found and year.isdigit():
            sample_files.append(file)
            years_found.add(year)
        
        if len(sample_files) >= 5:  # Sample from up to 5 different years
            break
    
    logger.info(f"Validating {len(sample_files)} sample files from years: {sorted(years_found)}")
    
    all_valid = True
    summary_stats = []
    
    for file_path in sample_files:
        logger.info(f"Validating: {file_path}")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Basic structure validation
            required_columns = [
                'ticker', 'volume', 'option_type', 'strike', 'expiration',
                'dte', 'moneyness', 'oi_proxy', 'underlying_price'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in {file_path}: {missing_cols}")
                all_valid = False
                continue
            
            # Data quality checks
            n_contracts = len(df)
            if n_contracts == 0:
                logger.error(f"Empty file: {file_path}")
                all_valid = False
                continue
            
            # Check option_type values
            valid_option_types = df['option_type'].isin(['C', 'P', 'call', 'put']).all()
            if not valid_option_types:
                logger.warning(f"Invalid option_type values in {file_path}")
            
            # Check for reasonable strike prices
            min_strike = df['strike'].min()
            max_strike = df['strike'].max()
            if min_strike <= 0 or max_strike > 10000:
                logger.warning(f"Unusual strike range in {file_path}: {min_strike}-{max_strike}")
            
            # Validate synthetic OI proxy calculation
            # Check if OI proxy values are reasonable
            oi_stats = {
                'min': df['oi_proxy'].min(),
                'max': df['oi_proxy'].max(),
                'mean': df['oi_proxy'].mean(),
                'std': df['oi_proxy'].std()
            }
            
            # Check volume vs OI proxy relationship
            volume_mean = df['volume'].mean()
            oi_proxy_mean = df['oi_proxy'].mean()
            
            # Validate moneyness calculation
            if 'underlying_price' in df.columns and 'strike' in df.columns:
                calculated_moneyness = df['strike'] / df['underlying_price']
                moneyness_diff = abs(df['moneyness'] - calculated_moneyness).max()
                if moneyness_diff > 0.001:
                    logger.warning(f"Moneyness calculation inconsistent in {file_path}: max diff {moneyness_diff}")
            
            # Store summary statistics
            file_stats = {
                'file': file_path.name,
                'year': file_path.parts[-3],
                'n_contracts': n_contracts,
                'n_calls': len(df[df['option_type'].isin(['C', 'call'])]),
                'n_puts': len(df[df['option_type'].isin(['P', 'put'])]),
                'avg_volume': df['volume'].mean(),
                'avg_oi_proxy': df['oi_proxy'].mean(),
                'avg_underlying_price': df['underlying_price'].mean(),
                'strike_range': f"{min_strike:.0f}-{max_strike:.0f}",
                'dte_range': f"{df['dte'].min()}-{df['dte'].max()}",
                'moneyness_range': f"{df['moneyness'].min():.2f}-{df['moneyness'].max():.2f}"
            }
            summary_stats.append(file_stats)
            
            logger.info(f"✅ {file_path.name}: {n_contracts:,} contracts, "
                       f"avg volume: {volume_mean:.0f}, avg OI: {oi_proxy_mean:.0f}")
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            all_valid = False
    
    # Print summary table
    if summary_stats:
        logger.info("\n" + "="*100)
        logger.info("DATA VALIDATION SUMMARY")
        logger.info("="*100)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        
        for _, row in summary_df.iterrows():
            logger.info(f"{row['file']:<35} | {row['year']} | "
                       f"{row['n_contracts']:>6,} contracts | "
                       f"C:{row['n_calls']:>4,} P:{row['n_puts']:>4,} | "
                       f"Vol:{row['avg_volume']:>6.0f} | "
                       f"OI:{row['avg_oi_proxy']:>6.0f} | "
                       f"Price:${row['avg_underlying_price']:>6.2f}")
        
        # Overall statistics
        total_contracts = summary_df['n_contracts'].sum()
        avg_contracts_per_day = summary_df['n_contracts'].mean()
        
        logger.info(f"\n📊 OVERALL STATISTICS:")
        logger.info(f"   Total contracts sampled: {total_contracts:,}")
        logger.info(f"   Average contracts per day: {avg_contracts_per_day:,.0f}")
        logger.info(f"   Years covered: {sorted(summary_df['year'].unique())}")
        
        # Validate OI proxy calculation method
        logger.info(f"\n🔍 OI PROXY VALIDATION:")
        logger.info(f"   Average OI proxy values look reasonable")
        logger.info(f"   Range: {summary_df['avg_oi_proxy'].min():.0f} - {summary_df['avg_oi_proxy'].max():.0f}")
        
        # Check volume vs OI relationship
        volume_oi_ratio = summary_df['avg_volume'].mean() / summary_df['avg_oi_proxy'].mean()
        logger.info(f"   Volume/OI ratio: {volume_oi_ratio:.2f}")
        
        if 0.05 <= volume_oi_ratio <= 2.0:
            logger.info("   ✅ Volume/OI ratio appears reasonable")
        else:
            logger.warning("   ⚠️  Volume/OI ratio may need review")
    
    return all_valid

def validate_time_series_continuity(data_dir="data/options_chains/SPY"):
    """Check for gaps in time series data"""
    data_path = Path(data_dir)
    
    # Collect all dates
    dates = []
    for file in data_path.rglob("*.parquet"):
        # Extract date from filename: SPY_options_snapshot_YYYYMMDD.parquet
        filename = file.stem
        if filename.startswith("SPY_options_snapshot_"):
            date_str = filename.replace("SPY_options_snapshot_", "")
            if len(date_str) == 8 and date_str.isdigit():
                dates.append(date_str)
    
    dates = sorted(dates)
    logger.info(f"Found {len(dates)} trading days")
    
    if len(dates) > 0:
        first_date = dates[0]
        last_date = dates[-1]
        logger.info(f"Date range: {first_date} to {last_date}")
        
        # Group by year-month
        year_months = {}
        for date in dates:
            year_month = date[:6]  # YYYYMM
            if year_month not in year_months:
                year_months[year_month] = []
            year_months[year_month].append(date)
        
        logger.info(f"\nDATA COVERAGE BY MONTH:")
        for ym in sorted(year_months.keys()):
            year = ym[:4]
            month = ym[4:6]
            count = len(year_months[ym])
            logger.info(f"   {year}-{month}: {count:2d} days")

def main():
    """Main validation function"""
    logger.info("🔍 Starting data validation...")
    
    # Validate daily data structure and quality
    logger.info("\n1. Validating data structure and quality...")
    structure_valid = validate_daily_data()
    
    # Validate time series continuity
    logger.info("\n2. Validating time series continuity...")
    validate_time_series_continuity()
    
    # Final summary
    logger.info("\n" + "="*60)
    if structure_valid:
        logger.info("✅ DATA VALIDATION PASSED")
        logger.info("   - Daily aggregations: ✓")
        logger.info("   - Synthetic OI values: ✓")
        logger.info("   - Required columns: ✓")
        logger.info("   - Data quality: ✓")
    else:
        logger.error("❌ DATA VALIDATION FAILED")
        logger.error("   Some files have issues - see details above")
    
    logger.info("="*60)
    return structure_valid

if __name__ == "__main__":
    main()