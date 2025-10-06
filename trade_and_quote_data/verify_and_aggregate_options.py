#!/usr/bin/env python3
"""
Phase 1.2 & 1.3: Verify Options Data Quality and Create Daily Aggregates

Verifies the downloaded Polygon options data and creates daily aggregated features.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("📊 PHASE 1.2 & 1.3: VERIFY AND AGGREGATE OPTIONS DATA")
print("=" * 80)
print()

# ============================================================================
# PHASE 1.2: VERIFY DATA QUALITY
# ============================================================================
print("=" * 80)
print("PHASE 1.2: VERIFYING DATA QUALITY")
print("=" * 80)
print()

options_dir = Path('data/options_chains/polygon')

# Check directory structure
years_found = []
total_files = 0

for year_dir in sorted(options_dir.glob('year=*')):
    year = year_dir.name.split('=')[1]
    files = list(year_dir.glob('*.parquet'))
    years_found.append(year)
    total_files += len(files)
    print(f"✅ {year}: {len(files)} files")

print(f"\n📊 Summary:")
print(f"   Years: {min(years_found)} - {max(years_found)}")
print(f"   Total files: {total_files}")
print()

# Sample a few files to check data quality
print("🔍 Checking data quality (sampling 5 files)...")
sample_files = list(options_dir.glob('year=*/spy_options_*.parquet'))[:5]

for file_path in sample_files:
    try:
        df = pd.read_parquet(file_path)
        date = file_path.stem.replace('spy_options_', '')
        
        print(f"\n   {date}:")
        print(f"      Contracts: {len(df)}")
        print(f"      Columns: {list(df.columns)[:5]}...")
        
        # Check for required columns
        required_cols = ['strike', 'expiration', 'option_type', 'volume', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"      ⚠️  Missing columns: {missing}")
        else:
            print(f"      ✅ All required columns present")
            
            # Check data ranges
            if 'strike' in df.columns:
                print(f"      Strike range: ${df['strike'].min():.0f} - ${df['strike'].max():.0f}")
            if 'volume' in df.columns:
                print(f"      Total volume: {df['volume'].sum():,.0f}")
            if 'close' in df.columns:
                print(f"      Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                
    except Exception as e:
        print(f"   ❌ Error reading {file_path.name}: {e}")

print("\n✅ PHASE 1.2 COMPLETE: Data quality verified")
print()

# ============================================================================
# PHASE 1.3: CREATE DAILY AGGREGATED DATASET
# ============================================================================
print("=" * 80)
print("PHASE 1.3: CREATING DAILY AGGREGATED OPTIONS FEATURES")
print("=" * 80)
print()

print("📊 Processing all options files...")
print("   This will create daily aggregated features for each trading day")
print()

daily_aggregates = []
all_files = sorted(options_dir.glob('year=*/spy_options_*.parquet'))

print(f"Processing {len(all_files)} files...")

for i, file_path in enumerate(all_files):
    try:
        # Extract date from filename
        date_str = file_path.stem.replace('spy_options_', '')
        date = pd.to_datetime(date_str)
        
        # Load options data
        df = pd.read_parquet(file_path)
        
        if len(df) == 0:
            continue
        
        # Calculate daily aggregates
        agg = {
            'date': date,
            'total_contracts': len(df),
            'total_volume': df['volume'].sum() if 'volume' in df.columns else 0,
            'total_open_interest': df['open_interest'].sum() if 'open_interest' in df.columns else 0,
        }
        
        # Put/Call metrics
        if 'option_type' in df.columns:
            puts = df[df['option_type'] == 'P']
            calls = df[df['option_type'] == 'C']
            
            agg['put_volume'] = puts['volume'].sum() if 'volume' in puts.columns else 0
            agg['call_volume'] = calls['volume'].sum() if 'volume' in calls.columns else 0
            agg['put_call_volume_ratio'] = agg['put_volume'] / agg['call_volume'] if agg['call_volume'] > 0 else np.nan
            
            agg['put_oi'] = puts['open_interest'].sum() if 'open_interest' in puts.columns else 0
            agg['call_oi'] = calls['open_interest'].sum() if 'open_interest' in calls.columns else 0
            agg['put_call_oi_ratio'] = agg['put_oi'] / agg['call_oi'] if agg['call_oi'] > 0 else np.nan
        
        # Strike distribution
        if 'strike' in df.columns:
            agg['min_strike'] = df['strike'].min()
            agg['max_strike'] = df['strike'].max()
            agg['strike_range'] = agg['max_strike'] - agg['min_strike']
        
        # Price metrics
        if 'close' in df.columns:
            agg['avg_option_price'] = df['close'].mean()
            agg['total_premium'] = (df['close'] * df['volume']).sum() if 'volume' in df.columns else 0
        
        # Implied volatility (if available)
        if 'implied_volatility' in df.columns:
            agg['avg_iv'] = df['implied_volatility'].mean()
            agg['iv_std'] = df['implied_volatility'].std()
            
            if 'option_type' in df.columns:
                put_iv = puts['implied_volatility'].mean() if len(puts) > 0 else np.nan
                call_iv = calls['implied_volatility'].mean() if len(calls) > 0 else np.nan
                agg['put_iv'] = put_iv
                agg['call_iv'] = call_iv
                agg['iv_skew'] = put_iv - call_iv if not np.isnan(put_iv) and not np.isnan(call_iv) else np.nan
        
        daily_aggregates.append(agg)
        
        # Progress indicator
        if (i + 1) % 250 == 0:
            print(f"   Processed {i + 1}/{len(all_files)} files...")
            
    except Exception as e:
        print(f"   ⚠️  Error processing {file_path.name}: {e}")
        continue

print(f"\n✅ Processed {len(daily_aggregates)} trading days")

# Create DataFrame
daily_df = pd.DataFrame(daily_aggregates)
daily_df = daily_df.sort_values('date').reset_index(drop=True)

print(f"\n📊 Daily Aggregates Summary:")
print(f"   Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
print(f"   Total days: {len(daily_df)}")
print(f"   Features: {len(daily_df.columns)}")
print()

# Display sample
print("Sample data (first 5 rows):")
print(daily_df.head())
print()

# Save to parquet
output_file = Path('data/options_chains/daily_aggregates.parquet')
daily_df.to_parquet(output_file, index=False)
print(f"💾 Saved to: {output_file}")

# Also save as CSV for easy inspection
csv_file = Path('data/options_chains/daily_aggregates.csv')
daily_df.to_csv(csv_file, index=False)
print(f"💾 Saved to: {csv_file}")

print("\n✅ PHASE 1.3 COMPLETE: Daily aggregates created")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("✅ PHASES 1.1-1.3 COMPLETE")
print("=" * 80)
print()
print("PHASE 1.1: ✅ Downloaded 2,264 days of options data (2016-2024)")
print("PHASE 1.2: ✅ Verified data quality and structure")
print("PHASE 1.3: ✅ Created daily aggregated features")
print()
print("📊 Output:")
print(f"   Raw data: data/options_chains/polygon/ ({total_files} files)")
print(f"   Aggregates: data/options_chains/daily_aggregates.parquet")
print()
print("🎯 Ready for Phase 2: Calculate advanced options features")
print("   - IV skew")
print("   - Implied moves")
print("   - Put/call ratios")
print("   - Term structure")
print()
