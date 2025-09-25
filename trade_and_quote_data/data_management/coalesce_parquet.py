#!/usr/bin/env python3
"""
Coalesce multiple Parquet files into a single file

This script combines all Parquet files from the SPY trades download into one consolidated file.
"""

import pandas as pd
import pyarrow.parquet as pq
import os
import sys
from pathlib import Path

def coalesce_parquet_files(input_dir, output_file):
    """Coalesce multiple Parquet files into a single file."""
    try:
        print(f"📊 Coalescing Parquet files from: {input_dir}")
        
        # Read all Parquet files
        parquet_files = list(Path(input_dir).glob("*.parquet"))
        parquet_files = [f for f in parquet_files if not f.name.startswith('.')]
        
        if not parquet_files:
            print("❌ No Parquet files found in directory")
            return False
        
        print(f"📁 Found {len(parquet_files)} Parquet files")
        
        # Read and combine all files
        dfs = []
        total_records = 0
        
        for i, file_path in enumerate(parquet_files):
            print(f"📖 Reading file {i+1}/{len(parquet_files)}: {file_path.name}")
            
            df = pd.read_parquet(file_path, engine='pyarrow')
            dfs.append(df)
            total_records += len(df)
            
            print(f"   Records: {len(df):,}")
        
        print(f"📊 Total records to combine: {total_records:,}")
        
        # Combine all DataFrames
        print("🔄 Combining DataFrames...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        print(f"✅ Combined DataFrame: {len(combined_df):,} records")
        print(f"📋 Columns: {list(combined_df.columns)}")
        
        # Save as single Parquet file
        print(f"💾 Saving to: {output_file}")
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        # Verify output file
        output_size = os.path.getsize(output_file)
        print(f"📁 Output file size: {output_size / (1024*1024):.1f} MB")
        
        # Verify record count
        verify_df = pd.read_parquet(output_file, engine='pyarrow')
        print(f"✅ Verification: {len(verify_df):,} records in output file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error coalescing Parquet files: {e}")
        return False

def main():
    """Main function."""
    print("🚀 PARQUET COALESCER")
    print("="*50)
    
    # Get input directory from command line or use default
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "/Users/varun/code/quant_final_final/trade_and_quote_data/data/spy_trades_2016_full_postgres"
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return
    
    # Output file
    output_file = "/Users/varun/code/quant_final_final/trade_and_quote_data/data/spy_trades_2016_consolidated.parquet"
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output file: {output_file}")
    
    # Coalesce files
    if coalesce_parquet_files(input_dir, output_file):
        print("🎉 Successfully coalesced all Parquet files!")
        print(f"📁 Single file created: {output_file}")
    else:
        print("❌ Failed to coalesce Parquet files")

if __name__ == "__main__":
    main()
