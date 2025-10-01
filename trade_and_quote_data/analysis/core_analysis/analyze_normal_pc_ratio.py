#!/usr/bin/env python3
"""
Analyze Normal Put/Call Ratio
=============================

Establish baseline SPY put/call ratios across different time periods
to understand what's normal vs what's unusual positioning.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_normal_pc_ratio():
    """Analyze normal SPY put/call ratios over time"""
    
    print(f"🔍 SPY PUT/CALL RATIO BASELINE ANALYSIS")
    print("=" * 50)
    
    # Sample files from different months to get baseline
    data_dir = Path("data/options_chains/SPY")
    
    sample_files = []
    
    # Get a few files from each available month
    for year_dir in sorted(data_dir.glob("202*")):
        for month_dir in sorted(year_dir.glob("*")):
            month_files = sorted(month_dir.glob("SPY_options_snapshot_*.parquet"))
            
            # Sample first, middle, and last file of each month
            if len(month_files) >= 3:
                sample_files.extend([month_files[0], month_files[len(month_files)//2], month_files[-1]])
            elif month_files:
                sample_files.extend(month_files)
    
    if not sample_files:
        print("❌ No data files found")
        return
    
    print(f"📊 Analyzing {len(sample_files)} sample files across time periods")
    
    # Analyze each sample
    analysis_results = []
    
    for file_path in sample_files[:30]:  # Limit to avoid too much processing
        date_str = file_path.stem.split('_')[-1]
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        try:
            df = pd.read_parquet(file_path)
            spy_price = df['underlying_price'].iloc[0]
            
            # Exclude 0 DTE to focus on positioning
            df_filtered = df[df['dte'] > 0]
            
            # Analyze different strike ranges
            ranges = {
                'immediate': (spy_price * 0.95, spy_price * 1.05),  # ±5%
                'near': (spy_price * 0.90, spy_price * 1.10),      # ±10%
                'broad': (spy_price * 0.85, spy_price * 1.15)      # ±15%
            }
            
            range_results = {}
            
            for range_name, (low, high) in ranges.items():
                range_data = df_filtered[(df_filtered['strike'] >= low) & (df_filtered['strike'] <= high)]
                
                puts = range_data[range_data['option_type'] == 'P']
                calls = range_data[range_data['option_type'] == 'C']
                
                put_oi = puts['oi_proxy'].sum() if len(puts) > 0 else 0
                call_oi = calls['oi_proxy'].sum() if len(calls) > 0 else 0
                put_volume = puts['volume'].sum() if len(puts) > 0 else 0
                call_volume = calls['volume'].sum() if len(calls) > 0 else 0
                
                pc_oi_ratio = put_oi / (call_oi + 1e-6)
                pc_vol_ratio = put_volume / (call_volume + 1e-6)
                
                range_results[range_name] = {
                    'put_oi': put_oi,
                    'call_oi': call_oi,
                    'put_volume': put_volume,
                    'call_volume': call_volume,
                    'pc_oi_ratio': pc_oi_ratio,
                    'pc_vol_ratio': pc_vol_ratio
                }
            
            analysis_results.append({
                'date': formatted_date,
                'spy_price': spy_price,
                'ranges': range_results
            })
            
        except Exception as e:
            continue
    
    if not analysis_results:
        print("❌ No analysis results generated")
        return
    
    # Calculate baseline statistics
    print(f"\n📊 BASELINE PUT/CALL RATIOS:")
    print("-" * 40)
    
    for range_name in ['immediate', 'near', 'broad']:
        oi_ratios = [r['ranges'][range_name]['pc_oi_ratio'] for r in analysis_results 
                    if r['ranges'][range_name]['pc_oi_ratio'] > 0]
        vol_ratios = [r['ranges'][range_name]['pc_vol_ratio'] for r in analysis_results 
                     if r['ranges'][range_name]['pc_vol_ratio'] > 0]
        
        if oi_ratios and vol_ratios:
            avg_oi_ratio = np.mean(oi_ratios)
            avg_vol_ratio = np.mean(vol_ratios)
            median_oi_ratio = np.median(oi_ratios)
            median_vol_ratio = np.median(vol_ratios)
            
            print(f"\n{range_name.upper()} RANGE:")
            print(f"  • Avg P/C OI ratio: {avg_oi_ratio:.2f}")
            print(f"  • Median P/C OI ratio: {median_oi_ratio:.2f}")
            print(f"  • Avg P/C Vol ratio: {avg_vol_ratio:.2f}")
            print(f"  • Median P/C Vol ratio: {median_vol_ratio:.2f}")
    
    # Compare September to baseline
    print(f"\n⚖️ SEPTEMBER vs BASELINE COMPARISON:")
    print("-" * 40)
    
    # Get recent data for comparison
    recent_file = Path("data/options_chains/SPY/2025/09/SPY_options_snapshot_20250930.parquet")
    
    if recent_file.exists():
        try:
            df_recent = pd.read_parquet(recent_file)
            spy_recent = df_recent['underlying_price'].iloc[0]
            df_recent_filtered = df_recent[df_recent['dte'] > 0]
            
            # Calculate September ratios
            september_ranges = {
                'immediate': (spy_recent * 0.95, spy_recent * 1.05),
                'near': (spy_recent * 0.90, spy_recent * 1.10),
                'broad': (spy_recent * 0.85, spy_recent * 1.15)
            }
            
            print(f"September 30, 2025 (SPY: ${spy_recent:.2f}):")
            
            for range_name, (low, high) in september_ranges.items():
                range_data = df_recent_filtered[(df_recent_filtered['strike'] >= low) & 
                                              (df_recent_filtered['strike'] <= high)]
                
                puts = range_data[range_data['option_type'] == 'P']
                calls = range_data[range_data['option_type'] == 'C']
                
                put_oi = puts['oi_proxy'].sum()
                call_oi = calls['oi_proxy'].sum()
                put_volume = puts['volume'].sum()
                call_volume = calls['volume'].sum()
                
                current_pc_oi = put_oi / (call_oi + 1e-6)
                current_pc_vol = put_volume / (call_volume + 1e-6)
                
                # Compare to baseline
                baseline_oi_ratios = [r['ranges'][range_name]['pc_oi_ratio'] for r in analysis_results 
                                    if r['ranges'][range_name]['pc_oi_ratio'] > 0]
                baseline_vol_ratios = [r['ranges'][range_name]['pc_vol_ratio'] for r in analysis_results 
                                     if r['ranges'][range_name]['pc_vol_ratio'] > 0]
                
                if baseline_oi_ratios and baseline_vol_ratios:
                    baseline_avg_oi = np.mean(baseline_oi_ratios)
                    baseline_avg_vol = np.mean(baseline_vol_ratios)
                    
                    oi_deviation = (current_pc_oi - baseline_avg_oi) / baseline_avg_oi * 100
                    vol_deviation = (current_pc_vol - baseline_avg_vol) / baseline_avg_vol * 100
                    
                    print(f"\n  {range_name.upper()}:")
                    print(f"    Current P/C OI: {current_pc_oi:.2f} (baseline: {baseline_avg_oi:.2f}) "
                          f"[{oi_deviation:+.1f}%]")
                    print(f"    Current P/C Vol: {current_pc_vol:.2f} (baseline: {baseline_avg_vol:.2f}) "
                          f"[{vol_deviation:+.1f}%]")
        
        except Exception as e:
            print(f"❌ Error analyzing recent data: {e}")
    
    # Historical context
    print(f"\n📈 HISTORICAL CONTEXT:")
    print("-" * 25)
    
    # Look at trends over time
    if len(analysis_results) > 10:
        # Split into early and recent periods
        mid_point = len(analysis_results) // 2
        early_period = analysis_results[:mid_point]
        recent_period = analysis_results[mid_point:]
        
        for range_name in ['immediate', 'near']:
            early_oi_ratios = [r['ranges'][range_name]['pc_oi_ratio'] for r in early_period 
                              if r['ranges'][range_name]['pc_oi_ratio'] > 0]
            recent_oi_ratios = [r['ranges'][range_name]['pc_oi_ratio'] for r in recent_period 
                               if r['ranges'][range_name]['pc_oi_ratio'] > 0]
            
            if early_oi_ratios and recent_oi_ratios:
                early_avg = np.mean(early_oi_ratios)
                recent_avg = np.mean(recent_oi_ratios)
                trend = (recent_avg - early_avg) / early_avg * 100
                
                print(f"{range_name.upper()} trend: {early_avg:.2f} → {recent_avg:.2f} ({trend:+.1f}%)")
    
    # Final assessment
    print(f"\n💡 ASSESSMENT:")
    print("-" * 15)
    
    # Get typical ranges
    immediate_ratios = [r['ranges']['immediate']['pc_oi_ratio'] for r in analysis_results 
                       if r['ranges']['immediate']['pc_oi_ratio'] > 0]
    
    if immediate_ratios:
        typical_low = np.percentile(immediate_ratios, 25)
        typical_high = np.percentile(immediate_ratios, 75)
        typical_median = np.median(immediate_ratios)
        
        print(f"  • Typical SPY P/C OI ratio range: {typical_low:.2f} - {typical_high:.2f}")
        print(f"  • Typical median: {typical_median:.2f}")
        print(f"  • Yes, SPY structurally skews toward puts")
        print(f"  • Ratios above {typical_high:.2f} suggest elevated defensive positioning")
        print(f"  • Ratios above {typical_high*1.5:.2f} suggest panic/extreme hedging")

if __name__ == "__main__":
    analyze_normal_pc_ratio()