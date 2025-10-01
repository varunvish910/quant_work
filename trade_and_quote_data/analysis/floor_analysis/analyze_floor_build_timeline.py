#!/usr/bin/env python3
"""
Analyze Floor Build Timeline
===========================

Track when the $600 floor and $660 floor were built by analyzing
historical OI development at these key strikes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_floor_build_timeline():
    """Analyze the timeline of floor building for $600 and $660 levels"""
    
    print(f"🔍 FLOOR BUILD TIMELINE ANALYSIS")
    print("=" * 50)
    
    # Find all available data files going back in time
    data_dir = Path("data/options_chains/SPY")
    all_files = []
    
    for year_dir in sorted(data_dir.glob("202*")):
        for month_dir in sorted(year_dir.glob("*")):
            for file in sorted(month_dir.glob("SPY_options_snapshot_*.parquet")):
                date_str = file.stem.split('_')[-1]
                all_files.append((date_str, file))
    
    # Sort by date
    all_files.sort(key=lambda x: x[0])
    
    if not all_files:
        print("❌ No data files found")
        return
    
    print(f"📊 Found {len(all_files)} total files")
    print(f"Date range: {all_files[0][0]} to {all_files[-1][0]}")
    
    # Track key strikes over time
    key_strikes = [600, 660]  # Focus on these two levels
    strike_history = {strike: [] for strike in key_strikes}
    
    # Sample every few days to avoid too much data
    sample_interval = max(1, len(all_files) // 100)  # Sample ~100 points
    sampled_files = all_files[::sample_interval]
    
    print(f"📈 Analyzing {len(sampled_files)} sampled dates...")
    
    for date_str, file_path in sampled_files:
        try:
            df = pd.read_parquet(file_path)
            spy_price = df['underlying_price'].iloc[0]
            
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            for strike in key_strikes:
                # Get put OI for this strike
                strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P')]
                
                if len(strike_puts) > 0:
                    total_oi = strike_puts['oi_proxy'].sum()
                    total_volume = strike_puts['volume'].sum()
                    vol_oi_ratio = total_volume / (total_oi + 1e-6)
                    
                    strike_history[strike].append({
                        'date': formatted_date,
                        'date_raw': date_str,
                        'spy_price': spy_price,
                        'oi': total_oi,
                        'volume': total_volume,
                        'vol_oi_ratio': vol_oi_ratio,
                        'distance': spy_price - strike
                    })
        
        except Exception as e:
            continue
    
    # Analyze each strike's development
    for strike in key_strikes:
        history = strike_history[strike]
        if not history:
            continue
        
        print(f"\n🏗️ ${strike} PUT FLOOR DEVELOPMENT:")
        print("-" * 40)
        
        # Find when significant OI first appeared
        significant_threshold = 10000  # OI threshold for "significant"
        first_significant = None
        
        for entry in history:
            if entry['oi'] > significant_threshold:
                first_significant = entry
                break
        
        if first_significant:
            print(f"  • First significant OI (>{significant_threshold:,}): {first_significant['date']}")
            print(f"    - OI: {first_significant['oi']:,.0f}")
            print(f"    - SPY was: ${first_significant['spy_price']:.2f}")
            print(f"    - Distance: {first_significant['distance']:.0f} points below")
        
        # Find peak OI and when it occurred
        peak_entry = max(history, key=lambda x: x['oi'])
        print(f"  • Peak OI: {peak_entry['oi']:,.0f} on {peak_entry['date']}")
        print(f"    - SPY was: ${peak_entry['spy_price']:.2f}")
        
        # Find current levels
        current_entry = history[-1]
        print(f"  • Current OI: {current_entry['oi']:,.0f} on {current_entry['date']}")
        print(f"    - SPY is: ${current_entry['spy_price']:.2f}")
        
        # Calculate build period
        if first_significant:
            first_date = datetime.strptime(first_significant['date'], '%Y-%m-%d')
            current_date = datetime.strptime(current_entry['date'], '%Y-%m-%d')
            build_days = (current_date - first_date).days
            
            print(f"  • Build period: {build_days} days ({build_days/30:.1f} months)")
        
        # Show key milestones in development
        print(f"\n  📈 Development milestones:")
        
        milestones = [10000, 25000, 50000, 75000]
        for milestone in milestones:
            for entry in history:
                if entry['oi'] >= milestone:
                    print(f"    - {milestone:,} OI reached: {entry['date']} (SPY: ${entry['spy_price']:.2f})")
                    break
        
        # Recent activity analysis (last 30 data points)
        recent_history = history[-30:] if len(history) > 30 else history
        recent_activity = []
        
        print(f"\n  ⚡ Recent activity pattern:")
        
        for i, entry in enumerate(recent_history[-10:]):  # Last 10 data points
            activity_type = "HIGH" if entry['vol_oi_ratio'] > 0.5 else "MODERATE" if entry['vol_oi_ratio'] > 0.2 else "LOW"
            print(f"    {entry['date']}: OI {entry['oi']:,.0f}, V/OI {entry['vol_oi_ratio']:.2f} ({activity_type})")
    
    # Comparative analysis
    print(f"\n🔍 COMPARATIVE TIMELINE ANALYSIS:")
    print("=" * 50)
    
    # Compare build periods
    for strike in key_strikes:
        history = strike_history[strike]
        if not history:
            continue
        
        # Find significant build start
        significant_start = None
        for entry in history:
            if entry['oi'] > 10000:
                significant_start = entry
                break
        
        if significant_start:
            start_date = datetime.strptime(significant_start['date'], '%Y-%m-%d')
            end_date = datetime.strptime(history[-1]['date'], '%Y-%m-%d')
            build_period = (end_date - start_date).days
            
            print(f"${strike} Floor:")
            print(f"  • Build start: {significant_start['date']}")
            print(f"  • Build period: {build_period} days ({build_period/30:.1f} months)")
            print(f"  • Peak OI: {max(history, key=lambda x: x['oi'])['oi']:,.0f}")
            print(f"  • Current OI: {history[-1]['oi']:,.0f}")
    
    # Speed of build analysis
    print(f"\n⚡ BUILD SPEED ANALYSIS:")
    
    for strike in key_strikes:
        history = strike_history[strike]
        if len(history) < 10:
            continue
        
        # Calculate OI growth rate
        start_oi = history[0]['oi'] if history[0]['oi'] > 1000 else 1000
        end_oi = history[-1]['oi']
        total_days = len(history) * sample_interval  # Approximate
        
        if total_days > 0:
            daily_growth = (end_oi / start_oi) ** (1/total_days) - 1
            print(f"${strike}: {daily_growth*100:.2f}% daily OI growth rate")
    
    # Recent acceleration analysis
    print(f"\n🚀 RECENT ACCELERATION:")
    
    for strike in key_strikes:
        history = strike_history[strike]
        if len(history) < 20:
            continue
        
        # Compare last month vs previous period
        recent_period = history[-10:]  # Last 10 data points
        previous_period = history[-20:-10]  # Previous 10 data points
        
        recent_avg_oi = np.mean([x['oi'] for x in recent_period])
        previous_avg_oi = np.mean([x['oi'] for x in previous_period])
        
        if previous_avg_oi > 0:
            acceleration = (recent_avg_oi / previous_avg_oi - 1) * 100
            
            if acceleration > 50:
                acceleration_level = "MASSIVE ACCELERATION"
            elif acceleration > 20:
                acceleration_level = "STRONG ACCELERATION"
            elif acceleration > 5:
                acceleration_level = "MODERATE GROWTH"
            elif acceleration > -5:
                acceleration_level = "STABLE"
            else:
                acceleration_level = "DECLINING"
            
            print(f"${strike}: {acceleration:+.1f}% recent change - {acceleration_level}")
    
    # Activity pattern analysis
    print(f"\n📊 ACTIVITY PATTERN COMPARISON:")
    
    for strike in key_strikes:
        history = strike_history[strike]
        if not history:
            continue
        
        # Average V/OI over time
        avg_vol_oi = np.mean([x['vol_oi_ratio'] for x in history])
        recent_vol_oi = np.mean([x['vol_oi_ratio'] for x in history[-5:]])
        
        if avg_vol_oi < 0.2:
            historical_pattern = "ACCUMULATED (low activity)"
        elif avg_vol_oi < 0.5:
            historical_pattern = "MIXED (moderate activity)"
        else:
            historical_pattern = "ACTIVE (high turnover)"
        
        if recent_vol_oi > avg_vol_oi * 2:
            recent_pattern = "SPIKING ACTIVITY"
        elif recent_vol_oi > avg_vol_oi * 1.5:
            recent_pattern = "ELEVATED ACTIVITY"
        else:
            recent_pattern = "NORMAL ACTIVITY"
        
        print(f"${strike}:")
        print(f"  • Historical: {historical_pattern} (avg V/OI: {avg_vol_oi:.2f})")
        print(f"  • Recent: {recent_pattern} (recent V/OI: {recent_vol_oi:.2f})")

if __name__ == "__main__":
    analyze_floor_build_timeline()