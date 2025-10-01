#!/usr/bin/env python3
"""
Analyze Current Put Floor
========================

Using the same methodology that identified the $520 floor in July 2024,
this analyzes current SPY options data to find where institutional 
put support levels are being built today.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_current_put_floor():
    """Analyze current put floor support levels"""
    
    print(f"🔍 CURRENT PUT FLOOR ANALYSIS")
    print("=" * 50)
    
    # Try to find the most recent data file
    data_dir = Path("data/options_chains/SPY")
    
    # Look for recent data files
    recent_files = []
    for year_dir in sorted(data_dir.glob("202*"), reverse=True):
        for month_dir in sorted(year_dir.glob("*"), reverse=True):
            for file in sorted(month_dir.glob("SPY_options_snapshot_*.parquet"), reverse=True):
                recent_files.append(file)
                if len(recent_files) >= 10:  # Get last 10 files
                    break
            if len(recent_files) >= 10:
                break
        if len(recent_files) >= 10:
            break
    
    if not recent_files:
        print("❌ No recent data files found")
        return
    
    print(f"📊 Found {len(recent_files)} recent files")
    
    # Analyze the most recent file for current levels
    latest_file = recent_files[0]
    date_str = latest_file.stem.split('_')[-1]
    
    print(f"📅 Most recent data: {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    
    try:
        # Load latest data
        df = pd.read_parquet(latest_file)
        spy_price = df['underlying_price'].iloc[0]
        
        print(f"💰 Current SPY Price: ${spy_price:.2f}")
        print(f"📊 Total contracts: {len(df):,}")
        
        # Define potential floor strikes based on current price
        # Look at strikes 5-15% below current price (similar to July 2024 pattern)
        floor_range_low = spy_price * 0.85  # 15% below
        floor_range_high = spy_price * 0.95  # 5% below
        
        # Round to nearest $5 strikes
        potential_floors = []
        current_strike = int(floor_range_low / 5) * 5
        while current_strike <= floor_range_high:
            potential_floors.append(current_strike)
            current_strike += 5
        
        print(f"\n🎯 ANALYZING POTENTIAL FLOOR STRIKES:")
        print(f"Range: ${floor_range_low:.0f} - ${floor_range_high:.0f}")
        print(f"Strikes: {potential_floors}")
        
        # Analyze put OI at potential floor levels
        floor_analysis = {}
        
        for strike in potential_floors:
            strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P')]
            
            if len(strike_puts) > 0:
                total_oi = strike_puts['oi_proxy'].sum()
                total_volume = strike_puts['volume'].sum()
                total_contracts = len(strike_puts)
                avg_dte = strike_puts['dte'].mean() if len(strike_puts) > 0 else 0
                
                # Calculate distance metrics
                distance_points = spy_price - strike
                distance_pct = distance_points / spy_price * 100
                
                floor_analysis[strike] = {
                    'oi': total_oi,
                    'volume': total_volume,
                    'contracts': total_contracts,
                    'avg_dte': avg_dte,
                    'distance_points': distance_points,
                    'distance_pct': distance_pct,
                    'vol_oi_ratio': total_volume / (total_oi + 1e-6)
                }
        
        # Sort by OI to find strongest levels
        sorted_floors = sorted(floor_analysis.items(), key=lambda x: x[1]['oi'], reverse=True)
        
        print(f"\n🏗️ PUT FLOOR ANALYSIS (Sorted by OI):")
        print(f"{'Strike':<8} {'OI':<12} {'Volume':<10} {'V/OI':<6} {'Distance':<12} {'DTE':<6}")
        print("-" * 70)
        
        significant_floors = []
        for strike, data in sorted_floors:
            if data['oi'] > 1000:  # Only show significant levels
                significant_floors.append((strike, data))
                
                print(f"${strike:<7} {data['oi']:<12,.0f} {data['volume']:<10,.0f} "
                      f"{data['vol_oi_ratio']:<6.2f} "
                      f"{data['distance_points']:.0f}pts ({data['distance_pct']:.1f}%) "
                      f"{data['avg_dte']:<6.0f}")
        
        # Identify the primary floor levels
        if significant_floors:
            print(f"\n🎯 PRIMARY FLOOR LEVELS:")
            
            # Top 3 by OI
            top_floors = significant_floors[:3]
            
            for i, (strike, data) in enumerate(top_floors, 1):
                strength = "STRONG" if data['oi'] > 50000 else "MODERATE" if data['oi'] > 20000 else "WEAK"
                positioning = "NEW" if data['vol_oi_ratio'] > 0.3 else "ACCUMULATED"
                
                print(f"  {i}. ${strike} - {strength} FLOOR")
                print(f"     • OI: {data['oi']:,.0f}")
                print(f"     • Distance: {data['distance_points']:.0f} points ({data['distance_pct']:.1f}%) below SPY")
                print(f"     • Positioning: {positioning} (V/OI: {data['vol_oi_ratio']:.2f})")
                print(f"     • Avg DTE: {data['avg_dte']:.0f} days")
        
        # Compare to July 2024 pattern
        print(f"\n📊 COMPARISON TO JULY 2024 PATTERN:")
        print(f"July 2024:")
        print(f"  • SPY: $555.28, Floor: $520 (35 points, 6.3% below)")
        print(f"  • Floor OI: ~189,000 combined")
        print(f"  • Accuracy: 99.5% (SPY bottomed at $517)")
        
        if significant_floors:
            primary_floor = significant_floors[0]
            primary_strike, primary_data = primary_floor
            
            print(f"Current:")
            print(f"  • SPY: ${spy_price:.2f}, Primary Floor: ${primary_strike} "
                  f"({primary_data['distance_points']:.0f} points, {primary_data['distance_pct']:.1f}% below)")
            print(f"  • Primary Floor OI: {primary_data['oi']:,.0f}")
            
            # Strength assessment
            if primary_data['oi'] > 150000:
                strength_assessment = "VERY STRONG - Similar to July 2024"
            elif primary_data['oi'] > 75000:
                strength_assessment = "STRONG - Significant institutional support"
            elif primary_data['oi'] > 30000:
                strength_assessment = "MODERATE - Some institutional support"
            else:
                strength_assessment = "WEAK - Limited institutional support"
            
            print(f"  • Strength: {strength_assessment}")
        
        # Trend analysis with multiple recent files
        if len(recent_files) > 1:
            print(f"\n📈 RECENT TREND ANALYSIS:")
            print("Analyzing last few days of data...")
            
            trend_data = []
            for file in recent_files[:5]:  # Last 5 files
                try:
                    file_date = file.stem.split('_')[-1]
                    df_temp = pd.read_parquet(file)
                    spy_temp = df_temp['underlying_price'].iloc[0]
                    
                    # Check primary floor strike
                    if significant_floors:
                        primary_strike = significant_floors[0][0]
                        primary_puts = df_temp[(df_temp['strike'] == primary_strike) & (df_temp['option_type'] == 'P')]
                        primary_oi = primary_puts['oi_proxy'].sum() if len(primary_puts) > 0 else 0
                        
                        trend_data.append({
                            'date': f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:8]}",
                            'spy': spy_temp,
                            'floor_oi': primary_oi
                        })
                except:
                    continue
            
            if trend_data:
                print(f"{'Date':<12} {'SPY':<8} {'Floor OI':<12} {'OI Change':<12}")
                print("-" * 50)
                
                prev_oi = None
                for data in reversed(trend_data):  # Show chronologically
                    oi_change = ""
                    if prev_oi:
                        change = data['floor_oi'] - prev_oi
                        oi_change = f"{change:+,.0f}"
                    
                    print(f"{data['date']:<12} ${data['spy']:<7.2f} {data['floor_oi']:<12,.0f} {oi_change:<12}")
                    prev_oi = data['floor_oi']
        
        # Final assessment
        print(f"\n🎯 CURRENT FLOOR ASSESSMENT:")
        if significant_floors:
            primary_strike, primary_data = significant_floors[0]
            
            # Risk assessment
            if primary_data['distance_pct'] > 10:
                risk_level = "LOW RISK - Floor far below current price"
            elif primary_data['distance_pct'] > 5:
                risk_level = "MODERATE RISK - Floor 5-10% below"
            else:
                risk_level = "HIGH RISK - Floor very close to current price"
            
            print(f"  • Primary Floor: ${primary_strike}")
            print(f"  • Current Risk Level: {risk_level}")
            print(f"  • If pattern holds, expect support around ${primary_strike}")
            
            # Confidence level
            if primary_data['oi'] > 100000 and primary_data['vol_oi_ratio'] < 0.5:
                confidence = "HIGH CONFIDENCE - Strong accumulated positions"
            elif primary_data['oi'] > 50000:
                confidence = "MODERATE CONFIDENCE - Decent institutional support"
            else:
                confidence = "LOW CONFIDENCE - Limited floor strength"
            
            print(f"  • Confidence Level: {confidence}")
        
    except Exception as e:
        print(f"❌ Error analyzing current data: {e}")

if __name__ == "__main__":
    analyze_current_put_floor()