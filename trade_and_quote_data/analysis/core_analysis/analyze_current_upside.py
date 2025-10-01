#!/usr/bin/env python3
"""
Analyze Current Upside Call Positioning
======================================

Check if there are new call positions being built on the upside
that could indicate institutional bullish positioning.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_current_upside():
    """Analyze current call positioning on the upside"""
    
    print(f"🔍 CURRENT UPSIDE CALL POSITIONING ANALYSIS")
    print("=" * 60)
    
    # Find most recent data
    data_dir = Path("data/options_chains/SPY")
    recent_files = []
    for year_dir in sorted(data_dir.glob("202*"), reverse=True):
        for month_dir in sorted(year_dir.glob("*"), reverse=True):
            for file in sorted(month_dir.glob("SPY_options_snapshot_*.parquet"), reverse=True):
                recent_files.append(file)
                if len(recent_files) >= 5:
                    break
            if len(recent_files) >= 5:
                break
        if len(recent_files) >= 5:
            break
    
    if not recent_files:
        print("❌ No recent data files found")
        return
    
    # Analyze latest file
    latest_file = recent_files[0]
    date_str = latest_file.stem.split('_')[-1]
    
    print(f"📅 Latest data: {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    
    try:
        df = pd.read_parquet(latest_file)
        spy_price = df['underlying_price'].iloc[0]
        
        print(f"💰 Current SPY: ${spy_price:.2f}")
        
        # Define upside strikes (5-20% above current price)
        upside_range_low = spy_price * 1.05   # 5% above
        upside_range_high = spy_price * 1.20  # 20% above
        
        # Round to nearest $5 strikes
        upside_strikes = []
        current_strike = int(upside_range_low / 5) * 5
        while current_strike <= upside_range_high:
            upside_strikes.append(current_strike)
            current_strike += 5
        
        print(f"\n🎯 ANALYZING UPSIDE CALL STRIKES:")
        print(f"Range: ${upside_range_low:.0f} - ${upside_range_high:.0f}")
        print(f"Strikes: {upside_strikes}")
        
        # Analyze call OI at upside levels
        upside_analysis = {}
        
        for strike in upside_strikes:
            strike_calls = df[(df['strike'] == strike) & (df['option_type'] == 'C')]
            
            if len(strike_calls) > 0:
                total_oi = strike_calls['oi_proxy'].sum()
                total_volume = strike_calls['volume'].sum()
                total_contracts = len(strike_calls)
                avg_dte = strike_calls['dte'].mean() if len(strike_calls) > 0 else 0
                
                distance_points = strike - spy_price
                distance_pct = distance_points / spy_price * 100
                
                upside_analysis[strike] = {
                    'oi': total_oi,
                    'volume': total_volume,
                    'contracts': total_contracts,
                    'avg_dte': avg_dte,
                    'distance_points': distance_points,
                    'distance_pct': distance_pct,
                    'vol_oi_ratio': total_volume / (total_oi + 1e-6)
                }
        
        # Sort by OI to find strongest levels
        sorted_upside = sorted(upside_analysis.items(), key=lambda x: x[1]['oi'], reverse=True)
        
        print(f"\n📈 UPSIDE CALL ANALYSIS (Sorted by OI):")
        print(f"{'Strike':<8} {'OI':<12} {'Volume':<10} {'V/OI':<6} {'Distance':<12} {'DTE':<6}")
        print("-" * 70)
        
        significant_upside = []
        for strike, data in sorted_upside:
            if data['oi'] > 1000:  # Only show significant levels
                significant_upside.append((strike, data))
                
                print(f"${strike:<7} {data['oi']:<12,.0f} {data['volume']:<10,.0f} "
                      f"{data['vol_oi_ratio']:<6.2f} "
                      f"+{data['distance_points']:.0f}pts (+{data['distance_pct']:.1f}%) "
                      f"{data['avg_dte']:<6.0f}")
        
        # Identify major upside targets
        if significant_upside:
            print(f"\n🚀 MAJOR UPSIDE TARGETS:")
            
            top_targets = significant_upside[:5]  # Top 5
            
            for i, (strike, data) in enumerate(top_targets, 1):
                strength = "VERY STRONG" if data['oi'] > 50000 else "STRONG" if data['oi'] > 20000 else "MODERATE"
                positioning = "NEW" if data['vol_oi_ratio'] > 0.3 else "ACCUMULATED"
                
                print(f"  {i}. ${strike} - {strength} TARGET")
                print(f"     • OI: {data['oi']:,.0f}")
                print(f"     • Distance: +{data['distance_points']:.0f} points (+{data['distance_pct']:.1f}%)")
                print(f"     • Positioning: {positioning} (V/OI: {data['vol_oi_ratio']:.2f})")
                print(f"     • Avg DTE: {data['avg_dte']:.0f} days")
        
        # Compare call vs put positioning
        print(f"\n⚖️ CALL vs PUT POSITIONING:")
        
        # Get put floor data for comparison
        floor_range_low = spy_price * 0.85
        floor_range_high = spy_price * 0.95
        
        total_put_oi = 0
        put_strikes = []
        current_strike = int(floor_range_low / 5) * 5
        while current_strike <= floor_range_high:
            put_strikes.append(current_strike)
            strike_puts = df[(df['strike'] == current_strike) & (df['option_type'] == 'P')]
            if len(strike_puts) > 0:
                total_put_oi += strike_puts['oi_proxy'].sum()
            current_strike += 5
        
        total_call_oi = sum([data['oi'] for _, data in upside_analysis.items()])
        
        print(f"  • Total upside call OI: {total_call_oi:,.0f}")
        print(f"  • Total downside put OI: {total_put_oi:,.0f}")
        print(f"  • Call/Put ratio: {total_call_oi / (total_put_oi + 1e-6):.2f}")
        
        if total_call_oi > total_put_oi:
            bias = "BULLISH - More upside positioning"
        elif total_put_oi > total_call_oi * 1.5:
            bias = "BEARISH - Heavy downside protection"
        else:
            bias = "NEUTRAL - Balanced positioning"
        
        print(f"  • Market bias: {bias}")
        
        # Recent trend analysis
        if len(recent_files) > 1:
            print(f"\n📈 RECENT UPSIDE TREND:")
            
            if significant_upside:
                primary_target = significant_upside[0][0]  # Strongest target
                
                print(f"Tracking ${primary_target} calls over recent days:")
                print(f"{'Date':<12} {'SPY':<8} {'Call_OI':<12} {'Volume':<10} {'V/OI':<6}")
                print("-" * 55)
                
                for file in reversed(recent_files):  # Chronological order
                    try:
                        file_date = file.stem.split('_')[-1]
                        df_temp = pd.read_parquet(file)
                        spy_temp = df_temp['underlying_price'].iloc[0]
                        
                        target_calls = df_temp[(df_temp['strike'] == primary_target) & (df_temp['option_type'] == 'C')]
                        if len(target_calls) > 0:
                            call_oi = target_calls['oi_proxy'].sum()
                            call_vol = target_calls['volume'].sum()
                            vol_oi = call_vol / (call_oi + 1e-6)
                            
                            print(f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:8]:<4} "
                                  f"${spy_temp:<7.2f} "
                                  f"{call_oi:<12,.0f} "
                                  f"{call_vol:<10,.0f} "
                                  f"{vol_oi:<6.2f}")
                    except:
                        continue
        
        # Gamma squeeze potential
        print(f"\n⚡ GAMMA SQUEEZE POTENTIAL:")
        
        if significant_upside:
            # Find the closest significant call level
            closest_target = None
            min_distance = float('inf')
            
            for strike, data in significant_upside:
                if data['distance_points'] < min_distance and data['oi'] > 10000:
                    min_distance = data['distance_points']
                    closest_target = (strike, data)
            
            if closest_target:
                strike, data = closest_target
                gamma_risk = "HIGH" if data['distance_points'] < 20 else "MODERATE" if data['distance_points'] < 40 else "LOW"
                
                print(f"  • Closest major target: ${strike} (+{data['distance_points']:.0f} pts)")
                print(f"  • OI at target: {data['oi']:,.0f}")
                print(f"  • Gamma squeeze risk: {gamma_risk}")
                
                if data['vol_oi_ratio'] > 0.5:
                    print(f"  • WARNING: High V/OI ({data['vol_oi_ratio']:.2f}) suggests active positioning")
        
        # Final assessment
        print(f"\n🎯 UPSIDE POSITIONING ASSESSMENT:")
        
        if total_call_oi > 100000:
            upside_strength = "STRONG institutional upside positioning"
        elif total_call_oi > 50000:
            upside_strength = "MODERATE institutional upside positioning"
        else:
            upside_strength = "WEAK institutional upside positioning"
        
        print(f"  • {upside_strength}")
        
        if significant_upside:
            primary_target = significant_upside[0]
            print(f"  • Primary upside target: ${primary_target[0]} (+{primary_target[1]['distance_pct']:.1f}%)")
            
            if primary_target[1]['vol_oi_ratio'] > 0.3:
                print(f"  • NEW positioning detected - institutions building upside exposure")
            else:
                print(f"  • ACCUMULATED positioning - existing upside bets")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_current_upside()