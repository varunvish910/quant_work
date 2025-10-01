#!/usr/bin/env python3
"""
Analyze Strikes Between $600-$660
=================================

Examine all strikes between the strategic $600 floor and 
the tactical $660 defense to understand the institutional
positioning landscape.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_between_floors():
    """Analyze all strikes between $600-660 floors"""
    
    print(f"🔍 STRIKES BETWEEN $600-$660 ANALYSIS")
    print("=" * 50)
    
    # Get recent data
    data_dir = Path("data/options_chains/SPY")
    recent_files = []
    for year_dir in sorted(data_dir.glob("202*"), reverse=True):
        for month_dir in sorted(year_dir.glob("*"), reverse=True):
            for file in sorted(month_dir.glob("SPY_options_snapshot_*.parquet"), reverse=True):
                recent_files.append(file)
                if len(recent_files) >= 1:
                    break
            if len(recent_files) >= 1:
                break
        if len(recent_files) >= 1:
            break
    
    if not recent_files:
        print("❌ No recent data files found")
        return
    
    latest_file = recent_files[0]
    date_str = latest_file.stem.split('_')[-1]
    
    print(f"📅 Analysis date: {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    
    try:
        df = pd.read_parquet(latest_file)
        spy_price = df['underlying_price'].iloc[0]
        
        print(f"💰 Current SPY: ${spy_price:.2f}")
        
        # Define the range between floors
        between_strikes = list(range(600, 665, 5))  # $600, $605, $610, ..., $660
        
        print(f"\n🎯 ANALYZING STRIKES: {between_strikes}")
        print(f"Range: ${between_strikes[0]} - ${between_strikes[-1]}")
        
        # Analyze each strike in detail
        strike_data = []
        
        for strike in between_strikes:
            # Exclude 0 DTE to focus on positioning buildup
            strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P') & (df['dte'] > 0)]
            strike_calls = df[(df['strike'] == strike) & (df['option_type'] == 'C') & (df['dte'] > 0)]
            
            if len(strike_puts) > 0:
                put_oi = strike_puts['oi_proxy'].sum()
                put_volume = strike_puts['volume'].sum()
                put_vol_oi = put_volume / (put_oi + 1e-6)
                put_avg_dte = strike_puts['dte'].mean()
            else:
                put_oi = put_volume = put_vol_oi = put_avg_dte = 0
            
            if len(strike_calls) > 0:
                call_oi = strike_calls['oi_proxy'].sum()
                call_volume = strike_calls['volume'].sum()
                call_vol_oi = call_volume / (call_oi + 1e-6)
                call_avg_dte = strike_calls['dte'].mean()
            else:
                call_oi = call_volume = call_vol_oi = call_avg_dte = 0
            
            # Calculate metrics
            distance = spy_price - strike
            distance_pct = distance / spy_price * 100
            pc_ratio = put_oi / (call_oi + 1e-6)
            total_oi = put_oi + call_oi
            total_volume = put_volume + call_volume
            
            strike_data.append({
                'strike': strike,
                'distance': distance,
                'distance_pct': distance_pct,
                'put_oi': put_oi,
                'call_oi': call_oi,
                'put_volume': put_volume,
                'call_volume': call_volume,
                'put_vol_oi': put_vol_oi,
                'call_vol_oi': call_vol_oi,
                'pc_ratio': pc_ratio,
                'total_oi': total_oi,
                'total_volume': total_volume,
                'put_avg_dte': put_avg_dte,
                'call_avg_dte': call_avg_dte
            })
        
        # Create summary table
        print(f"\n📊 COMPLETE STRIKE ANALYSIS:")
        print("-" * 90)
        print(f"{'Strike':<8} {'Dist':<6} {'Put_OI':<10} {'Call_OI':<10} {'P_V/OI':<7} {'C_V/OI':<7} {'P/C':<6} {'Level':<12}")
        print("-" * 90)
        
        for data in strike_data:
            if data['total_oi'] > 1000:  # Only show significant levels
                # Categorize the level
                if data['put_vol_oi'] > 3:
                    level_type = "PANIC PUTS"
                elif data['put_vol_oi'] > 1:
                    level_type = "ACTIVE PUTS"
                elif data['call_vol_oi'] > 1:
                    level_type = "ACTIVE CALLS"
                elif data['pc_ratio'] > 2:
                    level_type = "PUT WALL"
                elif data['pc_ratio'] < 0.5:
                    level_type = "CALL WALL"
                else:
                    level_type = "BALANCED"
                
                print(f"${data['strike']:<7} {data['distance']:<6.0f} "
                      f"{data['put_oi']:<10,.0f} {data['call_oi']:<10,.0f} "
                      f"{data['put_vol_oi']:<7.2f} {data['call_vol_oi']:<7.2f} "
                      f"{data['pc_ratio']:<6.2f} {level_type:<12}")
        
        # Identify key levels and patterns
        print(f"\n🔍 KEY LEVEL IDENTIFICATION:")
        print("-" * 35)
        
        # Sort by total OI to find strongest levels
        strong_levels = sorted(strike_data, key=lambda x: x['total_oi'], reverse=True)[:8]
        
        for i, level in enumerate(strong_levels, 1):
            if level['total_oi'] > 10000:
                # Characterize each level
                if level['distance'] > 40:
                    zone = "DEEP SUPPORT"
                elif level['distance'] > 20:
                    zone = "MEDIUM SUPPORT"  
                elif level['distance'] > 10:
                    zone = "NEAR SUPPORT"
                elif level['distance'] > 0:
                    zone = "IMMEDIATE SUPPORT"
                else:
                    zone = "RESISTANCE"
                
                activity = "HIGH" if max(level['put_vol_oi'], level['call_vol_oi']) > 1 else "MODERATE"
                
                print(f"  {i}. ${level['strike']} - {zone}")
                print(f"     • Total OI: {level['total_oi']:,.0f}")
                print(f"     • Distance: {level['distance']:.0f} points ({level['distance_pct']:.1f}%)")
                print(f"     • Activity: {activity}")
                print(f"     • P/C Ratio: {level['pc_ratio']:.2f}")
        
        # Find the defensive line pattern
        print(f"\n🛡️ DEFENSIVE LINE ANALYSIS:")
        print("-" * 30)
        
        # Group strikes by distance ranges
        immediate_defense = [d for d in strike_data if 0 <= d['distance'] <= 10]
        near_defense = [d for d in strike_data if 10 < d['distance'] <= 25]
        medium_defense = [d for d in strike_data if 25 < d['distance'] <= 45]
        deep_defense = [d for d in strike_data if d['distance'] > 45]
        
        def analyze_defense_zone(zone_data, zone_name):
            if not zone_data:
                return
            
            total_put_oi = sum(d['put_oi'] for d in zone_data)
            total_call_oi = sum(d['call_oi'] for d in zone_data)
            avg_put_vol_oi = np.mean([d['put_vol_oi'] for d in zone_data if d['put_oi'] > 1000])
            avg_call_vol_oi = np.mean([d['call_vol_oi'] for d in zone_data if d['call_oi'] > 1000])
            
            strikes_in_zone = [f"${d['strike']}" for d in zone_data if d['total_oi'] > 5000]
            
            print(f"\n{zone_name}:")
            print(f"  • Strikes: {', '.join(strikes_in_zone)}")
            print(f"  • Total Put OI: {total_put_oi:,.0f}")
            print(f"  • Total Call OI: {total_call_oi:,.0f}")
            if not np.isnan(avg_put_vol_oi):
                print(f"  • Avg Put V/OI: {avg_put_vol_oi:.2f}")
            if not np.isnan(avg_call_vol_oi):
                print(f"  • Avg Call V/OI: {avg_call_vol_oi:.2f}")
        
        analyze_defense_zone(immediate_defense, "IMMEDIATE DEFENSE (0-10 pts)")
        analyze_defense_zone(near_defense, "NEAR DEFENSE (10-25 pts)")
        analyze_defense_zone(medium_defense, "MEDIUM DEFENSE (25-45 pts)")
        analyze_defense_zone(deep_defense, "DEEP DEFENSE (45+ pts)")
        
        # Look for gaps in coverage
        print(f"\n🕳️ GAP ANALYSIS:")
        print("-" * 20)
        
        significant_strikes = [d for d in strike_data if d['total_oi'] > 20000]
        significant_strikes.sort(key=lambda x: x['strike'])
        
        gaps = []
        for i in range(len(significant_strikes) - 1):
            current_strike = significant_strikes[i]['strike']
            next_strike = significant_strikes[i + 1]['strike']
            gap_size = next_strike - current_strike
            
            if gap_size > 10:  # Gap larger than $10
                gap_midpoint = (current_strike + next_strike) / 2
                gap_distance = spy_price - gap_midpoint
                gaps.append({
                    'start': current_strike,
                    'end': next_strike,
                    'midpoint': gap_midpoint,
                    'distance': gap_distance,
                    'size': gap_size
                })
        
        if gaps:
            print(f"  • Coverage gaps detected:")
            for gap in gaps:
                print(f"    ${gap['start']} - ${gap['end']} (gap: ${gap['size']}, midpoint {gap['distance']:.0f} pts below SPY)")
        else:
            print(f"  • No significant gaps in institutional coverage")
        
        # Activity heat map
        print(f"\n🌡️ ACTIVITY HEAT MAP:")
        print("-" * 25)
        
        # Find most active strikes
        active_strikes = sorted(strike_data, key=lambda x: max(x['put_vol_oi'], x['call_vol_oi']), reverse=True)[:5]
        
        for i, strike in enumerate(active_strikes, 1):
            if strike['total_oi'] > 5000:
                max_activity = max(strike['put_vol_oi'], strike['call_vol_oi'])
                activity_type = "PUT" if strike['put_vol_oi'] > strike['call_vol_oi'] else "CALL"
                
                heat_level = "🔥🔥🔥" if max_activity > 3 else "🔥🔥" if max_activity > 1 else "🔥"
                
                print(f"  {i}. ${strike['strike']} {heat_level} - {activity_type} activity ({max_activity:.2f} V/OI)")
        
        # Strategic summary
        print(f"\n💡 STRATEGIC SUMMARY:")
        print("-" * 25)
        
        # Calculate total institutional positioning
        total_put_oi = sum(d['put_oi'] for d in strike_data)
        total_call_oi = sum(d['call_oi'] for d in strike_data)
        overall_pc_ratio = total_put_oi / (total_call_oi + 1e-6)
        
        # Find concentration points
        top_3_strikes = sorted(strike_data, key=lambda x: x['total_oi'], reverse=True)[:3]
        concentration_strikes = [f"${s['strike']}" for s in top_3_strikes]
        
        print(f"  • Total Put OI (600-660): {total_put_oi:,.0f}")
        print(f"  • Total Call OI (600-660): {total_call_oi:,.0f}")
        print(f"  • Overall P/C Ratio: {overall_pc_ratio:.2f}")
        print(f"  • Concentration at: {', '.join(concentration_strikes)}")
        
        # Institutional intent
        if overall_pc_ratio > 1.5:
            intent = "DEFENSIVE - Heavy put positioning suggests downside concern"
        elif overall_pc_ratio < 0.8:
            intent = "BULLISH - Heavy call positioning suggests upside expectations"
        else:
            intent = "BALANCED - Mixed positioning suggests uncertainty"
        
        print(f"  • Institutional intent: {intent}")
        
        # Final risk assessment
        highest_activity = max(d['put_vol_oi'] for d in strike_data)
        if highest_activity > 5:
            risk_assessment = "EXTREME RISK - Panic activity detected"
        elif highest_activity > 2:
            risk_assessment = "HIGH RISK - Elevated defensive activity"
        elif highest_activity > 1:
            risk_assessment = "MODERATE RISK - Active positioning"
        else:
            risk_assessment = "LOW RISK - Normal institutional activity"
        
        print(f"  • Risk assessment: {risk_assessment}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_between_floors()