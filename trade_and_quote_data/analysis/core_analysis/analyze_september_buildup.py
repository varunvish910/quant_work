#!/usr/bin/env python3
"""
Analyze September Put Floor Buildup
==================================

Track which strikes had the most put positioning activity during September,
excluding 0 DTE to focus on actual floor building rather than day trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_september_buildup():
    """Analyze September put floor building activity"""
    
    print(f"🔍 SEPTEMBER PUT FLOOR BUILDUP ANALYSIS")
    print("=" * 50)
    
    # Find September 2025 files
    data_dir = Path("data/options_chains/SPY/2025/09")
    
    if not data_dir.exists():
        print(f"❌ September 2025 data directory not found: {data_dir}")
        return
    
    # Get all September files
    september_files = sorted(data_dir.glob("SPY_options_snapshot_*.parquet"))
    
    if not september_files:
        print(f"❌ No September files found in {data_dir}")
        return
    
    print(f"📊 Found {len(september_files)} September files")
    print(f"Date range: {september_files[0].stem.split('_')[-1]} to {september_files[-1].stem.split('_')[-1]}")
    
    # Track activity by strike across September
    put_activity = {}
    call_activity = {}
    daily_spy_prices = {}
    
    # Analyze key strikes (focus on 600-680 range for puts, 670-720 for calls)
    target_put_strikes = list(range(600, 685, 5))
    target_call_strikes = list(range(670, 725, 5))
    
    for strike in target_put_strikes:
        put_activity[strike] = []
    
    for strike in target_call_strikes:
        call_activity[strike] = []
    
    print(f"\n🎯 Tracking put strikes: {target_put_strikes}")
    print(f"🎯 Tracking call strikes: {target_call_strikes}")
    
    # Process each day
    for i, file_path in enumerate(september_files):
        date_str = file_path.stem.split('_')[-1]
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        try:
            df = pd.read_parquet(file_path)
            spy_price = df['underlying_price'].iloc[0]
            daily_spy_prices[formatted_date] = spy_price
            
            # Analyze each target strike
            for strike in target_strikes:
                # Exclude 0 DTE to focus on positioning buildup
                strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P') & (df['dte'] > 0)]
                
                if len(strike_puts) > 0:
                    put_oi = strike_puts['oi_proxy'].sum()
                    put_volume = strike_puts['volume'].sum()
                    put_vol_oi = put_volume / (put_oi + 1e-6)
                    avg_dte = strike_puts['dte'].mean()
                    
                    strike_activity[strike].append({
                        'date': formatted_date,
                        'spy_price': spy_price,
                        'put_oi': put_oi,
                        'put_volume': put_volume,
                        'vol_oi_ratio': put_vol_oi,
                        'avg_dte': avg_dte,
                        'distance': spy_price - strike
                    })
        
        except Exception as e:
            print(f"  ⚠️ Error processing {formatted_date}: {e}")
            continue
    
    # Calculate total activity and changes for each strike
    strike_summary = {}
    
    for strike in target_strikes:
        if not strike_activity[strike]:
            continue
        
        activity_data = strike_activity[strike]
        
        # Calculate totals
        total_volume = sum(d['put_volume'] for d in activity_data)
        avg_vol_oi = np.mean([d['vol_oi_ratio'] for d in activity_data])
        max_vol_oi = max(d['vol_oi_ratio'] for d in activity_data)
        
        # Calculate OI change from start to end
        start_oi = activity_data[0]['put_oi'] if activity_data else 0
        end_oi = activity_data[-1]['put_oi'] if activity_data else 0
        oi_change = end_oi - start_oi
        oi_change_pct = (oi_change / start_oi * 100) if start_oi > 0 else 0
        
        # Calculate average distance from SPY
        avg_distance = np.mean([d['distance'] for d in activity_data])
        
        strike_summary[strike] = {
            'total_volume': total_volume,
            'avg_vol_oi': avg_vol_oi,
            'max_vol_oi': max_vol_oi,
            'oi_change': oi_change,
            'oi_change_pct': oi_change_pct,
            'start_oi': start_oi,
            'end_oi': end_oi,
            'avg_distance': avg_distance,
            'days_active': len(activity_data)
        }
    
    # Find most active strikes
    print(f"\n🏗️ SEPTEMBER PUT FLOOR BUILDING ACTIVITY:")
    print("-" * 70)
    
    # Sort by total volume (most activity)
    most_active = sorted(strike_summary.items(), 
                        key=lambda x: x[1]['total_volume'], 
                        reverse=True)[:10]
    
    print(f"{'Strike':<8} {'Total_Vol':<12} {'Avg_V/OI':<9} {'Max_V/OI':<9} {'OI_Change':<11} {'Avg_Dist':<9}")
    print("-" * 70)
    
    for strike, data in most_active:
        if data['total_volume'] > 0:
            print(f"${strike:<7} {data['total_volume']:<12,.0f} "
                  f"{data['avg_vol_oi']:<9.2f} {data['max_vol_oi']:<9.2f} "
                  f"{data['oi_change']:<11,.0f} {data['avg_distance']:<9.0f}")
    
    # Identify where the most building occurred
    print(f"\n🎯 MOST ACTIVE FLOOR BUILDING:")
    print("-" * 35)
    
    if most_active:
        top_strike = most_active[0][0]
        top_data = most_active[0][1]
        
        print(f"  • Most active strike: ${top_strike}")
        print(f"  • Total volume: {top_data['total_volume']:,.0f}")
        print(f"  • Average V/OI: {top_data['avg_vol_oi']:.2f}")
        print(f"  • OI change: {top_data['oi_change']:+,.0f} ({top_data['oi_change_pct']:+.1f}%)")
        print(f"  • Average distance from SPY: {top_data['avg_distance']:.0f} points")
    
    # Analyze OI growth patterns
    print(f"\n📈 OI GROWTH PATTERNS:")
    print("-" * 25)
    
    # Sort by OI growth
    oi_growers = sorted(strike_summary.items(), 
                       key=lambda x: x[1]['oi_change'], 
                       reverse=True)[:5]
    
    print(f"Top OI builders:")
    for strike, data in oi_growers:
        if data['oi_change'] > 1000:
            print(f"  • ${strike}: {data['oi_change']:+,.0f} OI ({data['oi_change_pct']:+.1f}%)")
    
    # Analyze activity intensity
    print(f"\n⚡ ACTIVITY INTENSITY:")
    print("-" * 25)
    
    # Sort by max V/OI (most intense activity)
    most_intense = sorted(strike_summary.items(), 
                         key=lambda x: x[1]['max_vol_oi'], 
                         reverse=True)[:5]
    
    print(f"Highest intensity strikes:")
    for strike, data in most_intense:
        if data['max_vol_oi'] > 0.5:
            print(f"  • ${strike}: Peak V/OI {data['max_vol_oi']:.2f}")
    
    # Daily timeline analysis
    if most_active:
        top_strike = most_active[0][0]
        print(f"\n📅 ${top_strike} DAILY TIMELINE (Top Active Strike):")
        print("-" * 50)
        
        timeline_data = strike_activity[top_strike]
        
        print(f"{'Date':<12} {'SPY':<8} {'Put_OI':<10} {'Volume':<10} {'V/OI':<6} {'Distance':<8}")
        print("-" * 60)
        
        # Show every few days to avoid clutter
        step = max(1, len(timeline_data) // 10)
        for i in range(0, len(timeline_data), step):
            data = timeline_data[i]
            print(f"{data['date']:<12} ${data['spy_price']:<7.2f} "
                  f"{data['put_oi']:<10,.0f} {data['put_volume']:<10,.0f} "
                  f"{data['vol_oi_ratio']:<6.2f} {data['distance']:<8.0f}")
        
        # Show final day
        if len(timeline_data) > 1 and step > 1:
            final_data = timeline_data[-1]
            print(f"{final_data['date']:<12} ${final_data['spy_price']:<7.2f} "
                  f"{final_data['put_oi']:<10,.0f} {final_data['put_volume']:<10,.0f} "
                  f"{final_data['vol_oi_ratio']:<6.2f} {final_data['distance']:<8.0f}")
    
    # Distance analysis
    print(f"\n📏 DISTANCE-BASED ACTIVITY:")
    print("-" * 30)
    
    # Group by distance ranges
    immediate_activity = {}  # 0-10 pts
    near_activity = {}       # 10-25 pts
    medium_activity = {}     # 25-50 pts
    deep_activity = {}       # 50+ pts
    
    for strike, data in strike_summary.items():
        avg_dist = data['avg_distance']
        if 0 <= avg_dist <= 10:
            immediate_activity[strike] = data
        elif 10 < avg_dist <= 25:
            near_activity[strike] = data
        elif 25 < avg_dist <= 50:
            medium_activity[strike] = data
        else:
            deep_activity[strike] = data
    
    def print_distance_group(group_data, group_name):
        if not group_data:
            return
        
        total_vol = sum(d['total_volume'] for d in group_data.values())
        most_active_strike = max(group_data.items(), key=lambda x: x[1]['total_volume'])
        
        print(f"\n{group_name}:")
        print(f"  • Total volume: {total_vol:,.0f}")
        print(f"  • Most active: ${most_active_strike[0]} ({most_active_strike[1]['total_volume']:,.0f} vol)")
    
    print_distance_group(immediate_activity, "IMMEDIATE (0-10 pts)")
    print_distance_group(near_activity, "NEAR (10-25 pts)")
    print_distance_group(medium_activity, "MEDIUM (25-50 pts)")
    print_distance_group(deep_activity, "DEEP (50+ pts)")
    
    # Final summary
    print(f"\n🎯 SEPTEMBER BUILDUP SUMMARY:")
    print("-" * 30)
    
    if most_active:
        top_3_strikes = [s[0] for s in most_active[:3]]
        top_3_volumes = [s[1]['total_volume'] for s in most_active[:3]]
        
        print(f"  • Primary building at: ${', $'.join(map(str, top_3_strikes))}")
        print(f"  • Total volume: {sum(top_3_volumes):,.0f}")
        
        # Characterize the pattern
        avg_distance_top3 = np.mean([most_active[i][1]['avg_distance'] for i in range(3)])
        
        if avg_distance_top3 < 15:
            pattern_type = "DEFENSIVE - Building close to current levels"
        elif avg_distance_top3 < 35:
            pattern_type = "TACTICAL - Standard correction protection"
        else:
            pattern_type = "STRATEGIC - Deep crash protection"
        
        print(f"  • Pattern type: {pattern_type}")
        print(f"  • Average distance: {avg_distance_top3:.0f} points from SPY")

if __name__ == "__main__":
    analyze_september_buildup()