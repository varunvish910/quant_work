#!/usr/bin/env python3
"""
Analyze September Positioning Buildup
====================================

Track put floor building and call upside positioning during September,
excluding 0 DTE to focus on actual institutional positioning.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_september_positioning():
    """Analyze September positioning buildup for puts and calls"""
    
    print(f"🔍 SEPTEMBER POSITIONING BUILDUP ANALYSIS")
    print("=" * 50)
    
    # Find September 2025 files
    data_dir = Path("data/options_chains/SPY/2025/09")
    
    if not data_dir.exists():
        print(f"❌ September 2025 data directory not found")
        return
    
    # Get all September files
    september_files = sorted(data_dir.glob("SPY_options_snapshot_*.parquet"))
    
    if not september_files:
        print(f"❌ No September files found")
        return
    
    print(f"📊 Found {len(september_files)} September files")
    
    # Track positioning by strike
    put_strikes = list(range(600, 685, 5))  # 600-680 puts
    call_strikes = list(range(670, 725, 5))  # 670-720 calls
    
    put_data = {strike: [] for strike in put_strikes}
    call_data = {strike: [] for strike in call_strikes}
    
    # Process each September day
    for file_path in september_files:
        date_str = file_path.stem.split('_')[-1]
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        try:
            df = pd.read_parquet(file_path)
            spy_price = df['underlying_price'].iloc[0]
            
            # Analyze puts (exclude 0 DTE)
            for strike in put_strikes:
                strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P') & (df['dte'] > 0)]
                
                if len(strike_puts) > 0:
                    put_oi = strike_puts['oi_proxy'].sum()
                    put_volume = strike_puts['volume'].sum()
                    put_vol_oi = put_volume / (put_oi + 1e-6)
                    
                    put_data[strike].append({
                        'date': formatted_date,
                        'spy_price': spy_price,
                        'oi': put_oi,
                        'volume': put_volume,
                        'vol_oi': put_vol_oi,
                        'distance': spy_price - strike
                    })
            
            # Analyze calls (exclude 0 DTE)
            for strike in call_strikes:
                strike_calls = df[(df['strike'] == strike) & (df['option_type'] == 'C') & (df['dte'] > 0)]
                
                if len(strike_calls) > 0:
                    call_oi = strike_calls['oi_proxy'].sum()
                    call_volume = strike_calls['volume'].sum()
                    call_vol_oi = call_volume / (call_oi + 1e-6)
                    
                    call_data[strike].append({
                        'date': formatted_date,
                        'spy_price': spy_price,
                        'oi': call_oi,
                        'volume': call_volume,
                        'vol_oi': call_vol_oi,
                        'distance': strike - spy_price
                    })
        
        except Exception as e:
            continue
    
    # Analyze put floor building
    print(f"\n📉 PUT FLOOR BUILDING ANALYSIS:")
    print("-" * 40)
    
    put_summary = {}
    for strike in put_strikes:
        if not put_data[strike]:
            continue
        
        data_points = put_data[strike]
        total_volume = sum(d['volume'] for d in data_points)
        avg_vol_oi = np.mean([d['vol_oi'] for d in data_points])
        max_vol_oi = max(d['vol_oi'] for d in data_points)
        
        # OI change
        start_oi = data_points[0]['oi'] if data_points else 0
        end_oi = data_points[-1]['oi'] if data_points else 0
        oi_change = end_oi - start_oi
        
        avg_distance = np.mean([d['distance'] for d in data_points])
        
        put_summary[strike] = {
            'total_volume': total_volume,
            'avg_vol_oi': avg_vol_oi,
            'max_vol_oi': max_vol_oi,
            'oi_change': oi_change,
            'avg_distance': avg_distance
        }
    
    # Sort puts by total volume (most active)
    most_active_puts = sorted(put_summary.items(), key=lambda x: x[1]['total_volume'], reverse=True)[:8]
    
    print(f"{'Strike':<8} {'Total_Vol':<12} {'Avg_V/OI':<9} {'Max_V/OI':<9} {'OI_Chg':<10} {'Avg_Dist':<8}")
    print("-" * 65)
    
    for strike, data in most_active_puts:
        if data['total_volume'] > 1000:
            print(f"${strike:<7} {data['total_volume']:<12,.0f} "
                  f"{data['avg_vol_oi']:<9.2f} {data['max_vol_oi']:<9.2f} "
                  f"{data['oi_change']:<10,.0f} {data['avg_distance']:<8.0f}")
    
    # Analyze call upside building
    print(f"\n📈 CALL UPSIDE BUILDING ANALYSIS:")
    print("-" * 40)
    
    call_summary = {}
    for strike in call_strikes:
        if not call_data[strike]:
            continue
        
        data_points = call_data[strike]
        total_volume = sum(d['volume'] for d in data_points)
        avg_vol_oi = np.mean([d['vol_oi'] for d in data_points])
        max_vol_oi = max(d['vol_oi'] for d in data_points)
        
        # OI change
        start_oi = data_points[0]['oi'] if data_points else 0
        end_oi = data_points[-1]['oi'] if data_points else 0
        oi_change = end_oi - start_oi
        
        avg_distance = np.mean([d['distance'] for d in data_points])
        
        call_summary[strike] = {
            'total_volume': total_volume,
            'avg_vol_oi': avg_vol_oi,
            'max_vol_oi': max_vol_oi,
            'oi_change': oi_change,
            'avg_distance': avg_distance
        }
    
    # Sort calls by total volume (most active)
    most_active_calls = sorted(call_summary.items(), key=lambda x: x[1]['total_volume'], reverse=True)[:8]
    
    print(f"{'Strike':<8} {'Total_Vol':<12} {'Avg_V/OI':<9} {'Max_V/OI':<9} {'OI_Chg':<10} {'Avg_Dist':<8}")
    print("-" * 65)
    
    for strike, data in most_active_calls:
        if data['total_volume'] > 1000:
            print(f"${strike:<7} {data['total_volume']:<12,.0f} "
                  f"{data['avg_vol_oi']:<9.2f} {data['max_vol_oi']:<9.2f} "
                  f"{data['oi_change']:<10,.0f} {data['avg_distance']:<8.0f}")
    
    # Compare put vs call activity
    print(f"\n⚖️ PUT vs CALL POSITIONING COMPARISON:")
    print("-" * 45)
    
    total_put_volume = sum(s[1]['total_volume'] for s in most_active_puts)
    total_call_volume = sum(s[1]['total_volume'] for s in most_active_calls)
    
    avg_put_vol_oi = np.mean([s[1]['avg_vol_oi'] for s in most_active_puts[:5]])
    avg_call_vol_oi = np.mean([s[1]['avg_vol_oi'] for s in most_active_calls[:5]])
    
    print(f"  • Total put volume: {total_put_volume:,.0f}")
    print(f"  • Total call volume: {total_call_volume:,.0f}")
    print(f"  • Volume ratio (P/C): {total_put_volume/(total_call_volume+1e-6):.2f}")
    print(f"  • Avg put V/OI: {avg_put_vol_oi:.2f}")
    print(f"  • Avg call V/OI: {avg_call_vol_oi:.2f}")
    
    # Identify primary positioning
    print(f"\n🎯 PRIMARY POSITIONING ACTIVITY:")
    print("-" * 35)
    
    if most_active_puts:
        top_put = most_active_puts[0]
        print(f"  • Most active put floor: ${top_put[0]}")
        print(f"    - Volume: {top_put[1]['total_volume']:,.0f}")
        print(f"    - Avg distance: {top_put[1]['avg_distance']:.0f} points below SPY")
        print(f"    - Activity level: {'HIGH' if top_put[1]['avg_vol_oi'] > 1 else 'MODERATE'}")
    
    if most_active_calls:
        top_call = most_active_calls[0]
        print(f"  • Most active call target: ${top_call[0]}")
        print(f"    - Volume: {top_call[1]['total_volume']:,.0f}")
        print(f"    - Avg distance: {top_call[1]['avg_distance']:.0f} points above SPY")
        print(f"    - Activity level: {'HIGH' if top_call[1]['avg_vol_oi'] > 1 else 'MODERATE'}")
    
    # Distance-based analysis
    print(f"\n📏 DISTANCE-BASED ACTIVITY:")
    print("-" * 30)
    
    # Group puts by distance
    immediate_puts = [s for s in most_active_puts if s[1]['avg_distance'] <= 15]
    near_puts = [s for s in most_active_puts if 15 < s[1]['avg_distance'] <= 35]
    deep_puts = [s for s in most_active_puts if s[1]['avg_distance'] > 35]
    
    # Group calls by distance  
    near_calls = [s for s in most_active_calls if s[1]['avg_distance'] <= 15]
    medium_calls = [s for s in most_active_calls if 15 < s[1]['avg_distance'] <= 35]
    far_calls = [s for s in most_active_calls if s[1]['avg_distance'] > 35]
    
    print(f"PUT ACTIVITY:")
    if immediate_puts:
        imm_vol = sum(s[1]['total_volume'] for s in immediate_puts)
        print(f"  • Immediate defense (≤15 pts): {imm_vol:,.0f} volume")
    if near_puts:
        near_vol = sum(s[1]['total_volume'] for s in near_puts)
        print(f"  • Near defense (15-35 pts): {near_vol:,.0f} volume")
    if deep_puts:
        deep_vol = sum(s[1]['total_volume'] for s in deep_puts)
        print(f"  • Deep defense (>35 pts): {deep_vol:,.0f} volume")
    
    print(f"CALL ACTIVITY:")
    if near_calls:
        near_call_vol = sum(s[1]['total_volume'] for s in near_calls)
        print(f"  • Near upside (≤15 pts): {near_call_vol:,.0f} volume")
    if medium_calls:
        med_call_vol = sum(s[1]['total_volume'] for s in medium_calls)
        print(f"  • Medium upside (15-35 pts): {med_call_vol:,.0f} volume")
    if far_calls:
        far_call_vol = sum(s[1]['total_volume'] for s in far_calls)
        print(f"  • Far upside (>35 pts): {far_call_vol:,.0f} volume")
    
    # Final assessment
    print(f"\n💡 SEPTEMBER POSITIONING ASSESSMENT:")
    print("-" * 40)
    
    if total_put_volume > total_call_volume * 1.5:
        bias = "DEFENSIVE - Heavy put floor building"
    elif total_call_volume > total_put_volume * 1.5:
        bias = "BULLISH - Heavy call upside building"
    else:
        bias = "BALANCED - Mixed positioning"
    
    print(f"  • Overall bias: {bias}")
    
    if most_active_puts and most_active_calls:
        put_intensity = most_active_puts[0][1]['avg_vol_oi']
        call_intensity = most_active_calls[0][1]['avg_vol_oi']
        
        if put_intensity > 2:
            put_urgency = "PANIC BUILDING"
        elif put_intensity > 1:
            put_urgency = "URGENT BUILDING"
        else:
            put_urgency = "STEADY BUILDING"
        
        if call_intensity > 2:
            call_urgency = "AGGRESSIVE BUYING"
        elif call_intensity > 1:
            call_urgency = "ACTIVE BUYING"
        else:
            call_urgency = "STEADY BUYING"
        
        print(f"  • Put floor urgency: {put_urgency}")
        print(f"  • Call upside urgency: {call_urgency}")

if __name__ == "__main__":
    analyze_september_positioning()