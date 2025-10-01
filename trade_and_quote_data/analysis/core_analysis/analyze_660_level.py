#!/usr/bin/env python3
"""
Analyze $660 Level
==================

Deep dive into the $660 level to understand what type of positioning
this represents compared to the $600 rolling floor.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_660_level():
    """Analyze the $660 level positioning and characteristics"""
    
    print(f"🔍 $660 LEVEL DEEP ANALYSIS")
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
        print(f"📏 Distance to $660: {spy_price - 660:.1f} points ({(spy_price - 660)/spy_price*100:.1f}%)")
        
        # Analyze $660 puts
        puts_660 = df[(df['strike'] == 660) & (df['option_type'] == 'P')]
        calls_660 = df[(df['strike'] == 660) & (df['option_type'] == 'C')]
        
        print(f"\n🎯 $660 PUT ANALYSIS:")
        print("-" * 30)
        
        if len(puts_660) > 0:
            put_oi = puts_660['oi_proxy'].sum()
            put_volume = puts_660['volume'].sum()
            put_vol_oi = put_volume / (put_oi + 1e-6)
            
            print(f"  • Put OI: {put_oi:,.0f}")
            print(f"  • Put Volume: {put_volume:,.0f}")
            print(f"  • Put V/OI: {put_vol_oi:.2f}")
            
            # Distance analysis
            distance = spy_price - 660
            print(f"  • Distance from SPY: {distance:.1f} points ({distance/spy_price*100:.1f}%)")
            
            # Intrinsic value (puts are OTM)
            intrinsic = max(0, 660 - spy_price)
            print(f"  • Intrinsic value: ${intrinsic:.2f}")
            
            # Expiration structure
            print(f"\n📅 $660 PUT EXPIRATION STRUCTURE:")
            
            # Show key expirations
            exp_summary = puts_660.groupby('exp_date').agg({
                'oi_proxy': 'sum',
                'volume': 'sum', 
                'dte': 'first'
            }).sort_values('oi_proxy', ascending=False)
            
            print(f"{'Expiration':<12} {'DTE':<5} {'OI':<12} {'Volume':<12} {'V/OI':<6}")
            print("-" * 55)
            
            for exp_date, data in exp_summary.head(8).iterrows():
                exp_str = f"{exp_date[:4]}-{exp_date[4:6]}-{exp_date[6:8]}"
                dte = int(data['dte'])
                oi = data['oi_proxy'] 
                volume = data['volume']
                vol_oi = volume / (oi + 1e-6)
                
                print(f"{exp_str:<12} {dte:<5} {oi:<12,.0f} {volume:<12,.0f} {vol_oi:<6.2f}")
            
            # Critical: Check for 0 DTE activity
            today_puts = puts_660[puts_660['dte'] == 0]
            tomorrow_puts = puts_660[puts_660['dte'] == 1]
            
            if len(today_puts) > 0:
                today_oi = today_puts['oi_proxy'].sum()
                today_vol = today_puts['volume'].sum()
                print(f"\n⚡ 0 DTE (TODAY) ACTIVITY:")
                print(f"  • OI: {today_oi:,.0f}")
                print(f"  • Volume: {today_vol:,.0f}")
                print(f"  • V/OI: {today_vol/(today_oi+1e-6):.2f}")
            
            if len(tomorrow_puts) > 0:
                tomorrow_oi = tomorrow_puts['oi_proxy'].sum()
                tomorrow_vol = tomorrow_puts['volume'].sum()
                print(f"\n⚡ 1 DTE (TOMORROW) ACTIVITY:")
                print(f"  • OI: {tomorrow_oi:,.0f}")
                print(f"  • Volume: {tomorrow_vol:,.0f}")
                print(f"  • V/OI: {tomorrow_vol/(tomorrow_oi+1e-6):.2f}")
        
        # Analyze $660 calls  
        print(f"\n🚀 $660 CALL ANALYSIS:")
        print("-" * 30)
        
        if len(calls_660) > 0:
            call_oi = calls_660['oi_proxy'].sum()
            call_volume = calls_660['volume'].sum()
            call_vol_oi = call_volume / (call_oi + 1e-6)
            
            print(f"  • Call OI: {call_oi:,.0f}")
            print(f"  • Call Volume: {call_volume:,.0f}")
            print(f"  • Call V/OI: {call_vol_oi:.2f}")
            
            # Put/Call ratio
            if put_oi > 0:
                pc_ratio = put_oi / call_oi
                print(f"  • Put/Call ratio: {pc_ratio:.2f}")
            
            # Check for 0 DTE call activity
            today_calls = calls_660[calls_660['dte'] == 0]
            if len(today_calls) > 0:
                today_call_oi = today_calls['oi_proxy'].sum()
                today_call_vol = today_calls['volume'].sum()
                print(f"\n⚡ 0 DTE CALL ACTIVITY:")
                print(f"  • Call OI: {today_call_oi:,.0f}")
                print(f"  • Call Volume: {today_call_vol:,.0f}")
        
        # Compare to $600 level
        puts_600 = df[(df['strike'] == 600) & (df['option_type'] == 'P')]
        
        print(f"\n📊 $660 vs $600 COMPARISON:")
        print("-" * 35)
        
        if len(puts_600) > 0:
            put_600_oi = puts_600['oi_proxy'].sum()
            put_600_vol = puts_600['volume'].sum()
            put_600_vol_oi = put_600_vol / (put_600_oi + 1e-6)
            
            print(f"{'Level':<6} {'Put_OI':<12} {'Put_Vol':<12} {'V/OI':<6} {'Distance':<10} {'Type':<15}")
            print("-" * 70)
            print(f"$600   {put_600_oi:<12,.0f} {put_600_vol:<12,.0f} {put_600_vol_oi:<6.2f} {spy_price-600:<10.0f} Strategic Floor")
            print(f"$660   {put_oi:<12,.0f} {put_volume:<12,.0f} {put_vol_oi:<6.2f} {spy_price-660:<10.0f} Tactical Defense")
        
        # Characterize $660 level
        print(f"\n💡 $660 LEVEL CHARACTERISTICS:")
        print("-" * 35)
        
        distance_pct = (spy_price - 660) / spy_price * 100
        
        if distance_pct < 2:
            proximity = "IMMEDIATE PROXIMITY"
        elif distance_pct < 5:
            proximity = "VERY CLOSE"
        else:
            proximity = "MODERATE DISTANCE"
        
        print(f"  • Proximity: {proximity} ({distance_pct:.1f}% below SPY)")
        
        if put_vol_oi > 3:
            activity_level = "PANIC BUYING"
        elif put_vol_oi > 1:
            activity_level = "HEAVY ACTIVITY"
        elif put_vol_oi > 0.5:
            activity_level = "ACTIVE POSITIONING"
        else:
            activity_level = "ACCUMULATED POSITIONS"
        
        print(f"  • Activity level: {activity_level} (V/OI: {put_vol_oi:.2f})")
        
        # Time horizon analysis
        if len(puts_660) > 0:
            avg_dte = puts_660['dte'].mean()
            
            if avg_dte < 30:
                time_horizon = "SHORT-TERM DEFENSE"
            elif avg_dte < 90:
                time_horizon = "MEDIUM-TERM HEDGE"
            else:
                time_horizon = "LONG-TERM PROTECTION"
            
            print(f"  • Time horizon: {time_horizon} (avg {avg_dte:.0f} DTE)")
        
        # Strategic interpretation
        print(f"\n🎯 STRATEGIC INTERPRETATION:")
        print("-" * 30)
        
        if put_vol_oi > 3 and distance_pct < 2:
            interpretation = "BATTLE LINE - Institutions defending current levels with heavy put buying"
        elif put_vol_oi > 1 and distance_pct < 5:
            interpretation = "TACTICAL SUPPORT - Active defense of nearby support level"
        elif call_vol_oi > 0.5 and distance_pct < 3:
            interpretation = "BOUNCE EXPECTATION - Betting on support holding with calls"
        else:
            interpretation = "STANDARD HEDGE - Normal risk management positioning"
        
        print(f"  • {interpretation}")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        print("-" * 20)
        
        if put_vol_oi > 5:
            risk_level = "EXTREME RISK - Institutions in full defensive mode"
        elif put_vol_oi > 2:
            risk_level = "HIGH RISK - Heavy protective buying"
        elif put_vol_oi > 1:
            risk_level = "ELEVATED RISK - Active positioning"
        else:
            risk_level = "MODERATE RISK - Normal hedging activity"
        
        print(f"  • {risk_level}")
        
        # Compare to historical patterns
        print(f"\n📈 PATTERN COMPARISON:")
        print("-" * 25)
        
        print(f"  • July 2024 $520 floor: V/OI ~0.3, Built over weeks, Held perfectly")
        print(f"  • Current $600 floor: V/OI ~0.3, Built over 1.5 years, Strategic level")
        print(f"  • Current $660 level: V/OI ~{put_vol_oi:.1f}, Recent panic build, Tactical defense")
        
        if put_vol_oi > 3:
            print(f"\n🚨 CONCLUSION: $660 shows PANIC CHARACTERISTICS")
            print(f"  • Similar to July 2024 immediate pre-crash activity")
            print(f"  • High V/OI suggests institutions scrambling for protection")
            print(f"  • Very close to current price = limited cushion")
            print(f"  • This is 'last line of defense' before major correction")
        else:
            print(f"\n✅ CONCLUSION: $660 shows CONTROLLED HEDGING")
            print(f"  • Professional risk management")
            print(f"  • Reasonable proximity defense")
            print(f"  • Confidence in near-term support")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_660_level()