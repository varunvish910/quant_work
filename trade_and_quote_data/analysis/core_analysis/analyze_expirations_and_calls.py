#!/usr/bin/env python3
"""
Analyze Expirations and Call Positioning
========================================

Examine the expiration dates for puts at $600 and $660 levels,
and analyze if there's corresponding call positioning building.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def analyze_expirations_and_calls():
    """Analyze expiration structure and call positioning"""
    
    print(f"🔍 EXPIRATION & CALL POSITIONING ANALYSIS")
    print("=" * 60)
    
    # Find most recent data
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
        
        # Analyze specific strikes
        key_strikes = [600, 660]
        
        for strike in key_strikes:
            print(f"\n🎯 ${strike} STRIKE ANALYSIS:")
            print("-" * 40)
            
            # Put analysis
            strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P')]
            
            if len(strike_puts) > 0:
                print(f"\n📉 PUT POSITIONING:")
                total_put_oi = strike_puts['oi_proxy'].sum()
                total_put_volume = strike_puts['volume'].sum()
                put_vol_oi = total_put_volume / (total_put_oi + 1e-6)
                
                print(f"  • Total Put OI: {total_put_oi:,.0f}")
                print(f"  • Total Put Volume: {total_put_volume:,.0f}")
                print(f"  • Put V/OI Ratio: {put_vol_oi:.2f}")
                
                # Expiration analysis for puts
                print(f"\n📅 PUT EXPIRATION BREAKDOWN:")
                exp_groups = strike_puts.groupby('exp_date').agg({
                    'oi_proxy': 'sum',
                    'volume': 'sum',
                    'dte': 'first'
                }).sort_values('dte')
                
                # Show top expiration dates by OI
                top_exps = exp_groups.nlargest(10, 'oi_proxy')
                
                print(f"{'Expiration':<12} {'DTE':<5} {'Put_OI':<12} {'Put_Vol':<10} {'V/OI':<6}")
                print("-" * 55)
                
                for exp_date, data in top_exps.iterrows():
                    dte = int(data['dte'])
                    oi = data['oi_proxy']
                    volume = data['volume']
                    vol_oi = volume / (oi + 1e-6)
                    
                    # Format expiration date
                    exp_str = f"{exp_date[:4]}-{exp_date[4:6]}-{exp_date[6:8]}"
                    
                    print(f"{exp_str:<12} {dte:<5} {oi:<12,.0f} {volume:<10,.0f} {vol_oi:<6.2f}")
                
                # Categorize by time to expiration
                weekly_puts = exp_groups[exp_groups['dte'] <= 7]
                monthly_puts = exp_groups[(exp_groups['dte'] > 7) & (exp_groups['dte'] <= 60)]
                quarterly_puts = exp_groups[(exp_groups['dte'] > 60) & (exp_groups['dte'] <= 120)]
                long_term_puts = exp_groups[exp_groups['dte'] > 120]
                
                print(f"\n📊 PUT EXPIRATION CATEGORIES:")
                print(f"  • Weekly (≤7 days): {weekly_puts['oi_proxy'].sum():,.0f} OI ({weekly_puts['oi_proxy'].sum()/total_put_oi*100:.1f}%)")
                print(f"  • Monthly (8-60 days): {monthly_puts['oi_proxy'].sum():,.0f} OI ({monthly_puts['oi_proxy'].sum()/total_put_oi*100:.1f}%)")
                print(f"  • Quarterly (61-120 days): {quarterly_puts['oi_proxy'].sum():,.0f} OI ({quarterly_puts['oi_proxy'].sum()/total_put_oi*100:.1f}%)")
                print(f"  • Long-term (>120 days): {long_term_puts['oi_proxy'].sum():,.0f} OI ({long_term_puts['oi_proxy'].sum()/total_put_oi*100:.1f}%)")
            
            # Call analysis
            strike_calls = df[(df['strike'] == strike) & (df['option_type'] == 'C')]
            
            if len(strike_calls) > 0:
                print(f"\n📈 CALL POSITIONING:")
                total_call_oi = strike_calls['oi_proxy'].sum()
                total_call_volume = strike_calls['volume'].sum()
                call_vol_oi = total_call_volume / (total_call_oi + 1e-6)
                
                print(f"  • Total Call OI: {total_call_oi:,.0f}")
                print(f"  • Total Call Volume: {total_call_volume:,.0f}")
                print(f"  • Call V/OI Ratio: {call_vol_oi:.2f}")
                
                # Put/Call ratio
                if total_put_oi > 0:
                    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else float('inf')
                    print(f"  • Put/Call OI Ratio: {pc_ratio:.2f}")
                
                # Expiration analysis for calls
                print(f"\n📅 CALL EXPIRATION BREAKDOWN:")
                call_exp_groups = strike_calls.groupby('exp_date').agg({
                    'oi_proxy': 'sum',
                    'volume': 'sum',
                    'dte': 'first'
                }).sort_values('dte')
                
                # Show top call expiration dates by OI
                top_call_exps = call_exp_groups.nlargest(10, 'oi_proxy')
                
                if len(top_call_exps) > 0:
                    print(f"{'Expiration':<12} {'DTE':<5} {'Call_OI':<12} {'Call_Vol':<10} {'V/OI':<6}")
                    print("-" * 55)
                    
                    for exp_date, data in top_call_exps.iterrows():
                        dte = int(data['dte'])
                        oi = data['oi_proxy']
                        volume = data['volume']
                        vol_oi = volume / (oi + 1e-6)
                        
                        exp_str = f"{exp_date[:4]}-{exp_date[4:6]}-{exp_date[6:8]}"
                        print(f"{exp_str:<12} {dte:<5} {oi:<12,.0f} {volume:<10,.0f} {vol_oi:<6.2f}")
                
                # Categorize call expirations
                weekly_calls = call_exp_groups[call_exp_groups['dte'] <= 7]
                monthly_calls = call_exp_groups[(call_exp_groups['dte'] > 7) & (call_exp_groups['dte'] <= 60)]
                quarterly_calls = call_exp_groups[(call_exp_groups['dte'] > 60) & (call_exp_groups['dte'] <= 120)]
                long_term_calls = call_exp_groups[call_exp_groups['dte'] > 120]
                
                print(f"\n📊 CALL EXPIRATION CATEGORIES:")
                print(f"  • Weekly (≤7 days): {weekly_calls['oi_proxy'].sum():,.0f} OI ({weekly_calls['oi_proxy'].sum()/total_call_oi*100:.1f}%)")
                print(f"  • Monthly (8-60 days): {monthly_calls['oi_proxy'].sum():,.0f} OI ({monthly_calls['oi_proxy'].sum()/total_call_oi*100:.1f}%)")
                print(f"  • Quarterly (61-120 days): {quarterly_calls['oi_proxy'].sum():,.0f} OI ({quarterly_calls['oi_proxy'].sum()/total_call_oi*100:.1f}%)")
                print(f"  • Long-term (>120 days): {long_term_calls['oi_proxy'].sum():,.0f} OI ({long_term_calls['oi_proxy'].sum()/total_call_oi*100:.1f}%)")
        
        # Broader call positioning analysis
        print(f"\n🚀 BROADER CALL POSITIONING ANALYSIS:")
        print("=" * 50)
        
        # Look for call building in nearby strikes
        call_strikes_600 = [595, 600, 605, 610, 615]  # Around $600
        call_strikes_660 = [655, 660, 665, 670, 675]  # Around $660
        
        def analyze_call_cluster(strikes, zone_name):
            print(f"\n📈 {zone_name} CALL CLUSTER:")
            
            total_cluster_call_oi = 0
            total_cluster_call_vol = 0
            
            for strike in strikes:
                strike_calls = df[(df['strike'] == strike) & (df['option_type'] == 'C')]
                if len(strike_calls) > 0:
                    call_oi = strike_calls['oi_proxy'].sum()
                    call_vol = strike_calls['volume'].sum()
                    call_vol_oi = call_vol / (call_oi + 1e-6)
                    
                    total_cluster_call_oi += call_oi
                    total_cluster_call_vol += call_vol
                    
                    if call_oi > 1000:  # Only show significant levels
                        distance = strike - spy_price
                        print(f"  ${strike}: OI {call_oi:,.0f}, Vol {call_vol:,.0f}, V/OI {call_vol_oi:.2f} "
                              f"({distance:+.0f} pts, {distance/spy_price*100:+.1f}%)")
            
            cluster_vol_oi = total_cluster_call_vol / (total_cluster_call_oi + 1e-6)
            print(f"  Total: OI {total_cluster_call_oi:,.0f}, Vol {total_cluster_call_vol:,.0f}, V/OI {cluster_vol_oi:.2f}")
            
            return total_cluster_call_oi, cluster_vol_oi
        
        call_600_oi, call_600_activity = analyze_call_cluster(call_strikes_600, "$600 ZONE")
        call_660_oi, call_660_activity = analyze_call_cluster(call_strikes_660, "$660 ZONE")
        
        # Compare call vs put positioning
        print(f"\n⚖️ CALL vs PUT POSITIONING SUMMARY:")
        print("-" * 50)
        
        # Get put totals for comparison
        put_600_total = df[(df['strike'] == 600) & (df['option_type'] == 'P')]['oi_proxy'].sum()
        put_660_total = df[(df['strike'] == 660) & (df['option_type'] == 'P')]['oi_proxy'].sum()
        
        print(f"$600 Zone:")
        print(f"  • Put OI: {put_600_total:,.0f}")
        print(f"  • Call cluster OI: {call_600_oi:,.0f}")
        print(f"  • P/C ratio: {put_600_total/(call_600_oi+1e-6):.2f}")
        print(f"  • Call activity: {'HIGH' if call_600_activity > 0.3 else 'MODERATE' if call_600_activity > 0.1 else 'LOW'}")
        
        print(f"$660 Zone:")
        print(f"  • Put OI: {put_660_total:,.0f}")
        print(f"  • Call cluster OI: {call_660_oi:,.0f}")
        print(f"  • P/C ratio: {put_660_total/(call_660_oi+1e-6):.2f}")
        print(f"  • Call activity: {'HIGH' if call_660_activity > 0.3 else 'MODERATE' if call_660_activity > 0.1 else 'LOW'}")
        
        # Strategic implications
        print(f"\n💡 STRATEGIC IMPLICATIONS:")
        
        if call_600_oi > 30000 and call_600_activity > 0.3:
            print(f"  • $600 zone: Strong call building suggests institutions expect bounce from this level")
        elif call_600_oi > 15000:
            print(f"  • $600 zone: Moderate call interest suggests some bounce expectations")
        else:
            print(f"  • $600 zone: Limited call interest - primarily defensive put positioning")
        
        if call_660_oi > 30000 and call_660_activity > 0.3:
            print(f"  • $660 zone: Strong call building suggests institutions expect support here")
        elif call_660_oi > 15000:
            print(f"  • $660 zone: Moderate call interest suggests some support expectations")
        else:
            print(f"  • $660 zone: Limited call interest - primarily defensive put positioning")
        
        # Time decay analysis
        print(f"\n⏰ TIME DECAY IMPLICATIONS:")
        
        # Check for near-term expirations with high volume
        all_options = df.copy()
        near_term = all_options[all_options['dte'] <= 30]  # Next 30 days
        
        if len(near_term) > 0:
            near_term_activity = near_term.groupby(['strike', 'option_type']).agg({
                'oi_proxy': 'sum',
                'volume': 'sum'
            }).reset_index()
            
            high_activity = near_term_activity[near_term_activity['volume'] > 10000]
            
            if len(high_activity) > 0:
                print(f"  • High volume near-term positions detected:")
                for _, row in high_activity.head(5).iterrows():
                    vol_oi = row['volume'] / (row['oi_proxy'] + 1e-6)
                    print(f"    ${row['strike']} {row['option_type']}: OI {row['oi_proxy']:,.0f}, Vol {row['volume']:,.0f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_expirations_and_calls()