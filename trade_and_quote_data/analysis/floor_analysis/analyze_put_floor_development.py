#!/usr/bin/env python3
"""
Analyze Put Floor Development
============================

This tracks when the $500-520 put floor support level first developed
by analyzing put open interest buildup at those specific strikes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_put_floor_development():
    """Track the development of put floor support levels"""
    
    print(f"🔍 PUT FLOOR DEVELOPMENT ANALYSIS")
    print("=" * 60)
    
    # Extended date range to see when floor first appeared
    dates = [
        '20240708', '20240709', '20240710', '20240711', '20240712',  # Pre-decline
        '20240715', '20240716', '20240717', '20240718', '20240719',  # Start of decline
        '20240722', '20240723', '20240724', '20240725', '20240726',  # Mid decline
        '20240729', '20240730', '20240731', '20240801', '20240802',  # Late decline
        '20240805'   # Bottom day
    ]
    
    # Track key put strikes around the eventual floor
    floor_strikes = [500, 505, 510, 515, 520, 525, 530]
    
    results = []
    
    for date_str in dates:
        print(f"\n📅 Analyzing {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}...")
        
        try:
            # Load data
            year = date_str[:4]
            month = date_str[4:6]
            file_path = Path(f"data/options_chains/SPY/{year}/{month}/SPY_options_snapshot_{date_str}.parquet")
            
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                continue
                
            df = pd.read_parquet(file_path)
            spy_price = df['underlying_price'].iloc[0]
            
            # Analyze put floor strikes
            floor_data = {}
            for strike in floor_strikes:
                strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P')]
                
                if len(strike_puts) > 0:
                    total_oi = strike_puts['oi_proxy'].sum()
                    total_volume = strike_puts['volume'].sum()
                    total_contracts = len(strike_puts)
                    
                    floor_data[strike] = {
                        'oi': total_oi,
                        'volume': total_volume,
                        'contracts': total_contracts
                    }
                else:
                    floor_data[strike] = {
                        'oi': 0,
                        'volume': 0,
                        'contracts': 0
                    }
            
            # Calculate distance to floor
            distance_to_520 = spy_price - 520
            distance_to_510 = spy_price - 510
            distance_to_500 = spy_price - 500
            
            result = {
                'date': date_str,
                'date_formatted': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                'spy_price': spy_price,
                'distance_to_520': distance_to_520,
                'distance_to_510': distance_to_510,
                'distance_to_500': distance_to_500,
                'floor_data': floor_data
            }
            
            results.append(result)
            
            # Show key metrics
            total_520_oi = floor_data[520]['oi']
            total_510_oi = floor_data[510]['oi']
            total_500_oi = floor_data[500]['oi']
            
            print(f"  ✅ SPY: ${spy_price:.2f}")
            print(f"      $520P OI: {total_520_oi:,.0f}, $510P OI: {total_510_oi:,.0f}, $500P OI: {total_500_oi:,.0f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if results:
        print(f"\n{'='*100}")
        print(f"  PUT FLOOR DEVELOPMENT TIMELINE")
        print(f"{'='*100}")
        
        # Track when significant OI first appeared
        print(f"{'Date':<12} {'SPY':<8} {'Dist520':<8} {'520P_OI':<10} {'510P_OI':<10} {'500P_OI':<10} {'Total_Floor':<12}")
        print("-" * 100)
        
        floor_oi_threshold = 1000  # Significant OI threshold
        first_floor_date = None
        
        for result in results:
            floor_520_oi = result['floor_data'][520]['oi']
            floor_510_oi = result['floor_data'][510]['oi']
            floor_500_oi = result['floor_data'][500]['oi']
            total_floor_oi = floor_520_oi + floor_510_oi + floor_500_oi
            
            # Track when floor first appeared
            if first_floor_date is None and total_floor_oi > floor_oi_threshold * 3:
                first_floor_date = result['date_formatted']
            
            print(f"{result['date_formatted']:<12} "
                  f"${result['spy_price']:<7.2f} "
                  f"{result['distance_to_520']:<8.0f} "
                  f"{floor_520_oi:<10,.0f} "
                  f"{floor_510_oi:<10,.0f} "
                  f"{floor_500_oi:<10,.0f} "
                  f"{total_floor_oi:<12,.0f}")
        
        # Analysis of floor development
        print(f"\n🏗️ FLOOR DEVELOPMENT ANALYSIS:")
        
        if first_floor_date:
            print(f"  • First significant floor OI appeared: {first_floor_date}")
        
        # Track OI buildup over time
        print(f"\n📈 OI BUILDUP TIMELINE:")
        
        prev_total = 0
        for i, result in enumerate(results):
            total_floor_oi = sum([result['floor_data'][strike]['oi'] for strike in [500, 510, 520]])
            
            if i > 0:
                oi_change = total_floor_oi - prev_total
                if abs(oi_change) > 500:  # Significant change
                    change_str = f"({oi_change:+,.0f})"
                    print(f"    {result['date_formatted']}: {total_floor_oi:,.0f} {change_str}")
            
            prev_total = total_floor_oi
        
        # Find when each strike became significant
        print(f"\n🎯 STRIKE-BY-STRIKE DEVELOPMENT:")
        
        for strike in [520, 510, 500]:
            significant_date = None
            peak_oi = 0
            peak_date = None
            
            for result in results:
                strike_oi = result['floor_data'][strike]['oi']
                
                if significant_date is None and strike_oi > floor_oi_threshold:
                    significant_date = result['date_formatted']
                
                if strike_oi > peak_oi:
                    peak_oi = strike_oi
                    peak_date = result['date_formatted']
            
            if significant_date:
                print(f"  • ${strike}P: First significant OI on {significant_date}, Peak {peak_oi:,.0f} on {peak_date}")
        
        # Distance analysis when floor was being built
        print(f"\n📏 DISTANCE ANALYSIS:")
        
        for result in results:
            if result['date_formatted'] == first_floor_date:
                print(f"  • When floor first appeared ({first_floor_date}):")
                print(f"    - SPY was ${result['spy_price']:.2f}")
                print(f"    - Distance to $520: {result['distance_to_520']:.0f} points")
                print(f"    - Distance to $510: {result['distance_to_510']:.0f} points")
                print(f"    - Distance to $500: {result['distance_to_500']:.0f} points")
                break
        
        # Volume vs OI analysis (accumulation vs new positioning)
        print(f"\n📊 ACCUMULATION vs NEW POSITIONING:")
        
        for i, result in enumerate(results[-5:], len(results)-5):  # Last 5 days
            date = result['date_formatted']
            
            for strike in [520, 510, 500]:
                oi = result['floor_data'][strike]['oi']
                volume = result['floor_data'][strike]['volume']
                
                if oi > 500:  # Only show significant levels
                    vol_oi_ratio = volume / (oi + 1e-6)
                    positioning_type = "NEW" if vol_oi_ratio > 0.2 else "ACCUMULATED"
                    
                    print(f"    {date} ${strike}P: OI {oi:,.0f}, Vol {volume:,.0f} "
                          f"(V/OI: {vol_oi_ratio:.2f}) - {positioning_type}")
        
        # Final insights
        bottom_result = results[-1]  # August 5
        bottom_spy = bottom_result['spy_price']
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"  • SPY bottomed at ${bottom_spy:.2f}")
        print(f"  • Closest floor strike: $520 ({520 - bottom_spy:.0f} points above bottom)")
        print(f"  • Floor accuracy: {(520 - bottom_spy) / bottom_spy * 100:.1f}% above actual bottom")
        
        if first_floor_date:
            first_result = next(r for r in results if r['date_formatted'] == first_floor_date)
            days_advance = len([r for r in results if r['date_formatted'] >= first_floor_date and r['date_formatted'] <= bottom_result['date_formatted']]) - 1
            print(f"  • Floor identified {days_advance} trading days before bottom")
            print(f"  • SPY was ${first_result['spy_price']:.2f} when floor identified")
            print(f"  • That was {first_result['spy_price'] - bottom_spy:.0f} points above the eventual bottom")

if __name__ == "__main__":
    analyze_put_floor_development()