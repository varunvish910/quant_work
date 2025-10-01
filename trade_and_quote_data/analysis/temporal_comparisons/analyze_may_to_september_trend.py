#!/usr/bin/env python3
"""
Analyze May to September 2025 Defensive Trend
=============================================

Track how put/call positioning has evolved from May through September 2025
to understand the character change and defensive buildup timeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_may_to_september_trend():
    """Analyze defensive positioning trend from May to September 2025"""
    
    print(f"🔍 MAY-SEPTEMBER 2025 DEFENSIVE TREND ANALYSIS")
    print("=" * 55)
    
    # Target months: May, June, July, August, September 2025
    target_months = ['05', '06', '07', '08', '09']
    monthly_data = {}
    
    for month in target_months:
        month_dir = Path(f"data/options_chains/SPY/2025/{month}")
        
        if not month_dir.exists():
            print(f"❌ {month} data not found")
            continue
        
        # Get all files for the month
        month_files = sorted(month_dir.glob("SPY_options_snapshot_*.parquet"))
        
        if not month_files:
            continue
        
        print(f"📊 Processing {month}/2025: {len(month_files)} files")
        
        # Sample files throughout the month (every few days)
        sample_interval = max(1, len(month_files) // 8)  # ~8 samples per month
        sampled_files = month_files[::sample_interval]
        
        month_results = []
        
        for file_path in sampled_files:
            date_str = file_path.stem.split('_')[-1]
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            try:
                df = pd.read_parquet(file_path)
                spy_price = df['underlying_price'].iloc[0]
                
                # Exclude 0 DTE to focus on positioning
                df_filtered = df[df['dte'] > 0]
                
                # Analyze different ranges
                immediate_range = (spy_price * 0.95, spy_price * 1.05)  # ±5%
                near_range = (spy_price * 0.90, spy_price * 1.10)       # ±10%
                
                results = {}
                
                for range_name, (low, high) in [('immediate', immediate_range), ('near', near_range)]:
                    range_data = df_filtered[(df_filtered['strike'] >= low) & 
                                           (df_filtered['strike'] <= high)]
                    
                    puts = range_data[range_data['option_type'] == 'P']
                    calls = range_data[range_data['option_type'] == 'C']
                    
                    put_oi = puts['oi_proxy'].sum()
                    call_oi = calls['oi_proxy'].sum()
                    put_volume = puts['volume'].sum()
                    call_volume = calls['volume'].sum()
                    
                    # Calculate ratios
                    pc_oi_ratio = put_oi / (call_oi + 1e-6)
                    pc_vol_ratio = put_volume / (call_volume + 1e-6)
                    
                    # Calculate activity levels
                    put_vol_oi = put_volume / (put_oi + 1e-6)
                    call_vol_oi = call_volume / (call_oi + 1e-6)
                    
                    results[range_name] = {
                        'put_oi': put_oi,
                        'call_oi': call_oi,
                        'put_volume': put_volume,
                        'call_volume': call_volume,
                        'pc_oi_ratio': pc_oi_ratio,
                        'pc_vol_ratio': pc_vol_ratio,
                        'put_vol_oi': put_vol_oi,
                        'call_vol_oi': call_vol_oi
                    }
                
                month_results.append({
                    'date': formatted_date,
                    'spy_price': spy_price,
                    'ranges': results
                })
                
            except Exception as e:
                continue
        
        if month_results:
            monthly_data[month] = month_results
    
    if not monthly_data:
        print("❌ No monthly data generated")
        return
    
    # Calculate monthly averages
    print(f"\n📊 MONTHLY DEFENSIVE POSITIONING TRENDS:")
    print("-" * 50)
    
    monthly_summary = {}
    
    for month, data in monthly_data.items():
        month_name = datetime.strptime(f"2025-{month}-01", "%Y-%m-%d").strftime("%B")
        
        # Calculate averages for the month
        immediate_pc_oi = np.mean([d['ranges']['immediate']['pc_oi_ratio'] for d in data])
        immediate_pc_vol = np.mean([d['ranges']['immediate']['pc_vol_ratio'] for d in data])
        immediate_put_activity = np.mean([d['ranges']['immediate']['put_vol_oi'] for d in data])
        
        near_pc_oi = np.mean([d['ranges']['near']['pc_oi_ratio'] for d in data])
        near_pc_vol = np.mean([d['ranges']['near']['pc_vol_ratio'] for d in data])
        near_put_activity = np.mean([d['ranges']['near']['put_vol_oi'] for d in data])
        
        avg_spy = np.mean([d['spy_price'] for d in data])
        
        monthly_summary[month] = {
            'month_name': month_name,
            'avg_spy': avg_spy,
            'immediate_pc_oi': immediate_pc_oi,
            'immediate_pc_vol': immediate_pc_vol,
            'immediate_put_activity': immediate_put_activity,
            'near_pc_oi': near_pc_oi,
            'near_pc_vol': near_pc_vol,
            'near_put_activity': near_put_activity,
            'sample_count': len(data)
        }
    
    # Display monthly progression
    print(f"{'Month':<10} {'SPY':<8} {'P/C_OI':<7} {'P/C_Vol':<8} {'Put_Act':<8} {'Trend':<15}")
    print("-" * 65)
    
    baseline_pc_vol = None
    
    for month in sorted(monthly_summary.keys()):
        data = monthly_summary[month]
        
        if baseline_pc_vol is None:
            baseline_pc_vol = data['immediate_pc_vol']
            trend = "BASELINE"
        else:
            change = (data['immediate_pc_vol'] - baseline_pc_vol) / baseline_pc_vol * 100
            if change > 15:
                trend = f"+{change:.1f}% DEFENSIVE"
            elif change > 5:
                trend = f"+{change:.1f}% BUILDING"
            elif change < -5:
                trend = f"{change:.1f}% REDUCING"
            else:
                trend = f"{change:.1f}% STABLE"
        
        print(f"{data['month_name']:<10} ${data['avg_spy']:<7.0f} "
              f"{data['immediate_pc_oi']:<7.2f} {data['immediate_pc_vol']:<8.2f} "
              f"{data['immediate_put_activity']:<8.2f} {trend:<15}")
    
    # Calculate overall trend
    print(f"\n📈 TREND ANALYSIS:")
    print("-" * 20)
    
    months_sorted = sorted(monthly_summary.keys())
    start_month = monthly_summary[months_sorted[0]]
    end_month = monthly_summary[months_sorted[-1]]
    
    pc_vol_change = (end_month['immediate_pc_vol'] - start_month['immediate_pc_vol']) / start_month['immediate_pc_vol'] * 100
    pc_oi_change = (end_month['immediate_pc_oi'] - start_month['immediate_pc_oi']) / start_month['immediate_pc_oi'] * 100
    activity_change = (end_month['immediate_put_activity'] - start_month['immediate_put_activity']) / start_month['immediate_put_activity'] * 100
    spy_change = (end_month['avg_spy'] - start_month['avg_spy']) / start_month['avg_spy'] * 100
    
    print(f"  • SPY change: {spy_change:+.1f}% (${start_month['avg_spy']:.0f} → ${end_month['avg_spy']:.0f})")
    print(f"  • P/C Volume ratio: {pc_vol_change:+.1f}% ({start_month['immediate_pc_vol']:.2f} → {end_month['immediate_pc_vol']:.2f})")
    print(f"  • P/C OI ratio: {pc_oi_change:+.1f}% ({start_month['immediate_pc_oi']:.2f} → {end_month['immediate_pc_oi']:.2f})")
    print(f"  • Put activity: {activity_change:+.1f}% ({start_month['immediate_put_activity']:.2f} → {end_month['immediate_put_activity']:.2f})")
    
    # Month-over-month changes
    print(f"\n📅 MONTH-OVER-MONTH CHANGES:")
    print("-" * 30)
    
    prev_data = None
    for month in sorted(monthly_summary.keys()):
        data = monthly_summary[month]
        
        if prev_data:
            mom_pc_vol = (data['immediate_pc_vol'] - prev_data['immediate_pc_vol']) / prev_data['immediate_pc_vol'] * 100
            mom_activity = (data['immediate_put_activity'] - prev_data['immediate_put_activity']) / prev_data['immediate_put_activity'] * 100
            mom_spy = (data['avg_spy'] - prev_data['avg_spy']) / prev_data['avg_spy'] * 100
            
            print(f"{prev_data['month_name']} → {data['month_name']}:")
            print(f"  SPY: {mom_spy:+.1f}%, P/C Vol: {mom_pc_vol:+.1f}%, Put Activity: {mom_activity:+.1f}%")
        
        prev_data = data
    
    # Identify key inflection points
    print(f"\n🔥 KEY INFLECTION POINTS:")
    print("-" * 30)
    
    max_pc_vol = max(monthly_summary.values(), key=lambda x: x['immediate_pc_vol'])
    max_activity = max(monthly_summary.values(), key=lambda x: x['immediate_put_activity'])
    
    print(f"  • Peak P/C Volume ratio: {max_pc_vol['immediate_pc_vol']:.2f} in {max_pc_vol['month_name']}")
    print(f"  • Peak Put activity: {max_activity['immediate_put_activity']:.2f} in {max_activity['month_name']}")
    
    # Correlation with SPY movement
    print(f"\n📊 SPY vs DEFENSIVE CORRELATION:")
    print("-" * 35)
    
    spy_prices = [data['avg_spy'] for data in monthly_summary.values()]
    pc_vol_ratios = [data['immediate_pc_vol'] for data in monthly_summary.values()]
    
    if len(spy_prices) > 3:
        correlation = np.corrcoef(spy_prices, pc_vol_ratios)[0,1]
        
        if correlation > 0.5:
            correlation_desc = "POSITIVE - More defensive as SPY rises"
        elif correlation < -0.5:
            correlation_desc = "NEGATIVE - Less defensive as SPY rises"
        else:
            correlation_desc = "WEAK - No clear pattern"
        
        print(f"  • Correlation: {correlation:.2f} ({correlation_desc})")
    
    # Final assessment
    print(f"\n💡 MAY-SEPTEMBER 2025 ASSESSMENT:")
    print("-" * 35)
    
    if pc_vol_change > 20:
        trend_assessment = "MAJOR DEFENSIVE BUILDUP"
    elif pc_vol_change > 10:
        trend_assessment = "SIGNIFICANT DEFENSIVE INCREASE"
    elif pc_vol_change > 5:
        trend_assessment = "MODERATE DEFENSIVE INCREASE"
    else:
        trend_assessment = "STABLE POSITIONING"
    
    print(f"  • Overall trend: {trend_assessment}")
    print(f"  • Duration: {len(monthly_summary)} months")
    print(f"  • Character change: {'YES' if pc_vol_change > 15 else 'MODERATE' if pc_vol_change > 10 else 'NO'}")
    
    if activity_change > 50:
        activity_assessment = "PANIC LEVELS"
    elif activity_change > 25:
        activity_assessment = "ELEVATED URGENCY"
    elif activity_change > 10:
        activity_assessment = "INCREASED ACTIVITY"
    else:
        activity_assessment = "NORMAL ACTIVITY"
    
    print(f"  • Activity urgency: {activity_assessment}")

if __name__ == "__main__":
    analyze_may_to_september_trend()