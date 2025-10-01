#!/usr/bin/env python3
"""
Analyze Pullback Probability When Defensive Positioning Builds
==============================================================

Examine historical patterns when defensive positioning builds systematically
to estimate probability and timing of market pullbacks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def analyze_pullback_probability():
    """Analyze historical pullback patterns following defensive buildups"""
    
    print(f"🔍 PULLBACK PROBABILITY ANALYSIS")
    print("=" * 40)
    
    # Get all available data files for historical analysis
    data_dir = Path("data/options_chains/SPY")
    all_files = []
    
    for year_dir in sorted(data_dir.glob("202*")):
        for month_dir in sorted(year_dir.glob("*")):
            month_files = sorted(month_dir.glob("SPY_options_snapshot_*.parquet"))
            if month_files:
                # Sample a few files from each month for efficiency
                sample_files = month_files[::max(1, len(month_files)//5)]
                all_files.extend(sample_files)
    
    if not all_files:
        print("❌ No data files found")
        return
    
    print(f"📊 Analyzing {len(all_files)} historical data points")
    
    # Track defensive positioning over time
    historical_data = []
    
    for file_path in all_files:
        date_str = file_path.stem.split('_')[-1]
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        try:
            df = pd.read_parquet(file_path)
            spy_price = df['underlying_price'].iloc[0]
            
            # Exclude 0 DTE
            df_filtered = df[df['dte'] > 0]
            
            # Calculate immediate range defensive positioning (±5%)
            immediate_range = (spy_price * 0.95, spy_price * 1.05)
            range_data = df_filtered[(df_filtered['strike'] >= immediate_range[0]) & 
                                   (df_filtered['strike'] <= immediate_range[1])]
            
            puts = range_data[range_data['option_type'] == 'P']
            calls = range_data[range_data['option_type'] == 'C']
            
            put_oi = puts['oi_proxy'].sum()
            call_oi = calls['oi_proxy'].sum()
            put_volume = puts['volume'].sum()
            call_volume = calls['volume'].sum()
            
            pc_vol_ratio = put_volume / (call_volume + 1e-6)
            pc_oi_ratio = put_oi / (call_oi + 1e-6)
            put_vol_oi = put_volume / (put_oi + 1e-6)
            
            historical_data.append({
                'date': formatted_date,
                'spy_price': spy_price,
                'pc_vol_ratio': pc_vol_ratio,
                'pc_oi_ratio': pc_oi_ratio,
                'put_vol_oi': put_vol_oi
            })
            
        except Exception as e:
            continue
    
    if len(historical_data) < 50:
        print("❌ Insufficient historical data")
        return
    
    # Sort by date
    historical_data.sort(key=lambda x: x['date'])
    
    print(f"✅ Processed {len(historical_data)} historical periods")
    
    # Identify defensive buildup periods
    print(f"\n🔍 IDENTIFYING DEFENSIVE BUILDUP PERIODS:")
    print("-" * 45)
    
    # Calculate rolling averages and identify buildups
    window = 10  # Look at 10-period moving averages
    buildup_periods = []
    
    for i in range(window, len(historical_data) - 30):  # Need 30 periods ahead to check outcomes
        # Calculate recent average vs baseline
        current_window = historical_data[i-window:i]
        baseline_window = historical_data[max(0, i-window*3):i-window*2] if i > window*3 else current_window
        
        current_avg_pc_vol = np.mean([d['pc_vol_ratio'] for d in current_window])
        baseline_avg_pc_vol = np.mean([d['pc_vol_ratio'] for d in baseline_window])
        
        current_avg_activity = np.mean([d['put_vol_oi'] for d in current_window])
        
        # Define "defensive buildup" criteria
        if len(baseline_window) > 5:
            pc_vol_increase = (current_avg_pc_vol - baseline_avg_pc_vol) / baseline_avg_pc_vol
            
            # Buildup criteria: 15%+ increase in P/C vol ratio + elevated activity
            if pc_vol_increase > 0.15 and current_avg_pc_vol > 1.1 and current_avg_activity > 0.6:
                
                current_spy = historical_data[i]['spy_price']
                current_date = historical_data[i]['date']
                
                # Look ahead 30 periods to see what happened
                future_spys = [historical_data[j]['spy_price'] for j in range(i+1, min(i+31, len(historical_data)))]
                
                if future_spys:
                    max_future_spy = max(future_spys)
                    min_future_spy = min(future_spys)
                    final_spy = future_spys[-1]  # 30 periods later
                    
                    # Calculate drawdown
                    max_drawdown = (current_spy - min_future_spy) / current_spy * 100
                    final_return = (final_spy - current_spy) / current_spy * 100
                    
                    # Check for various pullback thresholds
                    pullback_5pct = max_drawdown >= 5
                    pullback_10pct = max_drawdown >= 10
                    pullback_15pct = max_drawdown >= 15
                    
                    # Find days to peak drawdown
                    days_to_low = None
                    for j, spy_val in enumerate(future_spys):
                        if spy_val == min_future_spy:
                            days_to_low = j + 1
                            break
                    
                    buildup_periods.append({
                        'date': current_date,
                        'spy_price': current_spy,
                        'pc_vol_ratio': current_avg_pc_vol,
                        'pc_vol_increase': pc_vol_increase * 100,
                        'put_activity': current_avg_activity,
                        'max_drawdown': max_drawdown,
                        'final_return': final_return,
                        'pullback_5pct': pullback_5pct,
                        'pullback_10pct': pullback_10pct,
                        'pullback_15pct': pullback_15pct,
                        'days_to_low': days_to_low,
                        'min_future_spy': min_future_spy
                    })
    
    if not buildup_periods:
        print("❌ No defensive buildup periods identified with sufficient criteria")
        return
    
    print(f"✅ Identified {len(buildup_periods)} defensive buildup periods")
    
    # Calculate probabilities
    print(f"\n📊 PULLBACK PROBABILITIES:")
    print("-" * 30)
    
    total_periods = len(buildup_periods)
    pullbacks_5pct = sum(1 for p in buildup_periods if p['pullback_5pct'])
    pullbacks_10pct = sum(1 for p in buildup_periods if p['pullback_10pct'])
    pullbacks_15pct = sum(1 for p in buildup_periods if p['pullback_15pct'])
    
    prob_5pct = pullbacks_5pct / total_periods * 100
    prob_10pct = pullbacks_10pct / total_periods * 100
    prob_15pct = pullbacks_15pct / total_periods * 100
    
    print(f"  • 5%+ pullback: {pullbacks_5pct}/{total_periods} = {prob_5pct:.1f}%")
    print(f"  • 10%+ pullback: {pullbacks_10pct}/{total_periods} = {prob_10pct:.1f}%")
    print(f"  • 15%+ pullback: {pullbacks_15pct}/{total_periods} = {prob_15pct:.1f}%")
    
    # Timing analysis
    print(f"\n⏰ TIMING ANALYSIS:")
    print("-" * 20)
    
    if pullbacks_5pct > 0:
        significant_pullbacks = [p for p in buildup_periods if p['pullback_5pct']]
        avg_days_to_low = np.mean([p['days_to_low'] for p in significant_pullbacks if p['days_to_low']])
        max_drawdowns = [p['max_drawdown'] for p in significant_pullbacks]
        avg_max_drawdown = np.mean(max_drawdowns)
        
        print(f"  • Average days to low: {avg_days_to_low:.1f}")
        print(f"  • Average max drawdown: {avg_max_drawdown:.1f}%")
        print(f"  • Range of drawdowns: {min(max_drawdowns):.1f}% - {max(max_drawdowns):.1f}%")
    
    # Show historical examples
    print(f"\n📋 HISTORICAL EXAMPLES:")
    print("-" * 25)
    
    # Sort by max drawdown to show most significant examples
    examples = sorted(buildup_periods, key=lambda x: x['max_drawdown'], reverse=True)[:5]
    
    print(f"{'Date':<12} {'SPY':<8} {'P/C_Vol':<8} {'Increase':<9} {'Max_DD':<8} {'Days':<6}")
    print("-" * 60)
    
    for example in examples:
        print(f"{example['date']:<12} ${example['spy_price']:<7.0f} "
              f"{example['pc_vol_ratio']:<8.2f} {example['pc_vol_increase']:<9.1f}% "
              f"{example['max_drawdown']:<8.1f}% {example['days_to_low'] or 'N/A':<6}")
    
    # Current situation comparison
    print(f"\n🎯 CURRENT SITUATION ANALYSIS:")
    print("-" * 35)
    
    # Compare current levels to historical buildups
    current_pc_vol = 1.25  # From September analysis
    current_buildup = 17.9  # % increase from May
    
    similar_buildups = [p for p in buildup_periods if abs(p['pc_vol_increase'] - current_buildup) < 5]
    
    if similar_buildups:
        similar_prob_5pct = sum(1 for p in similar_buildups if p['pullback_5pct']) / len(similar_buildups) * 100
        similar_prob_10pct = sum(1 for p in similar_buildups if p['pullback_10pct']) / len(similar_buildups) * 100
        similar_avg_drawdown = np.mean([p['max_drawdown'] for p in similar_buildups])
        
        print(f"  • Current buildup: {current_buildup:.1f}% increase")
        print(f"  • Similar historical cases: {len(similar_buildups)}")
        print(f"  • Probability of 5%+ pullback: {similar_prob_5pct:.1f}%")
        print(f"  • Probability of 10%+ pullback: {similar_prob_10pct:.1f}%")
        print(f"  • Average drawdown in similar cases: {similar_avg_drawdown:.1f}%")
    
    # Risk assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    print("-" * 20)
    
    if prob_5pct > 70:
        risk_level = "HIGH RISK - Very likely pullback"
    elif prob_5pct > 50:
        risk_level = "ELEVATED RISK - Probable pullback"
    elif prob_5pct > 30:
        risk_level = "MODERATE RISK - Possible pullback"
    else:
        risk_level = "LOW RISK - Unlikely pullback"
    
    print(f"  • Risk level: {risk_level}")
    
    # Expected value calculation
    if pullbacks_5pct > 0:
        expected_drawdown = prob_5pct/100 * avg_max_drawdown
        print(f"  • Expected drawdown: {expected_drawdown:.1f}%")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 15)
    
    print(f"  • Defensive buildups are {prob_5pct:.0f}% likely to result in 5%+ pullbacks")
    print(f"  • When pullbacks occur, they average {avg_max_drawdown:.1f}% drawdown")
    print(f"  • Timing is typically {avg_days_to_low:.0f} periods from buildup peak")
    print(f"  • Current 17.9% buildup is {'significant' if current_buildup > 15 else 'moderate'}")
    
    if prob_5pct > 60:
        print(f"  • ⚠️ HIGH PROBABILITY scenario - prepare for correction")
    elif prob_5pct > 40:
        print(f"  • ⚡ ELEVATED PROBABILITY scenario - monitor closely")
    else:
        print(f"  • ✅ LOWER PROBABILITY scenario - normal positioning")

if __name__ == "__main__":
    analyze_pullback_probability()