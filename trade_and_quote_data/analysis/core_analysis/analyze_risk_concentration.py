#!/usr/bin/env python3
"""
Analyze Risk Concentration
==========================

Examine where pullback risk is most concentrated based on options positioning,
strike distribution, and institutional hedging patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_risk_concentration():
    """Analyze where pullback risk is concentrated"""
    
    print(f"🔍 PULLBACK RISK CONCENTRATION ANALYSIS")
    print("=" * 45)
    
    # Get latest data
    recent_file = Path("data/options_chains/SPY/2025/09/SPY_options_snapshot_20250930.parquet")
    
    if not recent_file.exists():
        print("❌ Recent data not found")
        return
    
    try:
        df = pd.read_parquet(recent_file)
        spy_price = df['underlying_price'].iloc[0]
        
        print(f"📅 Analysis date: 2025-09-30")
        print(f"💰 Current SPY: ${spy_price:.2f}")
        
        # Exclude 0 DTE to focus on positioning
        df_filtered = df[df['dte'] > 0]
        
        print(f"\n🎯 RISK CONCENTRATION BY STRIKE LEVELS:")
        print("-" * 45)
        
        # Define risk zones based on distance from current price
        zones = {
            'IMMEDIATE_RISK': (spy_price * 0.985, spy_price),           # 0-1.5% down
            'NEAR_RISK': (spy_price * 0.97, spy_price * 0.985),        # 1.5-3% down  
            'MODERATE_RISK': (spy_price * 0.94, spy_price * 0.97),     # 3-6% down
            'SIGNIFICANT_RISK': (spy_price * 0.90, spy_price * 0.94),  # 6-10% down
            'CRASH_RISK': (spy_price * 0.80, spy_price * 0.90),       # 10-20% down
            'EXTREME_RISK': (0, spy_price * 0.80)                      # >20% down
        }
        
        zone_analysis = {}
        
        for zone_name, (low, high) in zones.items():
            zone_data = df_filtered[(df_filtered['strike'] >= low) & 
                                   (df_filtered['strike'] <= high)]
            
            puts = zone_data[zone_data['option_type'] == 'P']
            calls = zone_data[zone_data['option_type'] == 'C']
            
            if len(puts) > 0:
                put_oi = puts['oi_proxy'].sum()
                put_volume = puts['volume'].sum()
                put_vol_oi = put_volume / (put_oi + 1e-6)
                avg_put_dte = puts['dte'].mean()
                
                # Calculate concentration metrics
                total_put_value = put_oi * (spy_price * 0.02)  # Rough premium estimate
                
                zone_analysis[zone_name] = {
                    'put_oi': put_oi,
                    'put_volume': put_volume,
                    'put_vol_oi': put_vol_oi,
                    'avg_dte': avg_put_dte,
                    'notional_value': total_put_value,
                    'strike_range': f"${low:.0f}-${high:.0f}",
                    'midpoint': (low + high) / 2,
                    'distance_pct': (spy_price - (low + high) / 2) / spy_price * 100
                }
        
        # Display zone analysis
        print(f"{'Zone':<18} {'Strike_Range':<15} {'Put_OI':<12} {'Volume':<10} {'V/OI':<6} {'NotionalB$':<10}")
        print("-" * 85)
        
        total_notional = 0
        for zone_name, data in zone_analysis.items():
            if data['put_oi'] > 1000:  # Only show significant zones
                notional_billions = data['notional_value'] / 1e9
                total_notional += data['notional_value']
                
                print(f"{zone_name:<18} {data['strike_range']:<15} "
                      f"{data['put_oi']:<12,.0f} {data['put_volume']:<10,.0f} "
                      f"{data['put_vol_oi']:<6.2f} ${notional_billions:<9.1f}")
        
        print(f"\nTotal estimated notional: ${total_notional/1e9:.1f}B")
        
        # Identify highest risk concentration
        print(f"\n🔥 HIGHEST RISK CONCENTRATION:")
        print("-" * 35)
        
        # Sort by volume (recent activity)
        active_zones = sorted([(k, v) for k, v in zone_analysis.items() if v['put_oi'] > 5000], 
                             key=lambda x: x[1]['put_volume'], reverse=True)
        
        if active_zones:
            top_zone = active_zones[0]
            zone_name, data = top_zone
            
            print(f"  • Highest activity zone: {zone_name}")
            print(f"  • Strike range: {data['strike_range']}")
            print(f"  • Distance: {data['distance_pct']:.1f}% below SPY")
            print(f"  • Put volume: {data['put_volume']:,.0f}")
            print(f"  • Activity level: {'PANIC' if data['put_vol_oi'] > 2 else 'HIGH' if data['put_vol_oi'] > 1 else 'MODERATE'}")
        
        # Analyze specific strike concentrations
        print(f"\n📍 TOP RISK CONCENTRATION STRIKES:")
        print("-" * 40)
        
        # Find strikes with highest put OI
        puts_only = df_filtered[df_filtered['option_type'] == 'P']
        strike_oi = puts_only.groupby('strike').agg({
            'oi_proxy': 'sum',
            'volume': 'sum'
        }).sort_values('oi_proxy', ascending=False)
        
        print(f"{'Strike':<8} {'Put_OI':<12} {'Volume':<10} {'Distance':<10} {'Risk_Level':<12}")
        print("-" * 65)
        
        for strike, data in strike_oi.head(10).iterrows():
            distance = spy_price - strike
            distance_pct = distance / spy_price * 100
            
            if distance_pct <= 1:
                risk_level = "IMMEDIATE"
            elif distance_pct <= 5:
                risk_level = "NEAR"
            elif distance_pct <= 10:
                risk_level = "MODERATE"
            elif distance_pct <= 20:
                risk_level = "SIGNIFICANT"
            else:
                risk_level = "EXTREME"
            
            print(f"${strike:<7} {data['oi_proxy']:<12,.0f} {data['volume']:<10,.0f} "
                  f"{distance_pct:<10.1f}% {risk_level:<12}")
        
        # Gamma exposure analysis
        print(f"\n⚡ GAMMA EXPOSURE CONCENTRATION:")
        print("-" * 35)
        
        # Analyze calls near current price for gamma squeeze potential
        near_calls = df_filtered[(df_filtered['option_type'] == 'C') & 
                                (df_filtered['strike'] >= spy_price * 0.98) &
                                (df_filtered['strike'] <= spy_price * 1.05)]
        
        if len(near_calls) > 0:
            call_gamma_exposure = near_calls.groupby('strike').agg({
                'oi_proxy': 'sum',
                'volume': 'sum'
            }).sort_values('oi_proxy', ascending=False)
            
            print(f"Near-the-money call concentrations:")
            for strike, data in call_gamma_exposure.head(5).iterrows():
                distance = strike - spy_price
                print(f"  • ${strike}: {data['oi_proxy']:,.0f} OI ({distance:+.0f} pts)")
        
        # Time decay risk analysis
        print(f"\n⏰ TIME DECAY RISK CONCENTRATION:")
        print("-" * 35)
        
        # Analyze by expiration timeframes
        dte_ranges = {
            'THIS_WEEK': (0, 7),
            'THIS_MONTH': (8, 30),
            'QUARTERLY': (31, 90),
            'LONG_TERM': (91, 999)
        }
        
        for range_name, (min_dte, max_dte) in dte_ranges.items():
            range_options = df_filtered[(df_filtered['dte'] >= min_dte) & 
                                       (df_filtered['dte'] <= max_dte)]
            
            range_puts = range_options[range_options['option_type'] == 'P']
            
            if len(range_puts) > 0:
                total_put_oi = range_puts['oi_proxy'].sum()
                total_put_volume = range_puts['volume'].sum()
                vol_oi_ratio = total_put_volume / (total_put_oi + 1e-6)
                
                print(f"  • {range_name}: {total_put_oi:,.0f} put OI, V/OI: {vol_oi_ratio:.2f}")
        
        # Institutional vs retail patterns
        print(f"\n🏛️ INSTITUTIONAL vs RETAIL PATTERNS:")
        print("-" * 40)
        
        # Large vs small trades (rough proxy)
        large_trades = df_filtered[df_filtered['volume'] > 100]  # Institutional proxy
        small_trades = df_filtered[df_filtered['volume'] <= 100]  # Retail proxy
        
        large_puts = large_trades[large_trades['option_type'] == 'P']
        small_puts = small_trades[small_trades['option_type'] == 'P']
        
        if len(large_puts) > 0 and len(small_puts) > 0:
            large_put_oi = large_puts['oi_proxy'].sum()
            small_put_oi = small_puts['oi_proxy'].sum()
            
            large_avg_strike = np.average(large_puts['strike'], weights=large_puts['oi_proxy'])
            small_avg_strike = np.average(small_puts['strike'], weights=small_puts['oi_proxy'])
            
            print(f"  • Large trades (institutional):")
            print(f"    - Put OI: {large_put_oi:,.0f}")
            print(f"    - Avg strike: ${large_avg_strike:.0f}")
            print(f"    - Distance: {(spy_price - large_avg_strike):.0f} pts")
            
            print(f"  • Small trades (retail):")
            print(f"    - Put OI: {small_put_oi:,.0f}")
            print(f"    - Avg strike: ${small_avg_strike:.0f}")
            print(f"    - Distance: {(spy_price - small_avg_strike):.0f} pts")
        
        # Volatility risk zones
        print(f"\n📊 VOLATILITY IMPACT ZONES:")
        print("-" * 30)
        
        # Calculate where most put OI is concentrated
        put_strikes = puts_only['strike'].values
        put_weights = puts_only['oi_proxy'].values
        
        weighted_avg_strike = np.average(put_strikes, weights=put_weights)
        weighted_distance = spy_price - weighted_avg_strike
        
        print(f"  • Volume-weighted avg put strike: ${weighted_avg_strike:.0f}")
        print(f"  • Distance from SPY: {weighted_distance:.0f} points ({weighted_distance/spy_price*100:.1f}%)")
        
        # Risk cascade analysis
        print(f"\n🌊 RISK CASCADE ANALYSIS:")
        print("-" * 25)
        
        # Identify potential cascade levels
        cascade_levels = []
        
        # Look for strikes with >20k OI that could trigger cascades
        significant_strikes = strike_oi[strike_oi['oi_proxy'] > 20000]
        
        for strike, data in significant_strikes.head(5).iterrows():
            distance_pct = (spy_price - strike) / spy_price * 100
            
            cascade_levels.append({
                'strike': strike,
                'distance_pct': distance_pct,
                'put_oi': data['oi_proxy'],
                'cascade_risk': 'HIGH' if distance_pct < 5 else 'MODERATE' if distance_pct < 10 else 'LOW'
            })
        
        print(f"Potential cascade trigger levels:")
        for level in cascade_levels:
            print(f"  • ${level['strike']:.0f}: {level['distance_pct']:.1f}% down, "
                  f"{level['put_oi']:,.0f} OI, {level['cascade_risk']} risk")
        
        # Final risk assessment
        print(f"\n🎯 RISK CONCENTRATION SUMMARY:")
        print("-" * 35)
        
        # Find the danger zone
        immediate_and_near = zone_analysis.get('IMMEDIATE_RISK', {}).get('put_oi', 0) + \
                           zone_analysis.get('NEAR_RISK', {}).get('put_oi', 0)
        
        moderate_risk = zone_analysis.get('MODERATE_RISK', {}).get('put_oi', 0)
        
        total_defensive_oi = sum(data.get('put_oi', 0) for data in zone_analysis.values())
        
        immediate_concentration = immediate_and_near / total_defensive_oi * 100 if total_defensive_oi > 0 else 0
        
        print(f"  • Immediate risk (0-3% down): {immediate_concentration:.1f}% of defensive OI")
        print(f"  • Primary danger zone: ${weighted_avg_strike:.0f} ({weighted_distance/spy_price*100:.1f}% down)")
        print(f"  • Cascade trigger risk: {'HIGH' if any(l['cascade_risk'] == 'HIGH' for l in cascade_levels) else 'MODERATE'}")
        
        if immediate_concentration > 40:
            risk_concentration = "EXTREME - Most risk in immediate proximity"
        elif immediate_concentration > 25:
            risk_concentration = "HIGH - Significant near-term risk"
        else:
            risk_concentration = "MODERATE - Risk spread across levels"
        
        print(f"  • Overall concentration: {risk_concentration}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_risk_concentration()