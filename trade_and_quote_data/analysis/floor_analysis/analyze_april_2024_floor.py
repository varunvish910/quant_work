#!/usr/bin/env python3
"""
Analyze April 2024 Put Floor Pattern
===================================

Check if the same institutional put floor building pattern that occurred
before July 2024 also happened before April 2024 market movements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_april_2024_pattern():
    """Analyze April 2024 for similar put floor patterns"""
    
    print(f"🔍 APRIL 2024 PUT FLOOR ANALYSIS")
    print("=" * 60)
    
    # April 2024 date range - look for pattern development
    dates = [
        # Pre-April buildup
        '20240325', '20240326', '20240327', '20240328',  # Late March
        # April dates
        '20240401', '20240402', '20240403', '20240404', '20240405',  # Early April
        '20240408', '20240409', '20240410', '20240411', '20240412',  # Mid April
        '20240415', '20240416', '20240417', '20240418', '20240419',  # Late April
        '20240422', '20240423', '20240424', '20240425', '20240426',  # End April
        '20240429', '20240430',  # April end
        # Early May to see the outcome
        '20240501', '20240502', '20240503'
    ]
    
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
            
            # Define potential floor strikes based on SPY price (similar methodology)
            floor_range_low = spy_price * 0.85  # 15% below
            floor_range_high = spy_price * 0.95  # 5% below
            
            # Round to nearest $5 strikes
            potential_floors = []
            current_strike = int(floor_range_low / 5) * 5
            while current_strike <= floor_range_high:
                potential_floors.append(current_strike)
                current_strike += 5
            
            # Analyze put OI at potential floor levels
            floor_data = {}
            total_floor_oi = 0
            
            for strike in potential_floors:
                strike_puts = df[(df['strike'] == strike) & (df['option_type'] == 'P')]
                
                if len(strike_puts) > 0:
                    total_oi = strike_puts['oi_proxy'].sum()
                    total_volume = strike_puts['volume'].sum()
                    
                    floor_data[strike] = {
                        'oi': total_oi,
                        'volume': total_volume,
                        'vol_oi_ratio': total_volume / (total_oi + 1e-6)
                    }
                    total_floor_oi += total_oi
                else:
                    floor_data[strike] = {'oi': 0, 'volume': 0, 'vol_oi_ratio': 0}
            
            # Find the strongest floor level
            strongest_floor = 0
            strongest_oi = 0
            for strike, data in floor_data.items():
                if data['oi'] > strongest_oi:
                    strongest_oi = data['oi']
                    strongest_floor = strike
            
            # Process features for anomaly detection
            fe = OptionsFeatureEngine()
            df_processed = fe.calculate_oi_features(df)
            df_processed = fe.calculate_volume_features(df_processed)
            df_processed = fe.calculate_price_features(df_processed)
            df_processed = fe.calculate_temporal_features(df_processed)
            df_processed = fe.calculate_anomaly_features(df_processed)
            
            # Detect anomalies
            detector = OptionsAnomalyDetector()
            features = detector.prepare_features(df_processed)
            
            anomaly_rate = 0
            direction = 'neutral'
            confidence = 0
            
            if len(features) > 0:
                detector.fit_models(features, contamination=0.1)
                anomaly_results = detector.ensemble_detection(features)
                signals = detector.generate_signals(df_processed, anomaly_results)
                
                ensemble_anomalies = anomaly_results.get('ensemble_anomaly', [])
                anomaly_rate = ensemble_anomalies.mean() if len(ensemble_anomalies) > 0 else 0
                direction = signals.get('direction', 'neutral')
                confidence = signals.get('confidence', 0)
            
            result = {
                'date': date_str,
                'date_formatted': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                'spy_price': spy_price,
                'total_floor_oi': total_floor_oi,
                'strongest_floor': strongest_floor,
                'strongest_oi': strongest_oi,
                'floor_data': floor_data,
                'anomaly_rate': anomaly_rate,
                'direction': direction,
                'confidence': confidence
            }
            
            results.append(result)
            
            print(f"  ✅ SPY: ${spy_price:.2f}, Floor OI: {total_floor_oi:,.0f}, "
                  f"Strongest: ${strongest_floor} ({strongest_oi:,.0f})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if results:
        print(f"\n{'='*100}")
        print(f"  APRIL 2024 ANALYSIS RESULTS")
        print(f"{'='*100}")
        
        # Summary table
        print(f"{'Date':<12} {'SPY':<8} {'Change':<8} {'Floor_OI':<12} {'Top_Floor':<10} {'Top_OI':<12} {'Anomaly':<8} {'Signal':<8}")
        print("-" * 100)
        
        prev_price = None
        for result in results:
            change = ""
            if prev_price:
                pct_change = (result['spy_price'] - prev_price) / prev_price * 100
                change = f"{pct_change:+.1f}%"
            
            print(f"{result['date_formatted']:<12} "
                  f"${result['spy_price']:<7.2f} "
                  f"{change:<8} "
                  f"{result['total_floor_oi']:<12,.0f} "
                  f"${result['strongest_floor']:<9} "
                  f"{result['strongest_oi']:<12,.0f} "
                  f"{result['anomaly_rate']:<8.1%} "
                  f"{result['direction']:<8}")
            
            prev_price = result['spy_price']
        
        # Find significant patterns
        print(f"\n🔍 PATTERN ANALYSIS:")
        
        # Track when floors first appeared
        significant_floors = {}
        for result in results:
            for strike, data in result['floor_data'].items():
                if data['oi'] > 10000:  # Significant threshold
                    if strike not in significant_floors:
                        significant_floors[strike] = {
                            'first_date': result['date_formatted'],
                            'first_spy': result['spy_price'],
                            'peak_oi': data['oi'],
                            'peak_date': result['date_formatted']
                        }
                    else:
                        if data['oi'] > significant_floors[strike]['peak_oi']:
                            significant_floors[strike]['peak_oi'] = data['oi']
                            significant_floors[strike]['peak_date'] = result['date_formatted']
        
        if significant_floors:
            print(f"\n🏗️ SIGNIFICANT FLOOR DEVELOPMENT:")
            print(f"{'Strike':<8} {'First_Date':<12} {'SPY_Then':<8} {'Peak_OI':<12} {'Peak_Date':<12}")
            print("-" * 60)
            
            for strike in sorted(significant_floors.keys()):
                data = significant_floors[strike]
                print(f"${strike:<7} {data['first_date']:<12} "
                      f"${data['first_spy']:<7.2f} "
                      f"{data['peak_oi']:<12,.0f} "
                      f"{data['peak_date']:<12}")
        
        # Find market bottom in April 2024
        spy_prices = [r['spy_price'] for r in results]
        min_price = min(spy_prices)
        min_idx = spy_prices.index(min_price)
        bottom_date = results[min_idx]['date_formatted']
        
        print(f"\n📉 APRIL 2024 MARKET BOTTOM:")
        print(f"  • Bottom: ${min_price:.2f} on {bottom_date}")
        
        # Check floor accuracy
        if significant_floors:
            # Find the strongest floor before the bottom
            pre_bottom_results = results[:min_idx+1]
            strongest_pre_bottom = None
            strongest_oi = 0
            
            for result in pre_bottom_results:
                if result['strongest_oi'] > strongest_oi:
                    strongest_oi = result['strongest_oi']
                    strongest_pre_bottom = result
            
            if strongest_pre_bottom:
                floor_strike = strongest_pre_bottom['strongest_floor']
                floor_accuracy = abs(floor_strike - min_price)
                floor_pct_accuracy = floor_accuracy / min_price * 100
                
                print(f"  • Predicted floor: ${floor_strike}")
                print(f"  • Accuracy: {floor_accuracy:.2f} points ({floor_pct_accuracy:.1f}% error)")
                
                # Compare to July 2024
                print(f"\n📊 COMPARISON:")
                print(f"July 2024:")
                print(f"  • Predicted: $520, Actual: $517.38 (0.5% error)")
                print(f"  • Floor OI: ~189,000")
                print(f"April 2024:")
                print(f"  • Predicted: ${floor_strike}, Actual: ${min_price:.2f} ({floor_pct_accuracy:.1f}% error)")
                print(f"  • Floor OI: {strongest_oi:,.0f}")
        
        # Anomaly rate analysis
        print(f"\n🚨 ANOMALY PATTERN:")
        anomaly_rates = [r['anomaly_rate'] for r in results if r['anomaly_rate'] > 0]
        if anomaly_rates:
            avg_anomaly = np.mean(anomaly_rates)
            max_anomaly = max(anomaly_rates)
            max_anomaly_idx = [r['anomaly_rate'] for r in results].index(max_anomaly)
            max_anomaly_date = results[max_anomaly_idx]['date_formatted']
            
            print(f"  • Average anomaly rate: {avg_anomaly:.1%}")
            print(f"  • Peak anomaly rate: {max_anomaly:.1%} on {max_anomaly_date}")
            
            # Compare to July 2024 pattern
            print(f"  • July 2024 sustained rate: 8.1-9.3%")
            
            if avg_anomaly > 0.07:  # 7%+ sustained
                print(f"  • April 2024 shows SIMILAR sustained anomaly pattern")
            else:
                print(f"  • April 2024 shows DIFFERENT pattern than July 2024")
        
        # Timeline insights
        if len(results) > 10:
            early_period = results[:len(results)//3]
            late_period = results[2*len(results)//3:]
            
            early_avg_oi = np.mean([r['total_floor_oi'] for r in early_period])
            late_avg_oi = np.mean([r['total_floor_oi'] for r in late_period])
            
            print(f"\n📈 TIMELINE INSIGHTS:")
            print(f"  • Early period avg floor OI: {early_avg_oi:,.0f}")
            print(f"  • Late period avg floor OI: {late_avg_oi:,.0f}")
            print(f"  • OI trend: {((late_avg_oi - early_avg_oi) / early_avg_oi * 100):+.1f}%")
    
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    analyze_april_2024_pattern()