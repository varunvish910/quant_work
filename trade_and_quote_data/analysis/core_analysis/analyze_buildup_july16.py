#!/usr/bin/env python3
"""
Analyze Buildup to July 16, 2024
================================

This analyzes anomaly patterns in the days leading up to July 16, 2024
to see if there was a building pattern that the models detected.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_date_range():
    """Analyze anomaly patterns from July 8-16, 2024"""
    
    print(f"🔍 ANALYZING BUILDUP TO JULY 16, 2024")
    print("=" * 60)
    
    # Date range (weekdays only)
    dates = [
        '20240708',  # Monday July 8
        '20240709',  # Tuesday July 9  
        '20240710',  # Wednesday July 10
        '20240711',  # Thursday July 11
        '20240712',  # Friday July 12
        # Weekend skip
        '20240715',  # Monday July 15
        '20240716'   # Tuesday July 16 (the target date)
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
            df['date'] = pd.to_datetime(date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8])
            
            # Process features
            fe = OptionsFeatureEngine()
            df_processed = fe.calculate_oi_features(df)
            df_processed = fe.calculate_volume_features(df_processed)
            df_processed = fe.calculate_price_features(df_processed)
            df_processed = fe.calculate_temporal_features(df_processed)
            df_processed = fe.calculate_anomaly_features(df_processed)
            
            # Detect anomalies
            detector = OptionsAnomalyDetector()
            features = detector.prepare_features(df_processed)
            
            if len(features) == 0:
                continue
                
            detector.fit_models(features, contamination=0.1)
            anomaly_results = detector.ensemble_detection(features)
            signals = detector.generate_signals(df_processed, anomaly_results)
            
            # Extract results
            ensemble_anomalies = anomaly_results.get('ensemble_anomaly', [])
            high_conf_anomalies = anomaly_results.get('high_confidence', [])
            
            # Calculate key metrics
            result = {
                'date': date_str,
                'date_formatted': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                'contracts': len(df),
                'spy_price': df['underlying_price'].iloc[0],
                'anomaly_count': ensemble_anomalies.sum(),
                'anomaly_rate': ensemble_anomalies.mean(),
                'high_conf_count': high_conf_anomalies.sum(),
                'high_conf_rate': high_conf_anomalies.mean(),
                'direction': signals.get('direction', 'neutral'),
                'strength': signals.get('strength', 0),
                'confidence': signals.get('confidence', 0),
                'quality': signals.get('quality', 'low'),
                
                # Additional metrics for anomalous contracts
                'avg_anomaly_volume': 0,
                'avg_anomaly_oi': 0,
                'anomaly_pc_ratio': 0,
                'avg_normal_volume': 0
            }
            
            # Analyze anomalous contracts if any exist
            if ensemble_anomalies.sum() > 0:
                anomaly_mask = ensemble_anomalies.astype(bool)
                anomaly_contracts = df_processed[anomaly_mask]
                normal_contracts = df_processed[~anomaly_mask]
                
                result['avg_anomaly_volume'] = anomaly_contracts['volume'].mean()
                result['avg_anomaly_oi'] = anomaly_contracts['oi_proxy'].mean()
                result['avg_normal_volume'] = normal_contracts['volume'].mean()
                
                # Put/Call ratio for anomalies
                anomaly_calls = len(anomaly_contracts[anomaly_contracts['option_type'] == 'C'])
                anomaly_puts = len(anomaly_contracts[anomaly_contracts['option_type'] == 'P'])
                result['anomaly_pc_ratio'] = anomaly_puts / (anomaly_calls + 1e-6)
            
            results.append(result)
            print(f"  ✅ {result['anomaly_count']} anomalies ({result['anomaly_rate']:.1%}), SPY: ${result['spy_price']:.2f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if results:
        print(f"\n{'='*80}")
        print(f"  BUILDUP ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        # Create summary table
        print(f"{'Date':<12} {'SPY':<8} {'Contracts':<10} {'Anomalies':<10} {'Rate':<8} {'HighConf':<9} {'AvgVol':<10} {'Signal':<8}")
        print("-" * 85)
        
        for result in results:
            print(f"{result['date_formatted']:<12} "
                  f"${result['spy_price']:<7.2f} "
                  f"{result['contracts']:<10,} "
                  f"{result['anomaly_count']:<10} "
                  f"{result['anomaly_rate']:<8.1%} "
                  f"{result['high_conf_count']:<9} "
                  f"{result['avg_anomaly_volume']:<10,.0f} "
                  f"{result['direction']:<8}")
        
        # Trend analysis
        print(f"\n📈 TREND ANALYSIS:")
        
        # Anomaly rate trend
        anomaly_rates = [r['anomaly_rate'] for r in results]
        if len(anomaly_rates) > 1:
            rate_change = anomaly_rates[-1] - anomaly_rates[0]
            print(f"  • Anomaly rate: {anomaly_rates[0]:.1%} → {anomaly_rates[-1]:.1%} (change: {rate_change:+.1%})")
        
        # Volume trend in anomalies
        anomaly_volumes = [r['avg_anomaly_volume'] for r in results if r['avg_anomaly_volume'] > 0]
        if len(anomaly_volumes) > 1:
            vol_change = anomaly_volumes[-1] / (anomaly_volumes[0] + 1e-6)
            print(f"  • Anomaly volume: {anomaly_volumes[0]:.0f} → {anomaly_volumes[-1]:.0f} (ratio: {vol_change:.1f}x)")
        
        # Confidence trend
        confidence_scores = [r['confidence'] for r in results]
        if len(confidence_scores) > 1:
            conf_change = confidence_scores[-1] - confidence_scores[0]
            print(f"  • Confidence: {confidence_scores[0]:.2f} → {confidence_scores[-1]:.2f} (change: {conf_change:+.2f})")
        
        # Daily changes
        print(f"\n📊 DAILY CHANGES:")
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            
            rate_delta = curr['anomaly_rate'] - prev['anomaly_rate']
            vol_ratio = curr['avg_anomaly_volume'] / (prev['avg_anomaly_volume'] + 1e-6) if prev['avg_anomaly_volume'] > 0 else 1
            
            print(f"  • {prev['date_formatted']} → {curr['date_formatted']}: "
                  f"Rate {rate_delta:+.1%}, Volume {vol_ratio:.1f}x, "
                  f"Signal: {prev['direction']} → {curr['direction']}")
        
        # July 16 standout metrics
        july16 = results[-1]
        prev_avg_rate = np.mean([r['anomaly_rate'] for r in results[:-1]])
        prev_avg_vol = np.mean([r['avg_anomaly_volume'] for r in results[:-1] if r['avg_anomaly_volume'] > 0])
        
        print(f"\n🎯 JULY 16 vs PREVIOUS AVERAGE:")
        print(f"  • Anomaly rate: {july16['anomaly_rate']:.1%} vs {prev_avg_rate:.1%} "
              f"(ratio: {july16['anomaly_rate']/(prev_avg_rate+1e-6):.1f}x)")
        if prev_avg_vol > 0:
            print(f"  • Volume: {july16['avg_anomaly_volume']:,.0f} vs {prev_avg_vol:,.0f} "
                  f"(ratio: {july16['avg_anomaly_volume']/(prev_avg_vol+1e-6):.1f}x)")
        print(f"  • Confidence: {july16['confidence']:.2f}")
        print(f"  • Signal quality: {july16['quality']}")
        
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    analyze_date_range()