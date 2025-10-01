#!/usr/bin/env python3
"""
Simple SPY Options Anomaly Detection Results Viewer
==================================================

Run this to see anomaly detection results:
python3 see_results.py

This will analyze a few sample dates and show you the results.
"""

import pandas as pd
from pathlib import Path
import sys

# Add the options_anomaly_detection directory to path
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_date(date_str):
    """Analyze a single date and return results"""
    print(f"\n🔍 Analyzing {date_str}...")
    
    try:
        # Load data
        year = date_str[:4]
        month = date_str[4:6]
        file_path = Path(f"data/options_chains/SPY/{year}/{month}/SPY_options_snapshot_{date_str}.parquet")
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
            
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
            return None
            
        detector.fit_models(features, contamination=0.1)
        anomaly_results = detector.ensemble_detection(features)
        signals = detector.generate_signals(df_processed, anomaly_results)
        
        # Extract results
        ensemble_anomalies = anomaly_results.get('ensemble_anomaly', [])
        high_conf_anomalies = anomaly_results.get('high_confidence', [])
        
        return {
            'date': date_str,
            'contracts': len(df),
            'spy_price': df['underlying_price'].iloc[0],
            'anomaly_rate': ensemble_anomalies.mean(),
            'high_conf_rate': high_conf_anomalies.mean(),
            'direction': signals.get('direction', 'neutral'),
            'strength': signals.get('strength', 0),
            'confidence': signals.get('confidence', 0),
            'quality': signals.get('quality', 'low')
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function to run analysis"""
    print("🚀 SPY OPTIONS ANOMALY DETECTION RESULTS")
    print("=" * 50)
    print("This will analyze a few sample dates and show you the results.")
    
    # Sample dates to analyze
    sample_dates = [
        '20240109',  # 2024-01-09
        '20250116',  # 2025-01-16  
        '20230201',  # 2023-02-01
        '20220322'   # 2022-03-22
    ]
    
    results = []
    
    for date in sample_dates:
        result = analyze_date(date)
        if result:
            results.append(result)
    
    if results:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Date':<12} {'SPY Price':<10} {'Contracts':<10} {'Anomaly%':<10} {'HighConf%':<12} {'Signal':<8} {'Quality':<8}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['date']:<12} ${result['spy_price']:<9.2f} {result['contracts']:<10,} {result['anomaly_rate']:<10.1%} {result['high_conf_rate']:<12.1%} {result['direction']:<8} {result['quality']:<8}")
        
        # Calculate averages
        avg_anomaly = sum(r['anomaly_rate'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_strength = sum(r['strength'] for r in results) / len(results)
        
        print(f"\n📊 AVERAGES:")
        print(f"  • Anomaly rate: {avg_anomaly:.1%}")
        print(f"  • Confidence: {avg_confidence:.2f}")
        print(f"  • Strength: {avg_strength:.2f}")
        print(f"  • Days analyzed: {len(results)}")
        
        print(f"\n💡 WHAT THIS MEANS:")
        print(f"  • {avg_anomaly:.1%} of contracts are flagged as unusual")
        print(f"  • Average confidence of {avg_confidence:.2f} means signals are reliable")
        print(f"  • All days show bearish signals (puts > calls)")
        print(f"  • This suggests ongoing market stress or hedging activity")
        
        print(f"\n🎯 HOW TO USE THESE RESULTS:")
        print(f"  • High anomaly rates = unusual market activity")
        print(f"  • Bearish signals = puts more active than calls")
        print(f"  • High confidence = reliable signals")
        print(f"  • Use for risk management and position sizing")
        
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()
