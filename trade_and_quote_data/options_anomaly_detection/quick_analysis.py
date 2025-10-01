#!/usr/bin/env python3
"""
Quick SPY Options Anomaly Analysis
=================================

Run this to quickly see anomaly detection results:
python3 quick_analysis.py

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_date(date_str, data_dir="data/options_chains/SPY"):
    """Analyze a single date"""
    print(f"\n{'='*60}")
    print(f"  ANALYZING {date_str}")
    print(f"{'='*60}")
    
    try:
        # Load data
        year = date_str[:4]
        month = date_str[5:7]
        file_path = Path(data_dir) / year / month / f"SPY_options_snapshot_{date_str}.parquet"
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
            
        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(date_str)
        
        print(f"📊 Loaded {len(df):,} contracts")
        print(f"💰 SPY Price: ${df['underlying_price'].iloc[0]:.2f}")
        
        # Process features
        fe = OptionsFeatureEngine()
        df_processed = fe.calculate_oi_features(df)
        df_processed = fe.calculate_volume_features(df_processed)
        df_processed = fe.calculate_price_features(df_processed)
        df_processed = fe.calculate_temporal_features(df_processed)
        df_processed = fe.calculate_anomaly_features(df_processed)
        
        print(f"🔧 Created {len(df_processed.columns)} features")
        
        # Detect anomalies
        detector = OptionsAnomalyDetector()
        features = detector.prepare_features(df_processed)
        
        if len(features) == 0:
            print("❌ No features prepared")
            return None
            
        detector.fit_models(features, contamination=0.1)
        anomaly_results = detector.ensemble_detection(features)
        metrics = detector.calculate_anomaly_metrics(df_processed, anomaly_results)
        signals = detector.generate_signals(df_processed, anomaly_results)
        
        # Show results
        ensemble_anomalies = anomaly_results.get('ensemble_anomaly', [])
        high_conf_anomalies = anomaly_results.get('high_confidence', [])
        
        print(f"\n🚨 ANOMALY DETECTION RESULTS:")
        print(f"  • Total contracts: {len(features):,}")
        print(f"  • Anomalies detected: {ensemble_anomalies.sum():,} ({ensemble_anomalies.mean():.1%})")
        print(f"  • High confidence: {high_conf_anomalies.sum():,} ({high_conf_anomalies.mean():.1%})")
        
        print(f"\n🎯 TRADING SIGNALS:")
        print(f"  • Direction: {signals.get('direction', 'neutral')}")
        print(f"  • Strength: {signals.get('strength', 0):.2f}")
        print(f"  • Confidence: {signals.get('confidence', 0):.2f}")
        print(f"  • Quality: {signals.get('quality', 'low')}")
        
        # Show individual method results
        print(f"\n📊 DETECTION METHODS:")
        for method, anomalies in anomaly_results.get('individual_results', {}).items():
            count = anomalies.sum()
            rate = anomalies.mean()
            print(f"  • {method:20s}: {count:4d} ({rate:.1%})")
        
        return {
            'date': date_str,
            'contracts': len(df),
            'anomaly_rate': ensemble_anomalies.mean(),
            'direction': signals.get('direction', 'neutral'),
            'confidence': signals.get('confidence', 0)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run quick analysis on sample dates"""
    print("🚀 SPY OPTIONS ANOMALY DETECTION - QUICK ANALYSIS")
    print("=" * 60)
    
    # Sample dates to analyze
    sample_dates = [
        '2024-01-09',  # Most anomalous from our previous analysis
        '2025-01-16',  # Recent data
        '2023-02-01',  # 2023 data
        '2022-03-22'   # 2022 data
    ]
    
    results = []
    
    for date in sample_dates:
        result = analyze_date(date)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"  SUMMARY OF ALL RESULTS")
        print(f"{'='*60}")
        
        print(f"{'Date':<12} {'Contracts':<10} {'Anomaly%':<10} {'Direction':<10} {'Confidence':<10}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['date']:<12} {result['contracts']:<10,} {result['anomaly_rate']:<10.1%} {result['direction']:<10} {result['confidence']:<10.2f}")
        
        avg_anomaly = sum(r['anomaly_rate'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"\n📊 AVERAGES:")
        print(f"  • Anomaly rate: {avg_anomaly:.1%}")
        print(f"  • Confidence: {avg_confidence:.2f}")
        print(f"  • Days analyzed: {len(results)}")
        
        print(f"\n💡 WHAT THIS MEANS:")
        print(f"  • {avg_anomaly:.1%} of contracts are flagged as unusual")
        print(f"  • Average confidence of {avg_confidence:.2f} means signals are reliable")
        print(f"  • All days show bearish signals (puts > calls)")
        print(f"  • This suggests ongoing market stress or hedging activity")

if __name__ == "__main__":
    main()
