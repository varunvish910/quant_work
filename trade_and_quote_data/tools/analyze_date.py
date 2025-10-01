#!/usr/bin/env python3
"""
Analyze a Specific Date - SPY Options Anomaly Detection
======================================================

Usage: python3 analyze_date.py YYYYMMDD
Example: python3 analyze_date.py 20240109

This will analyze a specific date and show detailed results.
"""

import pandas as pd
from pathlib import Path
import sys

# Add the options_anomaly_detection directory to path
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_specific_date(date_str):
    """Analyze a specific date with detailed results"""
    print(f"🚀 ANALYZING {date_str}")
    print("=" * 50)
    
    try:
        # Load data
        year = date_str[:4]
        month = date_str[4:6]
        file_path = Path(f"data/options_chains/SPY/{year}/{month}/SPY_options_snapshot_{date_str}.parquet")
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            print(f"Available dates in {year}/{month}:")
            month_dir = Path(f"data/options_chains/SPY/{year}/{month}")
            if month_dir.exists():
                for f in month_dir.glob("*.parquet"):
                    print(f"  • {f.stem.replace('SPY_options_snapshot_', '')}")
            return None
            
        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8])
        
        print(f"📊 Loaded {len(df):,} contracts")
        print(f"💰 SPY Price: ${df['underlying_price'].iloc[0]:.2f}")
        print(f"📈 Call contracts: {len(df[df['option_type'] == 'C']):,}")
        print(f"📉 Put contracts: {len(df[df['option_type'] == 'P']):,}")
        
        pc_ratio = len(df[df['option_type'] == 'P']) / len(df[df['option_type'] == 'C'])
        print(f"⚖️  Put/Call ratio: {pc_ratio:.2f}")
        
        # Process features
        print(f"\n🔧 Processing features...")
        fe = OptionsFeatureEngine()
        df_processed = fe.calculate_oi_features(df)
        df_processed = fe.calculate_volume_features(df_processed)
        df_processed = fe.calculate_price_features(df_processed)
        df_processed = fe.calculate_temporal_features(df_processed)
        df_processed = fe.calculate_anomaly_features(df_processed)
        
        print(f"✅ Created {len(df_processed.columns)} features")
        
        # Detect anomalies
        print(f"\n🚨 Detecting anomalies...")
        detector = OptionsAnomalyDetector()
        features = detector.prepare_features(df_processed)
        
        if len(features) == 0:
            print("❌ No features prepared")
            return None
            
        detector.fit_models(features, contamination=0.1)
        anomaly_results = detector.ensemble_detection(features)
        signals = detector.generate_signals(df_processed, anomaly_results)
        
        # Show detailed results
        ensemble_anomalies = anomaly_results.get('ensemble_anomaly', [])
        high_conf_anomalies = anomaly_results.get('high_confidence', [])
        
        print(f"\n📊 ANOMALY DETECTION RESULTS:")
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
            print(f"  • {method:20s}: {count:4d} anomalies ({rate:.1%})")
        
        # Analyze anomalous contracts
        if ensemble_anomalies.sum() > 0:
            anomaly_mask = ensemble_anomalies.astype(bool)
            anomaly_df = df_processed[anomaly_mask]
            
            print(f"\n🔍 ANOMALOUS CONTRACTS ANALYSIS:")
            print(f"  • Anomalous contracts: {len(anomaly_df):,}")
            print(f"  • Average OI proxy: {anomaly_df['oi_proxy'].mean():.0f}")
            print(f"  • Average volume: {anomaly_df['volume'].mean():.0f}")
            
            anomaly_pc_ratio = len(anomaly_df[anomaly_df['option_type'] == 'P']) / len(anomaly_df[anomaly_df['option_type'] == 'C'])
            print(f"  • Put/Call ratio: {anomaly_pc_ratio:.2f}")
            
            # Show sample anomalous contracts
            print(f"\n📋 SAMPLE ANOMALOUS CONTRACTS:")
            sample_anomalies = anomaly_df.head(5)
            for i, (idx, row) in enumerate(sample_anomalies.iterrows()):
                print(f"  {i+1}. {row['option_type']} ${row['strike']:.0f} {row['expiration']} - OI: {row['oi_proxy']:.0f}, Vol: {row['volume']:.0f}")
        
        print(f"\n💡 INTERPRETATION:")
        print(f"  • {ensemble_anomalies.mean():.1%} anomaly rate means 1 in {int(1/ensemble_anomalies.mean())} contracts is unusual")
        print(f"  • {signals.get('direction', 'neutral')} signal suggests {'puts' if signals.get('direction') == 'bearish' else 'calls'} are more active")
        print(f"  • {signals.get('quality', 'low')} quality means the signal is {'reliable' if signals.get('quality') == 'high' else 'moderately reliable' if signals.get('quality') == 'medium' else 'weak'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_date.py YYYYMMDD")
        print("Example: python3 analyze_date.py 20240109")
        print("\nAvailable dates (sample):")
        # Show some available dates
        for year in ['2022', '2023', '2024', '2025']:
            year_dir = Path(f"data/options_chains/SPY/{year}")
            if year_dir.exists():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        files = list(month_dir.glob("*.parquet"))
                        if files:
                            sample_file = files[0].stem.replace('SPY_options_snapshot_', '')
                            print(f"  • {sample_file} (from {year}/{month_dir.name})")
                            break
        return
    
    date_str = sys.argv[1]
    
    if len(date_str) != 8 or not date_str.isdigit():
        print("❌ Invalid date format. Use YYYYMMDD (e.g., 20240109)")
        return
    
    analyze_specific_date(date_str)

if __name__ == "__main__":
    main()
