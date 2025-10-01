#!/usr/bin/env python3
"""
Analyze Strike Distribution and Magnitude Prediction
==================================================

This analyzes the strike distribution on July 16, 2024 to understand
the lowest strikes traded and magnitude prediction capability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_strike_magnitude():
    """Analyze strike distribution and magnitude prediction signals"""
    
    date_str = '20240716'
    print(f"🎯 STRIKE DISTRIBUTION & MAGNITUDE ANALYSIS: July 16, 2024")
    print("=" * 70)
    
    # Load data
    year = date_str[:4]
    month = date_str[4:6]
    file_path = Path(f"data/options_chains/SPY/{year}/{month}/SPY_options_snapshot_{date_str}.parquet")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
        
    df = pd.read_parquet(file_path)
    df['date'] = pd.to_datetime(date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8])
    
    spy_price = df['underlying_price'].iloc[0]
    print(f"📊 SPY Price: ${spy_price:.2f}")
    print(f"📊 Total contracts: {len(df):,}")
    
    # Process features and detect anomalies
    fe = OptionsFeatureEngine()
    df_processed = fe.calculate_oi_features(df)
    df_processed = fe.calculate_volume_features(df_processed)
    df_processed = fe.calculate_price_features(df_processed)
    df_processed = fe.calculate_temporal_features(df_processed)
    df_processed = fe.calculate_anomaly_features(df_processed)
    
    detector = OptionsAnomalyDetector()
    features = detector.prepare_features(df_processed)
    detector.fit_models(features, contamination=0.1)
    anomaly_results = detector.ensemble_detection(features)
    
    # Strike distribution analysis
    print(f"\n🎯 FULL STRIKE DISTRIBUTION:")
    strike_stats = df['strike'].describe()
    print(f"  • Lowest strike: ${strike_stats['min']:.0f}")
    print(f"  • Highest strike: ${strike_stats['max']:.0f}")
    print(f"  • Range: ${strike_stats['max'] - strike_stats['min']:.0f}")
    print(f"  • Distance below SPY: ${spy_price - strike_stats['min']:.0f} points")
    print(f"  • Distance above SPY: ${strike_stats['max'] - spy_price:.0f} points")
    
    # Extreme strikes analysis
    print(f"\n📉 EXTREME DOWNSIDE STRIKES (< $300):")
    extreme_low = df[df['strike'] < 300].sort_values('strike')
    if len(extreme_low) > 0:
        print(f"  • Count: {len(extreme_low)} contracts")
        for _, row in extreme_low.head(10).iterrows():
            pct_below = (spy_price - row['strike']) / spy_price * 100
            print(f"    - ${row['strike']:.0f} {row['option_type']} "
                  f"({pct_below:.0f}% below SPY), "
                  f"Vol: {row['volume']:,}, OI: {row['oi_proxy']:,}")
    else:
        print("  • No contracts below $300")
    
    print(f"\n📈 EXTREME UPSIDE STRIKES (> $800):")
    extreme_high = df[df['strike'] > 800].sort_values('strike', ascending=False)
    if len(extreme_high) > 0:
        print(f"  • Count: {len(extreme_high)} contracts")
        for _, row in extreme_high.head(10).iterrows():
            pct_above = (row['strike'] - spy_price) / spy_price * 100
            print(f"    - ${row['strike']:.0f} {row['option_type']} "
                  f"({pct_above:.0f}% above SPY), "
                  f"Vol: {row['volume']:,}, OI: {row['oi_proxy']:,}")
    else:
        print("  • No contracts above $800")
    
    # Anomalous strikes analysis
    ensemble_mask = anomaly_results['ensemble_anomaly'].astype(bool)
    anomaly_contracts = df_processed[ensemble_mask].copy()
    
    print(f"\n🚨 ANOMALOUS STRIKE DISTRIBUTION:")
    anomaly_strike_stats = anomaly_contracts['strike'].describe()
    print(f"  • Anomalous strikes: {len(anomaly_contracts)} contracts")
    print(f"  • Lowest anomalous: ${anomaly_strike_stats['min']:.0f}")
    print(f"  • Highest anomalous: ${anomaly_strike_stats['max']:.0f}")
    print(f"  • Median anomalous: ${anomaly_strike_stats['50%']:.0f}")
    
    # Extreme anomalous strikes
    extreme_anomaly_low = anomaly_contracts[anomaly_contracts['strike'] < 300]
    extreme_anomaly_high = anomaly_contracts[anomaly_contracts['strike'] > 800]
    
    print(f"\n🔥 EXTREME ANOMALOUS STRIKES:")
    print(f"  • Anomalous strikes < $300: {len(extreme_anomaly_low)}")
    print(f"  • Anomalous strikes > $800: {len(extreme_anomaly_high)}")
    
    if len(extreme_anomaly_low) > 0:
        print(f"\n📉 LOWEST ANOMALOUS STRIKES:")
        for _, row in extreme_anomaly_low.sort_values('strike').head(5).iterrows():
            pct_below = (spy_price - row['strike']) / spy_price * 100
            print(f"    - ${row['strike']:.0f} {row['option_type']} "
                  f"({pct_below:.0f}% below), "
                  f"Vol: {row['volume']:,}, OI: {row['oi_proxy']:,}, "
                  f"DTE: {row['dte']:.0f}")
    
    if len(extreme_anomaly_high) > 0:
        print(f"\n📈 HIGHEST ANOMALOUS STRIKES:")
        for _, row in extreme_anomaly_high.sort_values('strike', ascending=False).head(5).iterrows():
            pct_above = (row['strike'] - spy_price) / spy_price * 100
            print(f"    - ${row['strike']:.0f} {row['option_type']} "
                  f"({pct_above:.0f}% above), "
                  f"Vol: {row['volume']:,}, OI: {row['oi_proxy']:,}, "
                  f"DTE: {row['dte']:.0f}")
    
    # Magnitude prediction analysis
    print(f"\n🎯 MAGNITUDE PREDICTION SIGNALS:")
    
    # Strike spread analysis
    strike_range = anomaly_strike_stats['max'] - anomaly_strike_stats['min']
    normal_mask = ~ensemble_mask
    normal_contracts = df_processed[normal_mask]
    normal_strike_range = normal_contracts['strike'].max() - normal_contracts['strike'].min()
    
    print(f"  • Anomalous strike range: ${strike_range:.0f}")
    print(f"  • Normal strike range: ${normal_strike_range:.0f}")
    print(f"  • Range ratio: {strike_range/normal_strike_range:.1f}x")
    
    # Distance from ATM analysis
    anomaly_contracts['distance_from_atm'] = abs(anomaly_contracts['strike'] - spy_price)
    normal_contracts['distance_from_atm'] = abs(normal_contracts['strike'] - spy_price)
    
    print(f"  • Avg anomalous distance from ATM: ${anomaly_contracts['distance_from_atm'].mean():.0f}")
    print(f"  • Avg normal distance from ATM: ${normal_contracts['distance_from_atm'].mean():.0f}")
    
    # Tail risk indicators
    far_otm_puts = anomaly_contracts[
        (anomaly_contracts['option_type'] == 'P') & 
        (anomaly_contracts['strike'] < spy_price * 0.8)
    ]
    far_otm_calls = anomaly_contracts[
        (anomaly_contracts['option_type'] == 'C') & 
        (anomaly_contracts['strike'] > spy_price * 1.2)
    ]
    
    print(f"\n🎪 TAIL RISK INDICATORS:")
    print(f"  • Far OTM puts (>20% below): {len(far_otm_puts)} anomalies")
    print(f"  • Far OTM calls (>20% above): {len(far_otm_calls)} anomalies")
    print(f"  • Tail risk ratio: {len(far_otm_puts)/len(far_otm_calls) if len(far_otm_calls) > 0 else 'inf':.1f}")
    
    if len(far_otm_puts) > 0:
        avg_put_volume = far_otm_puts['volume'].mean()
        avg_put_oi = far_otm_puts['oi_proxy'].mean()
        print(f"  • Avg far OTM put volume: {avg_put_volume:,.0f}")
        print(f"  • Avg far OTM put OI: {avg_put_oi:,.0f}")
    
    # Volume-weighted magnitude signal
    anomaly_contracts['volume_weight'] = anomaly_contracts['volume'] / anomaly_contracts['volume'].sum()
    anomaly_contracts['weighted_distance'] = anomaly_contracts['distance_from_atm'] * anomaly_contracts['volume_weight']
    volume_weighted_distance = anomaly_contracts['weighted_distance'].sum()
    
    print(f"\n📊 VOLUME-WEIGHTED MAGNITUDE SIGNALS:")
    print(f"  • Volume-weighted avg distance: ${volume_weighted_distance:.0f}")
    print(f"  • This suggests expected move magnitude of ~{volume_weighted_distance/spy_price*100:.1f}%")
    
    # Concentration analysis
    strike_buckets = pd.cut(anomaly_contracts['strike'], bins=10)
    bucket_volumes = anomaly_contracts.groupby(strike_buckets)['volume'].sum()
    max_bucket_vol = bucket_volumes.max()
    total_vol = bucket_volumes.sum()
    concentration = max_bucket_vol / total_vol if total_vol > 0 else 0
    
    print(f"  • Strike concentration (max bucket): {concentration:.1%}")
    print(f"  • High concentration suggests focused expectations")

if __name__ == "__main__":
    analyze_strike_magnitude()