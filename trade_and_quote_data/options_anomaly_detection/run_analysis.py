#!/usr/bin/env python3
"""
SPY Options Anomaly Detection - Interactive Analysis Tool
========================================================

This script allows you to easily run anomaly detection and see results.
Just run: python3 run_analysis.py

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector
from analysis_engine import OptionsAnalysisEngine

def print_header(title):
    """Print a nice header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a section header"""
    print(f"\n🔍 {title}")
    print("-" * 50)

def analyze_single_day(date_str, data_dir="data/options_chains/SPY"):
    """Analyze a single day and show detailed results"""
    print_header(f"ANALYZING {date_str}")
    
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
        
        print_section("Raw Data Overview")
        print(f"📊 Total contracts: {len(df):,}")
        print(f"📅 Date: {date_str}")
        print(f"💰 SPY Price: ${df['underlying_price'].iloc[0]:.2f}")
        print(f"📈 Call contracts: {len(df[df['option_type'] == 'C']):,}")
        print(f"📉 Put contracts: {len(df[df['option_type'] == 'P']):,}")
        
        pc_ratio = len(df[df['option_type'] == 'P']) / len(df[df['option_type'] == 'C'])
        print(f"⚖️  Put/Call ratio: {pc_ratio:.2f}")
        
        # Process features
        print_section("Feature Engineering")
        fe = OptionsFeatureEngine()
        df_processed = fe.calculate_oi_features(df)
        df_processed = fe.calculate_volume_features(df_processed)
        df_processed = fe.calculate_price_features(df_processed)
        df_processed = fe.calculate_temporal_features(df_processed)
        df_processed = fe.calculate_anomaly_features(df_processed)
        
        print(f"✅ Features created: {len(df_processed.columns)}")
        print(f"📊 OI proxy range: {df_processed['oi_proxy'].min():.0f} - {df_processed['oi_proxy'].max():.0f}")
        print(f"📊 Average OI proxy: {df_processed['oi_proxy'].mean():.0f}")
        
        if 'pc_oi_ratio' in df_processed.columns:
            print(f"⚖️  Put/Call OI ratio: {df_processed['pc_oi_ratio'].iloc[0]:.2f}")
        
        # Detect anomalies
        print_section("Anomaly Detection")
        detector = OptionsAnomalyDetector()
        features = detector.prepare_features(df_processed)
        
        if len(features) == 0:
            print("❌ No features prepared")
            return None
            
        detector.fit_models(features, contamination=0.1)
        anomaly_results = detector.ensemble_detection(features)
        metrics = detector.calculate_anomaly_metrics(df_processed, anomaly_results)
        signals = detector.generate_signals(df_processed, anomaly_results)
        
        print(f"🤖 Models trained on {len(features):,} contracts")
        print(f"📊 Feature dimensions: {features.shape[1]}")
        
        # Show individual method results
        print("\n📊 Individual Detection Methods:")
        for method, anomalies in anomaly_results.get('individual_results', {}).items():
            count = anomalies.sum()
            rate = anomalies.mean()
            print(f"  • {method:20s}: {count:4d} anomalies ({rate:.1%})")
        
        # Show ensemble results
        ensemble_anomalies = anomaly_results.get('ensemble_anomaly', [])
        high_conf_anomalies = anomaly_results.get('high_confidence', [])
        
        print(f"\n🎯 Final Results:")
        print(f"  • Ensemble anomalies: {ensemble_anomalies.sum():,} ({ensemble_anomalies.mean():.1%})")
        print(f"  • High confidence: {high_conf_anomalies.sum():,} ({high_conf_anomalies.mean():.1%})")
        
        # Show trading signals
        print_section("Trading Signals")
        print(f"📈 Direction: {signals.get('direction', 'neutral')}")
        print(f"💪 Strength: {signals.get('strength', 0):.2f}")
        print(f"🎯 Confidence: {signals.get('confidence', 0):.2f}")
        print(f"⭐ Quality: {signals.get('quality', 'low')}")
        
        # Analyze anomalous contracts
        if ensemble_anomalies.sum() > 0:
            anomaly_mask = ensemble_anomalies.astype(bool)
            anomaly_df = df_processed[anomaly_mask]
            
            print_section("Anomalous Contracts Analysis")
            print(f"🔍 Anomalous contracts: {len(anomaly_df):,}")
            print(f"📊 Average OI proxy: {anomaly_df['oi_proxy'].mean():.0f}")
            print(f"📊 Average volume: {anomaly_df['volume'].mean():.0f}")
            
            anomaly_pc_ratio = len(anomaly_df[anomaly_df['option_type'] == 'P']) / len(anomaly_df[anomaly_df['option_type'] == 'C'])
            print(f"⚖️  Put/Call ratio: {anomaly_pc_ratio:.2f}")
            
            # Show sample anomalous contracts
            print(f"\n📋 Sample Anomalous Contracts:")
            sample_anomalies = anomaly_df.head(5)
            for i, (idx, row) in enumerate(sample_anomalies.iterrows()):
                print(f"  {i+1}. {row['option_type']} ${row['strike']:.0f} {row['expiration']} - OI: {row['oi_proxy']:.0f}, Vol: {row['volume']:.0f}")
        
        return {
            'date': date_str,
            'contracts': len(df),
            'anomaly_rate': ensemble_anomalies.mean(),
            'high_conf_rate': high_conf_anomalies.mean(),
            'direction': signals.get('direction', 'neutral'),
            'strength': signals.get('strength', 0),
            'confidence': signals.get('confidence', 0),
            'quality': signals.get('quality', 'low')
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {date_str}: {e}")
        return None

def analyze_multiple_days(dates, data_dir="data/options_chains/SPY"):
    """Analyze multiple days and show summary"""
    print_header("MULTI-DAY ANALYSIS")
    
    results = []
    for date in dates:
        print(f"\n📅 Analyzing {date}...")
        result = analyze_single_day(date, data_dir)
        if result:
            results.append(result)
    
    if results:
        print_section("Summary Results")
        print(f"{'Date':<12} {'Contracts':<10} {'Anomaly%':<10} {'HighConf%':<12} {'Signal':<8} {'Quality':<8}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['date']:<12} {result['contracts']:<10,} {result['anomaly_rate']:<10.1%} {result['high_conf_rate']:<12.1%} {result['direction']:<8} {result['quality']:<8}")
        
        # Calculate averages
        avg_anomaly_rate = sum(r['anomaly_rate'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"\n📊 Averages:")
        print(f"  • Anomaly rate: {avg_anomaly_rate:.1%}")
        print(f"  • Confidence: {avg_confidence:.2f}")
        print(f"  • Days analyzed: {len(results)}")
        
        return results
    else:
        print("❌ No results generated")
        return []

def find_available_dates(data_dir="data/options_chains/SPY"):
    """Find available dates in the data directory"""
    data_path = Path(data_dir)
    dates = []
    
    for year_dir in data_path.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and month_dir.name.isdigit():
                    for file_path in month_dir.glob("*.parquet"):
                        # Extract date from filename
                        filename = file_path.stem
                        if filename.startswith("SPY_options_snapshot_"):
                            date_str = filename.replace("SPY_options_snapshot_", "")
                            dates.append(date_str)
    
    return sorted(dates)

def main():
    """Main interactive function"""
    print_header("SPY OPTIONS ANOMALY DETECTION")
    print("Welcome! This tool will help you analyze SPY options data.")
    
    # Find available dates
    print_section("Finding Available Data")
    available_dates = find_available_dates()
    print(f"📊 Found {len(available_dates)} available dates")
    
    if len(available_dates) == 0:
        print("❌ No data found. Make sure you have downloaded SPY options data.")
        return
    
    # Show some sample dates
    print(f"📅 Sample dates: {available_dates[:5]}")
    
    # Let user choose what to do
    print_section("Choose Analysis Type")
    print("1. Analyze a specific date")
    print("2. Analyze multiple dates")
    print("3. Analyze recent dates")
    print("4. Show all available dates")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        date = input(f"Enter date (YYYY-MM-DD) from available dates: ").strip()
        if date in available_dates:
            analyze_single_day(date)
        else:
            print(f"❌ Date {date} not found in available dates")
    
    elif choice == "2":
        print("Enter dates separated by commas (e.g., 2024-01-09,2024-01-10)")
        dates_input = input("Dates: ").strip()
        dates = [d.strip() for d in dates_input.split(",")]
        analyze_multiple_days(dates)
    
    elif choice == "3":
        # Analyze last 5 dates
        recent_dates = available_dates[-5:]
        print(f"📅 Analyzing recent dates: {recent_dates}")
        analyze_multiple_days(recent_dates)
    
    elif choice == "4":
        print_section("All Available Dates")
        for i, date in enumerate(available_dates, 1):
            print(f"{i:3d}. {date}")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
