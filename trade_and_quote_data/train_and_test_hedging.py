#!/usr/bin/env python3
"""
Train hedging signal detector on 2016-2023 data and test on 2024 targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.path.append('options_anomaly_detection')

from options_anomaly_detection.anomaly_detection import OptionsAnomalyDetector
from target_creator import CorrectionTargetCreator

def train_and_test_system():
    """
    Train the hedging system and identify 2024 target dates
    """
    print("🚀 TRAINING HEDGING SYSTEM AND IDENTIFYING 2024 TARGETS")
    print("=" * 60)
    
    # 1. Initialize systems
    detector = OptionsAnomalyDetector("data/options_chains/SPY")
    target_creator = CorrectionTargetCreator()
    
    # 2. Train on 2016-2023 data
    print("\n📚 TRAINING PHASE (2016-2023)")
    print("-" * 40)
    
    # Train the anomaly detector (includes hedging system)
    print("Training hedging intelligence system...")
    detector.train_on_historical_data("2016-01-01", "2023-12-31")
    
    # Also train the hedging detector directly for calibration
    print("Calibrating hedging signal thresholds...")
    hedging_results = detector.hedging_detector.train_on_historical_hedging("2016-01-01", "2023-12-31")
    
    if 'error' not in hedging_results:
        calibration = hedging_results['calibration']
        print(f"✅ Training complete: {calibration['total_days']} days processed")
        print(f"📊 Signal distribution: {calibration['signal_distribution']}")
        print(f"🎯 Average confidence: {calibration['avg_confidence']:.3f}")
        print(f"🔥 High confidence days: {calibration['high_confidence_days']}")
    
    # 3. Identify 2024 correction targets
    print("\n🎯 IDENTIFYING 2024 CORRECTION TARGETS")
    print("-" * 40)
    
    # Load 2024 price data and identify corrections
    print("Loading 2024 price data...")
    price_data_2024 = target_creator.load_price_data("2024-01-01", "2024-12-31")
    
    if price_data_2024 is not None and len(price_data_2024) > 0:
        print(f"✅ Loaded {len(price_data_2024)} days of 2024 price data")
        
        # Identify 2024 corrections
        corrections_2024 = target_creator.identify_corrections(price_data_2024)
        print(f"🔍 Found {len(corrections_2024)} correction events in 2024")
        
        if corrections_2024:
            # Create targets
            targets_2024 = target_creator.create_prediction_targets(corrections_2024)
            
            # Show correction events with dates and magnitudes
            print(f"\n📊 2024 CORRECTION EVENTS AND TARGET DATES:")
            print("=" * 50)
            
            for i, correction in enumerate(corrections_2024):
                peak_date = correction['peak_date']
                magnitude = correction['magnitude']
                duration = correction['duration_days']
                
                print(f"\n{i+1}. CORRECTION EVENT")
                print(f"   Peak Date: {peak_date}")
                print(f"   Magnitude: {magnitude:.2%}")
                print(f"   Duration: {duration} days")
                print(f"   Peak Price: ${correction['peak_price']:.2f}")
                print(f"   Trough Price: ${correction['trough_price']:.2f}")
                
                # Find target dates (1-3 days before peak)
                peak_dt = pd.to_datetime(peak_date)
                target_dates = []
                for days_before in range(1, 4):
                    target_date = peak_dt - timedelta(days=days_before)
                    if target_date.weekday() < 5:  # Skip weekends
                        target_dates.append(target_date.strftime('%Y-%m-%d'))
                
                print(f"   Target Dates (prediction days): {target_dates}")
            
            # Test hedging signals on target dates
            print(f"\n🔍 TESTING HEDGING SIGNALS ON TARGET DATES")
            print("=" * 50)
            
            target_mask = targets_2024['target'] == 1
            target_dates_df = targets_2024[target_mask]
            
            hedging_signal_results = []
            
            for _, row in target_dates_df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                magnitude = row['correction_magnitude']
                days_to_correction = row['days_to_correction']
                
                print(f"\n📅 Testing {date_str} (T-{days_to_correction}, {magnitude:.1%} correction ahead)")
                
                # Get hedging signals for this date
                try:
                    daily_results = detector.process_daily_anomalies(date_str)
                    
                    if daily_results and 'signals' in daily_results:
                        signals = daily_results['signals']
                        metrics = daily_results['metrics']
                        
                        signal_source = signals.get('signal_source', 'unknown')
                        direction = signals.get('direction', 'neutral')
                        confidence = signals.get('confidence', 0.0)
                        quality = signals.get('quality', 'low')
                        strength = signals.get('strength', 0.0)
                        
                        print(f"   🎯 Signal: {direction} ({signal_source})")
                        print(f"   💪 Strength: {strength:.3f}")
                        print(f"   🎪 Confidence: {confidence:.3f}")
                        print(f"   ⭐ Quality: {quality}")
                        
                        # If we have hedging-specific data
                        if 'hedging_trend' in signals:
                            print(f"   📈 Hedging Trend: {signals['hedging_trend']:.6f}")
                            print(f"   ⚡ Hedging Acceleration: {signals['hedging_acceleration']:.6f}")
                            print(f"   🏦 Institutional Momentum: {signals['institutional_momentum']:.6f}")
                        
                        hedging_signal_results.append({
                            'date': date_str,
                            'days_to_correction': days_to_correction,
                            'correction_magnitude': magnitude,
                            'signal_direction': direction,
                            'signal_source': signal_source,
                            'confidence': confidence,
                            'quality': quality,
                            'strength': strength
                        })
                        
                    else:
                        print(f"   ❌ No signals available")
                        
                except Exception as e:
                    print(f"   ⚠️  Error processing {date_str}: {e}")
            
            # Summary analysis
            if hedging_signal_results:
                print(f"\n📈 HEDGING SIGNAL PERFORMANCE SUMMARY")
                print("=" * 50)
                
                results_df = pd.DataFrame(hedging_signal_results)
                
                # Signal source distribution
                source_dist = results_df['signal_source'].value_counts()
                print(f"📊 Signal Sources: {source_dist.to_dict()}")
                
                # Quality distribution
                quality_dist = results_df['quality'].value_counts()
                print(f"⭐ Signal Quality: {quality_dist.to_dict()}")
                
                # Average confidence by days before correction
                conf_by_days = results_df.groupby('days_to_correction')['confidence'].mean()
                print(f"🎯 Avg Confidence by Days Before:")
                for days, conf in conf_by_days.items():
                    print(f"   T-{days}: {conf:.3f}")
                
                # Hedging intelligence performance
                hedging_signals = results_df[results_df['signal_source'] == 'hedging_intelligence']
                if len(hedging_signals) > 0:
                    print(f"\n🧠 HEDGING INTELLIGENCE PERFORMANCE:")
                    print(f"   📊 Coverage: {len(hedging_signals)}/{len(results_df)} targets ({len(hedging_signals)/len(results_df)*100:.1f}%)")
                    print(f"   🎯 Avg Confidence: {hedging_signals['confidence'].mean():.3f}")
                    print(f"   ⭐ High Quality Signals: {len(hedging_signals[hedging_signals['quality'] == 'high'])}")
                    print(f"   📉 Bearish Signals: {len(hedging_signals[hedging_signals['signal_direction'] == 'bearish'])}")
                
                # Save results
                output_path = "hedging_signal_test_results_2024.csv"
                results_df.to_csv(output_path, index=False)
                print(f"\n💾 Results saved to {output_path}")
                
                return results_df
                
        else:
            print("❌ No corrections found in 2024 data")
            return None
    else:
        print("❌ Could not load 2024 price data")
        return None

if __name__ == "__main__":
    results = train_and_test_system()
    
    if results is not None:
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"Found {len(results)} target dates in 2024 with hedging signal analysis")
    else:
        print("❌ Analysis failed")