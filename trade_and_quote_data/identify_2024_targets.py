#!/usr/bin/env python3
"""
Identify 2024 correction targets and their magnitudes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from target_creator import CorrectionTargetCreator

def identify_2024_targets():
    """
    Identify correction targets in 2024 data
    """
    print("🎯 IDENTIFYING 2024 CORRECTION TARGETS")
    print("=" * 50)
    
    # Initialize target creator
    target_creator = CorrectionTargetCreator()
    
    # Load 2024 price data
    print("📊 Loading 2024 price data...")
    price_data_2024 = target_creator.load_price_data("2024-01-01", "2024-12-31")
    
    if price_data_2024 is None or len(price_data_2024) == 0:
        print("❌ Could not load 2024 price data")
        return
    
    print(f"✅ Loaded {len(price_data_2024)} days of 2024 price data")
    print(f"📅 Date range: {price_data_2024['date'].min()} to {price_data_2024['date'].max()}")
    
    # Identify corrections
    print("\n🔍 Identifying 4%+ correction events...")
    corrections_2024 = target_creator.identify_corrections(price_data_2024)
    
    if not corrections_2024:
        print("❌ No corrections found in 2024 data")
        return
    
    print(f"✅ Found {len(corrections_2024)} correction events in 2024")
    
    # Create prediction targets
    print("\n🎯 Creating prediction targets...")
    targets_2024 = target_creator.create_prediction_targets(corrections_2024)
    
    # Validate targets
    validation = target_creator.validate_targets(targets_2024)
    
    print(f"✅ Created {targets_2024['target'].sum()} prediction targets")
    print(f"📊 Target ratio: {validation['target_ratio']:.3f}")
    
    # Show detailed correction information
    print(f"\n📈 DETAILED 2024 CORRECTION ANALYSIS")
    print("=" * 50)
    
    for i, correction in enumerate(corrections_2024):
        peak_date = pd.to_datetime(correction['peak_date'])
        trough_date = pd.to_datetime(correction['trough_date'])
        magnitude = correction['magnitude']
        duration = correction['duration_days']
        peak_price = correction['peak_price']
        trough_price = correction['trough_price']
        
        print(f"\n🚨 CORRECTION #{i+1}")
        print(f"   📅 Peak Date: {peak_date.strftime('%Y-%m-%d (%A)')}")
        print(f"   📅 Trough Date: {trough_date.strftime('%Y-%m-%d (%A)')}")
        print(f"   📉 Magnitude: {magnitude:.2%}")
        print(f"   ⏱️  Duration: {duration} days")
        print(f"   💰 Peak Price: ${peak_price:.2f}")
        print(f"   💰 Trough Price: ${trough_price:.2f}")
        print(f"   📊 Price Drop: ${peak_price - trough_price:.2f}")
        
        # Find target dates (1-3 days before peak)
        target_dates = []
        for days_before in range(1, 4):
            target_date = peak_date - timedelta(days=days_before)
            # Only include weekdays
            if target_date.weekday() < 5:
                target_dates.append(target_date.strftime('%Y-%m-%d'))
        
        print(f"   🎯 Target Dates: {target_dates}")
        
        # Severity classification
        if magnitude >= 0.15:
            severity = "🔴 SEVERE"
        elif magnitude >= 0.10:
            severity = "🟡 MAJOR"
        elif magnitude >= 0.06:
            severity = "🟢 MODERATE"
        else:
            severity = "🔵 MINOR"
        
        print(f"   🚦 Severity: {severity}")
    
    # Show all target dates in chronological order
    print(f"\n📅 ALL TARGET DATES (CHRONOLOGICAL ORDER)")
    print("=" * 50)
    
    target_mask = targets_2024['target'] == 1
    target_dates_df = targets_2024[target_mask].copy()
    target_dates_df = target_dates_df.sort_values('date')
    
    for i, (_, row) in enumerate(target_dates_df.iterrows()):
        date_str = row['date'].strftime('%Y-%m-%d (%A)')
        days_to_correction = int(row['days_to_correction'])
        magnitude = row['correction_magnitude']
        correction_date = row['correction_peak_date'].strftime('%Y-%m-%d')
        
        print(f"{i+1:2d}. {date_str} → T-{days_to_correction} → {magnitude:.1%} correction on {correction_date}")
    
    # Export results
    output_file = "2024_correction_targets.csv"
    targets_2024.to_csv(output_file, index=False)
    print(f"\n💾 Full results exported to {output_file}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 30)
    print(f"Total 2024 trading days: {len(targets_2024)}")
    print(f"Total correction events: {len(corrections_2024)}")
    print(f"Total target days: {targets_2024['target'].sum()}")
    print(f"Target density: {targets_2024['target'].sum() / len(targets_2024) * 100:.1f}%")
    
    # Magnitude distribution
    magnitudes = [c['magnitude'] for c in corrections_2024]
    print(f"Average correction magnitude: {np.mean(magnitudes):.1%}")
    print(f"Largest correction: {max(magnitudes):.1%}")
    print(f"Smallest correction: {min(magnitudes):.1%}")
    
    return targets_2024, corrections_2024

if __name__ == "__main__":
    targets, corrections = identify_2024_targets()
    print(f"\n🎉 ANALYSIS COMPLETE!")