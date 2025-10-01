#!/usr/bin/env python3
"""
Test the target_creator to see if it correctly identifies July 2024 correction
"""

import sys
sys.path.append('.')
from target_creator import CorrectionTargetCreator

def test_july_2024_correction():
    """Test if we can identify the July 2024 correction"""
    
    print("🔍 Testing target_creator with July 2024 correction...")
    
    # Create target creator
    creator = CorrectionTargetCreator(correction_threshold=0.04, lookback_days=20)
    
    # Load data covering July 2024 correction period
    price_data = creator.load_price_data('2024-06-01', '2024-08-31')
    print(f"📊 Loaded {len(price_data)} days of price data")
    
    # Quick price analysis
    max_price = price_data['underlying_price'].max()
    min_price = price_data['underlying_price'].min()
    max_idx = price_data['underlying_price'].idxmax()
    min_idx = price_data['underlying_price'].idxmin()
    
    print(f"📈 Price range: ${min_price:.2f} - ${max_price:.2f}")
    print(f"📅 Peak: {price_data.loc[max_idx, 'date']} at ${max_price:.2f}")
    print(f"📅 Trough: {price_data.loc[min_idx, 'date']} at ${min_price:.2f}")
    
    if max_idx < min_idx:
        max_drawdown = (max_price - min_price) / max_price
        print(f"📉 Max drawdown from peak: {max_drawdown:.1%}")
    
    # Identify corrections
    corrections = creator.identify_corrections(price_data)
    print(f"\n🎯 Found {len(corrections)} corrections:")
    
    if corrections:
        for i, c in enumerate(corrections):
            print(f"  Correction {i+1}:")
            print(f"    Peak: {c['peak_date']} at ${c['peak_price']:.2f}")
            print(f"    Trough: {c['trough_date']} at ${c['trough_price']:.2f}")
            print(f"    Magnitude: {c['magnitude']:.1%}")
            print(f"    Duration: {c['duration_days']} days")
            
        # Create prediction targets
        targets = creator.create_prediction_targets(corrections)
        print(f"\n🎯 Created {targets['target'].sum()} prediction targets")
        
        # Show target dates
        target_dates = targets[targets['target'] == 1]['date'].tolist()
        print("📅 Target dates (1-3 days before corrections):")
        for date in target_dates:
            print(f"    {date}")
            
        return True
    else:
        print("❌ No corrections found - algorithm may need adjustment")
        return False

if __name__ == "__main__":
    success = test_july_2024_correction()
    print(f"\n{'✅ Test passed' if success else '❌ Test failed'}")