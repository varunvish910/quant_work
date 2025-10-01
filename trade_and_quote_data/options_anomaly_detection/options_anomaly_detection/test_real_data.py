#!/usr/bin/env python3
"""
Test the complete pipeline with real July 16, 2024 data
"""

from feature_extractor import HistoricalFeatureExtractor
from target_creator import CorrectionTargetCreator

def test_july_16_features():
    """Test feature extraction for July 16, 2024 correction"""
    
    print("🧪 Testing feature extraction with July 16, 2024 data...")
    
    extractor = HistoricalFeatureExtractor()
    
    # Test with July 16, 2024 (the correction peak)
    features = extractor.extract_all_features('20240716')
    
    if features:
        print('✅ Successfully extracted features for July 16, 2024:')
        print(f'   SPY Price: ${features["spy_price"]:.2f}')
        print(f'   Total Contracts: {features["total_contracts"]:,}')
        print(f'   Downward Composite: {features["downward_composite"]}/6')
        print(f'   Big Move Composite: {features["bigmove_composite"]}/5')
        
        print('\n📊 Key downward signals:')
        for key in ['downward_distribution_score', 'downward_put_accumulation', 'downward_call_exit_signal']:
            if key in features:
                print(f'   {key}: {features[key]:.2f}')
        
        print('\n🎯 Key big move signals:')
        for key in ['bigmove_tension_index', 'bigmove_asymmetry_score', 'bigmove_coiling_pattern']:
            if key in features:
                print(f'   {key}: {features[key]:.2f}')
                
        return features
    else:
        print('❌ Failed to extract features')
        return None

def test_july_correction_period():
    """Test the days around July 16 correction"""
    
    print("\n🔍 Testing correction period July 15-17, 2024...")
    
    extractor = HistoricalFeatureExtractor()
    
    # Test July 15, 16, 17 (around the correction)
    dates = ['20240715', '20240716', '20240717']
    
    for date in dates:
        features = extractor.extract_all_features(date)
        if features:
            print(f"\n📅 {date}: SPY ${features['spy_price']:.2f}")
            print(f"   Downward signals: {features['downward_composite']}/6")
            print(f"   Big move signals: {features['bigmove_composite']}/5")
        else:
            print(f"❌ No data for {date}")

def test_complete_pipeline():
    """Test the complete prediction pipeline"""
    
    print("\n🚀 Testing complete prediction pipeline...")
    
    try:
        from correction_classifier import CorrectionPredictor
        
        predictor = CorrectionPredictor()
        
        # Test with a small date range around July correction
        print("📊 Testing with July 2024 correction period...")
        results = predictor.run_full_analysis("2024-07-01", "2024-07-31")
        
        if results:
            print("✅ Pipeline test successful!")
            dataset = results['dataset']
            print(f"   Dataset: {len(dataset)} days")
            print(f"   Positive targets: {dataset['target'].sum()}")
            
            # Check if July 16 is marked as a target
            july_16_mask = dataset['date_key'] == '2024-07-16'
            if july_16_mask.any():
                july_16_data = dataset[july_16_mask].iloc[0]
                print(f"   July 16 target: {july_16_data['target']}")
                print(f"   July 16 downward signals: {july_16_data.get('downward_composite', 'N/A')}")
        else:
            print("❌ Pipeline test failed")
            
    except ImportError as e:
        print(f"❌ Cannot import correction_classifier: {e}")

if __name__ == "__main__":
    
    # Test 1: July 16 feature extraction
    features = test_july_16_features()
    
    # Test 2: Correction period
    test_july_correction_period()
    
    # Test 3: Complete pipeline
    test_complete_pipeline()
    
    print("\n🎯 Tests complete!")