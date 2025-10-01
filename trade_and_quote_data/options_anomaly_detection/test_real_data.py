#!/usr/bin/env python3
"""
Test the complete pipeline with real July 16, 2024 data
"""

from feature_extractor import HistoricalFeatureExtractor

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

if __name__ == "__main__":
    test_july_16_features()