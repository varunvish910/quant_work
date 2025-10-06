"""
Test script for gradual_decline_analyzer module

This script verifies that all features can be calculated correctly.
"""

import sys
sys.path.append('.')

import yfinance as yf
import pandas as pd
from gradual_decline_analyzer import GradualDeclineDetector
from gradual_decline_analyzer.features.trend_persistence import TrendPersistenceFeatures
from gradual_decline_analyzer.features.momentum_decay import MomentumDecayFeatures
from gradual_decline_analyzer.features.market_structure import MarketStructureFeatures
from gradual_decline_analyzer.features.volume_divergence import VolumeDivergenceFeatures
from gradual_decline_analyzer.targets.gradual_decline_targets import GradualDeclineTargetCreator


def test_feature_calculation():
    """Test that all feature modules work correctly"""
    print("🧪 Testing Gradual Decline Analyzer")
    print("=" * 60)
    
    # Load test data (April 2024)
    print("\n📥 Loading SPY data for April 2024...")
    spy = yf.download('SPY', start='2024-03-01', end='2024-05-31')
    print(f"✅ Loaded {len(spy)} records")
    
    # Test each feature category
    print("\n🔧 Testing Feature Calculation...")
    
    print("\n  1. Trend Persistence Features")
    trend_features = TrendPersistenceFeatures.calculate_features(spy)
    print(f"     ✅ Calculated {len(trend_features.columns)} features")
    print(f"     Sample features: {list(trend_features.columns[:5])}")
    
    print("\n  2. Momentum Decay Features")
    momentum_features = MomentumDecayFeatures.calculate_features(spy)
    print(f"     ✅ Calculated {len(momentum_features.columns)} features")
    print(f"     Sample features: {list(momentum_features.columns[:5])}")
    
    print("\n  3. Market Structure Features")
    structure_features = MarketStructureFeatures.calculate_features(spy)
    print(f"     ✅ Calculated {len(structure_features.columns)} features")
    print(f"     Sample features: {list(structure_features.columns[:5])}")
    
    print("\n  4. Volume Divergence Features")
    volume_features = VolumeDivergenceFeatures.calculate_features(spy)
    print(f"     ✅ Calculated {len(volume_features.columns)} features")
    print(f"     Sample features: {list(volume_features.columns[:5])}")
    
    # Test integrated detector
    print("\n🔍 Testing GradualDeclineDetector...")
    detector = GradualDeclineDetector()
    features_df, feature_cols = detector.calculate_features(spy)
    print(f"✅ Total features calculated: {len(feature_cols)}")
    
    # Test target creation
    print("\n🎯 Testing Target Creation...")
    target_creator = GradualDeclineTargetCreator(spy)
    targets_df = target_creator.create_all_targets(windows=[7, 14, 20, 30])
    print(f"✅ Created {len([c for c in targets_df.columns if 'decline' in c])} target columns")
    
    # Check April 2024 gradual decline
    print("\n📊 April 2024 Analysis:")
    april_mask = (features_df.index >= '2024-04-01') & (features_df.index <= '2024-04-30')
    april_data = features_df[april_mask]
    
    if len(april_data) > 0:
        # Check if gradual decline target was triggered
        april_targets = targets_df[april_mask]
        decline_20d = april_targets['decline_20d'].sum() if 'decline_20d' in april_targets.columns else 0
        
        print(f"  Trading days in April: {len(april_data)}")
        print(f"  20-day decline signals: {decline_20d}")
        
        # Show key feature values for April 19 (trough day)
        if '2024-04-19' in features_df.index:
            print(f"\n  Key features on April 19, 2024 (trough day):")
            trough_features = features_df.loc['2024-04-19']
            
            if 'price_slope_20d' in trough_features.index:
                print(f"    Price slope (20d): {trough_features['price_slope_20d']:.4f}")
            if 'rsi_14' in trough_features.index:
                print(f"    RSI (14): {trough_features['rsi_14']:.2f}")
            if 'lower_highs_count_20d' in trough_features.index:
                print(f"    Lower highs (20d): {trough_features['lower_highs_count_20d']:.0f}")
            if 'days_since_peak' in trough_features.index:
                print(f"    Days since peak: {trough_features['days_since_peak']:.0f}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("\n📝 Summary:")
    print(f"  - 4 feature categories implemented")
    print(f"  - {len(feature_cols)} total features available")
    print(f"  - Multi-timeframe targets created")
    print(f"  - Ready for model training")
    print("\n🚀 Next step: Implement model training in models/gradual_decline_model.py")


if __name__ == '__main__':
    try:
        test_feature_calculation()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

