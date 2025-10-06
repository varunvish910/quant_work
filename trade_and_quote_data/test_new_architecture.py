#!/usr/bin/env python3
"""
Test New Architecture

This script tests the new refactored architecture to ensure it works.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("🧪 TESTING NEW ARCHITECTURE")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Test 1: Load data using existing data loader
print("\n[1/5] Testing Data Loader...")
try:
    from core.data_loader import DataLoader
    
    loader = DataLoader(start_date='2023-01-01', end_date='2024-10-01')
    data = loader.load_all_data(
        include_sectors=True,
        include_currency=False,
        include_volatility=False
    )
    
    print(f"✅ Data loaded: {len(data['spy'])} SPY records")
    print(f"✅ Sectors loaded: {len(data.get('sectors', {}))} sectors")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Test Unified Feature Engine
print("\n[2/5] Testing Unified Feature Engine...")
try:
    from engines.unified_engine import UnifiedFeatureEngine
    
    engine = UnifiedFeatureEngine(feature_sets=['baseline'])
    features_df = engine.calculate_all(
        spy_data=data['spy'],
        sector_data=data.get('sectors')
    )
    
    print(f"✅ Features calculated: {len(features_df)} rows")
    print(f"✅ Feature count: {len(engine.get_feature_names())} features")
    print(f"✅ Sample features: {engine.get_feature_names()[:5]}")
except Exception as e:
    print(f"❌ Feature engine failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Test Early Warning Target
print("\n[3/5] Testing Early Warning Target...")
try:
    from targets.early_warning import EarlyWarningTarget
    
    target = EarlyWarningTarget(drawdown_threshold=0.05)
    target_df = target.create(data['spy'])
    
    print(f"✅ Target created: {len(target_df)} rows")
    print(f"✅ Target column: {target.get_target_column()}")
    stats = target.get_target_stats()
    print(f"✅ Positive rate: {stats['positive_rate']:.1%}")
except Exception as e:
    print(f"❌ Target creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Combine features and targets
print("\n[4/5] Testing Feature + Target Integration...")
try:
    # Merge features with targets
    combined_df = features_df.join(
        target_df[target.get_target_column()],
        how='inner'
    )
    
    print(f"✅ Combined data: {len(combined_df)} rows")
    print(f"✅ Total columns: {len(combined_df.columns)}")
    print(f"✅ Features: {len(engine.get_feature_names())}")
    print(f"✅ Has target: {target.get_target_column() in combined_df.columns}")
except Exception as e:
    print(f"❌ Integration failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Quick model test
print("\n[5/5] Testing Model Training...")
try:
    from core.models import EarlyWarningModel
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    feature_cols = engine.get_feature_names()
    X = combined_df[feature_cols]
    y = combined_df[target.get_target_column()]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Train
    model = EarlyWarningModel(model_type='rf')
    model.fit(X_train, y_train, feature_cols)
    
    # Predict
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    
    print(f"✅ Model trained: {model.model_type}")
    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Test accuracy: {accuracy:.1%}")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"✅ Top 3 features:")
    for idx, row in importance.head(3).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
        
except Exception as e:
    print(f"❌ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("🎉 ALL TESTS PASSED!")
print("=" * 80)
print("\n✅ New architecture is working!")
print("✅ Compatible with existing code")
print("✅ Ready for incremental feature migration")
print("\n📝 Next steps:")
print("   1. Migrate individual features to new structure")
print("   2. Add new features using the new base classes")
print("   3. Refactor data management")
print("   4. Complete documentation")
print("=" * 80)
