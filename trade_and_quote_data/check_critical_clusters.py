#!/usr/bin/env python3
"""
Check if current model caught the 4 critical clusters in 2024
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from core.data_loader import DataLoader
from core.features import FeatureEngine

print("=" * 80)
print("🎯 CHECKING MODEL PERFORMANCE ON 4 CRITICAL CLUSTERS")
print("=" * 80)
print()

# Define the 4 critical clusters
critical_clusters = {
    'April Pullback': {
        'signal_dates': ['2024-04-08', '2024-04-09', '2024-04-10', '2024-04-11', '2024-04-12'],
        'max_drawdown': -5.35,
        'description': 'Tech rotation, rising bond yields'
    },
    'August Crash': {
        'signal_dates': ['2024-07-31', '2024-08-01', '2024-08-02'],
        'max_drawdown': -8.41,
        'description': 'Yen carry trade unwind (MOST CRITICAL)'
    },
    'September Pullback': {
        'signal_dates': ['2024-08-30'],
        'max_drawdown': -4.34,
        'description': 'Labor market concerns'
    },
    'December Pullback': {
        'signal_dates': ['2024-12-11', '2024-12-12', '2024-12-16'],
        'max_drawdown': -3.57,
        'description': 'Hawkish Fed'
    }
}

# Load 2024 data
data_loader = DataLoader(start_date='2024-01-01', end_date='2024-12-31')
spy_data = data_loader.load_spy_data()
sector_data = data_loader.load_sector_data()
currency_data = data_loader.load_currency_data()
volatility_data = data_loader.load_volatility_data()

# Load options features
options_features_path = 'data/options_chains/enhanced_options_features.parquet'
if os.path.exists(options_features_path):
    options_features = pd.read_parquet(options_features_path)
    options_features['date'] = pd.to_datetime(options_features['date'])
    options_features = options_features.set_index('date')
else:
    options_features = None

# Create features
feature_engine = FeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])
spy_with_features = feature_engine.calculate_features(
    spy_data=spy_data,
    sector_data=sector_data,
    currency_data=currency_data,
    volatility_data=volatility_data
)

# Merge options features
if options_features is not None:
    existing_cols = set(spy_with_features.columns)
    options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]
    if options_cols_to_add:
        spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')

# Load regularized model
model = lgb.Booster(model_file='models/trained/early_warning_regularized.txt')
with open('models/trained/early_warning_regularized_config.json', 'r') as f:
    config = json.load(f)

# Make predictions
feature_cols = config['features']
X = spy_with_features[feature_cols]
predictions = model.predict(X, num_iteration=model.best_iteration)

# Create results DataFrame
results = pd.DataFrame({
    'date': spy_with_features.index,
    'spy_close': spy_with_features['Close'],
    'probability': predictions
})

# Check each critical cluster
print("CRITICAL CLUSTER ANALYSIS")
print("=" * 80)
print()

total_clusters = len(critical_clusters)
caught_clusters = 0

for cluster_name, cluster_info in critical_clusters.items():
    print(f"📊 {cluster_name}")
    print(f"   Description: {cluster_info['description']}")
    print(f"   Max Drawdown: {cluster_info['max_drawdown']:.2f}%")
    print(f"   Signal Window: {cluster_info['signal_dates'][0]} to {cluster_info['signal_dates'][-1]}")
    print()
    
    # Check predictions for these dates
    cluster_dates = pd.to_datetime(cluster_info['signal_dates'])
    cluster_predictions = results[results['date'].isin(cluster_dates)]
    
    if len(cluster_predictions) > 0:
        max_prob = cluster_predictions['probability'].max()
        max_date = cluster_predictions.loc[cluster_predictions['probability'].idxmax(), 'date']
        avg_prob = cluster_predictions['probability'].mean()
        
        # Consider caught if max probability >= 60%
        caught = max_prob >= 0.60
        if caught:
            caught_clusters += 1
            status = "✅ CAUGHT"
        else:
            status = "❌ MISSED"
        
        print(f"   {status}")
        print(f"   Max Probability: {max_prob*100:.1f}% on {max_date.strftime('%Y-%m-%d')}")
        print(f"   Avg Probability: {avg_prob*100:.1f}%")
        
        # Show all predictions in window
        print(f"   Daily Predictions:")
        for _, row in cluster_predictions.iterrows():
            print(f"      {row['date'].strftime('%Y-%m-%d')}: {row['probability']*100:.1f}%")
    else:
        print(f"   ❌ NO DATA for these dates")
    
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"Critical Clusters Caught: {caught_clusters}/{total_clusters} ({caught_clusters/total_clusters*100:.0f}%)")
print()

if caught_clusters >= 3:
    print("✅ GOOD: Model caught most critical events")
elif caught_clusters >= 2:
    print("⚠️  FAIR: Model caught some critical events, needs improvement")
else:
    print("❌ POOR: Model missed most critical events, major improvements needed")
print()
print("Next: Implement improvements to catch all 4 clusters")
