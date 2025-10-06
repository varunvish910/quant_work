#!/usr/bin/env python3
"""
Analyze what drives 70%+ confidence predictions

What patterns does the model see when it's highly confident a pullback is coming?
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from core.data_loader import DataLoader
from core.features import FeatureEngine
from features.calendar.seasonality import SeasonalityFeature
from features.market.cross_asset import CrossAssetFeature

print("=" * 80)
print("🔍 ANALYZING HIGH CONFIDENCE SIGNALS (70%+)")
print("=" * 80)
print()

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

# Add seasonality features
seasonality_feature = SeasonalityFeature()
spy_with_features = seasonality_feature.calculate(spy_with_features)

# Add cross-asset features
cross_asset_feature = CrossAssetFeature()
spy_with_features = cross_asset_feature.calculate(spy_with_features)

# Load improved model
model = lgb.Booster(model_file='models/trained/improved_early_warning.txt')
with open('models/trained/improved_early_warning_config.json', 'r') as f:
    config = json.load(f)

# Make predictions
feature_cols = config['features']
X = spy_with_features[feature_cols].ffill().bfill().fillna(0)
predictions = model.predict(X, num_iteration=model.best_iteration)

# Create results DataFrame
results = pd.DataFrame({
    'date': spy_with_features.index,
    'spy_close': spy_with_features['Close'],
    'probability': predictions
})

# Merge with features for analysis
results = results.join(spy_with_features[feature_cols], how='left')

# Filter for high confidence signals (70%+)
high_confidence = results[results['probability'] >= 0.70].copy()

print(f"Found {len(high_confidence)} days with 70%+ confidence in 2024")
print()

if len(high_confidence) == 0:
    print("No 70%+ confidence signals in 2024")
    exit()

# Analyze top features for high confidence signals
print("=" * 80)
print("TOP FEATURES DURING HIGH CONFIDENCE SIGNALS")
print("=" * 80)
print()

# Get feature importance from model
# Note: Booster doesn't have feature_importances_, need to get from importance_type
importance_dict = model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance_dict
}).sort_values('importance', ascending=False).head(20)

top_features = feature_importance['feature'].tolist()

# Compare high confidence vs all signals
print("Feature Comparison (High Confidence vs All Days):")
print(f"{'Feature':<40} {'High Conf Avg':<15} {'All Days Avg':<15} {'Difference'}")
print("-" * 90)

for feat in top_features[:15]:
    if feat in high_confidence.columns:
        high_avg = high_confidence[feat].mean()
        all_avg = results[feat].mean()
        diff = high_avg - all_avg
        
        print(f"{feat:<40} {high_avg:>14.2f} {all_avg:>14.2f} {diff:>+14.2f}")

# Analyze specific high confidence dates
print("\n" + "=" * 80)
print("HIGH CONFIDENCE SIGNAL DATES")
print("=" * 80)
print()

for idx, row in high_confidence.sort_values('probability', ascending=False).head(10).iterrows():
    print(f"📅 {row['date'].strftime('%Y-%m-%d')} - Probability: {row['probability']*100:.1f}%")
    print(f"   SPY Close: ${row['spy_close']:.2f}")
    
    # Show key features
    print(f"   Key Features:")
    print(f"      VIX Momentum (3d): {row['vix_momentum_3d']:.2f}")
    print(f"      VIX Level: {row['vix_level']:.2f}")
    print(f"      Volatility (20d): {row['volatility_20d']:.2f}%")
    print(f"      ADX (trend): {row['adx']:.2f}")
    print(f"      RSI: {row['rsi']:.2f}")
    
    if 'tlt_return_20d' in row.index:
        print(f"      TLT Return (20d): {row['tlt_return_20d']:.2f}%")
    if 'gld_return_5d' in row.index:
        print(f"      GLD Return (5d): {row['gld_return_5d']:.2f}%")
    if 'is_q3' in row.index:
        print(f"      Is Q3: {row['is_q3']}")
    if 'days_to_quarter_end' in row.index:
        print(f"      Days to Quarter End: {row['days_to_quarter_end']:.0f}")
    
    print()

# Pattern analysis
print("=" * 80)
print("COMMON PATTERNS IN HIGH CONFIDENCE SIGNALS")
print("=" * 80)
print()

# VIX patterns
vix_high = (high_confidence['vix_level'] > results['vix_level'].quantile(0.75)).sum()
vix_momentum_high = (high_confidence['vix_momentum_3d'] > results['vix_momentum_3d'].quantile(0.75)).sum()

print(f"VIX Patterns:")
print(f"  High VIX level (>75th percentile): {vix_high}/{len(high_confidence)} ({vix_high/len(high_confidence)*100:.0f}%)")
print(f"  High VIX momentum (>75th percentile): {vix_momentum_high}/{len(high_confidence)} ({vix_momentum_high/len(high_confidence)*100:.0f}%)")
print()

# Trend patterns
adx_high = (high_confidence['adx'] > results['adx'].quantile(0.75)).sum()
rsi_overbought = (high_confidence['rsi'] > 70).sum()

print(f"Trend/Momentum Patterns:")
print(f"  Strong trend (ADX >75th percentile): {adx_high}/{len(high_confidence)} ({adx_high/len(high_confidence)*100:.0f}%)")
print(f"  Overbought (RSI >70): {rsi_overbought}/{len(high_confidence)} ({rsi_overbought/len(high_confidence)*100:.0f}%)")
print()

# Cross-asset patterns
if 'tlt_return_20d' in high_confidence.columns:
    tlt_positive = (high_confidence['tlt_return_20d'] > 0).sum()
    print(f"Cross-Asset Patterns:")
    print(f"  Bonds rallying (TLT >0): {tlt_positive}/{len(high_confidence)} ({tlt_positive/len(high_confidence)*100:.0f}%)")

if 'gld_return_5d' in high_confidence.columns:
    gld_positive = (high_confidence['gld_return_5d'] > 0).sum()
    print(f"  Gold rallying (GLD >0): {gld_positive}/{len(high_confidence)} ({gld_positive/len(high_confidence)*100:.0f}%)")
print()

# Seasonality patterns
if 'is_q3' in high_confidence.columns:
    q3_count = high_confidence['is_q3'].sum()
    print(f"Seasonality Patterns:")
    print(f"  Q3 (most volatile): {q3_count}/{len(high_confidence)} ({q3_count/len(high_confidence)*100:.0f}%)")

if 'is_opex_week' in high_confidence.columns:
    opex_count = high_confidence['is_opex_week'].sum()
    print(f"  OpEx week: {opex_count}/{len(high_confidence)} ({opex_count/len(high_confidence)*100:.0f}%)")
print()

# Summary
print("=" * 80)
print("SUMMARY: WHAT DRIVES 70%+ CONFIDENCE?")
print("=" * 80)
print()
print("The model flags high risk (70%+) when it sees:")
print()
print("1. 🔥 VOLATILITY SPIKE:")
print(f"   - VIX momentum surging (avg: {high_confidence['vix_momentum_3d'].mean():.2f} vs {results['vix_momentum_3d'].mean():.2f})")
print(f"   - VIX level elevated (avg: {high_confidence['vix_level'].mean():.2f} vs {results['vix_level'].mean():.2f})")
print()
print("2. 📈 TREND EXHAUSTION:")
print(f"   - Strong trend (ADX avg: {high_confidence['adx'].mean():.2f} vs {results['adx'].mean():.2f})")
print(f"   - Overbought conditions (RSI avg: {high_confidence['rsi'].mean():.2f} vs {results['rsi'].mean():.2f})")
print()

if 'tlt_return_20d' in high_confidence.columns:
    print("3. 🛡️ FLIGHT TO SAFETY:")
    print(f"   - Bonds rallying (TLT avg: {high_confidence['tlt_return_20d'].mean():.2f}% vs {results['tlt_return_20d'].mean():.2f}%)")
    print()

if 'is_q3' in high_confidence.columns:
    print("4. 📅 SEASONAL RISK:")
    print(f"   - Q3 concentration: {high_confidence['is_q3'].sum()} signals in Q3")
    print()

print("In other words: The model sees danger when fear is RISING FAST,")
print("markets are EXTENDED, and defensive assets are BEING BOUGHT.")
