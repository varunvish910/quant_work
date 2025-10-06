#!/usr/bin/env python3
"""
Analyze 2024 Drawdowns - Did the model predict them?

Key dates to check:
1. April 1, 2024 (before April drawdown)
2. Before July 16, 2024 (July crash)
3. Before October 17, 2024 (October correction)
4. Before December 6, 2024 (December selloff)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.features import FeatureEngine

print("=" * 80)
print("📊 ANALYZING 2024 DRAWDOWNS - DID WE PREDICT THEM?")
print("=" * 80)
print()

# ============================================================================
# IDENTIFY 2024 DRAWDOWNS
# ============================================================================
print("📊 Loading 2024 SPY data to identify drawdowns...")
spy_2024 = yf.download('SPY', start='2024-01-01', end='2024-12-31', progress=False)
if isinstance(spy_2024.columns, pd.MultiIndex):
    spy_2024.columns = spy_2024.columns.get_level_values(0)

print(f"✅ Loaded {len(spy_2024)} days of 2024 data")
print()

# Calculate rolling max and drawdown
spy_2024['rolling_max'] = spy_2024['Close'].cummax()
spy_2024['drawdown'] = (spy_2024['Close'] / spy_2024['rolling_max'] - 1) * 100

print("=" * 80)
print("📉 2024 MAJOR DRAWDOWNS")
print("=" * 80)
print()

# Find significant drawdowns (>3%)
significant_dd = spy_2024[spy_2024['drawdown'] < -3.0]
print(f"Days with >3% drawdown: {len(significant_dd)}")
print()

# Key periods
april_data = spy_2024.loc['2024-04-01':'2024-04-30']
july_data = spy_2024.loc['2024-07-01':'2024-08-15']
oct_data = spy_2024.loc['2024-10-01':'2024-10-31']
dec_data = spy_2024.loc['2024-12-01':'2024-12-31']

print("April 2024:")
print(f"  Start: ${april_data['Close'].iloc[0]:.2f} on {april_data.index[0].date()}")
print(f"  Low:   ${april_data['Close'].min():.2f} on {april_data['Close'].idxmin().date()}")
print(f"  Max DD: {april_data['drawdown'].min():.2f}%")
print()

print("July-August 2024:")
print(f"  Start: ${july_data['Close'].iloc[0]:.2f} on {july_data.index[0].date()}")
print(f"  Low:   ${july_data['Close'].min():.2f} on {july_data['Close'].idxmin().date()}")
print(f"  Max DD: {july_data['drawdown'].min():.2f}%")
print()

print("October 2024:")
print(f"  Start: ${oct_data['Close'].iloc[0]:.2f} on {oct_data.index[0].date()}")
print(f"  Low:   ${oct_data['Close'].min():.2f} on {oct_data['Close'].idxmin().date()}")
print(f"  Max DD: {oct_data['drawdown'].min():.2f}%")
print()

print("December 2024:")
print(f"  Start: ${dec_data['Close'].iloc[0]:.2f} on {dec_data.index[0].date()}")
print(f"  Low:   ${dec_data['Close'].min():.2f} on {dec_data['Close'].idxmin().date()}")
print(f"  Max DD: {dec_data['drawdown'].min():.2f}%")
print()

# ============================================================================
# LOAD DATA AND GENERATE PREDICTIONS FOR 2024
# ============================================================================
print("=" * 80)
print("📊 GENERATING MODEL PREDICTIONS FOR 2024")
print("=" * 80)
print()

loader = DataLoader(start_date='2024-01-01', end_date='2024-12-31')

spy_data = loader.load_spy_data()
sector_data = loader.load_sector_data()
currency_data = loader.load_currency_data()
volatility_data = loader.load_volatility_data()

print(f"✅ Loaded {len(spy_data)} days of data")

# Create features
print("\n📊 Creating features...")
feature_engine = FeatureEngine(feature_sets=['baseline', 'currency', 'volatility'])

spy_with_features = feature_engine.calculate_features(
    spy_data=spy_data,
    sector_data=sector_data,
    currency_data=currency_data,
    volatility_data=volatility_data
)

# Load options features
options_features = pd.read_parquet('data/options_chains/enhanced_options_features.parquet')
options_features = options_features.set_index('date')

# Merge
existing_cols = set(spy_with_features.columns)
options_cols_to_add = [col for col in options_features.columns if col not in existing_cols]

if options_cols_to_add:
    spy_with_features = spy_with_features.join(options_features[options_cols_to_add], how='left')

all_features = feature_engine.feature_columns + options_cols_to_add

print(f"✅ Created {len(all_features)} features")

# Load models
print("\n📊 Loading trained models...")
model_crash = lgb.Booster(model_file='models/trained/crash_detection.txt')
model_gradual = lgb.Booster(model_file='models/trained/gradual_pullback.txt')
model_time = lgb.Booster(model_file='models/trained/time_correction.txt')
print("✅ Loaded all 3 models")

# Generate predictions
print("\n🔄 Running predictions...")
X = spy_with_features[all_features].fillna(0)

pred_crash = model_crash.predict(X, num_iteration=model_crash.best_iteration)
pred_gradual = model_gradual.predict(X, num_iteration=model_gradual.best_iteration)
pred_time = model_time.predict(X, num_iteration=model_time.best_iteration)

results = pd.DataFrame({
    'date': spy_with_features.index,
    'spy_close': spy_with_features['Close'],
    'prob_crash': pred_crash,
    'prob_gradual_pullback': pred_gradual,
    'prob_time_correction': pred_time
})

print("✅ Predictions complete")

# ============================================================================
# ANALYZE KEY DATES
# ============================================================================
print("\n" + "=" * 80)
print("🎯 DID WE PREDICT THE DRAWDOWNS?")
print("=" * 80)
print()

# Define key dates to check (BEFORE the drawdowns)
key_dates = {
    'April 1, 2024 (before April drawdown)': '2024-04-01',
    'July 15, 2024 (before July crash)': '2024-07-15',
    'October 16, 2024 (before October correction)': '2024-10-16',
    'December 5, 2024 (before December selloff)': '2024-12-05'
}

for description, date_str in key_dates.items():
    print("=" * 80)
    print(f"📅 {description}")
    print("=" * 80)
    
    try:
        # Get prediction for this date
        date = pd.to_datetime(date_str)
        
        # Find the closest date in our results
        if date in results['date'].values:
            row = results[results['date'] == date].iloc[0]
        else:
            # Find nearest date
            idx = (results['date'] - date).abs().idxmin()
            row = results.iloc[idx]
            print(f"⚠️  Exact date not found, using nearest: {row['date'].date()}")
        
        print(f"SPY Close: ${row['spy_close']:.2f}")
        print()
        print(f"Model Predictions:")
        print(f"  💥 Crash Risk:           {row['prob_crash']*100:6.2f}% {'🚨 HIGH RISK' if row['prob_crash'] > 0.5 else '⚠️  ELEVATED' if row['prob_crash'] > 0.2 else '✅ LOW'}")
        print(f"  📉 Gradual Pullback:     {row['prob_gradual_pullback']*100:6.2f}% {'🚨 HIGH RISK' if row['prob_gradual_pullback'] > 0.5 else '⚠️  ELEVATED' if row['prob_gradual_pullback'] > 0.2 else '✅ LOW'}")
        print(f"  ⏸️  Time Correction:      {row['prob_time_correction']*100:6.2f}% {'🚨 HIGH RISK' if row['prob_time_correction'] > 0.5 else '⚠️  ELEVATED' if row['prob_time_correction'] > 0.2 else '✅ LOW'}")
        print()
        
        # Show what happened next (next 10 days)
        next_date = row['date'] + pd.Timedelta(days=10)
        future_data = spy_2024.loc[row['date']:next_date]
        
        if len(future_data) > 1:
            future_low = future_data['Close'].min()
            future_dd = (future_low / row['spy_close'] - 1) * 100
            print(f"What happened next (10 days):")
            print(f"  Lowest price: ${future_low:.2f}")
            print(f"  Max drawdown: {future_dd:.2f}%")
            
            if future_dd < -5:
                verdict = "🎯 CORRECT - Major crash occurred!"
            elif future_dd < -3:
                verdict = "✅ CORRECT - Significant pullback occurred"
            elif future_dd < -1:
                verdict = "⚠️  PARTIAL - Minor pullback occurred"
            else:
                verdict = "❌ FALSE ALARM - No significant drawdown"
            
            print(f"  Verdict: {verdict}")
        
    except Exception as e:
        print(f"❌ Error analyzing {date_str}: {e}")
    
    print()

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("=" * 80)
print("📊 2024 PREDICTION SUMMARY")
print("=" * 80)
print()

summary_data = []
for description, date_str in key_dates.items():
    try:
        date = pd.to_datetime(date_str)
        if date in results['date'].values:
            row = results[results['date'] == date].iloc[0]
        else:
            idx = (results['date'] - date).abs().idxmin()
            row = results.iloc[idx]
        
        summary_data.append({
            'Date': row['date'].date(),
            'SPY': f"${row['spy_close']:.2f}",
            'Crash %': f"{row['prob_crash']*100:.1f}%",
            'Gradual %': f"{row['prob_gradual_pullback']*100:.1f}%",
            'Time Corr %': f"{row['prob_time_correction']*100:.1f}%",
            'Max Signal': max(row['prob_crash'], row['prob_gradual_pullback'], row['prob_time_correction']) * 100
        })
    except:
        pass

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print()

# Save results
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
results.to_csv(output_dir / '2024_predictions.csv', index=False)
print(f"💾 Saved full 2024 predictions to: output/2024_predictions.csv")
