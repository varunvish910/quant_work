#!/usr/bin/env python3
"""
Analyze April 2024 to understand why it wasn't caught by the gradual pullback model
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("📊 ANALYZING APRIL 2024 PULLBACK")
print("=" * 80)
print()

# Load SPY data for March-May 2024
spy = yf.download('SPY', start='2024-03-01', end='2024-06-01', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print("📊 SPY Price Action (March-May 2024):")
print(spy[['Close', 'High', 'Low']].head(10))
print()

# Find the peak and trough
peak_date = spy['Close'].idxmax()
peak_price = spy['Close'].max()
trough_date = spy['Close'].idxmin()
trough_price = spy['Close'].min()

print(f"📈 Peak: {peak_date.date()} at ${peak_price:.2f}")
print(f"📉 Trough: {trough_date.date()} at ${trough_price:.2f}")
print()

# Calculate decline
decline_pct = (trough_price / peak_price - 1) * 100
days_between = (trough_date - peak_date).days

print(f"💥 Decline: {decline_pct:.2f}% over {days_between} days")
print()

# Check if it meets our gradual pullback criteria
print("=" * 80)
print("GRADUAL PULLBACK CRITERIA CHECK")
print("=" * 80)
print()

# Criteria 1: 4%+ decline
meets_decline = abs(decline_pct) >= 4.0
print(f"1. Decline >= 4%: {abs(decline_pct):.2f}% - {'✅ YES' if meets_decline else '❌ NO'}")

# Criteria 2: Timeframe 15-30 days
meets_timeframe = 15 <= days_between <= 30
print(f"2. Timeframe 15-30 days: {days_between} days - {'✅ YES' if meets_timeframe else '❌ NO'}")

# Criteria 3: Low daily volatility (< 2%)
if peak_date < trough_date:
    window = spy.loc[peak_date:trough_date]
else:
    window = spy.loc[trough_date:peak_date]

daily_returns = window['Close'].pct_change().dropna()
daily_volatility = daily_returns.std()
meets_volatility = daily_volatility <= 0.02

print(f"3. Daily volatility < 2%: {daily_volatility*100:.2f}% - {'✅ YES' if meets_volatility else '❌ NO'}")

# Criteria 4: Negative slope
if len(window) > 1:
    x = np.arange(len(window))
    y = window['Close'].values
    slope = np.polyfit(x, y, 1)[0]
    meets_slope = slope < 0
    print(f"4. Negative slope: {slope:.2f} - {'✅ YES' if meets_slope else '❌ NO'}")
else:
    meets_slope = False
    print(f"4. Negative slope: N/A - ❌ NO")

print()
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

if meets_decline and meets_timeframe and meets_volatility and meets_slope:
    print("✅ April 2024 SHOULD have been caught by the model!")
    print("   This might be a model prediction error.")
else:
    print("❌ April 2024 did NOT meet all criteria:")
    if not meets_decline:
        print(f"   - Decline was only {abs(decline_pct):.2f}% (need 4%+)")
    if not meets_timeframe:
        print(f"   - Timeframe was {days_between} days (need 15-30 days)")
    if not meets_volatility:
        print(f"   - Daily volatility was {daily_volatility*100:.2f}% (need < 2%)")
    if not meets_slope:
        print("   - Slope was not consistently negative")

print()

# Show daily price action
print("=" * 80)
print("DAILY PRICE ACTION (April 2024)")
print("=" * 80)
print()

april_data = spy.loc['2024-04-01':'2024-04-30']
print(april_data[['Close', 'High', 'Low', 'Volume']])
print()

# Calculate rolling statistics
april_data['daily_return'] = april_data['Close'].pct_change()
april_data['cumulative_return'] = (1 + april_data['daily_return']).cumprod() - 1

print("=" * 80)
print("APRIL 2024 STATISTICS")
print("=" * 80)
print()
print(f"Start price: ${april_data['Close'].iloc[0]:.2f}")
print(f"End price: ${april_data['Close'].iloc[-1]:.2f}")
print(f"Total return: {(april_data['Close'].iloc[-1] / april_data['Close'].iloc[0] - 1) * 100:.2f}%")
print(f"Max drawdown: {(april_data['Close'].min() / april_data['Close'].max() - 1) * 100:.2f}%")
print(f"Avg daily return: {april_data['daily_return'].mean() * 100:.2f}%")
print(f"Daily volatility: {april_data['daily_return'].std() * 100:.2f}%")
print()

# Check what the model predicted
print("=" * 80)
print("MODEL PREDICTIONS")
print("=" * 80)
print()

# Load the model and make predictions
import lightgbm as lgb
from core.data_loader import DataLoader
from core.features import FeatureEngine

# Load data
loader = DataLoader(start_date='2024-01-01', end_date='2024-06-01')
spy_data = loader.load_spy_data()
sector_data = loader.load_sector_data()
currency_data = loader.load_currency_data()
volatility_data = loader.load_volatility_data()

# Create features
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

# Load model
model = lgb.Booster(model_file='models/trained/gradual_pullback.txt')

# Make predictions for April 2024
april_features = spy_with_features.loc['2024-04-01':'2024-04-30']
X_april = april_features[all_features].fillna(0)
predictions = model.predict(X_april, num_iteration=model.best_iteration)

# Show predictions
results = pd.DataFrame({
    'date': april_features.index,
    'close': april_features['Close'],
    'prediction': predictions
})

print("Predictions for April 2024:")
print(results[['date', 'close', 'prediction']].to_string())
print()

print(f"Max prediction: {predictions.max():.4f}")
print(f"Mean prediction: {predictions.mean():.4f}")
print(f"Dates with prediction > 0.5: {(predictions > 0.5).sum()}")
print()

if predictions.max() > 0.3:
    print(f"✅ Model DID flag elevated risk in April 2024 (max: {predictions.max():.4f})")
else:
    print(f"❌ Model did NOT flag elevated risk in April 2024 (max: {predictions.max():.4f})")
