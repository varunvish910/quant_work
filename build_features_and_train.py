#!/usr/bin/env python3
"""
Feature Engineering and Model Training Script
Builds comprehensive features and trains early warning models
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
print("="*80)
print("FEATURE ENGINEERING & MODEL TRAINING")
print("="*80)

with open('data/market_data_cache.pkl', 'rb') as f:
    data = pickle.load(f)

spy = data['spy'].copy()
sectors = data['sectors']
currency = data['currency']
volatility = data['volatility']

print(f"\n📊 Data loaded: {len(spy)} SPY records")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("BUILDING FEATURES")
print("="*80)

df = spy.copy()

# Basic features
df['Returns'] = df['Close'].pct_change()
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']

# Price momentum features
for period in [5, 10, 20, 50]:
    df[f'Momentum_{period}d'] = df['Close'].pct_change(period)
    df[f'SMA_{period}d'] = df['Close'].rolling(period).mean()
    df[f'Price_to_SMA_{period}d'] = df['Close'] / df[f'SMA_{period}d'] - 1

# Volatility features
for period in [5, 10, 20]:
    df[f'Volatility_{period}d'] = df['Returns'].rolling(period).std()
    df[f'ATR_{period}d'] = (df['High'] - df['Low']).rolling(period).mean()

# RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_rsi(df['Close'], 14)

# MACD
ema_12 = df['Close'].ewm(span=12).mean()
ema_26 = df['Close'].ewm(span=26).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

# Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(20).mean()
df['BB_Std'] = df['Close'].rolling(20).std()
df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

# Volume features (if available)
if 'Volume' in df.columns:
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

print("✅ Technical features created")

# VIX features
vix = volatility['VIX']
vix = vix.reindex(df.index, method='ffill')
df['VIX'] = vix['Close']
df['VIX_Change'] = vix['Close'].pct_change()
df['VIX_SMA_20'] = vix['Close'].rolling(20).mean()
df['VIX_Spike'] = (vix['Close'] > vix['Close'].rolling(20).mean() + vix['Close'].rolling(20).std())

# VIX term structure (if VIX9D available)
if 'VIX9D' in volatility:
    vix9d = volatility['VIX9D']
    vix9d = vix9d.reindex(df.index, method='ffill')
    df['VIX_Term_Structure'] = vix9d['Close'] / vix['Close'] - 1

print("✅ Volatility features created")

# Currency features - USD/JPY (critical for carry trade detection)
usdjpy = currency['USDJPY']
usdjpy = usdjpy.reindex(df.index, method='ffill')
df['USDJPY'] = usdjpy['Close']
df['USDJPY_Change'] = usdjpy['Close'].pct_change()
df['USDJPY_Change_5d'] = usdjpy['Close'].pct_change(5)
df['USDJPY_Volatility_20d'] = usdjpy['Close'].pct_change().rolling(20).std()

# Carry trade stress indicator
df['Carry_Trade_Stress'] = (df['USDJPY_Change_5d'] < -0.02).astype(int)  # Sharp yen appreciation

print("✅ Currency features created")

# Sector rotation features
sector_returns = pd.DataFrame()
for sector_name, sector_data in sectors.items():
    if sector_name in ['XLU', 'XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB']:
        sector_aligned = sector_data.reindex(df.index, method='ffill')
        sector_returns[f'{sector_name}_Return'] = sector_aligned['Close'].pct_change(5)

# Defensive vs cyclical rotation
if 'XLU_Return' in sector_returns.columns and 'XLK_Return' in sector_returns.columns:
    df['Defensive_Rotation'] = sector_returns['XLU_Return'] - sector_returns['XLK_Return']

# Market breadth (if RSP available)
if 'RSP' in sectors:
    rsp = sectors['RSP'].reindex(df.index, method='ffill')
    df['Market_Breadth'] = rsp['Close'].pct_change(20) - df['Momentum_20d']

print("✅ Sector rotation features created")

# Lagged features for prediction
for lag in [1, 2, 3, 5]:
    df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    df[f'VIX_Lag_{lag}'] = df['VIX'].shift(lag)

print(f"\n✅ Total features created: {len(df.columns)}")

# Drop NaN rows
df_clean = df.dropna()
print(f"✅ Clean data: {len(df_clean)} rows (removed {len(df) - len(df_clean)} NaN rows)")

# Save features
df_clean.to_pickle('data/features_complete.pkl')
print("💾 Features saved to data/features_complete.pkl")

print(f"\n📊 Feature Summary:")
print(f"   Date range: {df_clean.index[0].date()} to {df_clean.index[-1].date()}")
print(f"   Total features: {len(df_clean.columns)}")
print(f"   Training data (2000-2022): {len(df_clean[df_clean.index <= '2022-12-31'])} rows")
print(f"   Validation data (2023): {len(df_clean[(df_clean.index > '2022-12-31') & (df_clean.index <= '2023-12-31')])} rows")
print(f"   Test data (2024): {len(df_clean[df_clean.index > '2023-12-31'])} rows")
