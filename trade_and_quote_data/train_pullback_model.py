#!/usr/bin/env python3
"""
Train the original-style pullback model (more sensitive, catches all corrections)
Target: 2% pullback within 20 days
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle

print('=' * 90)
print('TRAINING PULLBACK MODEL (2% in 20 days - Sensitive)')
print('=' * 90)

# Load data
spy_df = pd.read_parquet('data/ohlc/SPY.parquet')
spy_df.index = pd.to_datetime(spy_df.index)
if isinstance(spy_df.columns, pd.MultiIndex):
    spy_df.columns = spy_df.columns.get_level_values(0)

vix_df = pd.read_parquet('data/volatility/VIX.parquet')
vix_df.index = pd.to_datetime(vix_df.index)
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)

# Build features
df = pd.DataFrame(index=spy_df.index)
df['Close'] = spy_df['Close']
df['High'] = spy_df['High']
df['Low'] = spy_df['Low']
df['Volume'] = spy_df['Volume']
df['VIX'] = vix_df['Close']

# Calculate features
df['returns'] = df['Close'].pct_change()
df['returns_5d'] = df['Close'].pct_change(5)
df['returns_20d'] = df['Close'].pct_change(20)
df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
df['sma_20'] = df['Close'].rolling(20).mean()
df['sma_50'] = df['Close'].rolling(50).mean()
df['sma_200'] = df['Close'].rolling(200).mean()
df['distance_from_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']
df['distance_from_sma50'] = (df['Close'] - df['sma_50']) / df['sma_50']
df['distance_from_sma200'] = (df['Close'] - df['sma_200']) / df['sma_200']

# RSI
delta = df['returns']
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# VIX features
df['vix_change'] = df['VIX'].pct_change()
df['vix_sma_20'] = df['VIX'].rolling(20).mean()
df['vix_distance'] = (df['VIX'] - df['vix_sma_20']) / df['vix_sma_20']

# Create target: 2% pullback in next 20 days (more sensitive than 4% in 30 days)
df['future_low_20d'] = df['Low'].rolling(20).min().shift(-20)
df['target'] = ((df['future_low_20d'] - df['Close']) / df['Close']) <= -0.02

# Select features (NO sector rotation - this is the old model)
feature_cols = [
    'returns_5d', 'returns_20d', 'volatility_20d',
    'distance_from_sma20', 'distance_from_sma50', 'distance_from_sma200',
    'rsi', 'VIX', 'vix_change', 'vix_sma_20', 'vix_distance'
]

# Prepare data - train on pre-2024
df_clean = df[feature_cols + ['target']].dropna()
train_data = df_clean[df_clean.index < '2024-01-01']

X_train = train_data[feature_cols]
y_train = train_data['target'].astype(int)

print(f'\nTraining data (pre-2024): {len(X_train)} samples')
print(f'Positive samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)')

# Train ensemble
print('\nTraining ensemble model...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, 
                             class_weight='balanced', n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)

rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)

print('✓ Model trained')

# Save model
model_data = {
    'rf_model': rf,
    'gb_model': gb,
    'scaler': scaler,
    'feature_cols': feature_cols
}

with open('models/trained/pullback_2pct_20d_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print('✅ Model saved to: models/trained/pullback_2pct_20d_model.pkl')

# Generate predictions for 2024 and 2025
for year in [2024, 2025]:
    df_year = df_clean[df_clean.index.year == year]
    X_year = df_year[feature_cols]
    
    if len(X_year) > 0:
        X_year_scaled = scaler.transform(X_year)
        
        rf_proba = rf.predict_proba(X_year_scaled)[:, 1]
        gb_proba = gb.predict_proba(X_year_scaled)[:, 1]
        ensemble_proba = (rf_proba + gb_proba) / 2
        
        results = pd.DataFrame({
            'date': X_year.index,
            'spy_close': df.loc[X_year.index, 'Close'].values,
            'probability': ensemble_proba
        })
        
        results.to_csv(f'output/{year}_pullback_predictions.csv', index=False)
        
        high_prob = results[results['probability'] >= 0.7]
        print(f'\n{year}: {len(high_prob)} signals @ 70%')

print('\n✅ Pullback model predictions complete!')
