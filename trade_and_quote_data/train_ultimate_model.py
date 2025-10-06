#!/usr/bin/env python3
"""
Train ULTIMATE model with:
- Original features (overextension, momentum, VIX)
- Sector rotation (QQQ/QQQE, RSP/SPY, XLU/XLK)
- Distribution features (skew, kurtosis, entropy)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import skew, kurtosis, entropy
import pickle

print('=' * 90)
print('TRAINING ULTIMATE MODEL (All Features)')
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

# Load sector data
xlu = pd.read_parquet('data/sectors/XLU.parquet')
xlk = pd.read_parquet('data/sectors/XLK.parquet')
xlv = pd.read_parquet('data/sectors/XLV.parquet')
qqq = pd.read_parquet('data/rotation/QQQ.parquet')
qqqe = pd.read_parquet('data/rotation/QQQE.parquet')
rsp = pd.read_parquet('data/rotation/RSP.parquet')

# Build feature dataframe
df = pd.DataFrame(index=spy_df.index)
df['Close'] = spy_df['Close']
df['High'] = spy_df['High']
df['Low'] = spy_df['Low']
df['Volume'] = spy_df['Volume']
df['VIX'] = vix_df['Close']
df['XLU'] = xlu['Close']
df['XLK'] = xlk['Close']
df['XLV'] = xlv['Close']
df['QQQ'] = qqq['Close']
df['QQQE'] = qqqe['Close']
df['RSP'] = rsp['Close']

print('\n1️⃣ Calculating standard features...')

# Standard features
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

print('✓ Standard features done')

print('\n2️⃣ Calculating sector rotation features...')

# Sector rotation
df['xlu_xlk_ratio'] = df['XLU'] / df['XLK']
df['xlu_xlk_ratio_sma20'] = df['xlu_xlk_ratio'].rolling(20).mean()
df['xlu_xlk_momentum'] = df['xlu_xlk_ratio'] / df['xlu_xlk_ratio_sma20'] - 1

df['qqq_qqqe_ratio'] = df['QQQ'] / df['QQQE']
df['qqq_qqqe_ratio_sma20'] = df['qqq_qqqe_ratio'].rolling(20).mean()
df['qqq_qqqe_momentum'] = df['qqq_qqqe_ratio'] / df['qqq_qqqe_ratio_sma20'] - 1

df['rsp_spy_ratio'] = df['RSP'] / df['Close']
df['rsp_spy_ratio_sma20'] = df['rsp_spy_ratio'].rolling(20).mean()
df['rsp_spy_momentum'] = df['rsp_spy_ratio'] / df['rsp_spy_ratio_sma20'] - 1

df['xlv_xlk_ratio'] = df['XLV'] / df['XLK']

print('✓ Sector rotation done')

print('\n3️⃣ Calculating distribution features (skew, kurtosis, entropy)...')

# Rolling skew
df['returns_skew_20d'] = df['returns'].rolling(20).apply(lambda x: skew(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)
df['returns_skew_5d'] = df['returns'].rolling(5).apply(lambda x: skew(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)

# Rolling kurtosis
df['returns_kurtosis_20d'] = df['returns'].rolling(20).apply(lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)
df['returns_kurtosis_5d'] = df['returns'].rolling(5).apply(lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)

# Rolling entropy
def rolling_entropy(x):
    x = x.dropna()
    if len(x) < 3:
        return 0
    hist, _ = np.histogram(x, bins=5)
    hist = hist + 1
    probs = hist / hist.sum()
    return entropy(probs)

df['returns_entropy_20d'] = df['returns'].rolling(20).apply(rolling_entropy, raw=False)
df['returns_entropy_5d'] = df['returns'].rolling(5).apply(rolling_entropy, raw=False)

# VIX distribution
df['vix_skew_20d'] = df['VIX'].pct_change().rolling(20).apply(lambda x: skew(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)
df['vix_kurtosis_20d'] = df['VIX'].pct_change().rolling(20).apply(lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)

# Volatility regime change
df['vol_regime_change'] = (df['volatility_20d'] - df['volatility_20d'].rolling(60).mean()) / df['volatility_20d'].rolling(60).std()

print('✓ Distribution features done')

# ALL features
feature_cols = [
    # Standard
    'returns_5d', 'returns_20d', 'volatility_20d',
    'distance_from_sma20', 'distance_from_sma50', 'distance_from_sma200',
    'rsi', 'VIX', 'vix_change', 'vix_sma_20', 'vix_distance',
    # Sector rotation
    'xlu_xlk_ratio', 'xlu_xlk_momentum',
    'qqq_qqqe_ratio', 'qqq_qqqe_momentum',
    'rsp_spy_ratio', 'rsp_spy_momentum',
    'xlv_xlk_ratio',
    # Distribution
    'returns_skew_20d', 'returns_skew_5d',
    'returns_kurtosis_20d', 'returns_kurtosis_5d',
    'returns_entropy_20d', 'returns_entropy_5d',
    'vix_skew_20d', 'vix_kurtosis_20d',
    'vol_regime_change'
]

print(f'\n📊 Total features: {len(feature_cols)}')
print(f'   Standard: 11')
print(f'   Sector rotation: 7')
print(f'   Distribution: 9')

# Create target: 4% pullback in 30 days
df['future_low_30d'] = df['Low'].rolling(30).min().shift(-30)
df['target'] = ((df['future_low_30d'] - df['Close']) / df['Close']) <= -0.04

# Train on pre-2024
df_clean = df[feature_cols + ['target']].dropna()
train_data = df_clean[df_clean.index < '2024-01-01']

X_train = train_data[feature_cols]
y_train = train_data['target'].astype(int)

print(f'\nTraining data (pre-2024): {len(X_train)} samples')
print(f'Positive samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)')

# Train ensemble
print('\nTraining ultimate ensemble model...')
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=20,
                             random_state=42, class_weight='balanced', n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

print('✓ Model trained')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'rf_importance': rf.feature_importances_,
    'gb_importance': gb.feature_importances_
})
feature_importance['avg_importance'] = (feature_importance['rf_importance'] + feature_importance['gb_importance']) / 2
feature_importance = feature_importance.sort_values('avg_importance', ascending=False)

print('\n🏆 Top 20 Features:')
for idx, row in feature_importance.head(20).iterrows():
    if any(x in row['feature'] for x in ['skew', 'kurtosis', 'entropy', 'regime']):
        marker = '🆕'
    elif any(x in row['feature'] for x in ['qqq', 'rsp', 'xlu', 'xlv']):
        marker = '🔄'
    else:
        marker = '  '
    print(f"  {marker} {row['feature']:30s}: {row['avg_importance']:.4f}")

# Save model
model_data = {
    'rf_model': rf,
    'gb_model': gb,
    'feature_cols': feature_cols,
    'feature_importance': feature_importance
}

with open('models/trained/ultimate_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print('\n✅ Model saved to: models/trained/ultimate_model.pkl')

# Generate predictions for 2024 and 2025
for year in [2024, 2025]:
    df_year = df_clean[df_clean.index.year == year]
    X_year = df_year[feature_cols]
    
    if len(X_year) > 0:
        rf_proba = rf.predict_proba(X_year)[:, 1]
        gb_proba = gb.predict_proba(X_year)[:, 1]
        ensemble_proba = (rf_proba + gb_proba) / 2
        
        results = pd.DataFrame({
            'date': X_year.index,
            'spy_close': df.loc[X_year.index, 'Close'].values,
            'probability': ensemble_proba
        })
        
        results.to_csv(f'output/{year}_ultimate_predictions.csv', index=False)
        
        high_prob = results[results['probability'] >= 0.7]
        print(f'\n{year}: {len(high_prob)} signals @ 70%')
        
        if year == 2024:
            # Check specific periods
            march_apr = high_prob[(high_prob['date'] >= '2024-03-01') & (high_prob['date'] <= '2024-04-30')]
            july_aug = high_prob[(high_prob['date'] >= '2024-07-01') & (high_prob['date'] <= '2024-08-31')]
            aug_sep = high_prob[(high_prob['date'] >= '2024-08-20') & (high_prob['date'] <= '2024-09-15')]
            october = high_prob[(high_prob['date'] >= '2024-10-01') & (high_prob['date'] <= '2024-11-15')]
            
            print(f'  March-April: {len(march_apr)} signals')
            print(f'  July-August: {len(july_aug)} signals')
            print(f'  Aug-Sep: {len(aug_sep)} signals')
            print(f'  October: {len(october)} signals')

print('\n✅ Ultimate model training complete!')
