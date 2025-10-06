#!/usr/bin/env python3
"""
Train enhanced model with distribution features: skew, kurtosis, entropy
These features help detect regime changes and volatility events
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, entropy
import pickle

print('=' * 90)
print('TRAINING ENHANCED MODEL WITH DISTRIBUTION FEATURES')
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

print('\nCalculating standard features...')

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
df['vix_spike'] = (df['VIX'] - df['vix_sma_20']) / df['vix_sma_20']

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

print('✓ Standard features calculated')

# NEW: Distribution features
print('\nCalculating distribution features (skew, kurtosis, entropy)...')

# Rolling skew of returns (detects asymmetric moves)
df['returns_skew_20d'] = df['returns'].rolling(20).apply(lambda x: skew(x.dropna()), raw=False)
df['returns_skew_5d'] = df['returns'].rolling(5).apply(lambda x: skew(x.dropna()), raw=False)

# Rolling kurtosis (detects fat tails / extreme moves)
df['returns_kurtosis_20d'] = df['returns'].rolling(20).apply(lambda x: kurtosis(x.dropna()), raw=False)
df['returns_kurtosis_5d'] = df['returns'].rolling(5).apply(lambda x: kurtosis(x.dropna()), raw=False)

# Rolling entropy (detects randomness / uncertainty)
def rolling_entropy(x):
    """Calculate entropy of return distribution"""
    if len(x) < 3:
        return 0
    # Bin returns into 5 buckets
    hist, _ = np.histogram(x, bins=5)
    hist = hist + 1  # Add 1 to avoid log(0)
    probs = hist / hist.sum()
    return entropy(probs)

df['returns_entropy_20d'] = df['returns'].rolling(20).apply(rolling_entropy, raw=False)
df['returns_entropy_5d'] = df['returns'].rolling(5).apply(rolling_entropy, raw=False)

# VIX distribution features
df['vix_skew_20d'] = df['VIX'].pct_change().rolling(20).apply(lambda x: skew(x.dropna()), raw=False)
df['vix_kurtosis_20d'] = df['VIX'].pct_change().rolling(20).apply(lambda x: kurtosis(x.dropna()), raw=False)

# Volatility regime change detection
df['vol_regime_change'] = (df['volatility_20d'] - df['volatility_20d'].rolling(60).mean()) / df['volatility_20d'].rolling(60).std()

print('✓ Distribution features calculated')

# Feature list
feature_cols = [
    # Price/momentum
    'returns_5d', 'returns_20d', 'volatility_20d',
    'distance_from_sma20', 'distance_from_sma50', 'distance_from_sma200',
    'rsi',
    # VIX
    'VIX', 'vix_change', 'vix_sma_20', 'vix_spike',
    # Sector rotation
    'xlu_xlk_ratio', 'xlu_xlk_momentum',
    'qqq_qqqe_ratio', 'qqq_qqqe_momentum',
    'rsp_spy_ratio', 'rsp_spy_momentum',
    'xlv_xlk_ratio',
    # NEW: Distribution features
    'returns_skew_20d', 'returns_skew_5d',
    'returns_kurtosis_20d', 'returns_kurtosis_5d',
    'returns_entropy_20d', 'returns_entropy_5d',
    'vix_skew_20d', 'vix_kurtosis_20d',
    'vol_regime_change'
]

# Create target: 4% pullback in 30 days
df['future_low_30d'] = df['Low'].rolling(30).min().shift(-30)
df['target'] = ((df['future_low_30d'] - df['Close']) / df['Close']) <= -0.04

# Prepare training data (pre-2024)
df_clean = df[feature_cols + ['target']].dropna()
train_data = df_clean[df_clean.index < '2024-01-01']

X_train = train_data[feature_cols]
y_train = train_data['target'].astype(int)

print(f'\nTraining data (pre-2024): {len(X_train)} samples')
print(f'Positive samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)')
print(f'Total features: {len(feature_cols)} (added 9 distribution features)')

# Train ensemble
print('\nTraining enhanced ensemble model...')
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, 
                             class_weight='balanced', n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)

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

print('\nTop 15 Features:')
for idx, row in feature_importance.head(15).iterrows():
    marker = '🆕' if any(x in row['feature'] for x in ['skew', 'kurtosis', 'entropy', 'regime']) else '  '
    print(f"  {marker} {row['feature']:30s}: {row['avg_importance']:.4f}")

# Save model
model_data = {
    'rf_model': rf,
    'gb_model': gb,
    'feature_cols': feature_cols,
    'feature_importance': feature_importance
}

with open('models/trained/enhanced_distribution_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print('\n✅ Model saved to: models/trained/enhanced_distribution_model.pkl')

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
        
        results.to_csv(f'output/{year}_distribution_predictions.csv', index=False)
        
        high_prob = results[results['probability'] >= 0.7]
        print(f'\n{year}: {len(high_prob)} signals @ 70%')
        
        if len(high_prob) > 0 and year == 2024:
            print(f'  Signal dates:')
            for idx, row in high_prob.head(10).iterrows():
                print(f"    {row['date'].strftime('%Y-%m-%d')}  |  ${row['spy_close']:.2f}  |  {row['probability']*100:.1f}%")
            if len(high_prob) > 10:
                print(f'    ... and {len(high_prob)-10} more')

print('\n✅ Enhanced distribution model complete!')
