#!/usr/bin/env python3
"""
Generate 2025 predictions using the enhanced rotation model
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print('=' * 90)
print('GENERATING 2024 AND 2025 PREDICTIONS WITH ENHANCED MODEL')
print('=' * 90)

# Load model
with open('models/trained/enhanced_rotation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

rf_model = model_data['rf_model']
gb_model = model_data['gb_model']
feature_cols = model_data['feature_cols']

# Load all data
spy_df = pd.read_parquet('data/ohlc/SPY.parquet')
spy_df.index = pd.to_datetime(spy_df.index)

vix_df = pd.read_parquet('data/volatility/VIX.parquet')
vix_df.index = pd.to_datetime(vix_df.index)

xlu_df = pd.read_parquet('data/sectors/XLU.parquet')
xlu_df.index = pd.to_datetime(xlu_df.index)

xlk_df = pd.read_parquet('data/sectors/XLK.parquet')
xlk_df.index = pd.to_datetime(xlk_df.index)

xlv_df = pd.read_parquet('data/sectors/XLV.parquet')
xlv_df.index = pd.to_datetime(xlv_df.index)

qqq_df = pd.read_parquet('data/rotation/QQQ.parquet')
qqq_df.index = pd.to_datetime(qqq_df.index)

qqqe_df = pd.read_parquet('data/rotation/QQQE.parquet')
qqqe_df.index = pd.to_datetime(qqqe_df.index)

rsp_df = pd.read_parquet('data/rotation/RSP.parquet')
rsp_df.index = pd.to_datetime(rsp_df.index)

# Build feature dataframe
df = pd.DataFrame(index=spy_df.index)
df['Close'] = spy_df['Close']
df['High'] = spy_df['High']
df['Low'] = spy_df['Low']
df['Open'] = spy_df['Open']
df['Volume'] = spy_df['Volume']
df['VIX'] = vix_df['Close']
df['XLU'] = xlu_df['Close']
df['XLK'] = xlk_df['Close']
df['XLV'] = xlv_df['Close']
df['QQQ'] = qqq_df['Close']
df['QQQE'] = qqqe_df['Close']
df['RSP'] = rsp_df['Close']

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

delta = df['returns']
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

df['vix_change'] = df['VIX'].pct_change()
df['vix_sma_20'] = df['VIX'].rolling(20).mean()

# Sector rotation features
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

# Generate predictions for both 2024 and 2025
for year in [2024, 2025]:
    df_year = df[df.index.year == year].copy()
    df_year_clean = df_year[feature_cols].dropna()
    
    if len(df_year_clean) > 0:
        X = df_year_clean[feature_cols]
        
        # Ensemble predictions
        rf_proba = rf_model.predict_proba(X)[:, 1]
        gb_proba = gb_model.predict_proba(X)[:, 1]
        ensemble_proba = (rf_proba + gb_proba) / 2
        
        # Create results
        results = pd.DataFrame({
            'date': df_year_clean.index,
            'spy_close': df_year.loc[df_year_clean.index, 'Close'].values,
            'probability': ensemble_proba
        })
        
        # Save
        output_file = f'output/{year}_enhanced_predictions.csv'
        results.to_csv(output_file, index=False)
        
        # Analysis
        high_prob = results[results['probability'] >= 0.7]
        
        print(f'\n✅ {year} Predictions Generated')
        print(f'  Period: {results["date"].min().strftime("%Y-%m-%d")} to {results["date"].max().strftime("%Y-%m-%d")}')
        print(f'  Total days: {len(results)}')
        print(f'  High risk signals (≥70%): {len(high_prob)}')
        print(f'  Average probability: {results["probability"].mean():.1%}')
        print(f'  Latest SPY: ${results["spy_close"].iloc[-1]:.2f}')
        
        if len(high_prob) > 0:
            print(f'  🚨 HIGH RISK PERIODS:')
            for idx, row in high_prob.iterrows():
                print(f"    {row['date'].strftime('%Y-%m-%d')}  |  ${row['spy_close']:.2f}  |  {row['probability']*100:.1f}%")
        else:
            print(f'  ✅ No high risk signals')
        
        print(f'  Saved to: {output_file}')
