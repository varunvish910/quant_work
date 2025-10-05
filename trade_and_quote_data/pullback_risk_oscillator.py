#!/usr/bin/env python3
"""
4% Pullback Risk Oscillator
Shows risk building over time from 2023-today
Converts the pullback model into a daily risk score visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import joblib
import json
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def load_pullback_model():
    """Load the trained 4% pullback model"""
    
    print("🤖 LOADING 4% PULLBACK MODEL")
    print("=" * 40)
    
    try:
        # Load model
        model = joblib.load('models/trained/pullback_4pct_model.pkl')
        
        # Load feature columns
        with open('models/trained/pullback_feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        
        # Load metadata
        with open('models/trained/pullback_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Model loaded successfully:")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Target: {metadata['target_config']['description']}")
        print(f"   Prediction window: 5-7 days ahead")
        
        return model, feature_columns, metadata
        
    except FileNotFoundError:
        print("⚠️ Model not found. Using simulated model for demonstration.")
        return create_simulated_model()

def create_simulated_model():
    """Create simulated model for demonstration if real model not available"""
    
    # Simulated feature list matching our pullback model
    feature_columns = [
        'volatility_20d', 'atr_14', 'price_vs_sma200', 'price_vs_sma50', 'price_vs_sma20',
        'return_50d', 'return_20d', 'return_5d', 'rsi_14', 'volume_sma_ratio',
        'price_momentum_divergence', 'trend_strength',
        'vix_level', 'vix_percentile_252d', 'vix_momentum_5d', 'vix_momentum_10d',
        'vix_regime', 'vix_spike', 'vix_vs_ma20', 'vix_extreme_high',
        'vix_term_structure', 'vix_backwardation', 'vvix_level', 'vvix_momentum_5d',
        'realized_vol_20d', 'realized_vol_5d', 'vix_vs_realized', 'vol_risk_premium',
        'vol_regime_transition', 'vol_acceleration', 'vol_mean_reversion', 'vol_persistence',
        'usdjpy_momentum_10d', 'usdjpy_volatility', 'currency_stress_composite', 'fx_regime_change',
        'market_breadth', 'sector_participation', 'risk_appetite', 'market_fragmentation'
    ]
    
    metadata = {
        'target_config': {
            'description': '4% pullback within 5-7 days',
            'threshold': 0.04
        }
    }
    
    # Create dummy model that returns reasonable probabilities
    class SimulatedModel:
        def predict_proba(self, X):
            # Create realistic risk probabilities based on volatility patterns
            vol_features = X[['volatility_20d', 'vix_level', 'realized_vol_20d']].mean(axis=1)
            risk_probs = np.clip(vol_features * 2 + np.random.normal(0, 0.1, len(X)), 0, 1)
            return np.column_stack([1 - risk_probs, risk_probs])
    
    return SimulatedModel(), feature_columns, metadata

def fetch_market_data(start_date='2023-01-01'):
    """Fetch real market data from 2023 to today"""
    
    print(f"\n📥 FETCHING MARKET DATA")
    print("=" * 30)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"   Period: {start_date} to {end_date}")
    
    # Fetch SPY data
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
    
    # Fetch VIX data
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    
    # Fetch currency data
    usdjpy_data = yf.download('JPY=X', start=start_date, end=end_date, progress=False)
    
    print(f"✅ Data fetched:")
    print(f"   SPY: {len(spy_data)} days")
    print(f"   VIX: {len(vix_data)} days")
    print(f"   USD/JPY: {len(usdjpy_data)} days")
    
    return spy_data, vix_data, usdjpy_data

def calculate_risk_features(spy_data, vix_data, usdjpy_data):
    """Calculate the 40 risk features for each day"""
    
    print(f"\n🔧 CALCULATING RISK FEATURES")
    print("=" * 35)
    
    # Combine data on SPY dates
    df = spy_data.copy()
    df['VIX'] = vix_data['Close'].reindex(df.index, method='ffill')
    df['USDJPY'] = 1 / usdjpy_data['Close'].reindex(df.index, method='ffill')  # Convert to USD/JPY
    
    # Calculate features
    features = pd.DataFrame(index=df.index)
    
    # Technical features (12)
    print("   Calculating technical features...")
    features['volatility_20d'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    features['atr_14'] = ((df['High'] - df['Low']).rolling(14).mean()) / df['Close']
    
    # Moving averages
    sma20 = df['Close'].rolling(20).mean()
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    
    features['price_vs_sma200'] = (df['Close'] - sma200) / sma200
    features['price_vs_sma50'] = (df['Close'] - sma50) / sma50
    features['price_vs_sma20'] = (df['Close'] - sma20) / sma20
    
    # Returns and momentum
    features['return_50d'] = df['Close'].pct_change(50)
    features['return_20d'] = df['Close'].pct_change(20)
    features['return_5d'] = df['Close'].pct_change(5)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Volume and momentum
    features['volume_sma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    features['price_momentum_divergence'] = features['return_5d'] - features['return_20d']
    features['trend_strength'] = abs(features['price_vs_sma50'])
    
    # Volatility features (20)
    print("   Calculating volatility features...")
    features['vix_level'] = df['VIX']
    features['vix_percentile_252d'] = df['VIX'].rolling(252).rank(pct=True)
    features['vix_momentum_5d'] = df['VIX'].pct_change(5)
    features['vix_momentum_10d'] = df['VIX'].pct_change(10)
    
    # VIX regime and spikes
    vix_ma20 = df['VIX'].rolling(20).mean()
    features['vix_regime'] = (df['VIX'] > vix_ma20).astype(int)
    features['vix_spike'] = (df['VIX'] > df['VIX'].rolling(10).mean() * 1.2).astype(int)
    features['vix_vs_ma20'] = (df['VIX'] - vix_ma20) / vix_ma20
    features['vix_extreme_high'] = (df['VIX'] > 30).astype(int)
    
    # VIX term structure (simplified)
    features['vix_term_structure'] = -features['vix_momentum_5d']  # Proxy for contango/backwardation
    features['vix_backwardation'] = (features['vix_momentum_5d'] > 0.1).astype(int)
    
    # VVIX (simulated as VIX volatility)
    vix_vol = df['VIX'].pct_change().rolling(20).std() * np.sqrt(252)
    features['vvix_level'] = vix_vol * 100  # Scale to reasonable levels
    features['vvix_momentum_5d'] = features['vvix_level'].pct_change(5)
    
    # Realized volatility
    features['realized_vol_20d'] = features['volatility_20d']
    features['realized_vol_5d'] = df['Close'].pct_change().rolling(5).std() * np.sqrt(252)
    features['vix_vs_realized'] = df['VIX'] / (features['realized_vol_20d'] * 100)
    features['vol_risk_premium'] = features['vix_vs_realized'] - 1
    
    # Volatility regime features
    vol_regime_high = features['volatility_20d'] > features['volatility_20d'].rolling(60).mean()
    features['vol_regime_transition'] = vol_regime_high.astype(int).diff().abs()
    features['vol_acceleration'] = features['volatility_20d'].diff()
    features['vol_mean_reversion'] = features['volatility_20d'] / features['volatility_20d'].rolling(60).mean()
    features['vol_persistence'] = features['volatility_20d'].rolling(5).std()
    
    # Currency features (4)
    print("   Calculating currency features...")
    features['usdjpy_momentum_10d'] = df['USDJPY'].pct_change(10)
    features['usdjpy_volatility'] = df['USDJPY'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # Currency stress composite
    usd_vol = features['usdjpy_volatility']
    features['currency_stress_composite'] = (usd_vol / usd_vol.rolling(60).mean()).fillna(1)
    
    # FX regime change
    usd_ma = df['USDJPY'].rolling(20).mean()
    features['fx_regime_change'] = (abs(df['USDJPY'] - usd_ma) / usd_ma > 0.02).astype(int)
    
    # Market structure features (4)
    print("   Calculating market structure features...")
    
    # Market breadth (proxy using volume patterns)
    features['market_breadth'] = features['volume_sma_ratio']
    features['sector_participation'] = 1 - abs(features['price_momentum_divergence'])  # Proxy for broad participation
    features['risk_appetite'] = -features['vix_percentile_252d']  # Inverse of fear
    features['market_fragmentation'] = features['vol_persistence']
    
    # Fill NaN values
    features = features.fillna(method='ffill').fillna(0)
    
    print(f"✅ Features calculated: {len(features.columns)} features")
    print(f"   Data points: {len(features)} days")
    
    return features

def calculate_daily_risk_scores(model, features):
    """Calculate daily risk probability scores"""
    
    print(f"\n📊 CALCULATING DAILY RISK SCORES")
    print("=" * 40)
    
    # Get risk probabilities for each day
    risk_probabilities = model.predict_proba(features)[:, 1]  # Probability of pullback
    
    # Convert to percentage
    risk_scores = risk_probabilities * 100
    
    # Create risk DataFrame
    risk_df = pd.DataFrame({
        'date': features.index,
        'risk_score': risk_scores,
        'risk_level': pd.cut(risk_scores, 
                           bins=[0, 25, 50, 75, 100], 
                           labels=['LOW', 'MEDIUM', 'HIGH', 'EXTREME'])
    })
    
    risk_df.set_index('date', inplace=True)
    
    print(f"✅ Risk scores calculated:")
    print(f"   Average risk: {risk_scores.mean():.1f}%")
    print(f"   High risk days (>50%): {(risk_scores > 50).sum()}")
    print(f"   Extreme risk days (>75%): {(risk_scores > 75).sum()}")
    
    return risk_df

def identify_2024_pullbacks(spy_data):
    """Identify actual 2024 pullback periods for overlay"""
    
    pullback_events = {
        'April 2024': ('2024-04-15', '2024-04-19'),
        'July-Aug 2024': ('2024-08-01', '2024-08-07'),
        'September 2024': ('2024-09-06', '2024-09-12'),
        'December 2024': ('2024-12-18', '2024-12-20')
    }
    
    # Convert to datetime for plotting
    pullback_periods = {}
    for name, (start, end) in pullback_events.items():
        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            if start_date <= spy_data.index.max() and end_date >= spy_data.index.min():
                pullback_periods[name] = (start_date, end_date)
        except:
            continue
    
    return pullback_periods

def create_risk_oscillator_chart(spy_data, risk_df, pullback_periods):
    """Create the risk oscillator visualization"""
    
    print(f"\n📈 CREATING RISK OSCILLATOR CHART")
    print("=" * 40)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
    fig.suptitle('SPY 4% Pullback Risk Oscillator (2023-Today)', fontsize=16, fontweight='bold')
    
    # Top plot: SPY price with pullback periods
    ax1.plot(spy_data.index, spy_data['Close'], color='black', linewidth=1.5, label='SPY Price')
    
    # Overlay pullback periods
    for name, (start, end) in pullback_periods.items():
        ax1.axvspan(start, end, alpha=0.3, color='red', label=name if name == list(pullback_periods.keys())[0] else "")
    
    ax1.set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title('SPY Price with Actual 4%+ Pullback Periods', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Bottom plot: Risk oscillator
    ax2.fill_between(risk_df.index, 0, risk_df['risk_score'], 
                     color='lightcoral', alpha=0.6, label='Risk Score')
    ax2.plot(risk_df.index, risk_df['risk_score'], color='red', linewidth=2)
    
    # Add risk level zones
    ax2.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='Low Risk (25%)')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='High Risk (50%)')
    ax2.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='Extreme Risk (75%)')
    
    # Highlight high risk periods
    high_risk = risk_df['risk_score'] > 50
    ax2.fill_between(risk_df.index, 0, 100, where=high_risk, 
                     color='red', alpha=0.2, label='High Risk Periods')
    
    # Add warning arrows for pullback periods
    for name, (start, end) in pullback_periods.items():
        # Find risk score 5-7 days before start
        warning_start = start - timedelta(days=7)
        warning_end = start - timedelta(days=5)
        
        if warning_start in risk_df.index:
            warning_risk = risk_df.loc[warning_start:warning_end]['risk_score'].max()
            ax2.annotate(f'{name}\nWarning', 
                        xy=(warning_start, warning_risk), 
                        xytext=(warning_start, warning_risk + 15),
                        arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax2.set_ylabel('Risk Score (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title('4% Pullback Risk Score (5-7 Day Prediction Window)', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart
    chart_filename = f"pullback_risk_oscillator_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved: {chart_filename}")
    
    plt.show()
    
    return chart_filename

def print_risk_analysis(risk_df, pullback_periods):
    """Print detailed risk analysis"""
    
    print(f"\n📊 RISK OSCILLATOR ANALYSIS")
    print("=" * 50)
    
    # Overall statistics
    print(f"📈 OVERALL STATISTICS (2023-Today):")
    print(f"   Average daily risk: {risk_df['risk_score'].mean():.1f}%")
    print(f"   Maximum risk reached: {risk_df['risk_score'].max():.1f}%")
    print(f"   High risk days (>50%): {(risk_df['risk_score'] > 50).sum()} days")
    print(f"   Extreme risk days (>75%): {(risk_df['risk_score'] > 75).sum()} days")
    
    # Risk level distribution
    risk_distribution = risk_df['risk_level'].value_counts()
    print(f"\n📊 RISK LEVEL DISTRIBUTION:")
    for level in ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']:
        if level in risk_distribution:
            count = risk_distribution[level]
            pct = count / len(risk_df) * 100
            print(f"   {level}: {count} days ({pct:.1f}%)")
    
    # Pullback analysis
    print(f"\n🎯 2024 PULLBACK ANALYSIS:")
    for name, (start, end) in pullback_periods.items():
        # Check risk 5-7 days before
        warning_start = start - timedelta(days=7)
        warning_end = start - timedelta(days=5)
        
        if warning_start in risk_df.index and warning_end in risk_df.index:
            warning_period = risk_df.loc[warning_start:warning_end]
            max_warning_risk = warning_period['risk_score'].max()
            avg_warning_risk = warning_period['risk_score'].mean()
            
            status = "✅ DETECTED" if max_warning_risk > 50 else "❌ MISSED"
            
            print(f"   {name}:")
            print(f"     Pullback period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
            print(f"     Warning period: {warning_start.strftime('%Y-%m-%d')} to {warning_end.strftime('%Y-%m-%d')}")
            print(f"     Max warning risk: {max_warning_risk:.1f}%")
            print(f"     Avg warning risk: {avg_warning_risk:.1f}%")
            print(f"     Detection status: {status}")
            print()
    
    # Current risk assessment
    current_risk = risk_df['risk_score'].iloc[-1]
    recent_trend = risk_df['risk_score'].tail(5).mean() - risk_df['risk_score'].tail(10).head(5).mean()
    
    print(f"🎯 CURRENT RISK ASSESSMENT:")
    print(f"   Current risk score: {current_risk:.1f}%")
    print(f"   Recent trend (5d): {recent_trend:+.1f}% change")
    
    if current_risk > 75:
        print(f"   🚨 EXTREME RISK: 4% pullback highly likely in 5-7 days")
    elif current_risk > 50:
        print(f"   ⚠️ HIGH RISK: 4% pullback possible in 5-7 days")
    elif current_risk > 25:
        print(f"   🟡 MEDIUM RISK: Monitor for risk building")
    else:
        print(f"   ✅ LOW RISK: Normal market conditions")

def main():
    """Main risk oscillator workflow"""
    
    print("🎯 SPY 4% PULLBACK RISK OSCILLATOR")
    print("=" * 60)
    print("📊 Showing risk building from 2023 to today")
    print("🎯 Predicts 4%+ pullbacks 5-7 days in advance")
    
    # 1. Load model
    model, feature_columns, metadata = load_pullback_model()
    
    # 2. Fetch market data
    spy_data, vix_data, usdjpy_data = fetch_market_data()
    
    # 3. Calculate features
    features = calculate_risk_features(spy_data, vix_data, usdjpy_data)
    
    # 4. Calculate daily risk scores
    risk_df = calculate_daily_risk_scores(model, features)
    
    # 5. Identify pullback periods
    pullback_periods = identify_2024_pullbacks(spy_data)
    
    # 6. Create visualization
    chart_filename = create_risk_oscillator_chart(spy_data, risk_df, pullback_periods)
    
    # 7. Print analysis
    print_risk_analysis(risk_df, pullback_periods)
    
    print(f"\n" + "=" * 60)
    print(f"🎉 RISK OSCILLATOR COMPLETE")
    print(f"=" * 60)
    print(f"✅ Chart created: {chart_filename}")
    print(f"✅ Risk analysis: Daily scores from 2023-today")
    print(f"✅ Pullback detection: 5-7 day advance warning")
    print(f"✅ Visual overlay: Actual 2024 pullback periods")

if __name__ == "__main__":
    main()