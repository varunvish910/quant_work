#!/usr/bin/env python3
"""
2025 SPY Outlook and Predictions
"""

import pickle
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("2025 SPY MARKET OUTLOOK & PREDICTIONS")
print("="*80)

# Load models and features
early_warning_model = joblib.load('models/trained/early_warning_model.pkl')
pullback_model = joblib.load('models/trained/pullback_model.pkl')

with open('models/trained/early_warning_features.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Load complete feature data
df = pd.read_pickle('data/features_complete.pkl')

print(f"\n✅ Models and data loaded")
print(f"   Latest data point: {df.index[-1].date()}")

# ============================================================================
# CURRENT MARKET CONDITIONS (Last 30 days)
# ============================================================================

print("\n" + "="*80)
print("CURRENT MARKET CONDITIONS (Last 30 Days)")
print("="*80)

recent = df.tail(30)

print(f"\n📊 PRICE ACTION:")
print(f"   Current SPY: ${recent['Close'].iloc[-1]:.2f}")
print(f"   30-day change: {(recent['Close'].iloc[-1] / recent['Close'].iloc[0] - 1) * 100:+.2f}%")
print(f"   30-day high: ${recent['High'].max():.2f}")
print(f"   30-day low: ${recent['Low'].min():.2f}")
print(f"   Distance from 30-day high: {(recent['Close'].iloc[-1] / recent['High'].max() - 1) * 100:.2f}%")

print(f"\n📈 TECHNICAL INDICATORS:")
print(f"   RSI(14): {recent['RSI_14'].iloc[-1]:.1f}")
if recent['RSI_14'].iloc[-1] > 70:
    print(f"            → Overbought")
elif recent['RSI_14'].iloc[-1] < 30:
    print(f"            → Oversold")
else:
    print(f"            → Neutral")

print(f"   MACD: {recent['MACD'].iloc[-1]:.2f}")
print(f"   MACD Signal: {recent['MACD_Signal'].iloc[-1]:.2f}")
if recent['MACD'].iloc[-1] > recent['MACD_Signal'].iloc[-1]:
    print(f"            → Bullish crossover")
else:
    print(f"            → Bearish crossover")

print(f"   20-day volatility: {recent['Volatility_20d'].iloc[-1] * 100:.2f}%")

print(f"\n🔥 VOLATILITY REGIME:")
print(f"   VIX: {recent['VIX'].iloc[-1]:.1f}")
if recent['VIX'].iloc[-1] < 15:
    vix_regime = "Low (Complacent)"
elif recent['VIX'].iloc[-1] < 20:
    vix_regime = "Normal"
elif recent['VIX'].iloc[-1] < 30:
    vix_regime = "Elevated (Cautious)"
else:
    vix_regime = "High (Fear)"
print(f"            → {vix_regime}")

if 'VIX_Term_Structure' in recent.columns:
    vix_ts = recent['VIX_Term_Structure'].iloc[-1]
    print(f"   VIX Term Structure: {vix_ts*100:+.2f}%")
    if vix_ts < 0:
        print(f"            → Backwardation (RISK)")
    else:
        print(f"            → Contango (Normal)")

print(f"\n💱 CURRENCY SIGNALS:")
print(f"   USD/JPY: {recent['USDJPY'].iloc[-1]:.2f}")
print(f"   5-day change: {recent['USDJPY_Change_5d'].iloc[-1] * 100:+.2f}%")
if abs(recent['USDJPY_Change_5d'].iloc[-1]) > 0.02:
    print(f"            → ⚠️ Carry trade stress detected")

print(f"\n🔄 MARKET BREADTH:")
if 'Market_Breadth' in recent.columns:
    breadth = recent['Market_Breadth'].iloc[-1]
    print(f"   Breadth indicator: {breadth*100:+.2f}%")
    if breadth < -0.05:
        print(f"            → Narrowing (concentration risk)")
    elif breadth > 0.05:
        print(f"            → Broadening (healthy)")
    else:
        print(f"            → Neutral")

# ============================================================================
# CURRENT PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("CURRENT RISK ASSESSMENT")
print("="*80)

# Get latest features
latest_features = df[feature_cols].tail(10)

# Early warning predictions
early_warning_probs = early_warning_model.predict_proba(latest_features)[:, 1]
pullback_probs = pullback_model.predict_proba(latest_features)[:, 1]

print(f"\n📅 LAST 10 TRADING DAYS:")
print(f"{'Date':<12} {'SPY Close':>10} {'Early Warn':>12} {'Pullback':>10} {'Risk Level':<12}")
print("-" * 60)

for i in range(len(latest_features)):
    date = latest_features.index[i].date()
    close = df.loc[latest_features.index[i], 'Close']
    ew_prob = early_warning_probs[i]
    pb_prob = pullback_probs[i]

    if ew_prob > 0.7:
        risk = "🔴 HIGH"
    elif ew_prob > 0.5:
        risk = "🟡 MEDIUM"
    else:
        risk = "🟢 LOW"

    print(f"{date!s:<12} ${close:>9.2f} {ew_prob:>11.1%} {pb_prob:>9.1%} {risk:<12}")

# Latest prediction
latest_ew = early_warning_probs[-1]
latest_pb = pullback_probs[-1]

print(f"\n⚡ CURRENT RISK LEVEL:")
if latest_ew > 0.7:
    print(f"   🔴 HIGH RISK ({latest_ew:.0%})")
    print(f"   Model signals elevated probability of 2%+ drop in next 3-5 days")
elif latest_ew > 0.5:
    print(f"   🟡 MEDIUM RISK ({latest_ew:.0%})")
    print(f"   Model shows moderate warning signals")
else:
    print(f"   🟢 LOW RISK ({latest_ew:.0%})")
    print(f"   No immediate warning signals detected")

# ============================================================================
# 2025 OUTLOOK & WHAT TO WATCH
# ============================================================================

print("\n" + "="*80)
print("2025 MARKET OUTLOOK")
print("="*80)

print(f"\n📊 ENTRY TO 2025 POSITIONING:")

# Calculate year-end momentum
ytd_return = (df['Close'].iloc[-1] / df[df.index.year == 2024]['Close'].iloc[0] - 1) * 100
print(f"   2024 YTD Return: +{ytd_return:.1f}%")
print(f"   2024 was a STRONG year for SPY")

# Analyze current positioning relative to history
current_price = df['Close'].iloc[-1]
sma_200 = df['SMA_50d'].iloc[-1]
distance_from_ma = (current_price / sma_200 - 1) * 100

print(f"\n📈 TECHNICAL SETUP:")
print(f"   Distance from 50-day MA: {distance_from_ma:+.1f}%")
if distance_from_ma > 5:
    print(f"            → Extended above trend (pullback risk)")
elif distance_from_ma < -5:
    print(f"            → Well below trend (potential oversold)")
else:
    print(f"            → Near trend line (neutral)")

# Volatility regime
recent_vol = df['Volatility_20d'].tail(30).mean()
historical_vol = df['Volatility_20d'].mean()
print(f"\n🌊 VOLATILITY ENVIRONMENT:")
print(f"   Current 20-day vol: {recent_vol * 100:.2f}%")
print(f"   Historical avg: {historical_vol * 100:.2f}%")
if recent_vol < historical_vol * 0.8:
    print(f"            → Subdued volatility (may spike)")
elif recent_vol > historical_vol * 1.2:
    print(f"            → Elevated volatility (unstable)")
else:
    print(f"            → Normal volatility regime")

# ============================================================================
# KEY RISKS & WATCHLIST FOR 2025
# ============================================================================

print("\n" + "="*80)
print("⚠️  KEY RISKS TO MONITOR IN 2025")
print("="*80)

risks = []

# 1. Valuation/momentum risk
if ytd_return > 20:
    risks.append({
        'risk': 'Valuation Exhaustion',
        'severity': 'HIGH',
        'description': f'SPY up {ytd_return:.0f}% in 2024 - potential mean reversion',
        'watch': 'Earnings disappointments, P/E compression'
    })

# 2. VIX regime
current_vix = df['VIX'].iloc[-1]
if current_vix < 15:
    risks.append({
        'risk': 'Complacency (Low VIX)',
        'severity': 'MEDIUM',
        'description': f'VIX at {current_vix:.0f} suggests low fear - risk of sudden spike',
        'watch': 'VIX breaking above 20, term structure inversion'
    })

# 3. Currency/carry trade
recent_usdjpy_vol = df['USDJPY_Volatility_20d'].iloc[-1]
if recent_usdjpy_vol > 0.005:
    risks.append({
        'risk': 'Currency Volatility',
        'severity': 'HIGH',
        'description': 'USD/JPY showing elevated volatility - carry trade risk',
        'watch': 'Sharp yen appreciation (>2% in 5 days), Bank of Japan policy'
    })

# 4. Market breadth
if 'Market_Breadth' in df.columns:
    current_breadth = df['Market_Breadth'].iloc[-1]
    if current_breadth < -0.05:
        risks.append({
            'risk': 'Narrow Market Leadership',
            'severity': 'MEDIUM',
            'description': 'Market gains concentrated in few names',
            'watch': 'Mega-cap tech rotation, equal-weight vs cap-weight divergence'
        })

# 5. Technical exhaustion
if df['RSI_14'].iloc[-1] > 65:
    risks.append({
        'risk': 'Overbought Technicals',
        'severity': 'MEDIUM',
        'description': f'RSI at {df["RSI_14"].iloc[-1]:.0f} - potential pullback',
        'watch': 'Failed breakouts, negative divergences'
    })

# Display risks
for i, risk in enumerate(risks, 1):
    severity_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
    print(f"\n{i}. {risk['risk']} {severity_emoji[risk['severity']]} {risk['severity']}")
    print(f"   Description: {risk['description']}")
    print(f"   What to watch: {risk['watch']}")

# ============================================================================
# ACTIONABLE RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("💡 ACTIONABLE RECOMMENDATIONS FOR 2025")
print("="*80)

print(f"\n🎯 NEAR-TERM (Next 1-3 Months):")

if latest_ew > 0.6:
    print(f"   ⚠️  DEFENSIVE POSTURE RECOMMENDED")
    print(f"   • Consider reducing long exposure or taking profits")
    print(f"   • Increase cash position or buy protective puts")
    print(f"   • Model showing {latest_ew:.0%} probability of 2%+ drop")
else:
    print(f"   ✅ NEUTRAL TO MODERATELY BULLISH")
    print(f"   • Current risk levels manageable")
    print(f"   • Maintain diversified positions")
    print(f"   • Use pullbacks as entry opportunities")

print(f"\n📊 KEY LEVELS TO WATCH:")
recent_high = df.tail(60)['High'].max()
recent_low = df.tail(60)['Low'].min()
current = df['Close'].iloc[-1]

resistance = recent_high
support_1 = current * 0.98  # 2% down
support_2 = current * 0.95  # 5% down

print(f"   Resistance: ${resistance:.2f} (recent 60-day high)")
print(f"   Current:    ${current:.2f}")
print(f"   Support 1:  ${support_1:.2f} (-2%)")
print(f"   Support 2:  ${support_2:.2f} (-5%)")

print(f"\n🔍 DAILY MONITORING CHECKLIST:")
print(f"   [ ] VIX level and term structure")
print(f"   [ ] USD/JPY 5-day change (alert if >±2%)")
print(f"   [ ] Market breadth (RSP vs SPY)")
print(f"   [ ] Model early warning score")
print(f"   [ ] Sector rotation (defensive vs cyclical)")

print(f"\n⏰ POTENTIAL CATALYSTS IN Q1 2025:")
print(f"   • Fed policy decisions (rate path uncertainty)")
print(f"   • Q4 2024 earnings season (January)")
print(f"   • Geopolitical developments")
print(f"   • Seasonal pattern (January effect, tax-loss harvesting reversal)")

print(f"\n📈 BASE CASE SCENARIO (60% probability):")
print(f"   • Continued bull market with increased volatility")
print(f"   • 5-10% pullbacks are normal and healthy")
print(f"   • Early warning model provides 3-5 day advance notice")
print(f"   • Use systematic risk management")

print(f"\n⚠️  RISK SCENARIO (30% probability):")
print(f"   • 10-15% correction triggered by:")
print(f"     - Policy shock (Fed, fiscal, geopolitical)")
print(f"     - Valuation reset / profit-taking")
print(f"     - Currency volatility (carry trade unwind)")

print(f"\n🚀 BULLISH SCENARIO (10% probability):")
print(f"   • Melt-up continuation")
print(f"   • Low volatility, steady grind higher")
print(f"   • Model may show false positives")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Next update: Run this script with updated data")
