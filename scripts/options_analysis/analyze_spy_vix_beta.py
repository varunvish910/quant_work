#!/usr/bin/env python3
"""
Calculate and interpret rolling SPY/VIX beta
Analyze correlation regime changes and market conditions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SPY/VIX ROLLING BETA ANALYSIS")
print("="*80)

# Download data
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years of data

print(f"\n📊 Downloading SPY and VIX data...")
print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download SPY
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

# Download VIX (^VIX)
vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

print(f"   ✅ Downloaded {len(spy)} days of SPY data")
print(f"   ✅ Downloaded {len(vix)} days of VIX data")

# Calculate returns
spy['SPY_Return'] = spy['Close'].pct_change()
vix['VIX_Return'] = vix['Close'].pct_change()

# Merge data
data = pd.DataFrame({
    'SPY': spy['Close'],
    'VIX': vix['Close'],
    'SPY_Return': spy['SPY_Return'],
    'VIX_Return': vix['VIX_Return']
}).dropna()

print(f"   ✅ Merged data: {len(data)} trading days")

# =============================================================================
# CALCULATE ROLLING CORRELATIONS AND BETAS
# =============================================================================

print("\n" + "="*80)
print("CALCULATING ROLLING METRICS")
print("="*80)

# Calculate rolling correlations
data['Corr_20d'] = data['SPY_Return'].rolling(20).corr(data['VIX_Return'])
data['Corr_60d'] = data['SPY_Return'].rolling(60).corr(data['VIX_Return'])
data['Corr_120d'] = data['SPY_Return'].rolling(120).corr(data['VIX_Return'])

# Calculate rolling betas (VIX change per 1% SPY move)
def calculate_beta(window):
    """Calculate rolling beta between SPY and VIX"""
    cov = data['SPY_Return'].rolling(window).cov(data['VIX_Return'])
    var = data['SPY_Return'].rolling(window).var()
    return cov / var

data['Beta_20d'] = calculate_beta(20)
data['Beta_60d'] = calculate_beta(60)
data['Beta_120d'] = calculate_beta(120)

print(f"   ✅ Calculated 20-day, 60-day, and 120-day rolling metrics")

# =============================================================================
# CURRENT METRICS
# =============================================================================

print("\n" + "="*80)
print("CURRENT SPY/VIX RELATIONSHIP")
print("="*80)

latest = data.iloc[-1]

print(f"\n📊 CURRENT LEVELS:")
print(f"   SPY: ${latest['SPY']:.2f}")
print(f"   VIX: {latest['VIX']:.2f}")

print(f"\n📈 ROLLING CORRELATIONS:")
print(f"   20-day:  {latest['Corr_20d']:.3f}")
print(f"   60-day:  {latest['Corr_60d']:.3f}")
print(f"   120-day: {latest['Corr_120d']:.3f}")

# Interpret correlation
avg_corr = (latest['Corr_20d'] + latest['Corr_60d'] + latest['Corr_120d']) / 3
if avg_corr < -0.7:
    corr_interpretation = "Strong negative (normal market)"
elif avg_corr < -0.4:
    corr_interpretation = "Moderate negative (typical)"
elif avg_corr < -0.1:
    corr_interpretation = "Weak negative (regime change?)"
elif avg_corr < 0.1:
    corr_interpretation = "Near zero (decorrelated)"
else:
    corr_interpretation = "Positive (crisis/unusual)"

print(f"   → {corr_interpretation}")

print(f"\n📊 ROLLING BETAS (VIX change per 1% SPY move):")
print(f"   20-day:  {latest['Beta_20d']:.2f}")
print(f"   60-day:  {latest['Beta_60d']:.2f}")
print(f"   120-day: {latest['Beta_120d']:.2f}")

# Interpret beta
avg_beta = (latest['Beta_20d'] + latest['Beta_60d'] + latest['Beta_120d']) / 3
print(f"\n   Interpretation:")
print(f"   • For every 1% SPY decline, VIX increases by ~{abs(avg_beta):.1f}%")
print(f"   • For every 1% SPY rise, VIX decreases by ~{abs(avg_beta):.1f}%")

# =============================================================================
# HISTORICAL ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("HISTORICAL REGIME ANALYSIS")
print("="*80)

# Calculate percentiles
corr_percentile = (data['Corr_60d'] < latest['Corr_60d']).mean() * 100
beta_percentile = (data['Beta_60d'].abs() < abs(latest['Beta_60d'])).mean() * 100

print(f"\n📊 CURRENT VS HISTORICAL:")
print(f"   Current 60d correlation: {latest['Corr_60d']:.3f}")
print(f"   Historical percentile: {corr_percentile:.1f}%")
if corr_percentile > 80:
    print(f"   → Correlation weaker than usual (80th percentile)")
elif corr_percentile < 20:
    print(f"   → Correlation stronger than usual (20th percentile)")
else:
    print(f"   → Normal correlation range")

print(f"\n   Current 60d beta magnitude: {abs(latest['Beta_60d']):.2f}")
print(f"   Historical percentile: {beta_percentile:.1f}%")
if beta_percentile > 80:
    print(f"   → Beta higher than usual (more sensitive)")
elif beta_percentile < 20:
    print(f"   → Beta lower than usual (less sensitive)")
else:
    print(f"   → Normal beta range")

# =============================================================================
# REGIME DETECTION
# =============================================================================

print("\n" + "="*80)
print("MARKET REGIME DETECTION")
print("="*80)

# Recent trend analysis
recent_30d = data.tail(30)

print(f"\n📊 LAST 30 DAYS ANALYSIS:")

# Check for correlation breakdown
correlation_breakdown = False
if recent_30d['Corr_20d'].mean() > -0.3:
    correlation_breakdown = True
    print(f"   ⚠️ CORRELATION BREAKDOWN DETECTED")
    print(f"   • Average 20d correlation: {recent_30d['Corr_20d'].mean():.3f}")
    print(f"   • Normal range: -0.7 to -0.4")
    print(f"   • Implication: Market regime shift possible")

# Check for beta instability
beta_std = recent_30d['Beta_20d'].std()
if beta_std > 3:
    print(f"   ⚠️ HIGH BETA INSTABILITY")
    print(f"   • Beta standard deviation: {beta_std:.2f}")
    print(f"   • Implication: Volatile relationship, uncertain market")

# Trend in correlation
corr_trend = recent_30d['Corr_60d'].iloc[-1] - recent_30d['Corr_60d'].iloc[0]
if corr_trend > 0.2:
    print(f"   📈 WEAKENING NEGATIVE CORRELATION")
    print(f"   • 30-day change: {corr_trend:+.3f}")
    print(f"   • Implication: Risk of volatility spike")
elif corr_trend < -0.2:
    print(f"   📉 STRENGTHENING NEGATIVE CORRELATION")
    print(f"   • 30-day change: {corr_trend:+.3f}")
    print(f"   • Implication: Return to normal regime")
else:
    print(f"   → Stable correlation regime")

# =============================================================================
# KEY PERIODS ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("KEY PERIODS COMPARISON")
print("="*80)

# Find extreme correlation periods
high_corr_periods = data[data['Corr_60d'] > -0.2]
crisis_periods = data[data['Corr_60d'] > 0]

print(f"\n📊 HISTORICAL EXTREMES:")
print(f"   Days with weak correlation (>-0.2): {len(high_corr_periods)} ({len(high_corr_periods)/len(data)*100:.1f}%)")
print(f"   Days with positive correlation: {len(crisis_periods)} ({len(crisis_periods)/len(data)*100:.1f}%)")

# Recent extremes
if len(high_corr_periods) > 0:
    recent_extreme = high_corr_periods.tail(1).index[0]
    days_since = (data.index[-1] - recent_extreme).days
    print(f"   Last extreme weak correlation: {recent_extreme.strftime('%Y-%m-%d')} ({days_since} days ago)")

# =============================================================================
# SIGNALS AND INTERPRETATION
# =============================================================================

print("\n" + "="*80)
print("SIGNALS & MARKET INTERPRETATION")
print("="*80)

print(f"\n🎯 KEY SIGNALS:")

# Signal 1: Correlation regime
print(f"\n1. CORRELATION REGIME:")
if latest['Corr_60d'] < -0.6:
    print(f"   ✅ Normal inverse relationship intact")
    print(f"   • Market functioning normally")
    print(f"   • VIX providing portfolio hedge")
elif latest['Corr_60d'] < -0.3:
    print(f"   🟡 Weakening inverse relationship")
    print(f"   • Some regime uncertainty")
    print(f"   • Monitor for further breakdown")
else:
    print(f"   🔴 Correlation breakdown")
    print(f"   • Abnormal market conditions")
    print(f"   • VIX may not hedge effectively")
    print(f"   • Consider alternative hedges")

# Signal 2: Beta magnitude
print(f"\n2. BETA MAGNITUDE:")
abs_beta = abs(latest['Beta_60d'])
if abs_beta > 5:
    print(f"   🔴 Extreme sensitivity ({abs_beta:.1f})")
    print(f"   • Small SPY moves cause large VIX swings")
    print(f"   • High uncertainty/fear in market")
elif abs_beta > 3:
    print(f"   🟡 Elevated sensitivity ({abs_beta:.1f})")
    print(f"   • Above normal VIX reactions")
    print(f"   • Market on edge")
else:
    print(f"   ✅ Normal sensitivity ({abs_beta:.1f})")
    print(f"   • Typical VIX responses")

# Signal 3: Trend
print(f"\n3. TREND ANALYSIS:")
if correlation_breakdown:
    print(f"   🔴 WARNING: Correlation regime breaking down")
    print(f"   • Risk of sudden volatility spike")
    print(f"   • Traditional hedges may fail")
elif corr_trend > 0.1:
    print(f"   🟡 Correlation weakening")
    print(f"   • Building pressure")
    print(f"   • Watch for regime shift")
else:
    print(f"   ✅ Stable regime")
    print(f"   • Normal market dynamics")

# =============================================================================
# ACTIONABLE INSIGHTS
# =============================================================================

print("\n" + "="*80)
print("ACTIONABLE INSIGHTS")
print("="*80)

print(f"\n💡 PORTFOLIO IMPLICATIONS:")

if latest['Corr_60d'] < -0.5 and abs_beta < 4:
    print(f"\n✅ NORMAL MARKET CONDITIONS")
    print(f"   • VIX products work as portfolio hedge")
    print(f"   • Standard risk management applies")
    print(f"   • Typical 10-15% VIX allocation for hedging")

elif latest['Corr_60d'] > -0.3 or abs_beta > 5:
    print(f"\n⚠️ ABNORMAL CONDITIONS DETECTED")
    print(f"   • VIX hedging effectiveness reduced")
    print(f"   • Consider:")
    print(f"     - Reducing overall exposure")
    print(f"     - Alternative hedges (puts, gold, bonds)")
    print(f"     - Wider stop losses (high beta = whipsaws)")

if correlation_breakdown:
    print(f"\n🔴 REGIME BREAKDOWN")
    print(f"   • HIGH RISK of volatility event")
    print(f"   • Reduce leverage immediately")
    print(f"   • Increase cash allocation")
    print(f"   • Avoid VIX shorts")

# Trading signals based on beta
print(f"\n📈 TRADING SIGNALS:")
if abs_beta > 4:
    print(f"   • High beta ({abs_beta:.1f}) = Mean reversion trades")
    print(f"   • Sell VIX spikes, buy SPY dips")
elif abs_beta < 2:
    print(f"   • Low beta ({abs_beta:.1f}) = Trending environment")
    print(f"   • Follow momentum, avoid mean reversion")
else:
    print(f"   • Normal beta = Standard strategies")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n🎯 CURRENT STATE:")
print(f"   SPY/VIX Correlation: {latest['Corr_60d']:.3f} ({corr_interpretation})")
print(f"   Beta (60d): {latest['Beta_60d']:.2f}")
print(f"   Regime: ", end="")

if correlation_breakdown:
    print("🔴 ABNORMAL - High Risk")
elif latest['Corr_60d'] > -0.4:
    print("🟡 TRANSITIONAL - Caution")
else:
    print("✅ NORMAL - Standard Risk")

print(f"\n📊 INTERPRETATION:")
print(f"   • Every 1% SPY drop → VIX rises ~{abs(latest['Beta_60d']):.1f}%")
print(f"   • Historical context: {corr_percentile:.0f}th percentile correlation")
print(f"   • Beta sensitivity: {beta_percentile:.0f}th percentile")

if correlation_breakdown or latest['Corr_60d'] > -0.3:
    print(f"\n⚠️ ACTION REQUIRED:")
    print(f"   • Review hedging strategy")
    print(f"   • Consider reducing exposure")
    print(f"   • Monitor closely for regime confirmation")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")