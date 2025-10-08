#!/usr/bin/env python3
"""
Real-Time SPY and SMH Options Analysis
Using yfinance to get current options market signals

Questions to answer:
1. Is SPY options market showing defensive positioning?
2. Is SMH (semiconductors) showing warning signs?
3. Any unusual put hedging or call buying?
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SPY & SMH OPTIONS MARKET ANALYSIS - REAL-TIME")
print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

def analyze_options_chain(ticker_symbol: str) -> dict:
    """Analyze options chain for a given ticker"""
    print(f"\n{'='*80}")
    print(f"Analyzing {ticker_symbol} Options Market")
    print(f"{'='*80}")

    ticker = yf.Ticker(ticker_symbol)

    # Get current price
    hist = ticker.history(period='5d')
    current_price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
    daily_change = (current_price / prev_close - 1) * 100

    print(f"\n📊 Current Price: ${current_price:.2f} ({daily_change:+.2f}%)")

    # Get options expirations
    expirations = ticker.options

    if not expirations or len(expirations) == 0:
        print(f"   ❌ No options data available for {ticker_symbol}")
        return None

    print(f"   Available expirations: {len(expirations)}")

    # Analyze near-term options (first 3 expirations)
    results = {
        'ticker': ticker_symbol,
        'current_price': current_price,
        'daily_change_pct': daily_change,
        'expirations_analyzed': []
    }

    for i, exp_date in enumerate(expirations[:3]):
        print(f"\n   📅 Expiration {i+1}: {exp_date}")

        try:
            chain = ticker.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                print(f"      ⚠️  Empty options chain")
                continue

            # Filter for ATM options (within 5% of current price)
            atm_range = (current_price * 0.95, current_price * 1.05)

            atm_calls = calls[
                (calls['strike'] >= atm_range[0]) &
                (calls['strike'] <= atm_range[1])
            ]

            atm_puts = puts[
                (puts['strike'] >= atm_range[0]) &
                (puts['strike'] <= atm_range[1])
            ]

            # Calculate key metrics
            total_call_volume = calls['volume'].fillna(0).sum()
            total_put_volume = puts['volume'].fillna(0).sum()
            total_call_oi = calls['openInterest'].fillna(0).sum()
            total_put_oi = puts['openInterest'].fillna(0).sum()

            put_call_volume_ratio = total_put_volume / max(total_call_volume, 1)
            put_call_oi_ratio = total_put_oi / max(total_call_oi, 1)

            # ATM IV comparison
            atm_call_iv = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else np.nan
            atm_put_iv = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else np.nan
            iv_skew = atm_put_iv - atm_call_iv if not np.isnan(atm_put_iv) and not np.isnan(atm_call_iv) else np.nan

            print(f"      Call Volume: {total_call_volume:,.0f}")
            print(f"      Put Volume:  {total_put_volume:,.0f}")
            print(f"      P/C Volume Ratio: {put_call_volume_ratio:.2f}")
            print(f"      P/C OI Ratio: {put_call_oi_ratio:.2f}")
            print(f"      ATM Call IV: {atm_call_iv*100:.1f}%" if not np.isnan(atm_call_iv) else "      ATM Call IV: N/A")
            print(f"      ATM Put IV: {atm_put_iv*100:.1f}%" if not np.isnan(atm_put_iv) else "      ATM Put IV: N/A")
            print(f"      IV Skew: {iv_skew*100:+.2f}%" if not np.isnan(iv_skew) else "      IV Skew: N/A")

            results['expirations_analyzed'].append({
                'expiration': exp_date,
                'call_volume': total_call_volume,
                'put_volume': total_put_volume,
                'pc_volume_ratio': put_call_volume_ratio,
                'call_oi': total_call_oi,
                'put_oi': total_put_oi,
                'pc_oi_ratio': put_call_oi_ratio,
                'atm_call_iv': atm_call_iv,
                'atm_put_iv': atm_put_iv,
                'iv_skew': iv_skew
            })

        except Exception as e:
            print(f"      ❌ Error analyzing {exp_date}: {e}")
            continue

    return results

# Analyze SPY
spy_results = analyze_options_chain('SPY')

# Analyze SMH (Semiconductors)
smh_results = analyze_options_chain('SMH')

# ============================================================================
# Comparative Analysis
# ============================================================================

print(f"\n{'='*80}")
print("COMPARATIVE ANALYSIS & SIGNALS")
print(f"{'='*80}")

if spy_results and smh_results:
    if spy_results['expirations_analyzed'] and smh_results['expirations_analyzed']:
        # Get nearest expiration data
        spy_near = spy_results['expirations_analyzed'][0]
        smh_near = smh_results['expirations_analyzed'][0]

        print(f"\n📊 SPY vs SMH Options Comparison (Nearest Expiration):")
        print(f"\n   Put/Call Volume Ratio:")
        print(f"      SPY: {spy_near['pc_volume_ratio']:.2f}")
        print(f"      SMH: {smh_near['pc_volume_ratio']:.2f}")

        if smh_near['pc_volume_ratio'] > spy_near['pc_volume_ratio'] * 1.3:
            print(f"      🚨 SMH showing significantly more put hedging than SPY!")
            print(f"         → Semiconductors may be leading a correction")
        elif smh_near['pc_volume_ratio'] > 1.2 and spy_near['pc_volume_ratio'] < 1.0:
            print(f"      ⚠️  SMH defensive while SPY still call-heavy")
            print(f"         → Divergence: Semis may be warning signal")
        else:
            print(f"      ✅ No significant divergence")

        print(f"\n   Put/Call Open Interest Ratio:")
        print(f"      SPY: {spy_near['pc_oi_ratio']:.2f}")
        print(f"      SMH: {smh_near['pc_oi_ratio']:.2f}")

        if smh_near['pc_oi_ratio'] > 1.3:
            print(f"      ⚠️  High SMH put OI - traders positioned defensively")
        if spy_near['pc_oi_ratio'] > 1.2:
            print(f"      ⚠️  High SPY put OI - defensive positioning")

        print(f"\n   Implied Volatility Skew (Put IV - Call IV):")
        if not np.isnan(spy_near['iv_skew']):
            print(f"      SPY: {spy_near['iv_skew']*100:+.2f}%")
        if not np.isnan(smh_near['iv_skew']):
            print(f"      SMH: {smh_near['iv_skew']*100:+.2f}%")

        if not np.isnan(smh_near['iv_skew']) and smh_near['iv_skew'] > 0.05:
            print(f"      🚨 High SMH put skew - fear premium in semis")
        if not np.isnan(spy_near['iv_skew']) and spy_near['iv_skew'] > 0.03:
            print(f"      ⚠️  Elevated SPY put skew - market pricing downside risk")

print(f"\n{'='*80}")
print("INTERPRETATION & SIGNALS")
print(f"{'='*80}")

if spy_results and spy_results['expirations_analyzed']:
    spy_near = spy_results['expirations_analyzed'][0]

    print(f"\n🔍 SPY Options Market Signals:")

    # Put/Call Volume interpretation
    if spy_near['pc_volume_ratio'] > 1.2:
        print(f"   🔴 HIGH PUT BUYING (P/C: {spy_near['pc_volume_ratio']:.2f})")
        print(f"      → Traders actively hedging / defensive positioning")
        print(f"      → Elevated fear in options market")
    elif spy_near['pc_volume_ratio'] > 0.9:
        print(f"   🟡 BALANCED PUT/CALL ACTIVITY (P/C: {spy_near['pc_volume_ratio']:.2f})")
        print(f"      → Neutral positioning")
    else:
        print(f"   🟢 CALL BUYING DOMINATES (P/C: {spy_near['pc_volume_ratio']:.2f})")
        print(f"      → Traders still bullish / chasing upside")
        print(f"      → Low hedging activity (complacent?)")

    # Open Interest interpretation
    if spy_near['pc_oi_ratio'] > 1.3:
        print(f"\n   🔴 HIGH PUT OPEN INTEREST (P/C OI: {spy_near['pc_oi_ratio']:.2f})")
        print(f"      → Large protective positions in place")
        print(f"      → Market participants positioned for downside")
    elif spy_near['pc_oi_ratio'] > 1.0:
        print(f"\n   🟡 MODERATE PUT POSITIONS (P/C OI: {spy_near['pc_oi_ratio']:.2f})")
        print(f"      → Some hedging in place")
    else:
        print(f"\n   🟢 CALL POSITIONS DOMINANT (P/C OI: {spy_near['pc_oi_ratio']:.2f})")
        print(f"      → Bullish positioning still prevalent")

if smh_results and smh_results['expirations_analyzed']:
    smh_near = smh_results['expirations_analyzed'][0]

    print(f"\n🔍 SMH (Semiconductors) Warning Signs:")

    warning_count = 0

    # Check for warning signs
    if smh_near['pc_volume_ratio'] > 1.2:
        print(f"   ⚠️  Warning 1: High put volume (P/C: {smh_near['pc_volume_ratio']:.2f})")
        warning_count += 1

    if smh_near['pc_oi_ratio'] > 1.3:
        print(f"   ⚠️  Warning 2: High put OI (P/C OI: {smh_near['pc_oi_ratio']:.2f})")
        warning_count += 1

    if not np.isnan(smh_near['iv_skew']) and smh_near['iv_skew'] > 0.05:
        print(f"   ⚠️  Warning 3: Elevated put IV skew ({smh_near['iv_skew']*100:.1f}%)")
        warning_count += 1

    if smh_results['daily_change_pct'] < -1.0:
        print(f"   ⚠️  Warning 4: Negative price action ({smh_results['daily_change_pct']:.2f}%)")
        warning_count += 1

    if spy_results and smh_results['daily_change_pct'] < spy_results['daily_change_pct'] - 0.5:
        print(f"   ⚠️  Warning 5: Underperforming SPY (SMH: {smh_results['daily_change_pct']:.2f}% vs SPY: {spy_results['daily_change_pct']:.2f}%)")
        warning_count += 1

    print(f"\n   📊 Total Warning Signs: {warning_count}/5")

    if warning_count >= 3:
        print(f"   🚨 SMH SHOWING MULTIPLE WARNING SIGNS")
        print(f"      → Semiconductors may be leading a correction")
        print(f"      → Historically, SMH weakness precedes SPY pullbacks")
        print(f"      → Consider this a leading indicator")
    elif warning_count >= 1:
        print(f"   ⚠️  Some SMH warning signs present")
        print(f"      → Monitor for deterioration")
    else:
        print(f"   ✅ No significant SMH warning signs")

print(f"\n{'='*80}")
print(f"Analysis Complete")
print(f"{'='*80}")
