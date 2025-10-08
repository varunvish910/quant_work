#!/usr/bin/env python3
"""
Calculate SPY implied move and realized volatility metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gzip
from pathlib import Path

def load_oct6_trades():
    """Load October 6 SPY options trades"""
    file_path = Path('trade_and_quote_data/data_management/flatfiles/2025-10-06.csv.gz')

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path, compression='gzip')
    df['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')
    df['date'] = df['timestamp'].dt.date

    return df

def parse_spy_ticker(ticker):
    """Parse SPY option ticker: O:SPY241011C00670000"""
    if not ticker.startswith('O:SPY'):
        return None

    parts = ticker.replace('O:SPY', '')

    try:
        # Extract expiry (YYMMDD)
        expiry_str = parts[:6]
        year = int('20' + expiry_str[:2])
        month = int(expiry_str[2:4])
        day = int(expiry_str[4:6])
        expiry = datetime(year, month, day).date()

        # Extract type and strike
        option_type = parts[6].lower()
        strike = int(parts[7:]) / 1000.0

        return {
            'expiry': expiry,
            'option_type': option_type,
            'strike': strike,
            'dte': (expiry - datetime.now().date()).days
        }
    except:
        return None

def calculate_implied_move(trades, current_price=669.21):
    """
    Calculate implied move from ATM straddle

    Implied Move = (ATM Call Premium + ATM Put Premium) / Stock Price
    """
    print("\n" + "="*80)
    print("CALCULATING IMPLIED MOVE FROM SPY OPTIONS")
    print("="*80)

    # Parse all tickers
    trades['parsed'] = trades['ticker'].apply(parse_spy_ticker)
    trades = trades[trades['parsed'].notna()].copy()

    # Extract parsed fields
    trades['expiry'] = trades['parsed'].apply(lambda x: x['expiry'])
    trades['option_type'] = trades['parsed'].apply(lambda x: x['option_type'])
    trades['strike'] = trades['parsed'].apply(lambda x: x['strike'])
    trades['dte'] = trades['parsed'].apply(lambda x: x['dte'])

    # Focus on this week's expiry (closest)
    min_dte = trades[trades['dte'] > 0]['dte'].min()
    weekly = trades[trades['dte'] == min_dte].copy()

    expiry_date = weekly['expiry'].iloc[0]
    dte = min_dte

    print(f"\n📅 WEEKLY EXPIRY: {expiry_date} ({dte} DTE)")
    print(f"📍 Current SPY: ${current_price:.2f}")

    # Find ATM strike (closest to current price)
    strikes = weekly['strike'].unique()
    atm_strike = min(strikes, key=lambda x: abs(x - current_price))

    print(f"🎯 ATM Strike: ${atm_strike:.2f}")

    # Get ATM call and put prices (use VWAP)
    atm_calls = weekly[(weekly['strike'] == atm_strike) & (weekly['option_type'] == 'c')]
    atm_puts = weekly[(weekly['strike'] == atm_strike) & (weekly['option_type'] == 'p')]

    if len(atm_calls) > 0 and len(atm_puts) > 0:
        # Volume-weighted average price
        call_vwap = (atm_calls['price'] * atm_calls['size']).sum() / atm_calls['size'].sum()
        put_vwap = (atm_puts['price'] * atm_puts['size']).sum() / atm_puts['size'].sum()

        straddle_price = call_vwap + put_vwap
        implied_move_pct = (straddle_price / current_price) * 100
        implied_move_pts = straddle_price

        # Annualized IV approximation (simplified)
        # IV ≈ Straddle / (Stock * sqrt(DTE/365))
        iv_approx = (straddle_price / current_price) / np.sqrt(dte / 365) * 100

        # Daily implied move
        daily_move = implied_move_pts / np.sqrt(dte)
        daily_move_pct = (daily_move / current_price) * 100

        print(f"\n📊 ATM STRADDLE PRICES:")
        print(f"   Call ({atm_strike}): ${call_vwap:.2f}")
        print(f"   Put ({atm_strike}): ${put_vwap:.2f}")
        print(f"   Straddle: ${straddle_price:.2f}")

        print(f"\n📏 IMPLIED MOVE (to {expiry_date}):")
        print(f"   ±${implied_move_pts:.2f} (±{implied_move_pct:.2f}%)")
        print(f"   Range: ${current_price - implied_move_pts:.2f} - ${current_price + implied_move_pts:.2f}")

        print(f"\n📅 DAILY IMPLIED MOVE:")
        print(f"   ±${daily_move:.2f} (±{daily_move_pct:.2f}%) per day")

        print(f"\n📈 IMPLIED VOLATILITY (Approximate):")
        print(f"   ~{iv_approx:.1f}% annualized")

        return {
            'expiry': expiry_date,
            'dte': dte,
            'atm_strike': atm_strike,
            'call_price': call_vwap,
            'put_price': put_vwap,
            'straddle_price': straddle_price,
            'implied_move_pct': implied_move_pct,
            'implied_move_pts': implied_move_pts,
            'daily_move_pct': daily_move_pct,
            'daily_move_pts': daily_move,
            'iv_approx': iv_approx
        }
    else:
        print(f"❌ Could not find ATM call/put prices for strike ${atm_strike}")
        return None

def analyze_realized_volatility():
    """Calculate realized volatility from recent price action"""
    import yfinance as yf

    print("\n" + "="*80)
    print("REALIZED VOLATILITY ANALYSIS")
    print("="*80)

    # Get recent SPY data
    spy = yf.download('SPY', period='60d', progress=False)

    # Calculate returns
    spy['Return'] = spy['Close'].pct_change()

    # Calculate realized volatility (different periods)
    rv_5d = spy['Return'].tail(5).std() * np.sqrt(252) * 100
    rv_10d = spy['Return'].tail(10).std() * np.sqrt(252) * 100
    rv_20d = spy['Return'].tail(20).std() * np.sqrt(252) * 100
    rv_30d = spy['Return'].tail(30).std() * np.sqrt(252) * 100

    print(f"\n📊 REALIZED VOLATILITY (Annualized):")
    print(f"   5-day:  {rv_5d:.1f}%")
    print(f"   10-day: {rv_10d:.1f}%")
    print(f"   20-day: {rv_20d:.1f}%")
    print(f"   30-day: {rv_30d:.1f}%")

    # Get VIX
    vix = yf.download('^VIX', period='5d', progress=False)
    current_vix = float(vix['Close'].iloc[-1])

    print(f"\n📈 CURRENT VIX: {current_vix:.1f}%")

    # Calculate vol premium
    vol_premium = current_vix - rv_20d

    print(f"\n💰 VOLATILITY PREMIUM (VIX - RV20):")
    print(f"   {vol_premium:+.1f}% ({vol_premium/rv_20d*100:+.1f}% relative)")

    if vol_premium > 5:
        print(f"   → HIGH premium: Options expensive relative to realized")
    elif vol_premium > 2:
        print(f"   → NORMAL premium: Fair pricing")
    elif vol_premium < 0:
        print(f"   → NEGATIVE premium: Options cheap, expect volatility spike")

    return {
        'rv_5d': rv_5d,
        'rv_10d': rv_10d,
        'rv_20d': rv_20d,
        'rv_30d': rv_30d,
        'vix': current_vix,
        'vol_premium': vol_premium
    }

def momentum_indicators(implied_data, rv_data):
    """Determine bullish/bearish momentum indicators"""
    print("\n" + "="*80)
    print("MOMENTUM INDICATORS")
    print("="*80)

    print(f"\n🐂 BULLISH SIGNALS (would indicate upward momentum):")
    print(f"   ✓ SPY breaks above {implied_data['atm_strike'] + implied_data['implied_move_pts']:.2f}")
    print(f"   ✓ VIX drops below {rv_data['vix'] - 2:.1f}")
    print(f"   ✓ Realized vol stays below {rv_data['rv_20d']:.1f}%")
    print(f"   ✓ Daily moves < ±{implied_data['daily_move_pct']:.2f}% (calm market)")

    print(f"\n🐻 BEARISH SIGNALS (would indicate downward momentum):")
    print(f"   ✓ SPY breaks below {implied_data['atm_strike'] - implied_data['implied_move_pts']:.2f}")
    print(f"   ✓ VIX spikes above {rv_data['vix'] + 3:.1f}")
    print(f"   ✓ Realized vol rises above {rv_data['rv_20d'] * 1.3:.1f}%")
    print(f"   ✓ Daily moves > ±{implied_data['daily_move_pct'] * 1.5:.2f}% (increased volatility)")

    print(f"\n📈 WHAT WOULD CAUSE MEANINGFUL RV RISE:")
    print(f"   1. Event catalyst:")
    print(f"      • Economic data surprise (CPI, jobs, Fed)")
    print(f"      • Geopolitical shock")
    print(f"      • Corporate earnings miss")
    print(f"   ")
    print(f"   2. Technical breakdown:")
    print(f"      • Break below ${implied_data['atm_strike'] - implied_data['implied_move_pts']:.2f}")
    print(f"      • Multiple days outside implied move range")
    print(f"      • Gap down >1.5%")
    print(f"   ")
    print(f"   3. VIX term structure shift:")
    print(f"      • VIX >20 (from current {rv_data['vix']:.1f})")
    print(f"      • Backwardation appears")
    print(f"      • Vol premium turns negative")

    # Calculate what RV needs to reach to match VIX
    rv_target = rv_data['vix']
    current_rv = rv_data['rv_20d']
    rv_increase_needed = rv_target - current_rv

    print(f"\n🎯 RV CONVERGENCE TO VIX:")
    print(f"   Current RV (20d): {current_rv:.1f}%")
    print(f"   Current VIX:      {rv_data['vix']:.1f}%")
    print(f"   Gap to close:     {rv_increase_needed:+.1f}%")

    if rv_increase_needed > 3:
        daily_move_needed = rv_increase_needed / np.sqrt(252) * np.sqrt(5)  # 5-day move
        print(f"   → Would need ~{daily_move_needed:.1f}% daily moves for next week")

def main():
    # Load data
    trades = load_oct6_trades()

    if trades is None:
        print("❌ Could not load trade data")
        return

    # Calculate implied move
    implied_data = calculate_implied_move(trades)

    if implied_data is None:
        print("❌ Could not calculate implied move")
        return

    # Analyze realized volatility
    rv_data = analyze_realized_volatility()

    # Momentum indicators
    momentum_indicators(implied_data, rv_data)

    print("\n" + "="*80)
    print("✅ Analysis complete")
    print("="*80)

if __name__ == '__main__':
    main()
