#!/usr/bin/env python3
"""
PHASES 2.1-2.5: Implement Advanced Options Features

2.1: IV Calculator (Black-Scholes)
2.2: IV Skew Features (25-delta put/call differential)
2.3: Weekly Implied Moves (ATM straddle)
2.4: Put/Call Volume and OI Ratios
2.5: IV Term Structure Features (VIX vs VIX3M)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("📊 PHASES 2.1-2.5: ADVANCED OPTIONS FEATURES")
print("=" * 80)
print()

# ============================================================================
# PHASE 2.1: IV CALCULATOR (BLACK-SCHOLES)
# ============================================================================
print("=" * 80)
print("PHASE 2.1: IMPLEMENTING IV CALCULATOR (BLACK-SCHOLES)")
print("=" * 80)
print()

class BlackScholesIV:
    """
    Black-Scholes Implied Volatility Calculator
    """
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate (can be made dynamic)
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes call option price"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Black-Scholes put option price"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return put_price
    
    def calculate_iv(self, option_price, S, K, T, option_type='call', max_iter=100, tolerance=1e-6):
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            option_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            option_type: 'call' or 'put'
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
        
        Returns:
            Implied volatility (annualized)
        """
        if T <= 0:
            return np.nan
        
        # Initial guess
        sigma = 0.2
        
        for i in range(max_iter):
            if option_type.lower() == 'call':
                price = self.black_scholes_call(S, K, T, self.risk_free_rate, sigma)
                # Vega (derivative of price w.r.t. volatility)
                d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                vega = S * np.sqrt(T) * norm.pdf(d1)
            else:
                price = self.black_scholes_put(S, K, T, self.risk_free_rate, sigma)
                d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                vega = S * np.sqrt(T) * norm.pdf(d1)
            
            # Newton-Raphson update
            price_diff = price - option_price
            if abs(price_diff) < tolerance:
                break
            
            if vega == 0:
                break
            
            sigma = sigma - price_diff / vega
            
            # Keep sigma positive
            sigma = max(sigma, 0.001)
        
        return sigma if abs(price_diff) < tolerance else np.nan

print("✅ Black-Scholes IV Calculator implemented")
print()

# ============================================================================
# PHASE 2.2: IV SKEW FEATURES
# ============================================================================
print("=" * 80)
print("PHASE 2.2: CALCULATING IV SKEW FEATURES")
print("=" * 80)
print()

def calculate_iv_skew_features(options_data, spot_price, risk_free_rate=0.05):
    """
    Calculate IV skew features for a given options chain
    
    Args:
        options_data: DataFrame with options data
        spot_price: Current SPY price
        risk_free_rate: Risk-free rate
    
    Returns:
        Dictionary of IV skew features
    """
    iv_calc = BlackScholesIV()
    iv_calc.risk_free_rate = risk_free_rate
    
    features = {}
    
    # Filter for liquid options (volume > 0)
    liquid_options = options_data[options_data['volume'] > 0].copy()
    
    if len(liquid_options) == 0:
        return {key: np.nan for key in [
            'iv_skew_25d', 'iv_skew_10d', 'iv_skew_50d',
            'put_iv_25d', 'call_iv_25d', 'iv_smile_curvature',
            'iv_slope_25d', 'iv_slope_10d'
        ]}
    
    # Calculate time to expiration
    liquid_options['days_to_exp'] = (pd.to_datetime(liquid_options['expiration']) - pd.Timestamp.now()).dt.days
    liquid_options['time_to_exp'] = liquid_options['days_to_exp'] / 365.25
    
    # Filter for reasonable time to expiration (7-60 days)
    liquid_options = liquid_options[
        (liquid_options['time_to_exp'] >= 7/365.25) & 
        (liquid_options['time_to_exp'] <= 60/365.25)
    ]
    
    if len(liquid_options) == 0:
        return {key: np.nan for key in [
            'iv_skew_25d', 'iv_skew_10d', 'iv_skew_50d',
            'put_iv_25d', 'call_iv_25d', 'iv_smile_curvature',
            'iv_slope_25d', 'iv_slope_10d'
        ]}
    
    # Calculate implied volatilities
    liquid_options['implied_vol'] = np.nan
    
    for idx, row in liquid_options.iterrows():
        if row['time_to_exp'] > 0:
            iv = iv_calc.calculate_iv(
                option_price=row['close'],
                S=spot_price,
                K=row['strike'],
                T=row['time_to_exp'],
                option_type=row['option_type']
            )
            liquid_options.loc[idx, 'implied_vol'] = iv
    
    # Remove invalid IVs
    liquid_options = liquid_options.dropna(subset=['implied_vol'])
    
    if len(liquid_options) == 0:
        return {key: np.nan for key in [
            'iv_skew_25d', 'iv_skew_10d', 'iv_skew_50d',
            'put_iv_25d', 'call_iv_25d', 'iv_smile_curvature',
            'iv_slope_25d', 'iv_slope_10d'
        ]}
    
    # Calculate moneyness (strike/spot)
    liquid_options['moneyness'] = liquid_options['strike'] / spot_price
    
    # Separate puts and calls
    puts = liquid_options[liquid_options['option_type'] == 'P'].copy()
    calls = liquid_options[liquid_options['option_type'] == 'C'].copy()
    
    # Calculate IV skew features
    try:
        # 25-delta skew (approximate using moneyness)
        put_25d = puts[puts['moneyness'] >= 0.95].sort_values('moneyness')
        call_25d = calls[calls['moneyness'] <= 1.05].sort_values('moneyness')
        
        if len(put_25d) > 0 and len(call_25d) > 0:
            put_iv_25d = put_25d['implied_vol'].iloc[0]
            call_iv_25d = call_25d['implied_vol'].iloc[0]
            features['iv_skew_25d'] = put_iv_25d - call_iv_25d
            features['put_iv_25d'] = put_iv_25d
            features['call_iv_25d'] = call_iv_25d
        else:
            features['iv_skew_25d'] = np.nan
            features['put_iv_25d'] = np.nan
            features['call_iv_25d'] = np.nan
        
        # 10-delta skew (more extreme)
        put_10d = puts[puts['moneyness'] >= 0.90].sort_values('moneyness')
        call_10d = calls[calls['moneyness'] <= 1.10].sort_values('moneyness')
        
        if len(put_10d) > 0 and len(call_10d) > 0:
            put_iv_10d = put_10d['implied_vol'].iloc[0]
            call_iv_10d = call_10d['implied_vol'].iloc[0]
            features['iv_skew_10d'] = put_iv_10d - call_iv_10d
        else:
            features['iv_skew_10d'] = np.nan
        
        # 50-delta (ATM) skew
        atm_puts = puts[(puts['moneyness'] >= 0.98) & (puts['moneyness'] <= 1.02)]
        atm_calls = calls[(calls['moneyness'] >= 0.98) & (calls['moneyness'] <= 1.02)]
        
        if len(atm_puts) > 0 and len(atm_calls) > 0:
            put_iv_50d = atm_puts['implied_vol'].mean()
            call_iv_50d = atm_calls['implied_vol'].mean()
            features['iv_skew_50d'] = put_iv_50d - call_iv_50d
        else:
            features['iv_skew_50d'] = np.nan
        
        # IV smile curvature (second derivative approximation)
        all_options = liquid_options.sort_values('moneyness')
        if len(all_options) >= 3:
            # Use 3 points to estimate curvature
            mid_idx = len(all_options) // 2
            left_iv = all_options['implied_vol'].iloc[max(0, mid_idx-1)]
            mid_iv = all_options['implied_vol'].iloc[mid_idx]
            right_iv = all_options['implied_vol'].iloc[min(len(all_options)-1, mid_idx+1)]
            
            features['iv_smile_curvature'] = (left_iv - 2*mid_iv + right_iv) / 2
        else:
            features['iv_smile_curvature'] = np.nan
        
        # IV slope (first derivative approximation)
        if len(all_options) >= 2:
            features['iv_slope_25d'] = (features['put_iv_25d'] - features['call_iv_25d']) / 0.1
            features['iv_slope_10d'] = (features['iv_skew_10d']) / 0.2
        else:
            features['iv_slope_25d'] = np.nan
            features['iv_slope_10d'] = np.nan
            
    except Exception as e:
        print(f"   ⚠️  Error calculating IV skew: {e}")
        features = {key: np.nan for key in [
            'iv_skew_25d', 'iv_skew_10d', 'iv_skew_50d',
            'put_iv_25d', 'call_iv_25d', 'iv_smile_curvature',
            'iv_slope_25d', 'iv_slope_10d'
        ]}
    
    return features

print("✅ IV Skew features calculator implemented")
print()

# ============================================================================
# PHASE 2.3: WEEKLY IMPLIED MOVES (ATM STRADDLE)
# ============================================================================
print("=" * 80)
print("PHASE 2.3: CALCULATING WEEKLY IMPLIED MOVES")
print("=" * 80)
print()

def calculate_implied_moves(options_data, spot_price, risk_free_rate=0.05):
    """
    Calculate implied moves from ATM straddles
    
    Args:
        options_data: DataFrame with options data
        spot_price: Current SPY price
        risk_free_rate: Risk-free rate
    
    Returns:
        Dictionary of implied move features
    """
    features = {}
    
    # Filter for liquid options
    liquid_options = options_data[options_data['volume'] > 0].copy()
    
    if len(liquid_options) == 0:
        return {key: np.nan for key in [
            'atm_straddle_price', 'implied_move_1w', 'implied_move_2w', 'implied_move_1m',
            'straddle_skew', 'strangle_skew'
        ]}
    
    # Calculate time to expiration
    liquid_options['days_to_exp'] = (pd.to_datetime(liquid_options['expiration']) - pd.Timestamp.now()).dt.days
    liquid_options['time_to_exp'] = liquid_options['days_to_exp'] / 365.25
    
    # Group by expiration
    expirations = liquid_options.groupby('expiration')
    
    for exp_date, exp_options in expirations:
        days_to_exp = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
        
        if days_to_exp <= 0:
            continue
        
        # Find ATM strike (closest to spot)
        exp_options['strike_diff'] = abs(exp_options['strike'] - spot_price)
        atm_strike = exp_options.loc[exp_options['strike_diff'].idxmin(), 'strike']
        
        # Get ATM put and call
        atm_put = exp_options[
            (exp_options['option_type'] == 'P') & 
            (exp_options['strike'] == atm_strike)
        ]
        atm_call = exp_options[
            (exp_options['option_type'] == 'C') & 
            (exp_options['strike'] == atm_strike)
        ]
        
        if len(atm_put) > 0 and len(atm_call) > 0:
            straddle_price = atm_put['close'].iloc[0] + atm_call['close'].iloc[0]
            
            # Calculate implied move using straddle approximation
            # Implied move ≈ sqrt(2π) * straddle_price / spot_price / sqrt(T)
            time_to_exp = days_to_exp / 365.25
            if time_to_exp > 0:
                implied_move = np.sqrt(2 * np.pi) * straddle_price / spot_price / np.sqrt(time_to_exp)
                
                # Store by time horizon
                if 5 <= days_to_exp <= 10:  # 1 week
                    features['implied_move_1w'] = implied_move
                    features['atm_straddle_price_1w'] = straddle_price
                elif 10 <= days_to_exp <= 20:  # 2 weeks
                    features['implied_move_2w'] = implied_move
                    features['atm_straddle_price_2w'] = straddle_price
                elif 20 <= days_to_exp <= 35:  # 1 month
                    features['implied_move_1m'] = implied_move
                    features['atm_straddle_price_1m'] = straddle_price
        
        # Calculate strangle skew (25-delta put vs 25-delta call)
        put_25d = exp_options[
            (exp_options['option_type'] == 'P') & 
            (exp_options['strike'] < spot_price * 0.95)
        ].sort_values('strike', ascending=False)
        
        call_25d = exp_options[
            (exp_options['option_type'] == 'C') & 
            (exp_options['strike'] > spot_price * 1.05)
        ].sort_values('strike')
        
        if len(put_25d) > 0 and len(call_25d) > 0:
            strangle_price = put_25d['close'].iloc[0] + call_25d['close'].iloc[0]
            if days_to_exp <= 10:
                features['strangle_skew_1w'] = strangle_price - straddle_price if 'atm_straddle_price_1w' in features else np.nan
            elif days_to_exp <= 20:
                features['strangle_skew_2w'] = strangle_price - straddle_price if 'atm_straddle_price_2w' in features else np.nan
    
    # Fill missing values
    for key in [
        'atm_straddle_price_1w', 'atm_straddle_price_2w', 'atm_straddle_price_1m',
        'implied_move_1w', 'implied_move_2w', 'implied_move_1m',
        'strangle_skew_1w', 'strangle_skew_2w'
    ]:
        if key not in features:
            features[key] = np.nan
    
    return features

print("✅ Implied moves calculator implemented")
print()

# ============================================================================
# PHASE 2.4: PUT/CALL VOLUME AND OI RATIOS
# ============================================================================
print("=" * 80)
print("PHASE 2.4: CALCULATING PUT/CALL RATIOS")
print("=" * 80)
print()

def calculate_put_call_ratios(options_data):
    """
    Calculate various put/call ratios
    
    Args:
        options_data: DataFrame with options data
    
    Returns:
        Dictionary of put/call ratio features
    """
    features = {}
    
    # Separate puts and calls
    puts = options_data[options_data['option_type'] == 'P']
    calls = options_data[options_data['option_type'] == 'C']
    
    # Volume ratios
    put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
    call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
    
    features['put_call_volume_ratio'] = put_volume / call_volume if call_volume > 0 else np.nan
    features['put_call_volume_ratio_log'] = np.log(features['put_call_volume_ratio']) if not np.isnan(features['put_call_volume_ratio']) else np.nan
    
    # Open Interest ratios
    put_oi = puts['open_interest'].sum() if 'open_interest' in puts.columns else 0
    call_oi = calls['open_interest'].sum() if 'open_interest' in calls.columns else 0
    
    features['put_call_oi_ratio'] = put_oi / call_oi if call_oi > 0 else np.nan
    features['put_call_oi_ratio_log'] = np.log(features['put_call_oi_ratio']) if not np.isnan(features['put_call_oi_ratio']) else np.nan
    
    # Premium ratios (volume-weighted)
    if 'close' in puts.columns and 'volume' in puts.columns:
        put_premium = (puts['close'] * puts['volume']).sum()
        call_premium = (calls['close'] * calls['volume']).sum()
        features['put_call_premium_ratio'] = put_premium / call_premium if call_premium > 0 else np.nan
    else:
        features['put_call_premium_ratio'] = np.nan
    
    # Moneyness-based ratios
    if 'strike' in options_data.columns:
        # Get current price (approximate from ATM options)
        atm_options = options_data[
            (options_data['strike'] >= 400) & (options_data['strike'] <= 500)
        ]
        if len(atm_options) > 0:
            spot_price = atm_options['strike'].median()
            
            # ITM puts vs OTM calls (bearish sentiment)
            itm_puts = puts[puts['strike'] > spot_price]
            otm_calls = calls[calls['strike'] > spot_price]
            
            itm_put_volume = itm_puts['volume'].sum() if 'volume' in itm_puts.columns else 0
            otm_call_volume = otm_calls['volume'].sum() if 'volume' in otm_calls.columns else 0
            
            features['itm_put_otm_call_ratio'] = itm_put_volume / otm_call_volume if otm_call_volume > 0 else np.nan
            
            # OTM puts vs ITM calls (bullish sentiment)
            otm_puts = puts[puts['strike'] < spot_price]
            itm_calls = calls[calls['strike'] < spot_price]
            
            otm_put_volume = otm_puts['volume'].sum() if 'volume' in otm_puts.columns else 0
            itm_call_volume = itm_calls['volume'].sum() if 'volume' in itm_calls.columns else 0
            
            features['otm_put_itm_call_ratio'] = otm_put_volume / itm_call_volume if itm_call_volume > 0 else np.nan
        else:
            features['itm_put_otm_call_ratio'] = np.nan
            features['otm_put_itm_call_ratio'] = np.nan
    else:
        features['itm_put_otm_call_ratio'] = np.nan
        features['otm_put_itm_call_ratio'] = np.nan
    
    return features

print("✅ Put/Call ratios calculator implemented")
print()

# ============================================================================
# PHASE 2.5: IV TERM STRUCTURE FEATURES
# ============================================================================
print("=" * 80)
print("PHASE 2.5: CREATING IV TERM STRUCTURE FEATURES")
print("=" * 80)
print()

def download_vix_data(start_date='2016-01-01', end_date='2024-12-31'):
    """Download VIX and VIX3M data from Yahoo Finance"""
    print("📊 Downloading VIX term structure data...")
    
    try:
        # Download VIX (30-day)
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        # Download VIX3M (3-month)
        vix3m = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)
        if isinstance(vix3m.columns, pd.MultiIndex):
            vix3m.columns = vix3m.columns.get_level_values(0)
        
        # Download VIX9D (9-day)
        vix9d = yf.download('^VIX9D', start=start_date, end=end_date, progress=False)
        if isinstance(vix9d.columns, pd.MultiIndex):
            vix9d.columns = vix9d.columns.get_level_values(0)
        
        # Combine data
        term_structure = pd.DataFrame(index=vix.index)
        term_structure['vix'] = vix['Close']
        term_structure['vix3m'] = vix3m['Close']
        term_structure['vix9d'] = vix9d['Close']
        
        # Calculate term structure features
        term_structure['vix_term_structure'] = term_structure['vix3m'] / term_structure['vix']
        term_structure['vix_short_term'] = term_structure['vix9d'] / term_structure['vix']
        term_structure['vix_contango'] = (term_structure['vix3m'] > term_structure['vix']).astype(int)
        term_structure['vix_backwardation'] = (term_structure['vix3m'] < term_structure['vix']).astype(int)
        
        # VIX changes
        term_structure['vix_change'] = term_structure['vix'].pct_change()
        term_structure['vix_spike'] = (term_structure['vix_change'] > 0.1).astype(int)
        
        # VIX percentiles
        term_structure['vix_percentile'] = term_structure['vix'].rolling(252, min_periods=20).rank(pct=True)
        term_structure['vix3m_percentile'] = term_structure['vix3m'].rolling(252, min_periods=20).rank(pct=True)
        
        print(f"✅ Downloaded {len(term_structure)} days of VIX term structure data")
        return term_structure
        
    except Exception as e:
        print(f"❌ Error downloading VIX data: {e}")
        return pd.DataFrame()

def calculate_term_structure_features(term_structure_data):
    """
    Calculate IV term structure features
    
    Args:
        term_structure_data: DataFrame with VIX, VIX3M, VIX9D data
    
    Returns:
        Dictionary of term structure features
    """
    features = {}
    
    if len(term_structure_data) == 0:
        return {key: np.nan for key in [
            'vix_term_structure', 'vix_short_term', 'vix_contango', 'vix_backwardation',
            'vix_change', 'vix_spike', 'vix_percentile', 'vix3m_percentile',
            'vix_volatility', 'vix_mean_reversion'
        ]}
    
    # Basic term structure ratios
    features['vix_term_structure'] = term_structure_data['vix_term_structure'].iloc[-1] if 'vix_term_structure' in term_structure_data.columns else np.nan
    features['vix_short_term'] = term_structure_data['vix_short_term'].iloc[-1] if 'vix_short_term' in term_structure_data.columns else np.nan
    
    # Contango/Backwardation
    features['vix_contango'] = term_structure_data['vix_contango'].iloc[-1] if 'vix_contango' in term_structure_data.columns else np.nan
    features['vix_backwardation'] = term_structure_data['vix_backwardation'].iloc[-1] if 'vix_backwardation' in term_structure_data.columns else np.nan
    
    # VIX changes
    features['vix_change'] = term_structure_data['vix_change'].iloc[-1] if 'vix_change' in term_structure_data.columns else np.nan
    features['vix_spike'] = term_structure_data['vix_spike'].iloc[-1] if 'vix_spike' in term_structure_data.columns else np.nan
    
    # Percentiles
    features['vix_percentile'] = term_structure_data['vix_percentile'].iloc[-1] if 'vix_percentile' in term_structure_data.columns else np.nan
    features['vix3m_percentile'] = term_structure_data['vix3m_percentile'].iloc[-1] if 'vix3m_percentile' in term_structure_data.columns else np.nan
    
    # VIX volatility (vol of vol)
    if 'vix' in term_structure_data.columns:
        vix_returns = term_structure_data['vix'].pct_change().dropna()
        features['vix_volatility'] = vix_returns.std() * np.sqrt(252)  # Annualized
    else:
        features['vix_volatility'] = np.nan
    
    # VIX mean reversion (autocorrelation)
    if 'vix' in term_structure_data.columns and len(term_structure_data) > 20:
        vix_series = term_structure_data['vix'].dropna()
        if len(vix_series) > 20:
            features['vix_mean_reversion'] = vix_series.autocorr(lag=5)  # 5-day autocorrelation
        else:
            features['vix_mean_reversion'] = np.nan
    else:
        features['vix_mean_reversion'] = np.nan
    
    return features

print("✅ IV term structure features calculator implemented")
print()

# ============================================================================
# PROCESS ALL OPTIONS DATA
# ============================================================================
print("=" * 80)
print("PROCESSING ALL OPTIONS DATA WITH NEW FEATURES")
print("=" * 80)
print()

# Load daily aggregates
daily_aggregates = pd.read_parquet('data/options_chains/daily_aggregates.parquet')
print(f"📊 Loaded {len(daily_aggregates)} days of daily aggregates")

# Download VIX term structure data
vix_data = download_vix_data()

# Process each day
enhanced_features = []

print(f"\n🔄 Processing {len(daily_aggregates)} days...")

for i, row in daily_aggregates.iterrows():
    date = row['date']
    
    # Load options data for this day
    year = date.year
    date_str = date.strftime('%Y-%m-%d')
    
    options_file = Path(f'data/options_chains/polygon/year={year}/spy_options_{date_str}.parquet')
    
    if not options_file.exists():
        continue
    
    try:
        # Load options data
        options_data = pd.read_parquet(options_file)
        
        # Get SPY price (approximate from ATM options)
        atm_options = options_data[
            (options_data['strike'] >= 400) & (options_data['strike'] <= 500)
        ]
        spot_price = atm_options['strike'].median() if len(atm_options) > 0 else 400
        
        # Calculate all new features
        features = {}
        features['date'] = date
        
        # IV Skew features
        iv_skew = calculate_iv_skew_features(options_data, spot_price)
        features.update(iv_skew)
        
        # Implied moves
        implied_moves = calculate_implied_moves(options_data, spot_price)
        features.update(implied_moves)
        
        # Put/Call ratios
        put_call_ratios = calculate_put_call_ratios(options_data)
        features.update(put_call_ratios)
        
        # Term structure features
        if len(vix_data) > 0 and date in vix_data.index:
            term_structure = calculate_term_structure_features(vix_data.loc[[date]])
            features.update(term_structure)
        else:
            # Fill with NaN if no VIX data
            term_keys = [
                'vix_term_structure', 'vix_short_term', 'vix_contango', 'vix_backwardation',
                'vix_change', 'vix_spike', 'vix_percentile', 'vix3m_percentile',
                'vix_volatility', 'vix_mean_reversion'
            ]
            for key in term_keys:
                features[key] = np.nan
        
        enhanced_features.append(features)
        
        # Progress indicator
        if (i + 1) % 250 == 0:
            print(f"   Processed {i + 1}/{len(daily_aggregates)} days...")
            
    except Exception as e:
        print(f"   ⚠️  Error processing {date_str}: {e}")
        continue

print(f"\n✅ Processed {len(enhanced_features)} days with enhanced features")

# Create enhanced DataFrame
enhanced_df = pd.DataFrame(enhanced_features)
enhanced_df = enhanced_df.sort_values('date').reset_index(drop=True)

print(f"\n📊 Enhanced Features Summary:")
print(f"   Date range: {enhanced_df['date'].min().date()} to {enhanced_df['date'].max().date()}")
print(f"   Total days: {len(enhanced_df)}")
print(f"   Features: {len(enhanced_df.columns)}")
print()

# Display sample
print("Sample enhanced features (first 3 rows):")
print(enhanced_df.head(3))
print()

# Save enhanced data
output_file = Path('data/options_chains/enhanced_options_features.parquet')
enhanced_df.to_parquet(output_file, index=False)
print(f"💾 Saved enhanced features to: {output_file}")

# Also save as CSV
csv_file = Path('data/options_chains/enhanced_options_features.csv')
enhanced_df.to_csv(csv_file, index=False)
print(f"💾 Saved enhanced features to: {csv_file}")

print("\n✅ PHASES 2.1-2.5 COMPLETE!")
print()
print("📊 New Features Added:")
print("   - IV Skew (25d, 10d, 50d)")
print("   - Implied Moves (1w, 2w, 1m)")
print("   - Put/Call Ratios (volume, OI, premium)")
print("   - IV Term Structure (VIX, VIX3M, VIX9D)")
print("   - VIX Percentiles and Changes")
print()
print("🎯 Ready for Phase 2.6-2.10: Cycle Features")
