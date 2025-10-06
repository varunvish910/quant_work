#!/usr/bin/env python3
"""
SPY Options Greeks Calculator
Calculates Black-Scholes Greeks and aggregates dealer positioning for SPY options
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, date
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class GreeksCalculator:
    """Calculates option Greeks using Black-Scholes model and aggregates dealer positioning"""
    
    def __init__(self, spot: float, rate: float = 0.05, dividend_yield: float = 0.015):
        """
        Initialize Greeks calculator
        
        Args:
            spot: Current underlying price
            rate: Risk-free rate (default 5%)
            dividend_yield: Dividend yield for SPY (default 1.5%)
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield
    
    def calculate_greeks(self, strikes: np.ndarray, expiries: np.ndarray, 
                        option_types: np.ndarray, volumes: np.ndarray = None,
                        implied_vols: np.ndarray = None) -> pd.DataFrame:
        """
        Calculate Black-Scholes Greeks for arrays of options
        
        Args:
            strikes: Array of strike prices
            expiries: Array of expiry dates
            option_types: Array of 'c' or 'p'
            volumes: Array of trade volumes (optional)
            implied_vols: Array of implied volatilities (optional, defaults to 0.20)
        
        Returns:
            DataFrame with Greeks for each option
        """
        
        if implied_vols is None:
            implied_vols = np.full(len(strikes), 0.20)  # Default 20% IV
        
        if volumes is None:
            volumes = np.ones(len(strikes))
        
        # Calculate time to expiry in years
        current_date = datetime.now().date()
        
        if isinstance(expiries[0], str):
            expiry_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expiries]
        elif isinstance(expiries[0], date):
            expiry_dates = expiries
        else:
            expiry_dates = [exp.date() if hasattr(exp, 'date') else exp for exp in expiries]
        
        time_to_expiry = np.array([(exp - current_date).days / 365.0 for exp in expiry_dates])
        
        # Ensure minimum time to expiry (1 day)
        time_to_expiry = np.maximum(time_to_expiry, 1/365.0)
        
        results = []
        
        for i in range(len(strikes)):
            strike = strikes[i]
            ttm = time_to_expiry[i]
            vol = implied_vols[i]
            opt_type = option_types[i].lower()
            volume = volumes[i]
            
            try:
                greeks = self._calculate_single_greeks(
                    self.spot, strike, ttm, vol, self.rate, self.dividend_yield, opt_type
                )
                
                greeks.update({
                    'strike': strike,
                    'expiry': expiry_dates[i],
                    'time_to_expiry': ttm,
                    'option_type': opt_type,
                    'implied_vol': vol,
                    'volume': volume,
                    'underlying_price': self.spot
                })
                
                results.append(greeks)
                
            except Exception as e:
                # Handle calculation errors
                print(f"Error calculating Greeks for strike {strike}, expiry {expiry_dates[i]}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _calculate_single_greeks(self, spot: float, strike: float, time_to_expiry: float,
                               vol: float, rate: float, dividend_yield: float, 
                               option_type: str) -> Dict:
        """Calculate Greeks for a single option using Black-Scholes"""
        
        # Ensure positive inputs
        vol = max(vol, 0.01)  # Minimum 1% volatility
        time_to_expiry = max(time_to_expiry, 1/365.0)  # Minimum 1 day
        
        # Black-Scholes d1 and d2
        d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * vol**2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
        d2 = d1 - vol * np.sqrt(time_to_expiry)
        
        # Standard normal CDF and PDF
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        n_d2 = norm.pdf(d2)
        
        # Discount factors
        exp_rt = np.exp(-rate * time_to_expiry)
        exp_dt = np.exp(-dividend_yield * time_to_expiry)
        
        if option_type == 'c':  # Call option
            # Price
            price = spot * exp_dt * nd1 - strike * exp_rt * nd2
            
            # Delta
            delta = exp_dt * nd1
            
            # Theta (per day)
            theta = ((-spot * n_d1 * vol * exp_dt) / (2 * np.sqrt(time_to_expiry)) 
                    - rate * strike * exp_rt * nd2 
                    + dividend_yield * spot * exp_dt * nd1) / 365.0
            
        else:  # Put option
            # Price
            price = strike * exp_rt * norm.cdf(-d2) - spot * exp_dt * norm.cdf(-d1)
            
            # Delta
            delta = exp_dt * (nd1 - 1)
            
            # Theta (per day)
            theta = ((-spot * n_d1 * vol * exp_dt) / (2 * np.sqrt(time_to_expiry)) 
                    + rate * strike * exp_rt * norm.cdf(-d2) 
                    - dividend_yield * spot * exp_dt * norm.cdf(-d1)) / 365.0
        
        # Greeks that are the same for calls and puts
        gamma = exp_dt * n_d1 / (spot * vol * np.sqrt(time_to_expiry))
        vega = spot * exp_dt * n_d1 * np.sqrt(time_to_expiry) / 100.0  # Per 1% vol change
        
        # Higher-order Greeks
        # Vanna (sensitivity of delta to volatility)
        vanna = -exp_dt * n_d1 * d2 / vol / 100.0
        
        # Charm (sensitivity of delta to time)
        if option_type == 'c':
            charm = -exp_dt * (n_d1 * (2 * (rate - dividend_yield) * time_to_expiry - d2 * vol * np.sqrt(time_to_expiry)) / (2 * time_to_expiry * vol * np.sqrt(time_to_expiry)) + dividend_yield * nd1)
        else:
            charm = -exp_dt * (n_d1 * (2 * (rate - dividend_yield) * time_to_expiry - d2 * vol * np.sqrt(time_to_expiry)) / (2 * time_to_expiry * vol * np.sqrt(time_to_expiry)) - dividend_yield * norm.cdf(-d1))
        
        # Vomma (sensitivity of vega to volatility)
        vomma = vega * d1 * d2 / vol / 100.0
        
        # Speed (sensitivity of gamma to underlying price)
        speed = -gamma / spot * (d1 / (vol * np.sqrt(time_to_expiry)) + 1)
        
        # Zomma (sensitivity of gamma to volatility)
        zomma = gamma * (d1 * d2 - 1) / vol / 100.0
        
        return {
            'theoretical_price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'vanna': vanna,
            'charm': charm,
            'vomma': vomma,
            'speed': speed,
            'zomma': zomma,
            'd1': d1,
            'd2': d2
        }
    
    def aggregate_dealer_greeks(self, classified_trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate dealer Greeks exposure by strike and expiry
        
        Args:
            classified_trades: DataFrame with classified trades
        
        Returns:
            Tuple of (aggregated_greeks_by_strike, trade_level_greeks)
        """
        
        print(f"Calculating Greeks for {len(classified_trades):,} classified trades...")
        
        # Filter out invalid data
        valid_trades = classified_trades.dropna(subset=['strike', 'expiry', 'option_type', 'customer_action'])
        print(f"Valid trades after filtering: {len(valid_trades):,}")
        
        if len(valid_trades) == 0:
            print("No valid trades for Greeks calculation")
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate Greeks for all trades
        trade_greeks = self.calculate_greeks(
            strikes=valid_trades['strike'].values,
            expiries=valid_trades['expiry'].values,
            option_types=valid_trades['option_type'].values,
            volumes=valid_trades['size'].values,
            implied_vols=valid_trades.get('implied_vol', np.full(len(valid_trades), 0.20))
        )
        
        if len(trade_greeks) == 0:
            print("No Greeks calculated")
            return pd.DataFrame(), pd.DataFrame()
        
        # Add trade metadata - use correct column names
        confidence_col = 'classification_confidence' if 'classification_confidence' in valid_trades.columns else 'confidence'
        trade_greeks = trade_greeks.merge(
            valid_trades[['strike', 'expiry', 'option_type', 'size', 'customer_action', confidence_col]].rename(columns={confidence_col: 'confidence'}),
            on=['strike', 'expiry', 'option_type'],
            how='left'
        )
        
        # Map customer trades to dealer positioning
        dealer_multiplier = trade_greeks['customer_action'].map({
            'BTO': -1,  # Customer buying = Dealer short
            'STO': 1,   # Customer selling = Dealer long
            'BTC': 1,   # Customer covering = Dealer short
            'STC': -1   # Customer selling existing = Dealer long
        }).fillna(0)
        
        # For puts, flip the gamma sign (customer buying puts = dealer long gamma)
        put_multiplier = np.where(trade_greeks['option_type'] == 'p', -1, 1)
        dealer_multiplier = dealer_multiplier * put_multiplier
        
        # Calculate dealer exposure
        greeks_cols = ['delta', 'gamma', 'theta', 'vega', 'vanna', 'charm', 'vomma', 'speed', 'zomma']
        
        for greek in greeks_cols:
            trade_greeks[f'dealer_{greek}'] = trade_greeks[greek] * trade_greeks['size'] * dealer_multiplier
        
        # Aggregate by strike and expiry
        aggregation_cols = ['strike', 'expiry', 'option_type']
        
        aggregated = trade_greeks.groupby(aggregation_cols).agg({
            'volume': 'sum',
            'theoretical_price': 'mean',
            'implied_vol': 'mean',
            'time_to_expiry': 'mean',
            **{f'dealer_{greek}': 'sum' for greek in greeks_cols}
        }).reset_index()
        
        # Calculate net positioning by strike (combining calls and puts)
        net_by_strike = aggregated.groupby('strike').agg({
            'volume': 'sum',
            **{f'dealer_{greek}': 'sum' for greek in greeks_cols}
        }).reset_index()
        
        # Add strike-level analytics
        net_by_strike['distance_from_spot'] = net_by_strike['strike'] - self.spot
        net_by_strike['distance_pct'] = (net_by_strike['strike'] - self.spot) / self.spot * 100
        net_by_strike['moneyness'] = self.spot / net_by_strike['strike']
        
        # Calculate gamma exposure levels
        total_gamma = net_by_strike['dealer_gamma'].sum()
        net_by_strike['gamma_weight'] = net_by_strike['dealer_gamma'] / total_gamma if total_gamma != 0 else 0
        
        print(f"✓ Greeks calculated for {len(trade_greeks):,} trades")
        print(f"✓ Aggregated to {len(net_by_strike)} strikes")
        print(f"✓ Total dealer gamma exposure: {total_gamma:,.0f}")
        
        return net_by_strike, trade_greeks
    
    def calculate_gamma_profile(self, aggregated_greeks: pd.DataFrame, 
                              strike_range: Tuple[float, float] = None) -> pd.DataFrame:
        """Calculate gamma exposure profile across strike range"""
        
        if strike_range is None:
            # Default to ±15% from spot
            strike_range = (self.spot * 0.85, self.spot * 1.15)
        
        # Filter to strike range
        profile = aggregated_greeks[
            (aggregated_greeks['strike'] >= strike_range[0]) & 
            (aggregated_greeks['strike'] <= strike_range[1])
        ].copy()
        
        # Sort by strike
        profile = profile.sort_values('strike').reset_index(drop=True)
        
        # Calculate cumulative gamma
        profile['cumulative_gamma'] = profile['dealer_gamma'].cumsum()
        
        # Identify key levels
        max_gamma_idx = profile['dealer_gamma'].abs().idxmax()
        max_gamma_strike = profile.loc[max_gamma_idx, 'strike']
        max_gamma_value = profile.loc[max_gamma_idx, 'dealer_gamma']
        
        # Zero gamma crossover
        zero_crossings = []
        for i in range(len(profile) - 1):
            if (profile.loc[i, 'cumulative_gamma'] * profile.loc[i + 1, 'cumulative_gamma']) < 0:
                # Linear interpolation for zero crossing
                x1, y1 = profile.loc[i, 'strike'], profile.loc[i, 'cumulative_gamma']
                x2, y2 = profile.loc[i + 1, 'strike'], profile.loc[i + 1, 'cumulative_gamma']
                zero_strike = x1 - y1 * (x2 - x1) / (y2 - y1)
                zero_crossings.append(zero_strike)
        
        # Add metadata
        profile.attrs = {
            'spot_price': self.spot,
            'max_gamma_strike': max_gamma_strike,
            'max_gamma_value': max_gamma_value,
            'zero_crossings': zero_crossings,
            'total_gamma': profile['dealer_gamma'].sum(),
            'gamma_centroid': np.average(profile['strike'], weights=np.abs(profile['dealer_gamma']))
        }
        
        return profile
    
    def estimate_implied_volatility(self, market_price: float, strike: float, 
                                  expiry: date, option_type: str) -> float:
        """
        Estimate implied volatility using Newton-Raphson method
        (Simplified implementation for demonstration)
        """
        
        # Convert expiry to time to expiry
        current_date = datetime.now().date()
        time_to_expiry = max((expiry - current_date).days / 365.0, 1/365.0)
        
        # Initial guess
        vol = 0.20
        
        for i in range(10):  # Maximum 10 iterations
            try:
                greeks = self._calculate_single_greeks(
                    self.spot, strike, time_to_expiry, vol, 
                    self.rate, self.dividend_yield, option_type
                )
                
                price_diff = greeks['theoretical_price'] - market_price
                vega = greeks['vega'] * 100  # Convert to actual vega
                
                if abs(price_diff) < 0.01 or abs(vega) < 0.001:
                    break
                
                # Newton-Raphson update
                vol = vol - price_diff / vega
                vol = max(0.01, min(vol, 5.0))  # Keep vol between 1% and 500%
                
            except:
                break
        
        return vol


def main():
    """Example usage of the GreeksCalculator"""
    
    # Example with SPY at $669
    calculator = GreeksCalculator(spot=669.21, rate=0.05, dividend_yield=0.015)
    
    # Sample option data
    sample_data = pd.DataFrame({
        'strike': [665, 670, 675, 665, 670, 675],
        'expiry': ['2025-10-10', '2025-10-10', '2025-10-10', '2025-10-10', '2025-10-10', '2025-10-10'],
        'option_type': ['c', 'c', 'c', 'p', 'p', 'p'],
        'size': [100, 200, 150, 80, 120, 90],
        'trade_classification': ['BTO', 'STO', 'BTC', 'BTO', 'STO', 'STC'],
        'confidence': [0.8, 0.9, 0.7, 0.85, 0.95, 0.75]
    })
    
    # Calculate Greeks
    aggregated, trade_level = calculator.aggregate_dealer_greeks(sample_data)
    
    print("Aggregated Greeks by Strike:")
    print(aggregated[['strike', 'dealer_gamma', 'dealer_delta', 'dealer_vega']].head())
    
    print("\nGamma Profile:")
    profile = calculator.calculate_gamma_profile(aggregated)
    print(f"Gamma centroid: ${profile.attrs['gamma_centroid']:.2f}")
    print(f"Max gamma strike: ${profile.attrs['max_gamma_strike']:.2f}")


if __name__ == "__main__":
    main()