#!/usr/bin/env python3
"""
Create sample SPY options data for the last 7 days with Oct 6-10 expiries
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
from pathlib import Path


def create_recent_sample_data():
    """Create sample data for last 7 days with Oct 6-10 expiries"""
    
    # Current date is Oct 5, 2025, so last 7 days would be Sep 29 - Oct 5
    target_dates = []
    current_date = datetime(2025, 10, 5)
    
    for i in range(7):
        date_to_add = current_date - timedelta(days=i)
        # Skip weekends for market data
        if date_to_add.weekday() < 5:  # Monday=0, Friday=4
            target_dates.append(date_to_add.strftime('%Y-%m-%d'))
    
    target_dates.reverse()  # Chronological order
    
    # Target expiries Oct 6-10, 2025
    expiry_dates = ["2025-10-06", "2025-10-08", "2025-10-10"]
    
    print(f"Creating sample data for dates: {target_dates}")
    print(f"With expiries: {expiry_dates}")
    
    # Current SPY price
    spot_price = 669.21
    
    # Create output directories
    data_dir = Path("data/spy_options")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "trades").mkdir(exist_ok=True)
    
    for target_date in target_dates:
        print(f"\nGenerating data for {target_date}...")
        
        # Create strikes around current price
        strikes = list(range(int(spot_price - 20), int(spot_price + 21), 1))
        
        trades = []
        num_trades = np.random.randint(800, 1500)  # Realistic daily volume
        
        for i in range(num_trades):
            # Random selections with realistic distributions
            strike = np.random.choice(strikes, p=get_strike_probabilities(strikes, spot_price))
            expiry = np.random.choice(expiry_dates, p=[0.4, 0.3, 0.3])  # More Monday expiry
            option_type = np.random.choice(['c', 'p'], p=[0.52, 0.48])  # Slight call bias
            
            # Create ticker
            exp_dt = datetime.strptime(expiry, '%Y-%m-%d')
            exp_str = exp_dt.strftime('%y%m%d')
            ticker = f"O:SPY{exp_str}{option_type.upper()}{int(strike * 1000):08d}"
            
            # Calculate realistic option price
            price = calculate_option_price(strike, spot_price, option_type, expiry, target_date)
            
            # Volume based on moneyness
            distance = abs(strike - spot_price)
            if distance <= 2:
                volume = np.random.randint(50, 800)  # Heavy volume ATM
            elif distance <= 5:
                volume = np.random.randint(20, 300)  # Good volume near money
            elif distance <= 10:
                volume = np.random.randint(5, 100)   # Light volume OTM
            else:
                volume = np.random.randint(1, 20)    # Very light volume far OTM
            
            # Random time during market hours
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            market_open = target_dt.replace(hour=9, minute=30)
            market_close = target_dt.replace(hour=16, minute=0)
            trade_time = market_open + timedelta(
                seconds=np.random.randint(0, int((market_close - market_open).total_seconds()))
            )
            timestamp_ns = int(trade_time.timestamp() * 1e9)
            
            # Realistic bid/ask spread
            if price < 0.5:
                spread = 0.01
            elif price < 2:
                spread = 0.05
            elif price < 10:
                spread = price * 0.02
            else:
                spread = price * 0.015
            
            bid = max(0.01, price - spread/2)
            ask = price + spread/2
            
            trade = {
                'ticker': ticker,
                'timestamp': timestamp_ns,
                'sip_timestamp': timestamp_ns,
                'price': price,
                'size': volume,
                'exchange': np.random.choice(['CBOE', 'NASDAQ', 'PHLX', 'AMEX']),
                'conditions': [],
                'date': target_date,
                'expiry': exp_dt.date(),
                'strike': strike,
                'option_type': option_type,
                'underlying': 'SPY',
                'bid': bid,
                'ask': ask,
                'bid_size': np.random.randint(5, 200),
                'ask_size': np.random.randint(5, 200),
                'quote_timestamp': timestamp_ns,
                'time_diff_seconds': np.random.uniform(0, 0.5)  # Realistic quote timing
            }
            
            trades.append(trade)
        
        # Create DataFrame and save
        df = pd.DataFrame(trades)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Save as enriched trades
        output_file = data_dir / "trades" / f"{target_date}_enriched_trades.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"✅ Created {len(df)} trades for {target_date}")
        print(f"   Strikes: {df['strike'].min():.0f} - {df['strike'].max():.0f}")
        print(f"   Call/Put: {df['option_type'].value_counts().to_dict()}")


def get_strike_probabilities(strikes, spot_price):
    """Get realistic probability distribution for strike selection"""
    probs = []
    for strike in strikes:
        distance = abs(strike - spot_price)
        if distance <= 2:
            prob = 0.15  # High probability ATM
        elif distance <= 5:
            prob = 0.08  # Good probability near money
        elif distance <= 10:
            prob = 0.03  # Lower probability OTM
        else:
            prob = 0.01  # Very low probability far OTM
        probs.append(prob)
    
    # Normalize to sum to 1
    total = sum(probs)
    return [p/total for p in probs]


def calculate_option_price(strike, spot, option_type, expiry_str, trade_date_str):
    """Calculate realistic option price using simplified Black-Scholes"""
    
    trade_date = datetime.strptime(trade_date_str, '%Y-%m-%d')
    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
    
    # Time to expiry in years
    tte = max((expiry_date - trade_date).days / 365.0, 1/365.0)
    
    # Implied volatility varies by moneyness
    moneyness = spot / strike
    if option_type == 'c':
        if moneyness > 1.05:  # Deep ITM
            iv = 0.15
        elif moneyness > 1.02:  # ITM
            iv = 0.18
        elif moneyness > 0.98:  # ATM
            iv = 0.22
        elif moneyness > 0.95:  # OTM
            iv = 0.25
        else:  # Deep OTM
            iv = 0.30
    else:  # Put
        if moneyness < 0.95:  # Deep ITM
            iv = 0.15
        elif moneyness < 0.98:  # ITM
            iv = 0.18
        elif moneyness < 1.02:  # ATM
            iv = 0.22
        elif moneyness < 1.05:  # OTM
            iv = 0.25
        else:  # Deep OTM
            iv = 0.30
    
    # Simplified Black-Scholes calculation
    from scipy.stats import norm
    import math
    
    r = 0.05  # Risk-free rate
    
    d1 = (math.log(spot/strike) + (r + 0.5*iv**2)*tte) / (iv*math.sqrt(tte))
    d2 = d1 - iv*math.sqrt(tte)
    
    if option_type == 'c':
        price = spot * norm.cdf(d1) - strike * math.exp(-r*tte) * norm.cdf(d2)
    else:
        price = strike * math.exp(-r*tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    # Add some noise and ensure minimum price
    price *= np.random.uniform(0.95, 1.05)
    price = max(0.01, round(price, 2))
    
    return price


if __name__ == "__main__":
    create_recent_sample_data()
    print("\n✅ Sample data creation complete!")