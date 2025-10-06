#!/usr/bin/env python3
"""
Create sample SPY options data for testing the dealer positioning pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
from pathlib import Path


def create_sample_spy_trades(target_date: str, expiry_dates: list, num_trades: int = 1000) -> pd.DataFrame:
    """Create realistic sample SPY options trades data"""
    
    # Parse target date
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Current SPY price (using realistic value)
    spot_price = 569.21
    
    # Create strike range around current price
    strikes = []
    for exp in expiry_dates:
        exp_dt = datetime.strptime(exp, '%Y-%m-%d')
        # More strikes near the money for closer expiries
        if (exp_dt - target_dt).days <= 3:
            # Next day expiry - tight strikes
            strike_range = np.arange(spot_price - 15, spot_price + 16, 1)
        else:
            # Weekly expiry - wider strikes
            strike_range = np.arange(spot_price - 30, spot_price + 31, 1)
        strikes.extend(strike_range)
    
    # Remove duplicates and sort
    strikes = sorted(set(strikes))
    
    trades = []
    trade_id = 1
    
    for i in range(num_trades):
        # Random selections
        strike = np.random.choice(strikes)
        expiry = np.random.choice(expiry_dates)
        option_type = np.random.choice(['c', 'p'])
        
        # Create realistic SPY option ticker
        exp_dt = datetime.strptime(expiry, '%Y-%m-%d')
        exp_str = exp_dt.strftime('%y%m%d')  # YYMMDD format
        ticker = f"O:SPY{exp_str}{option_type.upper()}{int(strike * 1000):08d}"
        
        # Distance from money affects pricing and volume
        distance = abs(strike - spot_price)
        
        # Option pricing - rough estimate
        if option_type == 'c':
            if strike <= spot_price:
                # ITM call
                intrinsic = spot_price - strike
                time_value = max(0.5, 5 - distance * 0.2)
                price = intrinsic + time_value
            else:
                # OTM call
                price = max(0.01, 5 - distance * 0.3)
        else:
            if strike >= spot_price:
                # ITM put
                intrinsic = strike - spot_price
                time_value = max(0.5, 5 - distance * 0.2)
                price = intrinsic + time_value
            else:
                # OTM put
                price = max(0.01, 5 - distance * 0.3)
        
        # Add some noise to price
        price *= np.random.uniform(0.8, 1.2)
        price = max(0.01, round(price, 2))
        
        # Volume - more volume near the money
        if distance <= 5:
            volume = np.random.randint(10, 500)
        elif distance <= 15:
            volume = np.random.randint(5, 200)
        else:
            volume = np.random.randint(1, 50)
        
        # Timestamp during market hours (9:30 AM - 4:00 PM ET)
        market_open = target_dt.replace(hour=9, minute=30, second=0)
        market_close = target_dt.replace(hour=16, minute=0, second=0)
        trade_time = market_open + timedelta(
            seconds=np.random.randint(0, int((market_close - market_open).total_seconds()))
        )
        
        # Convert to nanoseconds (Polygon format)
        timestamp_ns = int(trade_time.timestamp() * 1e9)
        
        # Create bid/ask spread
        spread = max(0.01, price * 0.05)  # 5% spread
        bid = max(0.01, price - spread/2)
        ask = price + spread/2
        
        trade = {
            'ticker': ticker,
            'timestamp': timestamp_ns,
            'sip_timestamp': timestamp_ns,
            'price': price,
            'size': volume,
            'exchange': 'CBOE',
            'conditions': [],
            'date': target_date,
            'expiry': datetime.strptime(expiry, '%Y-%m-%d').date(),
            'strike': strike,
            'option_type': option_type,
            'underlying': 'SPY',
            'bid': bid,
            'ask': ask,
            'bid_size': np.random.randint(10, 100),
            'ask_size': np.random.randint(10, 100),
            'quote_timestamp': timestamp_ns,
            'time_diff_seconds': 0.0
        }
        
        trades.append(trade)
        trade_id += 1
    
    df = pd.DataFrame(trades)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Created {len(df)} sample SPY options trades")
    print(f"Strikes: {df['strike'].min():.0f} - {df['strike'].max():.0f}")
    print(f"Expiries: {sorted(df['expiry'].unique())}")
    print(f"Types: {df['option_type'].value_counts().to_dict()}")
    
    return df


def save_sample_data(target_date: str, expiry_dates: list):
    """Create and save sample data for testing"""
    
    # Create output directories
    data_dir = Path("data/spy_options")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "trades").mkdir(exist_ok=True)
    (data_dir / "raw_files").mkdir(exist_ok=True)
    
    # Generate sample trades
    print(f"Generating sample SPY options data for {target_date}")
    trades_df = create_sample_spy_trades(target_date, expiry_dates, num_trades=2000)
    
    # Save as enriched trades (already has bid/ask)
    output_file = data_dir / "trades" / f"{target_date}_enriched_trades.parquet"
    trades_df.to_parquet(output_file, index=False)
    print(f"✅ Saved sample data to {output_file}")
    
    return trades_df


if __name__ == "__main__":
    # Create sample data for the dates in PLAN.md
    target_date = "2024-09-20"  # Use a past date that should work
    expiry_dates = ["2024-09-23", "2024-09-25", "2024-09-27"]
    
    sample_df = save_sample_data(target_date, expiry_dates)
    
    print("\nSample data preview:")
    print(sample_df[['ticker', 'strike', 'option_type', 'price', 'size', 'bid', 'ask']].head(10))