#!/usr/bin/env python3
"""
Demo Data Generator for SPY Options Analysis
Creates synthetic SPY options trades for testing the analysis pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
from pathlib import Path


def generate_demo_spy_data(target_date: str = "2025-10-05", 
                          expiry_date: str = "2025-10-10",
                          spy_price: float = 669.21,
                          num_trades: int = 5000) -> pd.DataFrame:
    """Generate synthetic SPY options trades data"""
    
    print(f"Generating {num_trades:,} demo SPY option trades...")
    
    # Strike range around SPY price
    strikes = np.arange(int(spy_price * 0.9), int(spy_price * 1.1), 1)
    
    # Generate random trades
    trades = []
    
    for i in range(num_trades):
        # Random strike selection (more volume near ATM)
        strike_weights = np.exp(-0.5 * ((strikes - spy_price) / (spy_price * 0.05))**2)
        strike = np.random.choice(strikes, p=strike_weights / strike_weights.sum())
        
        # Random option type (slightly more calls)
        option_type = np.random.choice(['c', 'p'], p=[0.55, 0.45])
        
        # Estimate fair value using simple approximation
        moneyness = strike / spy_price
        time_to_exp = 5 / 365  # 5 days
        iv = 0.20 + abs(moneyness - 1) * 0.5  # Simple IV skew
        
        if option_type == 'c':
            intrinsic = max(0, spy_price - strike)
            time_value = max(0.01, iv * spy_price * np.sqrt(time_to_exp) * np.random.uniform(0.5, 1.5))
        else:
            intrinsic = max(0, strike - spy_price)
            time_value = max(0.01, iv * spy_price * np.sqrt(time_to_exp) * np.random.uniform(0.5, 1.5))
        
        fair_value = intrinsic + time_value
        
        # Generate bid/ask around fair value
        spread_pct = np.random.uniform(0.02, 0.10)  # 2-10% spread
        spread = fair_value * spread_pct
        
        bid = max(0.01, fair_value - spread / 2)
        ask = fair_value + spread / 2
        
        # Trade price - sometimes at bid/ask, sometimes mid-market
        price_type = np.random.choice(['bid', 'ask', 'mid'], p=[0.3, 0.3, 0.4])
        if price_type == 'bid':
            trade_price = bid
        elif price_type == 'ask':
            trade_price = ask
        else:
            trade_price = (bid + ask) / 2 + np.random.normal(0, spread * 0.1)
        
        trade_price = max(0.01, round(trade_price, 2))
        bid = round(bid, 2)
        ask = round(ask, 2)
        
        # Random trade size (log-normal distribution)
        size = int(np.random.lognormal(mean=3, sigma=1))
        size = max(1, min(size, 1000))  # Cap at 1000 contracts
        
        # Random timestamp during trading hours
        base_time = datetime.strptime(f"{target_date} 09:30:00", "%Y-%m-%d %H:%M:%S")
        trading_minutes = np.random.randint(0, 390)  # 6.5 hours * 60 minutes
        trade_time = base_time + timedelta(minutes=trading_minutes)
        timestamp = int(trade_time.timestamp() * 1_000_000_000)  # nanoseconds
        
        # Create SPY option ticker
        exp_str = expiry_date.replace('-', '')[2:]  # YYMMDD format
        ticker = f"O:SPY{exp_str}{option_type.upper()}{int(strike * 100):08d}"
        
        trade = {
            'ticker': ticker,
            'timestamp': timestamp,
            'price': trade_price,
            'size': size,
            'exchange': np.random.choice(['CBOE', 'PHLX', 'ISE', 'ARCA']),
            'conditions': [],
            'date': target_date,
            'expiry': datetime.strptime(expiry_date, '%Y-%m-%d').date(),
            'strike': strike,
            'option_type': option_type,
            'underlying': 'SPY',
            'bid': bid,
            'ask': ask,
            'bid_size': np.random.randint(1, 50),
            'ask_size': np.random.randint(1, 50),
            'quote_timestamp': timestamp,
            'time_diff_seconds': np.random.uniform(0, 0.5)
        }
        
        trades.append(trade)
    
    trades_df = pd.DataFrame(trades)
    
    print(f"✓ Generated {len(trades_df):,} demo trades")
    print(f"  • Strike range: ${trades_df['strike'].min():.0f} - ${trades_df['strike'].max():.0f}")
    print(f"  • Price range: ${trades_df['price'].min():.2f} - ${trades_df['price'].max():.2f}")
    print(f"  • Call/Put split: {(trades_df['option_type'] == 'c').sum():,} / {(trades_df['option_type'] == 'p').sum():,}")
    
    return trades_df


def save_demo_data(trades_df: pd.DataFrame, target_date: str):
    """Save demo data in the expected directory structure"""
    
    # Create directories
    data_dir = Path("data/spy_options")
    (data_dir / "trades").mkdir(parents=True, exist_ok=True)
    
    # Save enriched trades data (what the pipeline expects)
    output_file = data_dir / "trades" / f"{target_date}_enriched_trades.parquet"
    trades_df.to_parquet(output_file, index=False)
    
    print(f"✓ Saved demo data to {output_file}")
    
    return output_file


def main():
    """Generate demo data for testing"""
    
    target_date = "2025-10-05"
    expiry_date = "2025-10-10"
    spy_price = 669.21
    
    print("=== SPY Demo Data Generator ===")
    
    # Generate demo trades
    demo_trades = generate_demo_spy_data(
        target_date=target_date,
        expiry_date=expiry_date,
        spy_price=spy_price,
        num_trades=5000
    )
    
    # Save demo data
    save_demo_data(demo_trades, target_date)
    
    print(f"\nDemo data generated successfully!")
    print(f"You can now run: python3 main_spy_positioning.py --skip-download --spot {spy_price}")


if __name__ == "__main__":
    main()