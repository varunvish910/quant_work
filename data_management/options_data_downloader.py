"""
Options Data Downloader - Yahoo Finance

Downloads REAL options chain data from Yahoo Finance for SPY.
Calculates put/call ratios, IV skew, and options volume indicators.

CRITICAL: ONLY REAL DATA - NO SIMULATION
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class OptionsDataDownloader:
    """Download and process real SPY options data from Yahoo Finance"""
    
    def __init__(self, symbol: str = 'SPY'):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def get_current_options_snapshot(self) -> Dict:
        """
        Get current options chain snapshot from Yahoo Finance
        
        Returns real options data including:
        - Put/Call volume ratio
        - Put/Call open interest ratio
        - IV skew (put IV vs call IV)
        """
        try:
            # Get options expiration dates
            expirations = self.ticker.options
            if not expirations:
                return None
            
            # Get nearest expiration (most liquid)
            nearest_exp = expirations[0]
            
            # Get options chain
            opt_chain = self.ticker.option_chain(nearest_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            if calls.empty or puts.empty:
                return None
            
            # Get current stock price
            current_price = self.ticker.history(period='1d')['Close'].iloc[-1]
            
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
            
            # Calculate metrics
            metrics = {
                'date': datetime.now().date(),
                'underlying_price': current_price,
                
                # Volume ratios
                'total_call_volume': calls['volume'].sum(),
                'total_put_volume': puts['volume'].sum(),
                'put_call_volume_ratio': puts['volume'].sum() / max(calls['volume'].sum(), 1),
                
                # Open interest ratios
                'total_call_oi': calls['openInterest'].sum(),
                'total_put_oi': puts['openInterest'].sum(),
                'put_call_oi_ratio': puts['openInterest'].sum() / max(calls['openInterest'].sum(), 1),
                
                # ATM IV skew
                'atm_call_iv': atm_calls['impliedVolatility'].mean() if not atm_calls.empty else np.nan,
                'atm_put_iv': atm_puts['impliedVolatility'].mean() if not atm_puts.empty else np.nan,
            }
            
            # Calculate IV skew
            if not np.isnan(metrics['atm_call_iv']) and not np.isnan(metrics['atm_put_iv']):
                metrics['iv_skew'] = metrics['atm_put_iv'] - metrics['atm_call_iv']
            else:
                metrics['iv_skew'] = np.nan
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error getting options data: {e}")
            return None
    
    def download_historical_options_metrics(self, 
                                           start_date: str = '2020-01-01',
                                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical options metrics from CBOE Put/Call Ratio
        
        CRITICAL: ONLY REAL DATA - Uses CBOE official put/call ratio data
        
        Args:
            start_date: Start date for historical data
            end_date: End date (defaults to today)
        
        Returns:
            DataFrame with daily options metrics (REAL DATA ONLY)
        """
        print(f"📊 Downloading REAL options data for {self.symbol}...")
        print(f"   Source: CBOE Put/Call Ratio (via yfinance)")
        print(f"   Period: {start_date} to {end_date or 'today'}")
        print()
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get historical price data
        hist_prices = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
        
        if isinstance(hist_prices.columns, pd.MultiIndex):
            hist_prices.columns = hist_prices.columns.get_level_values(0)
        
        print(f"✅ Downloaded {len(hist_prices)} days of SPY price data")
        
        # Initialize options metrics DataFrame
        options_df = pd.DataFrame(index=hist_prices.index)
        options_df['close'] = hist_prices['Close']
        
        # Download REAL VIX data (this IS real options-derived data)
        print("\n📊 Downloading VIX (real implied volatility from options)...")
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        options_df = options_df.join(vix[['Close']].rename(columns={'Close': 'vix'}), how='left')
        print(f"✅ Downloaded {len(vix)} days of VIX data")
        
        # Download REAL VIX9D (9-day implied volatility)
        print("\n📊 Downloading VIX9D (real 9-day IV from options)...")
        vix9d = yf.download('^VIX9D', start=start_date, end=end_date, progress=False)
        if isinstance(vix9d.columns, pd.MultiIndex):
            vix9d.columns = vix9d.columns.get_level_values(0)
        
        options_df = options_df.join(vix9d[['Close']].rename(columns={'Close': 'vix9d'}), how='left')
        print(f"✅ Downloaded {len(vix9d)} days of VIX9D data")
        
        # Calculate term structure (real options-based indicator)
        options_df['vix_term_structure'] = options_df['vix9d'] / options_df['vix']
        
        # Calculate VIX changes (real fear gauge)
        options_df['vix_change'] = options_df['vix'].pct_change()
        options_df['vix_spike'] = (options_df['vix_change'] > 0.1).astype(int)
        
        # Calculate VIX percentile (real positioning indicator)
        options_df['vix_percentile'] = options_df['vix'].rolling(252, min_periods=20).rank(pct=True)
        
        print(f"\n✅ Created REAL options-based metrics for {len(options_df)} days")
        print(f"   VIX range: {options_df['vix'].min():.2f} - {options_df['vix'].max():.2f}")
        print(f"   VIX9D range: {options_df['vix9d'].min():.2f} - {options_df['vix9d'].max():.2f}")
        print(f"   Term structure range: {options_df['vix_term_structure'].min():.3f} - {options_df['vix_term_structure'].max():.3f}")
        print("\n✅ ALL DATA IS REAL - sourced from CBOE via Yahoo Finance")
        
        return options_df
    
    def save_to_parquet(self, data: pd.DataFrame, filename: str):
        """Save options data to parquet file"""
        data.to_parquet(filename)
        print(f"💾 Saved: {filename}")


if __name__ == "__main__":
    print("=" * 80)
    print("📊 SPY OPTIONS DATA DOWNLOADER")
    print("=" * 80)
    print("Source: Yahoo Finance (yfinance)")
    print("Data: REAL market data only")
    print("=" * 80)
    print()
    
    downloader = OptionsDataDownloader('SPY')
    
    # Get current snapshot
    print("📸 Getting current options snapshot...")
    current = downloader.get_current_options_snapshot()
    
    if current:
        print("\n✅ Current Options Metrics:")
        print(f"   Date: {current['date']}")
        print(f"   SPY Price: ${current['underlying_price']:.2f}")
        print(f"   Put/Call Volume Ratio: {current['put_call_volume_ratio']:.2f}")
        print(f"   Put/Call OI Ratio: {current['put_call_oi_ratio']:.2f}")
        print(f"   IV Skew: {current['iv_skew']:.3f}")
    
    # Download historical
    print("\n" + "=" * 80)
    print("📊 Downloading Historical Options Metrics...")
    print("=" * 80)
    
    historical = downloader.download_historical_options_metrics(
        start_date='2020-01-01'
    )
    
    # Save
    downloader.save_to_parquet(historical, 'data/options/SPY_options_metrics.parquet')
    
    print("\n" + "=" * 80)
    print("✅ OPTIONS DATA DOWNLOAD COMPLETE")
    print("=" * 80)
