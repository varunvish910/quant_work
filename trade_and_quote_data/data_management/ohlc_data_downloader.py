#!/usr/bin/env python3
"""
OHLC Data Downloader with Weekly Aggregation

A comprehensive class for downloading OHLC (Open, High, Low, Close) data
for specific tickers and dates using Yahoo Finance, with support for
both daily and weekly aggregation.

Features:
- Get OHLC data for a specific date
- Get weekly OHLC data (aggregated from daily data)
- Raw prices (not adjusted for splits/dividends)
- Automatic handling of weekends/holidays
- Simple, clean interface

Usage:
    from data_management.ohlc_data_downloader import OHLCDataDownloader
    
    downloader = OHLCDataDownloader()
    ohlc = downloader.get_ohlc('SPY', '2025-01-13')
    weekly_ohlc = downloader.get_weekly_ohlc('SPY', '2025-01-13')
    print(f"Close: {ohlc['close']}")
    print(f"Weekly Close: {weekly_ohlc['close']}")
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf

# Setup logging
logger = logging.getLogger(__name__)


class OHLCDataDownloader:
    """Simple OHLC data downloader using Yahoo Finance"""
    
    def __init__(self):
        """Initialize the OHLC downloader"""
        self.logger = logger
    
    def get_ohlc(self, ticker: str, date: Union[str, datetime]) -> Optional[Dict[str, float]]:
        """
        Get OHLC data for a specific ticker and date
        
        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')
            date: Target date as string 'YYYY-MM-DD' or datetime object
            
        Returns:
            Dictionary with keys: 'open', 'high', 'low', 'close', 'volume'
            Returns None if no data available
        """
        try:
            # Convert date to datetime if string
            if isinstance(date, str):
                target_date = pd.to_datetime(date).date()
            else:
                target_date = date.date() if hasattr(date, 'date') else date
            
            # Get data for a range around the target date to handle weekends/holidays
            start_date = target_date - timedelta(days=5)
            end_date = target_date + timedelta(days=5)
            
            # Download raw data (not adjusted for splits/dividends)
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if hist.empty:
                self.logger.warning(f"No OHLC data found for {ticker} around {target_date}")
                return None
            
            # Find the closest trading day to target date
            hist.index = hist.index.date
            if target_date in hist.index:
                # Exact date match
                row = hist.loc[target_date]
            else:
                # Find closest trading day
                closest_date = min(hist.index, key=lambda x: abs((x - target_date).days))
                row = hist.loc[closest_date]
                self.logger.info(f"Used closest trading day {closest_date} for target {target_date}")
            
            # Extract OHLC data
            ohlc_data = {
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume']) if 'Volume' in row else 0
            }
            
            self.logger.debug(f"OHLC for {ticker} on {target_date}: {ohlc_data}")
            return ohlc_data
            
        except Exception as e:
            self.logger.error(f"Error getting OHLC data for {ticker} on {date}: {e}")
            return None
    
    def get_close_price(self, ticker: str, date: Union[str, datetime]) -> Optional[float]:
        """
        Get just the closing price for a specific ticker and date
        
        Args:
            ticker: Stock ticker symbol
            date: Target date
            
        Returns:
            Closing price as float, or None if not available
        """
        ohlc = self.get_ohlc(ticker, date)
        return ohlc['close'] if ohlc else None
    
    def get_weekly_ohlc(self, ticker: str, date: Union[str, datetime]) -> Optional[Dict[str, float]]:
        """
        Get weekly OHLC data for a specific ticker and date (aggregated from daily data)
        
        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')
            date: Target date as string 'YYYY-MM-DD' or datetime object
            
        Returns:
            Dictionary with keys: 'open', 'high', 'low', 'close', 'volume', 'week_start', 'week_end'
            Returns None if no data available
        """
        try:
            # Convert date to datetime if string
            if isinstance(date, str):
                target_date = pd.to_datetime(date).date()
            else:
                target_date = date.date() if hasattr(date, 'date') else date
            
            # Calculate the start of the week (Monday)
            week_start = target_date - timedelta(days=target_date.weekday())
            week_end = week_start + timedelta(days=6)
            
            # Get data for the entire week
            start_date = week_start - timedelta(days=2)  # Buffer for weekends
            end_date = week_end + timedelta(days=2)      # Buffer for weekends
            
            # Download raw data (not adjusted for splits/dividends)
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if hist.empty:
                self.logger.warning(f"No OHLC data found for {ticker} around week {week_start} to {week_end}")
                return None
            
            # Filter to only trading days within the week
            hist.index = hist.index.date
            week_data = hist[(hist.index >= week_start) & (hist.index <= week_end)]
            
            if week_data.empty:
                self.logger.warning(f"No trading days found for {ticker} in week {week_start} to {week_end}")
                return None
            
            # Aggregate weekly OHLC data
            weekly_ohlc = {
                'open': float(week_data['Open'].iloc[0]),      # First day's open
                'high': float(week_data['High'].max()),        # Highest high of the week
                'low': float(week_data['Low'].min()),          # Lowest low of the week
                'close': float(week_data['Close'].iloc[-1]),   # Last day's close
                'volume': int(week_data['Volume'].sum()),      # Total volume for the week
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'trading_days': len(week_data)
            }
            
            self.logger.debug(f"Weekly OHLC for {ticker} week {week_start} to {week_end}: {weekly_ohlc}")
            return weekly_ohlc
            
        except Exception as e:
            self.logger.error(f"Error getting weekly OHLC data for {ticker} on {date}: {e}")
            return None
    
    def get_weekly_close_price(self, ticker: str, date: Union[str, datetime]) -> Optional[float]:
        """
        Get just the weekly closing price for a specific ticker and date
        
        Args:
            ticker: Stock ticker symbol
            date: Target date
            
        Returns:
            Weekly closing price as float, or None if not available
        """
        weekly_ohlc = self.get_weekly_ohlc(ticker, date)
        return weekly_ohlc['close'] if weekly_ohlc else None
    
    def get_weekly_dates_for_period(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> list:
        """
        Get a list of weekly dates (Fridays) for a given period
        
        Args:
            start_date: Start date for the period
            end_date: End date for the period
            
        Returns:
            List of date strings in YYYY-MM-DD format for each Friday in the period
        """
        try:
            # Convert dates to datetime if strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            else:
                start_date = start_date.date() if hasattr(start_date, 'date') else start_date
                
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()
            else:
                end_date = end_date.date() if hasattr(end_date, 'date') else end_date
            
            weekly_dates = []
            current_date = start_date
            
            # Find the first Friday on or after start_date
            while current_date <= end_date:
                if current_date.weekday() == 4:  # Friday
                    weekly_dates.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
            
            # Continue adding Fridays until we exceed end_date
            while current_date <= end_date:
                if current_date.weekday() == 4:  # Friday
                    weekly_dates.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=7)
            
            self.logger.info(f"Generated {len(weekly_dates)} weekly dates from {start_date} to {end_date}")
            return weekly_dates
            
        except Exception as e:
            self.logger.error(f"Error generating weekly dates for period {start_date} to {end_date}: {e}")
            return []


def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OHLC Data Downloader with Weekly Support')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker (e.g., SPY)')
    parser.add_argument('--date', type=str, required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--weekly', action='store_true', help='Get weekly OHLC data instead of daily')
    parser.add_argument('--period', action='store_true', help='Show weekly dates for a period (requires --end-date)')
    parser.add_argument('--end-date', type=str, help='End date for period analysis (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Setup logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print(f"🚀 OHLC DATA DOWNLOADER")
    print(f"📊 Ticker: {args.ticker}")
    print(f"📅 Date: {args.date}")
    if args.weekly:
        print("📅 Mode: Weekly Aggregation")
    if args.period and args.end_date:
        print(f"📅 Period: {args.date} to {args.end_date}")
    print("=" * 50)
    
    try:
        downloader = OHLCDataDownloader()
        
        if args.period and args.end_date:
            # Show weekly dates for period
            weekly_dates = downloader.get_weekly_dates_for_period(args.date, args.end_date)
            print(f"📅 Weekly dates (Fridays) from {args.date} to {args.end_date}:")
            for i, date in enumerate(weekly_dates, 1):
                print(f"   {i:2d}. {date}")
            print(f"\nTotal: {len(weekly_dates)} weeks")
            
        elif args.weekly:
            # Get weekly OHLC data
            weekly_ohlc = downloader.get_weekly_ohlc(args.ticker, args.date)
            
            if weekly_ohlc:
                print(f"✅ Weekly OHLC Data for {args.ticker} (week of {args.date}):")
                print(f"   Week: {weekly_ohlc['week_start']} to {weekly_ohlc['week_end']}")
                print(f"   Trading Days: {weekly_ohlc['trading_days']}")
                print(f"   Open:   ${weekly_ohlc['open']:.2f}")
                print(f"   High:   ${weekly_ohlc['high']:.2f}")
                print(f"   Low:    ${weekly_ohlc['low']:.2f}")
                print(f"   Close:  ${weekly_ohlc['close']:.2f}")
                print(f"   Volume: {weekly_ohlc['volume']:,}")
            else:
                print(f"❌ No weekly data available for {args.ticker} for week of {args.date}")
        else:
            # Get daily OHLC data
            ohlc = downloader.get_ohlc(args.ticker, args.date)
            
            if ohlc:
                print(f"✅ Daily OHLC Data for {args.ticker} on {args.date}:")
                print(f"   Open:   ${ohlc['open']:.2f}")
                print(f"   High:   ${ohlc['high']:.2f}")
                print(f"   Low:    ${ohlc['low']:.2f}")
                print(f"   Close:  ${ohlc['close']:.2f}")
                print(f"   Volume: {ohlc['volume']:,}")
            else:
                print(f"❌ No data available for {args.ticker} on {args.date}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()