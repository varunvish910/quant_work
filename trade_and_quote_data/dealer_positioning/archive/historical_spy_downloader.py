#!/usr/bin/env python3
"""
Historical SPY Options Data Downloader
Downloads SPY options data for the last 30 days for specific expiry dates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
from spy_trades_downloader import SPYTradesDownloader
from trade_classifier import TradeClassifier  
from greeks_calculator import GreeksCalculator
from market_structure_analyzer import MarketStructureAnalyzer
import yfinance as yf


class HistoricalSPYDownloader:
    """Downloads and processes historical SPY options data"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/historical_spy"):
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "daily_data").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        
        self.downloader = SPYTradesDownloader(api_key, str(self.output_dir))
        
    def get_date_range(self, days_back: int = 30, start_date: str = None) -> list:
        """Get list of trading dates for the last N days (excluding weekends)"""
        dates = []
        
        if start_date:
            current_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            # Try different recent periods to find available data
            current_date = datetime(2024, 9, 30).date()  # Try September 2024
        
        while len(dates) < days_back:
            # Only include weekdays (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                dates.append(current_date.strftime('%Y-%m-%d'))
            current_date -= timedelta(days=1)
            
        return list(reversed(dates))  # Return in chronological order
    
    def get_current_spy_price(self) -> float:
        """Get current SPY price"""
        try:
            spy = yf.Ticker('SPY')
            current_price = spy.history(period="1d")["Close"].iloc[-1]
            print(f"✓ Current SPY price: ${current_price:.2f}")
            return current_price
        except Exception as e:
            print(f"Warning: Could not fetch SPY price: {e}")
            fallback_price = 569.21
            print(f"Using fallback SPY price: ${fallback_price:.2f}")
            return fallback_price
    
    def download_historical_data(self, expiry_dates: list, days_back: int = 30) -> dict:
        """Download SPY options data for the last N days"""
        print(f"\n{'='*60}")
        print(f"DOWNLOADING HISTORICAL SPY OPTIONS DATA")
        print(f"Days back: {days_back}")
        print(f"Target expiries: {', '.join(expiry_dates)}")
        print(f"{'='*60}")
        
        # Get trading dates
        trading_dates = self.get_date_range(days_back)
        current_price = self.get_current_spy_price()
        
        results = {
            'dates_processed': [],
            'total_trades': 0,
            'failed_dates': [],
            'expiry_dates': expiry_dates,
            'data_summary': {}
        }
        
        for i, date in enumerate(trading_dates, 1):
            print(f"\n[{i}/{len(trading_dates)}] Processing {date}...")
            
            try:
                # Check if data already exists
                existing_file = self.output_dir / "daily_data" / f"{date}_enriched_trades.parquet"
                
                if existing_file.exists():
                    print(f"   ✅ Found existing data for {date}")
                    trades_df = pd.read_parquet(existing_file)
                else:
                    # Download new data
                    trades_df = self.downloader.download_trades(date, expiry_dates, current_price)
                    
                    if len(trades_df) == 0:
                        print(f"   ⚠️  No trades found for {date}")
                        results['failed_dates'].append(date)
                        continue
                    
                    # Enrich and save
                    trades_df = self.downloader.enrich_and_save(trades_df, date)
                    
                    # Save to our historical directory
                    daily_file = self.output_dir / "daily_data" / f"{date}_enriched_trades.parquet"
                    trades_df.to_parquet(daily_file, index=False)
                
                # Track results
                results['dates_processed'].append(date)
                results['total_trades'] += len(trades_df)
                
                # Store summary for this date
                results['data_summary'][date] = {
                    'total_trades': len(trades_df),
                    'unique_strikes': trades_df['strike'].nunique() if len(trades_df) > 0 else 0,
                    'expiries_found': trades_df['expiry'].nunique() if len(trades_df) > 0 else 0
                }
                
                print(f"   ✅ Processed {len(trades_df):,} trades for {date}")
                
            except Exception as e:
                print(f"   ❌ Failed to process {date}: {e}")
                results['failed_dates'].append(date)
                continue
        
        # Save overall summary
        self._save_download_summary(results)
        
        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETE")
        print(f"✅ Successfully processed: {len(results['dates_processed'])} dates")
        print(f"❌ Failed: {len(results['failed_dates'])} dates")
        print(f"📊 Total trades: {results['total_trades']:,}")
        print(f"{'='*60}")
        
        return results
    
    def process_daily_greeks(self, date: str, expiry_dates: list) -> dict:
        """Process Greeks for a specific date"""
        try:
            # Load trades data
            trades_file = self.output_dir / "daily_data" / f"{date}_enriched_trades.parquet"
            if not trades_file.exists():
                return None
                
            trades_df = pd.read_parquet(trades_file)
            
            if len(trades_df) == 0:
                return None
            
            # Get SPY price for this date (or use current as fallback)
            current_price = self.get_current_spy_price()
            
            # Classify trades
            classifier = TradeClassifier()
            classified_df = classifier.classify_all_trades(trades_df)
            
            # Calculate Greeks
            calculator = GreeksCalculator(
                spot=current_price,
                rate=0.05,
                dividend_yield=0.015
            )
            
            aggregated_greeks, trade_greeks = calculator.aggregate_dealer_greeks(classified_df)
            
            # Market structure analysis
            analyzer = MarketStructureAnalyzer(spot_price=current_price)
            analysis = analyzer.analyze_full_structure(aggregated_greeks)
            
            return {
                'date': date,
                'spot_price': current_price,
                'aggregated_greeks': aggregated_greeks,
                'trade_greeks': trade_greeks,
                'analysis': analysis,
                'trades_count': len(classified_df),
                'strikes_count': len(aggregated_greeks)
            }
            
        except Exception as e:
            print(f"Error processing Greeks for {date}: {e}")
            return None
    
    def process_all_historical_greeks(self, expiry_dates: list) -> dict:
        """Process Greeks for all historical dates"""
        print(f"\n{'='*60}")
        print(f"PROCESSING HISTORICAL GREEKS")
        print(f"{'='*60}")
        
        # Get list of available dates
        daily_files = list((self.output_dir / "daily_data").glob("*_enriched_trades.parquet"))
        dates = [f.stem.replace('_enriched_trades', '') for f in daily_files]
        dates.sort()
        
        historical_data = {}
        
        for i, date in enumerate(dates, 1):
            print(f"\n[{i}/{len(dates)}] Processing Greeks for {date}...")
            
            # Check if already processed
            processed_file = self.output_dir / "processed" / f"{date}_greeks.parquet"
            
            if processed_file.exists():
                print(f"   ✅ Found existing Greeks for {date}")
                continue
            
            result = self.process_daily_greeks(date, expiry_dates)
            
            if result:
                # Save aggregated Greeks
                result['aggregated_greeks'].to_parquet(processed_file, index=False)
                
                # Save analysis summary
                analysis_file = self.output_dir / "processed" / f"{date}_analysis.json"
                with open(analysis_file, 'w') as f:
                    # Convert analysis to JSON-serializable format
                    json_analysis = self._serialize_analysis(result['analysis'])
                    json.dump({
                        'date': date,
                        'spot_price': result['spot_price'],
                        'trades_count': result['trades_count'],
                        'strikes_count': result['strikes_count'],
                        'analysis': json_analysis
                    }, f, indent=2, default=str)
                
                historical_data[date] = result
                print(f"   ✅ Processed {result['strikes_count']} strikes, {result['trades_count']} trades")
            else:
                print(f"   ❌ Failed to process {date}")
        
        print(f"\n✅ Historical Greeks processing complete")
        return historical_data
    
    def create_historical_summary(self) -> pd.DataFrame:
        """Create a summary dataframe of historical positioning data"""
        print("\n📊 Creating historical summary...")
        
        # Load all analysis files
        analysis_files = list((self.output_dir / "processed").glob("*_analysis.json"))
        
        summary_data = []
        
        for file in sorted(analysis_files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract key metrics
                analysis = data.get('analysis', {})
                market_regime = analysis.get('market_regime', {})
                key_levels = analysis.get('key_levels', {})
                greeks_summary = analysis.get('greeks_summary', {})
                
                summary_data.append({
                    'date': data['date'],
                    'spot_price': data.get('spot_price', 0),
                    'trades_count': data.get('trades_count', 0),
                    'strikes_count': data.get('strikes_count', 0),
                    'market_regime': getattr(market_regime, 'regime_type', 'unknown') if hasattr(market_regime, 'regime_type') else market_regime.get('regime_type', 'unknown'),
                    'regime_confidence': getattr(market_regime, 'confidence', 0) if hasattr(market_regime, 'confidence') else market_regime.get('confidence', 0),
                    'gamma_centroid': getattr(key_levels, 'gamma_centroid', 0) if hasattr(key_levels, 'gamma_centroid') else key_levels.get('gamma_centroid', 0),
                    'upside_pivot': getattr(key_levels, 'upside_pivot', None) if hasattr(key_levels, 'upside_pivot') else key_levels.get('upside_pivot', None),
                    'downside_pivot': getattr(key_levels, 'downside_pivot', None) if hasattr(key_levels, 'downside_pivot') else key_levels.get('downside_pivot', None),
                    'total_gamma': greeks_summary.get('total_gamma', 0),
                    'total_delta': greeks_summary.get('total_delta', 0),
                    'total_vega': greeks_summary.get('total_vega', 0),
                    'total_theta': greeks_summary.get('total_theta', 0)
                })
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df['date'] = pd.to_datetime(summary_df['date'])
            summary_df = summary_df.sort_values('date')
            
            # Save summary
            summary_file = self.output_dir / "historical_summary.parquet"
            summary_df.to_parquet(summary_file, index=False)
            
            print(f"✅ Created historical summary with {len(summary_df)} dates")
            return summary_df
        else:
            print("❌ No data found for historical summary")
            return pd.DataFrame()
    
    def _serialize_analysis(self, analysis: dict) -> dict:
        """Convert analysis dict for JSON serialization"""
        serialized = {}
        
        for key, value in analysis.items():
            if hasattr(value, '__dict__'):
                # Convert dataclass to dict
                serialized[key] = value.__dict__
            elif isinstance(value, dict):
                # Recursively serialize nested dicts
                serialized[key] = self._serialize_analysis(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def _save_download_summary(self, results: dict):
        """Save download summary"""
        summary_file = self.output_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Saved download summary to {summary_file}")


def main():
    """Main execution"""
    # Let's try getting recent 2024 data first to test the system
    # Target expiry dates for Oct 2024 (the most recent complete week)
    expiry_dates = [
        "2024-10-04",  # Friday
        "2024-10-07",  # Monday
        "2024-10-08",  # Tuesday  
        "2024-10-09",  # Wednesday
        "2024-10-10",  # Thursday
        "2024-10-11"   # Friday
    ]
    
    # Initialize downloader
    downloader = HistoricalSPYDownloader()
    
    print("🚀 Starting historical SPY options analysis...")
    print(f"Target expiries: {', '.join(expiry_dates)}")
    
    # Step 1: Download historical data (last 10 days for testing)
    download_results = downloader.download_historical_data(expiry_dates, days_back=10)
    
    # Step 2: Process Greeks for all dates
    historical_data = downloader.process_all_historical_greeks(expiry_dates)
    
    # Step 3: Create summary
    summary_df = downloader.create_historical_summary()
    
    print(f"\n🎉 Historical analysis complete!")
    print(f"📁 Data saved to: {downloader.output_dir}")
    
    if len(summary_df) > 0:
        print(f"\n📊 Summary statistics:")
        print(f"   • Date range: {summary_df['date'].min()} to {summary_df['date'].max()}")
        print(f"   • Total trading days: {len(summary_df)}")
        print(f"   • Total trades processed: {summary_df['trades_count'].sum():,}")
        print(f"   • Avg daily strikes: {summary_df['strikes_count'].mean():.0f}")


if __name__ == "__main__":
    main()