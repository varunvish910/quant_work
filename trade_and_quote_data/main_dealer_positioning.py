#!/usr/bin/env python3
"""
SPX Weekly Options Dealer Positioning Analysis - Main Entry Point
Orchestrates the complete analysis pipeline from data collection to report generation
"""

import os
import sys
from pathlib import Path
import pandas as pd
import argparse
from datetime import datetime, date
import json

# Import our analysis modules
from spx_trades_downloader import SPXTradesDownloader
from trade_classifier import TradeClassifier
from greeks_calculator import GreeksCalculator
from market_structure_analyzer import MarketStructureAnalyzer
from dealer_positioning_visualizer import DealerPositioningVisualizer
from dealer_positioning_report import DealerPositioningReport


class DealerPositioningPipeline:
    """Main pipeline for SPX dealer positioning analysis"""
    
    def __init__(self, api_key: str, spot_price: float, output_dir: str = "outputs/dealer_positioning"):
        self.api_key = api_key
        self.spot_price = spot_price
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data directory structure
        self.data_dir = Path("data/spx_options")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.downloader = None
        self.classifier = None
        self.calculator = None
        self.analyzer = None
        self.visualizer = None
        self.report_gen = None
        
        print(f"Initialized SPX Dealer Positioning Pipeline")
        print(f"Output directory: {self.output_dir}")
        print(f"Spot price: {self.spot_price}")
    
    def run_full_analysis(self, target_date: str, expiry_dates: list, 
                         skip_download: bool = False) -> dict:
        """Run the complete analysis pipeline"""
        
        print(f"\n{'='*60}")
        print(f"SPX DEALER POSITIONING ANALYSIS")
        print(f"Date: {target_date}")
        print(f"Expiries: {', '.join(expiry_dates)}")
        print(f"Spot: {self.spot_price}")
        print(f"{'='*60}")
        
        results = {}
        
        try:
            # Phase 1: Data Collection
            if not skip_download:
                print(f"\n[PHASE 1] DATA COLLECTION")
                print("-" * 40)
                trades_df = self._run_data_collection(target_date, expiry_dates)
                results['trades_collected'] = len(trades_df)
            else:
                print(f"\n[PHASE 1] LOADING EXISTING DATA")
                print("-" * 40)
                trades_df = self._load_existing_data(target_date)
                results['trades_loaded'] = len(trades_df)
            
            # Phase 2: Trade Classification
            print(f"\n[PHASE 2] TRADE CLASSIFICATION")
            print("-" * 40)
            classified_df = self._run_trade_classification(trades_df, target_date)
            results['trades_classified'] = len(classified_df)
            
            # Phase 3: Greeks Calculation
            print(f"\n[PHASE 3] GREEKS CALCULATION")
            print("-" * 40)
            aggregated_greeks, trade_greeks = self._run_greeks_calculation(classified_df)
            results['strikes_analyzed'] = len(aggregated_greeks)
            results['trade_greeks'] = len(trade_greeks)
            
            # Phase 4: Market Structure Analysis
            print(f"\n[PHASE 4] MARKET STRUCTURE ANALYSIS")
            print("-" * 40)
            analysis = self._run_market_analysis(aggregated_greeks)
            results['analysis_complete'] = True
            
            # Phase 5: Visualization
            print(f"\n[PHASE 5] VISUALIZATION")
            print("-" * 40)
            chart_files = self._run_visualization(aggregated_greeks, trade_greeks, analysis)
            results['charts_created'] = len(chart_files)
            
            # Phase 6: Report Generation
            print(f"\n[PHASE 6] REPORT GENERATION")
            print("-" * 40)
            report_files = self._run_report_generation(aggregated_greeks, trade_greeks, analysis)
            results['reports_generated'] = len(report_files)
            
            # Save summary
            self._save_analysis_summary(results, analysis, target_date)
            
            print(f"\n{'='*60}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*60}")
            self._print_summary(results, analysis)
            
            return results
            
        except Exception as e:
            print(f"\nERROR in analysis pipeline: {e}")
            raise
    
    def _run_data_collection(self, target_date: str, expiry_dates: list) -> pd.DataFrame:
        """Phase 1: Download trades and quotes data"""
        
        self.downloader = SPXTradesDownloader(self.api_key)
        
        # Download raw trades
        print(f"Downloading SPX options trades for {target_date}...")
        trades_df = self.downloader.download_trades(target_date, expiry_dates)
        
        if len(trades_df) == 0:
            raise ValueError("No trades data downloaded. Check API key and date.")
        
        print(f"✓ Downloaded {len(trades_df):,} trades")
        
        # Enrich with quotes
        print(f"Enriching trades with quote data...")
        enriched_df = self.downloader.enrich_and_save(trades_df, target_date)
        
        print(f"✓ Enriched {len(enriched_df):,} trades with quotes")
        
        return enriched_df
    
    def _load_existing_data(self, target_date: str) -> pd.DataFrame:
        """Load existing enriched trades data"""
        
        try:
            file_path = f"data/spx_options/trades/{target_date}_enriched_trades.parquet"
            trades_df = pd.read_parquet(file_path)
            print(f"✓ Loaded {len(trades_df):,} existing trades from {file_path}")
            return trades_df
        except FileNotFoundError:
            raise FileNotFoundError(f"No existing data found for {target_date}. Set skip_download=False to download fresh data.")
    
    def _run_trade_classification(self, trades_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Phase 2: Classify trades as BTO/STO/BTC/STC"""
        
        self.classifier = TradeClassifier()
        
        print(f"Classifying {len(trades_df):,} trades...")
        classified_df = self.classifier.classify_all_trades(trades_df)
        
        # Save classified trades
        output_file = f"data/spx_options/classified/{target_date}_classified_trades.parquet"
        classified_df.to_parquet(output_file, index=False)
        
        print(f"✓ Classified trades saved to {output_file}")
        
        return classified_df
    
    def _run_greeks_calculation(self, classified_df: pd.DataFrame) -> tuple:
        """Phase 3: Calculate all option Greeks"""
        
        self.calculator = GreeksCalculator(
            spot=self.spot_price, 
            rate=0.05, 
            dividend_yield=0.015
        )
        
        print(f"Calculating Greeks for {len(classified_df):,} trades...")
        aggregated_greeks, trade_greeks = self.calculator.aggregate_dealer_greeks(classified_df)
        
        if len(aggregated_greeks) == 0:
            raise ValueError("No valid Greeks calculated. Check input data.")
        
        # Save Greeks data
        aggregated_greeks.to_parquet("data/spx_options/greeks/aggregated_greeks.parquet", index=False)
        trade_greeks.to_parquet("data/spx_options/greeks/trade_level_greeks.parquet", index=False)
        
        print(f"✓ Greeks calculated for {len(aggregated_greeks)} strikes")
        print(f"✓ Trade-level Greeks: {len(trade_greeks)} records")
        
        return aggregated_greeks, trade_greeks
    
    def _run_market_analysis(self, aggregated_greeks: pd.DataFrame) -> dict:
        """Phase 4: Analyze market structure and positioning patterns"""
        
        self.analyzer = MarketStructureAnalyzer(spot_price=self.spot_price)
        
        print(f"Analyzing market structure...")
        analysis = self.analyzer.analyze_full_structure(aggregated_greeks)
        
        # Save analysis
        with open("data/spx_options/market_structure_analysis.json", "w") as f:
            # Convert any dataclass objects to dicts for JSON serialization
            analysis_json = self._serialize_analysis(analysis)
            json.dump(analysis_json, f, indent=2, default=str)
        
        print(f"✓ Market structure analysis complete")
        print(f"✓ Regime identified: {getattr(analysis.get('market_regime', {}), 'regime_type', 'unknown')}")
        
        return analysis
    
    def _run_visualization(self, aggregated_greeks: pd.DataFrame, 
                          trade_greeks: pd.DataFrame, analysis: dict) -> dict:
        """Phase 5: Create all visualization charts"""
        
        self.visualizer = DealerPositioningVisualizer(
            spot_price=self.spot_price, 
            theme="dark"
        )
        
        print(f"Creating visualization charts...")
        chart_files = self.visualizer.create_all_charts(
            aggregated_greeks, trade_greeks, analysis, 
            output_dir=str(self.output_dir)
        )
        
        print(f"✓ Created {len(chart_files)} visualization files")
        
        return chart_files
    
    def _run_report_generation(self, aggregated_greeks: pd.DataFrame,
                              trade_greeks: pd.DataFrame, analysis: dict) -> list:
        """Phase 6: Generate comprehensive reports"""
        
        self.report_gen = DealerPositioningReport(spot_price=self.spot_price)
        
        print(f"Generating comprehensive report...")
        
        # Full report
        full_report = self.report_gen.generate_full_report(
            aggregated_greeks, trade_greeks, analysis,
            output_file=str(self.output_dir / "comprehensive_report.md")
        )
        
        # Executive summary
        summary_sections = [
            self.report_gen._generate_header(),
            self.report_gen._generate_executive_summary(analysis),
            self.report_gen._generate_trading_implications(analysis),
            self.report_gen._generate_quantitative_summary(aggregated_greeks, analysis)
        ]
        
        summary_report = "\n\n".join(summary_sections)
        
        with open(self.output_dir / "executive_summary.md", 'w') as f:
            f.write(summary_report)
        
        report_files = [
            str(self.output_dir / "comprehensive_report.md"),
            str(self.output_dir / "executive_summary.md")
        ]
        
        print(f"✓ Generated {len(report_files)} report files")
        
        return report_files
    
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
    
    def _save_analysis_summary(self, results: dict, analysis: dict, target_date: str):
        """Save high-level analysis summary"""
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'target_date': target_date,
            'spot_price': self.spot_price,
            'pipeline_results': results,
            'key_findings': {
                'market_regime': getattr(analysis.get('market_regime', {}), 'regime_type', 'unknown'),
                'regime_confidence': getattr(analysis.get('market_regime', {}), 'confidence', 0.0),
                'gamma_centroid': getattr(analysis.get('key_levels', {}), 'gamma_centroid', self.spot_price),
                'upside_pivot': getattr(analysis.get('key_levels', {}), 'upside_pivot', None),
                'downside_pivot': getattr(analysis.get('key_levels', {}), 'downside_pivot', None),
                'primary_insights': analysis.get('trading_insights', [])[:3]
            }
        }
        
        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _print_summary(self, results: dict, analysis: dict):
        """Print analysis summary to console"""
        
        regime = analysis.get('market_regime', {})
        key_levels = analysis.get('key_levels', {})
        insights = analysis.get('trading_insights', [])
        
        print(f"📊 Trades processed: {results.get('trades_classified', 0):,}")
        print(f"📈 Strikes analyzed: {results.get('strikes_analyzed', 0)}")
        print(f"🎯 Market regime: {getattr(regime, 'regime_type', 'unknown').title()}")
        print(f"⚖️  Gamma centroid: {getattr(key_levels, 'gamma_centroid', 0):.0f}")
        
        if hasattr(key_levels, 'upside_pivot') and key_levels.upside_pivot:
            print(f"⬆️  Upside pivot: {key_levels.upside_pivot:.0f}")
        
        if hasattr(key_levels, 'downside_pivot') and key_levels.downside_pivot:
            print(f"⬇️  Downside pivot: {key_levels.downside_pivot:.0f}")
        
        print(f"\n💡 Key insights:")
        for i, insight in enumerate(insights[:3], 1):
            print(f"   {i}. {insight}")
        
        print(f"\n📁 Output files in: {self.output_dir}")
        print(f"   • Charts: greeks_panel.html, dashboard.html")
        print(f"   • Reports: comprehensive_report.md, executive_summary.md")


def main():
    """Main entry point with command line interface"""
    
    parser = argparse.ArgumentParser(description='SPX Weekly Options Dealer Positioning Analysis')
    parser.add_argument('--date', type=str, default='2025-10-05', 
                       help='Target analysis date (YYYY-MM-DD)')
    parser.add_argument('--expiry', type=str, nargs='+', default=['2025-10-11'],
                       help='Option expiry dates (YYYY-MM-DD)')
    parser.add_argument('--spot', type=float, default=5800.0,
                       help='Current SPX spot price')
    parser.add_argument('--api-key', type=str, 
                       help='Polygon API key (or set POLYGON_API_KEY env var)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download, use existing data')
    parser.add_argument('--output', type=str, default='outputs/dealer_positioning',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('POLYGON_API_KEY')
    if not api_key and not args.skip_download:
        print("ERROR: Polygon API key required. Set --api-key or POLYGON_API_KEY environment variable")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = DealerPositioningPipeline(
            api_key=api_key,
            spot_price=args.spot,
            output_dir=args.output
        )
        
        # Run analysis
        results = pipeline.run_full_analysis(
            target_date=args.date,
            expiry_dates=args.expiry,
            skip_download=args.skip_download
        )
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()