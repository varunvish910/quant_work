#!/usr/bin/env python3
"""
Real SPY Options Analyzer
Uses the actual SPY data we have and processes it for different expiry combinations to show real positioning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import json
from trade_classifier import TradeClassifier
from greeks_calculator import GreeksCalculator
from market_structure_analyzer import MarketStructureAnalyzer
import yfinance as yf


class RealSPYAnalyzer:
    """Analyzes real SPY options data for different expiry combinations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("outputs/real_spy_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_real_spy_data(self) -> pd.DataFrame:
        """Load the actual SPY trades data we have"""
        
        spy_file = self.data_dir / "spy_options" / "trades" / "2025-09-12_enriched_trades.parquet"
        
        if not spy_file.exists():
            raise FileNotFoundError(f"Real SPY data not found at {spy_file}")
        
        trades_df = pd.read_parquet(spy_file)
        print(f"✓ Loaded real SPY data: {len(trades_df):,} trades")
        print(f"✓ Date range: {trades_df['expiry'].min()} to {trades_df['expiry'].max()}")
        print(f"✓ Strike range: ${trades_df['strike'].min():.0f} to ${trades_df['strike'].max():.0f}")
        print(f"✓ Unique expiries: {trades_df['expiry'].nunique()}")
        
        return trades_df
    
    def analyze_by_expiry_groups(self, trades_df: pd.DataFrame) -> dict:
        """Analyze the real data by grouping different expiry combinations"""
        
        print("\n📊 Analyzing real SPY positioning by expiry groups...")
        
        # Get all available expiries from the real data
        available_expiries = sorted(trades_df['expiry'].unique())
        print(f"Available expiries in real data: {[str(exp) for exp in available_expiries]}")
        
        # Create different expiry groupings for analysis
        expiry_groups = {
            "All_Expiries": available_expiries,
            "Near_Term": available_expiries[:3] if len(available_expiries) >= 3 else available_expiries,
            "Mid_Term": available_expiries[1:4] if len(available_expiries) >= 4 else available_expiries[1:],
            "Far_Term": available_expiries[-3:] if len(available_expiries) >= 3 else available_expiries[-2:],
            "Weekly_Focus": [exp for exp in available_expiries if exp.weekday() == 4][:3]  # Fridays only
        }
        
        # Get current SPY price
        try:
            spy = yf.Ticker('SPY')
            current_price = spy.history(period="1d")["Close"].iloc[-1]
            print(f"✓ Current SPY price: ${current_price:.2f}")
        except:
            current_price = 569.21
            print(f"Using fallback SPY price: ${current_price:.2f}")
        
        analysis_results = {}
        
        for group_name, expiry_list in expiry_groups.items():
            if not expiry_list:
                continue
                
            print(f"\n🔍 Analyzing {group_name}: {[str(exp) for exp in expiry_list]}")
            
            # Filter trades for this expiry group
            group_trades = trades_df[trades_df['expiry'].isin(expiry_list)].copy()
            
            if len(group_trades) == 0:
                print(f"   ⚠️  No trades found for {group_name}")
                continue
            
            print(f"   📈 Processing {len(group_trades):,} trades for {group_name}")
            
            try:
                # Classify trades
                classifier = TradeClassifier()
                classified_df = classifier.classify_all_trades(group_trades)
                
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
                
                # Store results
                analysis_results[group_name] = {
                    'expiry_list': [str(exp) for exp in expiry_list],
                    'spot_price': current_price,
                    'aggregated_greeks': aggregated_greeks,
                    'trade_greeks': trade_greeks,
                    'analysis': analysis,
                    'trades_count': len(classified_df),
                    'strikes_count': len(aggregated_greeks)
                }
                
                print(f"   ✅ Completed: {len(aggregated_greeks)} strikes analyzed")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {group_name}: {e}")
                continue
        
        print(f"\n✅ Analyzed {len(analysis_results)} expiry groups")
        return analysis_results
    
    def save_analysis_results(self, analysis_results: dict):
        """Save the analysis results for visualization"""
        
        print("\n💾 Saving analysis results...")
        
        # Save each group's data
        for group_name, results in analysis_results.items():
            # Save aggregated Greeks
            greeks_file = self.output_dir / f"{group_name}_greeks.parquet"
            results['aggregated_greeks'].to_parquet(greeks_file, index=False)
            
            # Save analysis summary
            analysis_file = self.output_dir / f"{group_name}_analysis.json"
            
            # Convert analysis to JSON-serializable format
            json_analysis = self._serialize_analysis(results['analysis'])
            
            summary = {
                'group_name': group_name,
                'expiry_list': results['expiry_list'],
                'spot_price': results['spot_price'],
                'trades_count': results['trades_count'],
                'strikes_count': results['strikes_count'],
                'analysis': json_analysis
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"   ✅ Saved {group_name}: {results['strikes_count']} strikes")
        
        # Create overall summary
        self._create_summary_report(analysis_results)
        
        print(f"✅ All results saved to {self.output_dir}")
    
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
    
    def _create_summary_report(self, analysis_results: dict):
        """Create a summary report of all expiry group analyses"""
        
        print("\n📋 Creating summary report...")
        
        summary_data = []
        
        for group_name, results in analysis_results.items():
            analysis = results['analysis']
            market_regime = analysis.get('market_regime', {})
            key_levels = analysis.get('key_levels', {})
            greeks_summary = analysis.get('greeks_summary', {})
            
            summary_data.append({
                'expiry_group': group_name,
                'expiry_list': ', '.join(results['expiry_list']),
                'trades_count': results['trades_count'],
                'strikes_count': results['strikes_count'],
                'market_regime': getattr(market_regime, 'regime_type', 'unknown') if hasattr(market_regime, 'regime_type') else market_regime.get('regime_type', 'unknown'),
                'regime_confidence': getattr(market_regime, 'confidence', 0) if hasattr(market_regime, 'confidence') else market_regime.get('confidence', 0),
                'gamma_centroid': getattr(key_levels, 'gamma_centroid', results['spot_price']) if hasattr(key_levels, 'gamma_centroid') else key_levels.get('gamma_centroid', results['spot_price']),
                'upside_pivot': getattr(key_levels, 'upside_pivot', None) if hasattr(key_levels, 'upside_pivot') else key_levels.get('upside_pivot', None),
                'downside_pivot': getattr(key_levels, 'downside_pivot', None) if hasattr(key_levels, 'downside_pivot') else key_levels.get('downside_pivot', None),
                'total_gamma': greeks_summary.get('total_gamma', 0),
                'total_delta': greeks_summary.get('total_delta', 0),
                'total_vega': greeks_summary.get('total_vega', 0)
            })
        
        # Save summary as CSV and JSON
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / "expiry_groups_summary.csv", index=False)
        summary_df.to_json(self.output_dir / "expiry_groups_summary.json", orient='records', indent=2)
        
        print(f"✅ Summary report saved")
        
        # Print summary to console
        print(f"\n{'='*80}")
        print(f"REAL SPY OPTIONS POSITIONING SUMMARY")
        print(f"{'='*80}")
        
        for _, row in summary_df.iterrows():
            print(f"\n📊 {row['expiry_group']}:")
            print(f"   Expiries: {row['expiry_list']}")
            print(f"   Trades: {row['trades_count']:,} | Strikes: {row['strikes_count']}")
            print(f"   Market Regime: {row['market_regime']} ({row['regime_confidence']:.1%} confidence)")
            print(f"   Gamma Centroid: ${row['gamma_centroid']:.2f}")
            print(f"   Total Gamma: {row['total_gamma']:.0f} | Delta: {row['total_delta']:.0f}")
        
        return summary_df


def main():
    """Main execution"""
    
    print("🚀 Real SPY Options Positioning Analysis")
    print("=" * 50)
    
    analyzer = RealSPYAnalyzer()
    
    try:
        # Load real SPY data
        real_trades = analyzer.load_real_spy_data()
        
        # Analyze by different expiry groupings
        analysis_results = analyzer.analyze_by_expiry_groups(real_trades)
        
        if not analysis_results:
            print("❌ No analysis results generated")
            return
        
        # Save results
        analyzer.save_analysis_results(analysis_results)
        
        print(f"\n🎉 Real SPY analysis complete!")
        print(f"📁 Results saved to: {analyzer.output_dir}")
        print(f"\n💡 This shows how dealer positioning varies across different expiry combinations")
        print(f"📊 You can see how the same underlying data creates different positioning profiles")
        print(f"🔍 Each expiry group represents a different 'time slice' of market structure")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()