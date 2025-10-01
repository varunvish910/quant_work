"""
SPY Hedging Pattern Matcher
===========================

This script compares current hedging patterns with historical training data
to identify similar signals and their outcomes. It focuses on:

1. Major market events (COVID crash, 2022 bear market, etc.)
2. Hedging buildup patterns before corrections
3. Risk concentration similarities
4. Signal strength comparisons

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HedgingPatternMatcher:
    """
    Matches current hedging patterns with historical training data
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.historical_patterns = {}
        
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """Load options data for a specific date"""
        try:
            year = date[:4]
            month = date[5:7]
            
            # Convert date from YYYY-MM-DD to YYYYMMDD format
            date_formatted = date.replace('-', '')
            file_path = self.data_dir / year / month / f"SPY_options_snapshot_{date_formatted}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                else:
                    df['date'] = pd.to_datetime(date)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            return pd.DataFrame()
    
    def calculate_hedging_metrics(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate hedging metrics for pattern matching"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        metrics = {}
        
        # 1. Deep OTM put activity
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.90]
        if not deep_otm_puts.empty:
            metrics['deep_otm_oi'] = deep_otm_puts['oi_proxy'].sum()
            metrics['deep_otm_volume'] = deep_otm_puts['volume'].sum()
            metrics['deep_otm_vol_oi'] = deep_otm_puts['volume'].sum() / (deep_otm_puts['oi_proxy'].sum() + 1e-6)
        
        # 2. Strike concentration
        total_put_oi = puts['oi_proxy'].sum()
        if total_put_oi > 0:
            top_5_oi = puts.nlargest(5, 'oi_proxy')['oi_proxy'].sum()
            max_single_oi = puts['oi_proxy'].max()
            
            metrics['top_5_oi_pct'] = top_5_oi / total_put_oi
            metrics['max_strike_oi_pct'] = max_single_oi / total_put_oi
        
        # 3. Volume/OI ratio (hedging indicator)
        metrics['put_vol_oi_ratio'] = puts['volume'].sum() / (total_put_oi + 1e-6)
        
        # 4. Long-dated options
        if 'dte' in puts.columns:
            long_dated_puts = puts[puts['dte'] > 60]
            if not long_dated_puts.empty:
                metrics['long_dated_oi_pct'] = long_dated_puts['oi_proxy'].sum() / total_put_oi
            else:
                metrics['long_dated_oi_pct'] = 0
        
        # 5. Put/Call ratio
        calls = df[df['option_type'] == 'C']
        if not calls.empty:
            metrics['pc_ratio_oi'] = total_put_oi / calls['oi_proxy'].sum()
        else:
            metrics['pc_ratio_oi'] = 1.0
        
        # 6. Strike range concentrations
        strike_ranges = [
            (spy_price * 0.95, spy_price * 1.05, "near_money"),
            (spy_price * 0.90, spy_price * 0.95, "otm_5_10"),
            (spy_price * 0.80, spy_price * 0.90, "otm_10_20"),
            (0, spy_price * 0.80, "deep_otm")
        ]
        
        for min_strike, max_strike, range_name in strike_ranges:
            range_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] < max_strike)]
            if not range_puts.empty:
                range_oi = range_puts['oi_proxy'].sum()
                range_vol_oi = range_puts['volume'].sum() / (range_oi + 1e-6)
                
                metrics[f'{range_name}_oi_pct'] = range_oi / total_put_oi
                metrics[f'{range_name}_vol_oi'] = range_vol_oi
        
        return metrics
    
    def analyze_historical_periods(self) -> dict:
        """Analyze key historical periods for pattern matching"""
        
        # Define key historical periods
        historical_periods = {
            'COVID_Crash_2020': {
                'start': '2020-02-01',
                'end': '2020-04-30',
                'description': 'COVID-19 market crash and recovery'
            },
            '2022_Bear_Market': {
                'start': '2022-01-01',
                'end': '2022-12-31',
                'description': '2022 bear market with inflation concerns'
            },
            '2018_Volatility': {
                'start': '2018-10-01',
                'end': '2018-12-31',
                'description': '2018 Q4 volatility spike'
            },
            '2016_Election': {
                'start': '2016-10-01',
                'end': '2016-12-31',
                'description': '2016 election period volatility'
            },
            '2024_Recent': {
                'start': '2024-01-01',
                'end': '2024-12-31',
                'description': '2024 baseline period'
            }
        }
        
        print("🔍 ANALYZING HISTORICAL HEDGING PATTERNS")
        print("=" * 50)
        
        period_analysis = {}
        
        for period_name, period_info in historical_periods.items():
            print(f"\n📊 Analyzing {period_name}: {period_info['description']}")
            print("-" * 60)
            
            start_date = pd.to_datetime(period_info['start'])
            end_date = pd.to_datetime(period_info['end'])
            
            period_metrics = []
            current_date = start_date
            
            # Sample every 5 trading days for efficiency
            sample_count = 0
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Skip weekends
                    if sample_count % 5 == 0:  # Sample every 5th trading day
                        date_str = current_date.strftime('%Y-%m-%d')
                        df = self.load_daily_data(date_str)
                        
                        if not df.empty and 'underlying_price' in df.columns:
                            spy_price = df['underlying_price'].iloc[0]
                            metrics = self.calculate_hedging_metrics(df, spy_price)
                            
                            if metrics:
                                metrics['date'] = current_date
                                metrics['spy_price'] = spy_price
                                period_metrics.append(metrics)
                    
                    sample_count += 1
                
                current_date += timedelta(days=1)
            
            if period_metrics:
                period_df = pd.DataFrame(period_metrics)
                period_analysis[period_name] = {
                    'data': period_df,
                    'summary': self._summarize_period(period_df),
                    'description': period_info['description']
                }
                print(f"✅ Analyzed {len(period_df)} data points")
            else:
                print(f"❌ No data found for {period_name}")
        
        return period_analysis
    
    def _summarize_period(self, df: pd.DataFrame) -> dict:
        """Summarize hedging patterns for a historical period"""
        summary = {}
        
        # Calculate averages for key metrics
        key_metrics = [
            'deep_otm_oi', 'deep_otm_vol_oi', 'put_vol_oi_ratio',
            'top_5_oi_pct', 'max_strike_oi_pct', 'long_dated_oi_pct',
            'pc_ratio_oi', 'otm_5_10_oi_pct', 'otm_10_20_oi_pct',
            'deep_otm_oi_pct'
        ]
        
        for metric in key_metrics:
            if metric in df.columns:
                summary[f'{metric}_avg'] = df[metric].mean()
                summary[f'{metric}_std'] = df[metric].std()
                summary[f'{metric}_max'] = df[metric].max()
                summary[f'{metric}_min'] = df[metric].min()
        
        # Calculate hedging activity score
        if 'deep_otm_vol_oi' in df.columns and 'otm_5_10_vol_oi' in df.columns:
            # Low V/OI ratios indicate hedging activity
            hedging_score = 0
            if df['deep_otm_vol_oi'].mean() < 0.3:
                hedging_score += 25
            if df['otm_5_10_vol_oi'].mean() < 0.5:
                hedging_score += 25
            if df['otm_10_20_vol_oi'].mean() < 0.5:
                hedging_score += 25
            if df['long_dated_oi_pct'].mean() > 0.3:
                hedging_score += 25
            
            summary['hedging_activity_score'] = hedging_score
        
        return summary
    
    def match_current_patterns(self, current_metrics: dict, historical_analysis: dict) -> dict:
        """Match current hedging patterns with historical data"""
        
        print("\n🎯 PATTERN MATCHING ANALYSIS")
        print("=" * 40)
        
        matches = {}
        
        for period_name, period_data in historical_analysis.items():
            summary = period_data['summary']
            matches[period_name] = {
                'similarity_score': 0,
                'matches': [],
                'differences': [],
                'risk_level': 'UNKNOWN'
            }
            
            # Compare key metrics
            comparisons = [
                ('deep_otm_vol_oi', 'Deep OTM V/OI Ratio'),
                ('otm_5_10_vol_oi', '5-10% OTM V/OI Ratio'),
                ('otm_10_20_vol_oi', '10-20% OTM V/OI Ratio'),
                ('put_vol_oi_ratio', 'Overall Put V/OI Ratio'),
                ('long_dated_oi_pct', 'Long-dated Options %'),
                ('pc_ratio_oi', 'Put/Call Ratio'),
                ('max_strike_oi_pct', 'Max Strike Concentration')
            ]
            
            similarity_score = 0
            total_comparisons = 0
            
            for metric_key, display_name in comparisons:
                if metric_key in current_metrics and f'{metric_key}_avg' in summary:
                    current_val = current_metrics[metric_key]
                    historical_avg = summary[f'{metric_key}_avg']
                    historical_std = summary.get(f'{metric_key}_std', 0.1)
                    
                    # Calculate similarity (within 1 standard deviation = good match)
                    if historical_std > 0:
                        z_score = abs(current_val - historical_avg) / historical_std
                        similarity = max(0, 1 - z_score)  # Higher similarity for lower z-scores
                    else:
                        similarity = 1 if abs(current_val - historical_avg) < 0.1 else 0
                    
                    similarity_score += similarity
                    total_comparisons += 1
                    
                    if similarity > 0.7:
                        matches[period_name]['matches'].append(f"{display_name}: Similar (z-score: {z_score:.2f})")
                    elif similarity < 0.3:
                        matches[period_name]['differences'].append(f"{display_name}: Different (z-score: {z_score:.2f})")
            
            if total_comparisons > 0:
                matches[period_name]['similarity_score'] = similarity_score / total_comparisons
            
            # Determine risk level based on historical context
            if period_name in ['COVID_Crash_2020', '2022_Bear_Market']:
                if matches[period_name]['similarity_score'] > 0.6:
                    matches[period_name]['risk_level'] = 'HIGH'
                elif matches[period_name]['similarity_score'] > 0.4:
                    matches[period_name]['risk_level'] = 'MEDIUM'
                else:
                    matches[period_name]['risk_level'] = 'LOW'
            else:
                matches[period_name]['risk_level'] = 'NORMAL'
        
        return matches
    
    def generate_pattern_report(self, current_metrics: dict, matches: dict, historical_analysis: dict) -> str:
        """Generate comprehensive pattern matching report"""
        
        report = []
        report.append("=" * 70)
        report.append("SPY HEDGING PATTERN MATCHING REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current metrics summary
        report.append("🔍 CURRENT HEDGING METRICS:")
        report.append("-" * 30)
        report.append(f"Deep OTM V/OI Ratio: {current_metrics.get('deep_otm_vol_oi', 0):.3f}")
        report.append(f"5-10% OTM V/OI Ratio: {current_metrics.get('otm_5_10_vol_oi', 0):.3f}")
        report.append(f"10-20% OTM V/OI Ratio: {current_metrics.get('otm_10_20_vol_oi', 0):.3f}")
        report.append(f"Overall Put V/OI Ratio: {current_metrics.get('put_vol_oi_ratio', 0):.3f}")
        report.append(f"Long-dated Options %: {current_metrics.get('long_dated_oi_pct', 0):.1%}")
        report.append(f"Put/Call Ratio: {current_metrics.get('pc_ratio_oi', 0):.2f}")
        report.append("")
        
        # Pattern matches
        report.append("🎯 HISTORICAL PATTERN MATCHES:")
        report.append("-" * 35)
        
        # Sort by similarity score
        sorted_matches = sorted(matches.items(), key=lambda x: x[1]['similarity_score'], reverse=True)
        
        for period_name, match_data in sorted_matches:
            similarity = match_data['similarity_score']
            risk_level = match_data['risk_level']
            
            report.append(f"\n📊 {period_name.replace('_', ' ').title()}:")
            report.append(f"   Similarity Score: {similarity:.2f}/1.0")
            report.append(f"   Risk Level: {risk_level}")
            
            if match_data['matches']:
                report.append("   ✅ Similar Patterns:")
                for match in match_data['matches'][:3]:  # Show top 3 matches
                    report.append(f"      • {match}")
            
            if match_data['differences']:
                report.append("   ⚠️  Different Patterns:")
                for diff in match_data['differences'][:3]:  # Show top 3 differences
                    report.append(f"      • {diff}")
        
        # Risk assessment
        report.append("\n⚠️  RISK ASSESSMENT:")
        report.append("-" * 20)
        
        high_risk_matches = [name for name, data in matches.items() if data['risk_level'] == 'HIGH']
        medium_risk_matches = [name for name, data in matches.items() if data['risk_level'] == 'MEDIUM']
        
        if high_risk_matches:
            report.append(f"🔴 HIGH RISK: Similar to {', '.join(high_risk_matches)}")
            report.append("   • Monitor for potential market weakness")
            report.append("   • Consider defensive positioning")
        elif medium_risk_matches:
            report.append(f"🟡 MEDIUM RISK: Similar to {', '.join(medium_risk_matches)}")
            report.append("   • Watch for pattern acceleration")
            report.append("   • Prepare for potential volatility")
        else:
            report.append("✅ LOW RISK: No significant matches to high-risk periods")
            report.append("   • Current patterns appear normal")
            report.append("   • Continue regular monitoring")
        
        # Historical context
        report.append("\n📚 HISTORICAL CONTEXT:")
        report.append("-" * 25)
        
        for period_name, period_data in historical_analysis.items():
            description = period_data['description']
            summary = period_data['summary']
            hedging_score = summary.get('hedging_activity_score', 0)
            
            report.append(f"\n{period_name.replace('_', ' ').title()}:")
            report.append(f"   {description}")
            report.append(f"   Historical Hedging Score: {hedging_score}/100")
            
            if 'deep_otm_vol_oi_avg' in summary:
                report.append(f"   Avg Deep OTM V/OI: {summary['deep_otm_vol_oi_avg']:.3f}")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    
    # Initialize pattern matcher
    matcher = HedgingPatternMatcher()
    
    print("🔍 SPY HEDGING PATTERN MATCHING ANALYSIS")
    print("=" * 50)
    
    # Load current metrics (September 30, 2025)
    print("Loading current hedging metrics...")
    current_df = matcher.load_daily_data('2025-09-30')
    
    if current_df.empty:
        print("❌ No current data found")
        return
    
    spy_price = current_df['underlying_price'].iloc[0]
    current_metrics = matcher.calculate_hedging_metrics(current_df, spy_price)
    
    print(f"✅ Current SPY Price: ${spy_price:.2f}")
    print(f"✅ Current metrics calculated")
    
    # Analyze historical periods
    print("\nAnalyzing historical periods...")
    historical_analysis = matcher.analyze_historical_periods()
    
    # Match patterns
    print("\nMatching current patterns with historical data...")
    matches = matcher.match_current_patterns(current_metrics, historical_analysis)
    
    # Generate report
    report = matcher.generate_pattern_report(current_metrics, matches, historical_analysis)
    print("\n" + report)
    
    # Save results
    with open('hedging_pattern_matches.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Pattern matching report saved to 'hedging_pattern_matches.txt'")


if __name__ == "__main__":
    main()
