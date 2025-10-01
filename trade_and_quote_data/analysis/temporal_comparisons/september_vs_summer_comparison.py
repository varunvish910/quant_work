"""
September vs Summer 2025 Comparison
===================================

Compares September 2025 hedging patterns to summer months (June, July, August 2025)
to identify any significant changes in hedging behavior.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SeptemberVsSummerComparison:
    """
    Compares September 2025 to summer 2025 hedging patterns
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """Load options data for a specific date"""
        try:
            year = date[:4]
            month = date[5:7]
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
    
    def analyze_hedging_patterns(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze hedging patterns for a specific day"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return {}
        
        analysis = {}
        analysis['spy_price'] = spy_price
        
        # Basic metrics
        total_put_oi = puts['oi_proxy'].sum()
        total_call_oi = calls['oi_proxy'].sum() if not calls.empty else 0
        total_put_vol = puts['volume'].sum()
        
        analysis['total_put_oi'] = total_put_oi
        analysis['pc_ratio'] = total_put_oi / (total_call_oi + 1e-6)
        analysis['vol_oi_ratio'] = total_put_vol / (total_put_oi + 1e-6)
        
        # Institutional analysis
        if 'dte' in df.columns:
            institutional_puts = puts[puts['dte'] > 7]
            analysis['institutional_pct'] = institutional_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        else:
            analysis['institutional_pct'] = 0
        
        # Hedging intensity
        analysis['hedging_intensity'] = min(1.0, (analysis['pc_ratio'] - 1) * 2)
        
        # Defensive positioning
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.85]
        deep_otm_pct = deep_otm_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        analysis['defensive_positioning'] = min(1.0, deep_otm_pct * 3)
        
        # Strike concentration
        strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
        top_5_oi = strike_oi.head(5).sum()
        analysis['strike_concentration'] = top_5_oi / (total_put_oi + 1e-6)
        
        # Long-term positioning
        if 'dte' in df.columns:
            long_term_puts = puts[puts['dte'] > 60]
            analysis['long_term_pct'] = long_term_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        else:
            analysis['long_term_pct'] = 0
        
        return analysis
    
    def build_comparison_data(self) -> dict:
        """Build comparison data for September vs Summer 2025"""
        
        print("📊 SEPTEMBER vs SUMMER 2025 COMPARISON")
        print("Building comparison data...")
        print("=" * 50)
        
        # Define periods
        periods = {
            'June_2025': ('2025-06-01', '2025-06-30'),
            'July_2025': ('2025-07-01', '2025-07-31'),
            'August_2025': ('2025-08-01', '2025-08-31'),
            'September_2025': ('2025-09-01', '2025-09-30')
        }
        
        all_data = {}
        
        for period_name, (start_date, end_date) in periods.items():
            print(f"\n🔍 Analyzing {period_name}...")
            
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            period_data = []
            current_date = start
            
            while current_date <= end:
                if current_date.weekday() < 5:  # Skip weekends
                    date_str = current_date.strftime('%Y-%m-%d')
                    df = self.load_daily_data(date_str)
                    
                    if not df.empty and 'underlying_price' in df.columns:
                        spy_price = df['underlying_price'].iloc[0]
                        analysis = self.analyze_hedging_patterns(df, spy_price)
                        if analysis:
                            analysis['date'] = current_date
                            analysis['period'] = period_name
                            period_data.append(analysis)
                
                current_date += timedelta(days=1)
            
            if period_data:
                df_period = pd.DataFrame(period_data)
                all_data[period_name] = df_period
                print(f"   ✅ {period_name}: {len(df_period)} trading days")
            else:
                print(f"   ❌ {period_name}: No data available")
        
        return all_data
    
    def calculate_comparison_metrics(self, all_data: dict) -> dict:
        """Calculate comparison metrics between September and Summer"""
        
        if not all_data:
            return {}
        
        comparison = {}
        
        # Calculate monthly averages
        monthly_stats = {}
        for period_name, df in all_data.items():
            if not df.empty:
                monthly_stats[period_name] = {
                    'trading_days': len(df),
                    'avg_spy_price': df['spy_price'].mean(),
                    'avg_hedging_intensity': df['hedging_intensity'].mean(),
                    'avg_pc_ratio': df['pc_ratio'].mean(),
                    'avg_institutional_pct': df['institutional_pct'].mean(),
                    'avg_defensive_positioning': df['defensive_positioning'].mean(),
                    'avg_vol_oi_ratio': df['vol_oi_ratio'].mean(),
                    'avg_total_put_oi': df['total_put_oi'].mean(),
                    'avg_strike_concentration': df['strike_concentration'].mean(),
                    'avg_long_term_pct': df['long_term_pct'].mean(),
                    'max_hedging_intensity': df['hedging_intensity'].max(),
                    'max_pc_ratio': df['pc_ratio'].max(),
                    'max_defensive_positioning': df['defensive_positioning'].max()
                }
        
        comparison['monthly_stats'] = monthly_stats
        
        # Calculate summer averages (June, July, August combined)
        summer_data = []
        for month in ['June_2025', 'July_2025', 'August_2025']:
            if month in all_data and not all_data[month].empty:
                summer_data.append(all_data[month])
        
        if summer_data:
            df_summer = pd.concat(summer_data, ignore_index=True)
            comparison['summer_stats'] = {
                'trading_days': len(df_summer),
                'avg_spy_price': df_summer['spy_price'].mean(),
                'avg_hedging_intensity': df_summer['hedging_intensity'].mean(),
                'avg_pc_ratio': df_summer['pc_ratio'].mean(),
                'avg_institutional_pct': df_summer['institutional_pct'].mean(),
                'avg_defensive_positioning': df_summer['defensive_positioning'].mean(),
                'avg_vol_oi_ratio': df_summer['vol_oi_ratio'].mean(),
                'avg_total_put_oi': df_summer['total_put_oi'].mean(),
                'avg_strike_concentration': df_summer['strike_concentration'].mean(),
                'avg_long_term_pct': df_summer['long_term_pct'].mean(),
                'max_hedging_intensity': df_summer['hedging_intensity'].max(),
                'max_pc_ratio': df_summer['pc_ratio'].max(),
                'max_defensive_positioning': df_summer['defensive_positioning'].max()
            }
        
        # Calculate September vs Summer changes
        if 'summer_stats' in comparison and 'September_2025' in monthly_stats:
            summer = comparison['summer_stats']
            september = monthly_stats['September_2025']
            
            changes = {}
            for metric in ['avg_hedging_intensity', 'avg_pc_ratio', 'avg_institutional_pct', 
                          'avg_defensive_positioning', 'avg_vol_oi_ratio', 'avg_total_put_oi',
                          'avg_strike_concentration', 'avg_long_term_pct']:
                if metric in summer and metric in september:
                    summer_val = summer[metric]
                    september_val = september[metric]
                    
                    if summer_val != 0:
                        change_pct = (september_val - summer_val) / summer_val * 100
                    else:
                        change_pct = 0
                    
                    changes[metric] = {
                        'summer': summer_val,
                        'september': september_val,
                        'change': september_val - summer_val,
                        'change_pct': change_pct
                    }
            
            comparison['september_vs_summer_changes'] = changes
        
        return comparison
    
    def generate_comparison_report(self, all_data: dict, comparison: dict) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("📊 SEPTEMBER 2025 vs SUMMER 2025 COMPARISON")
        report.append("=" * 60)
        report.append("")
        
        if not all_data or not comparison:
            report.append("❌ No comparison data available")
            return "\n".join(report)
        
        # Overall summary
        report.append("📈 OVERALL SUMMARY:")
        report.append("-" * 18)
        
        if 'monthly_stats' in comparison:
            for period, stats in comparison['monthly_stats'].items():
                report.append(f"• {period.replace('_', ' ').title()}: {stats['trading_days']} trading days")
        
        if 'summer_stats' in comparison:
            summer_days = comparison['summer_stats']['trading_days']
            report.append(f"• Summer 2025 Total: {summer_days} trading days")
        
        report.append("")
        
        # Key metrics comparison
        if 'september_vs_summer_changes' in comparison:
            report.append("🔍 KEY METRICS COMPARISON:")
            report.append("-" * 25)
            report.append("September 2025 vs Summer 2025 Average")
            report.append("")
            
            changes = comparison['september_vs_summer_changes']
            
            # Hedging Intensity
            if 'avg_hedging_intensity' in changes:
                c = changes['avg_hedging_intensity']
                report.append(f"📊 Hedging Intensity:")
                report.append(f"  • Summer: {c['summer']:.3f}")
                report.append(f"  • September: {c['september']:.3f}")
                report.append(f"  • Change: {c['change']:+.3f} ({c['change_pct']:+.1f}%)")
                report.append("")
            
            # Put/Call Ratio
            if 'avg_pc_ratio' in changes:
                c = changes['avg_pc_ratio']
                report.append(f"📊 Put/Call Ratio:")
                report.append(f"  • Summer: {c['summer']:.3f}")
                report.append(f"  • September: {c['september']:.3f}")
                report.append(f"  • Change: {c['change']:+.3f} ({c['change_pct']:+.1f}%)")
                report.append("")
            
            # Institutional %
            if 'avg_institutional_pct' in changes:
                c = changes['avg_institutional_pct']
                report.append(f"📊 Institutional %:")
                report.append(f"  • Summer: {c['summer']:.1%}")
                report.append(f"  • September: {c['september']:.1%}")
                report.append(f"  • Change: {c['change']:+.1%} ({c['change_pct']:+.1f}%)")
                report.append("")
            
            # Defensive Positioning
            if 'avg_defensive_positioning' in changes:
                c = changes['avg_defensive_positioning']
                report.append(f"📊 Defensive Positioning:")
                report.append(f"  • Summer: {c['summer']:.3f}")
                report.append(f"  • September: {c['september']:.3f}")
                report.append(f"  • Change: {c['change']:+.3f} ({c['change_pct']:+.1f}%)")
                report.append("")
            
            # Volume/OI Ratio
            if 'avg_vol_oi_ratio' in changes:
                c = changes['avg_vol_oi_ratio']
                report.append(f"📊 Volume/OI Ratio:")
                report.append(f"  • Summer: {c['summer']:.3f}")
                report.append(f"  • September: {c['september']:.3f}")
                report.append(f"  • Change: {c['change']:+.3f} ({c['change_pct']:+.1f}%)")
                report.append("")
            
            # Total Put OI
            if 'avg_total_put_oi' in changes:
                c = changes['avg_total_put_oi']
                report.append(f"📊 Total Put OI:")
                report.append(f"  • Summer: {c['summer']:,.0f}")
                report.append(f"  • September: {c['september']:,.0f}")
                report.append(f"  • Change: {c['change']:+,.0f} ({c['change_pct']:+.1f}%)")
                report.append("")
        
        # Monthly breakdown
        if 'monthly_stats' in comparison:
            report.append("📅 MONTHLY BREAKDOWN:")
            report.append("-" * 20)
            
            for period, stats in comparison['monthly_stats'].items():
                report.append(f"\n{period.replace('_', ' ').title()}:")
                report.append(f"  • Hedging Intensity: {stats['avg_hedging_intensity']:.3f}")
                report.append(f"  • P/C Ratio: {stats['avg_pc_ratio']:.3f}")
                report.append(f"  • Institutional %: {stats['avg_institutional_pct']:.1%}")
                report.append(f"  • Defensive Positioning: {stats['avg_defensive_positioning']:.3f}")
                report.append(f"  • Volume/OI Ratio: {stats['avg_vol_oi_ratio']:.3f}")
                report.append(f"  • Total Put OI: {stats['avg_total_put_oi']:,.0f}")
        
        report.append("")
        
        # Key findings
        report.append("🎯 KEY FINDINGS:")
        report.append("-" * 15)
        
        if 'september_vs_summer_changes' in comparison:
            changes = comparison['september_vs_summer_changes']
            
            # Identify significant changes
            significant_changes = []
            
            for metric, change_data in changes.items():
                change_pct = change_data['change_pct']
                if abs(change_pct) > 10:  # 10%+ change
                    metric_name = metric.replace('avg_', '').replace('_', ' ').title()
                    significant_changes.append(f"{metric_name}: {change_pct:+.1f}%")
            
            if significant_changes:
                report.append("✅ SIGNIFICANT CHANGES DETECTED:")
                for change in significant_changes:
                    report.append(f"  • {change}")
            else:
                report.append("ℹ️  NO SIGNIFICANT CHANGES DETECTED")
                report.append("  • September patterns similar to summer")
                report.append("  • Consistent hedging behavior")
                report.append("  • No major shifts in risk positioning")
        
        report.append("")
        
        # Conclusion
        report.append("💡 CONCLUSION:")
        report.append("-" * 12)
        
        if 'september_vs_summer_changes' in comparison:
            changes = comparison['september_vs_summer_changes']
            
            # Check for increases in hedging activity
            hedging_increase = changes.get('avg_hedging_intensity', {}).get('change_pct', 0)
            pc_increase = changes.get('avg_pc_ratio', {}).get('change_pct', 0)
            defensive_increase = changes.get('avg_defensive_positioning', {}).get('change_pct', 0)
            
            if hedging_increase > 10 or pc_increase > 10 or defensive_increase > 10:
                report.append("⚠️  SEPTEMBER SHOWS INCREASED HEDGING ACTIVITY")
                report.append("• Options market became more defensive")
                report.append("• May indicate elevated risk concerns")
                report.append("• Monitor for continuation into October")
            elif hedging_increase < -10 or pc_increase < -10 or defensive_increase < -10:
                report.append("✅ SEPTEMBER SHOWS DECREASED HEDGING ACTIVITY")
                report.append("• Options market became less defensive")
                report.append("• May indicate reduced risk concerns")
                report.append("• More optimistic positioning")
            else:
                report.append("✅ SEPTEMBER PATTERNS SIMILAR TO SUMMER")
                report.append("• Consistent hedging behavior")
                report.append("• No major changes in risk positioning")
                report.append("• Normal institutional hedging patterns")
        
        return "\n".join(report)


def main():
    """Main September vs Summer comparison"""
    
    # Initialize analyzer
    analyzer = SeptemberVsSummerComparison()
    
    print("📊 SEPTEMBER 2025 vs SUMMER 2025 COMPARISON")
    print("How does September compare to June, July, August?")
    print("=" * 70)
    
    # Build comparison data
    all_data = analyzer.build_comparison_data()
    
    if not all_data:
        print("❌ No comparison data available")
        return
    
    # Calculate comparison metrics
    comparison = analyzer.calculate_comparison_metrics(all_data)
    
    # Generate report
    report = analyzer.generate_comparison_report(all_data, comparison)
    print(report)
    
    # Save analysis
    with open('september_vs_summer_comparison.txt', 'w') as f:
        f.write(report)
    
    # Save data
    import json
    comparison_data = {
        'monthly_stats': comparison.get('monthly_stats', {}),
        'summer_stats': comparison.get('summer_stats', {}),
        'september_vs_summer_changes': comparison.get('september_vs_summer_changes', {})
    }
    
    with open('september_vs_summer_data.json', 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to files")


if __name__ == "__main__":
    main()
