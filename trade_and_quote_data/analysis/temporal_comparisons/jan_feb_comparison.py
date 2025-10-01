"""
SPY January/February 2025 vs September 2025 Comparison
=====================================================

This script compares hedging patterns between January/February 2025 and September 2025
to identify similarities and differences that might explain market behavior.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class JanFebComparison:
    """
    Compares hedging patterns between January/February and September 2025
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
    
    def calculate_hedging_metrics(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate comprehensive hedging metrics"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        metrics = {}
        
        # 1. Deep OTM put analysis
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.90]
        if not deep_otm_puts.empty:
            metrics['deep_otm_oi'] = deep_otm_puts['oi_proxy'].sum()
            metrics['deep_otm_volume'] = deep_otm_puts['volume'].sum()
            metrics['deep_otm_vol_oi'] = deep_otm_puts['volume'].sum() / (deep_otm_puts['oi_proxy'].sum() + 1e-6)
            metrics['deep_otm_contracts'] = len(deep_otm_puts)
        
        # 2. Strike range concentrations
        total_put_oi = puts['oi_proxy'].sum()
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
                range_vol = range_puts['volume'].sum()
                range_vol_oi = range_vol / (range_oi + 1e-6)
                
                metrics[f'{range_name}_oi'] = range_oi
                metrics[f'{range_name}_oi_pct'] = range_oi / total_put_oi
                metrics[f'{range_name}_vol_oi'] = range_vol_oi
                metrics[f'{range_name}_contracts'] = len(range_puts)
        
        # 3. Put/Call analysis
        calls = df[df['option_type'] == 'C']
        if not calls.empty:
            metrics['pc_ratio_oi'] = total_put_oi / calls['oi_proxy'].sum()
            metrics['pc_ratio_volume'] = puts['volume'].sum() / calls['volume'].sum()
        else:
            metrics['pc_ratio_oi'] = 1.0
            metrics['pc_ratio_volume'] = 1.0
        
        # 4. Volume/OI analysis
        metrics['put_vol_oi_ratio'] = puts['volume'].sum() / (total_put_oi + 1e-6)
        
        # 5. Long-dated options
        if 'dte' in puts.columns:
            long_dated_puts = puts[puts['dte'] > 60]
            if not long_dated_puts.empty:
                metrics['long_dated_oi_pct'] = long_dated_puts['oi_proxy'].sum() / total_put_oi
                metrics['long_dated_contracts'] = len(long_dated_puts)
            else:
                metrics['long_dated_oi_pct'] = 0
                metrics['long_dated_contracts'] = 0
        
        # 6. Strike concentration
        top_5_oi = puts.nlargest(5, 'oi_proxy')['oi_proxy'].sum()
        max_single_oi = puts['oi_proxy'].max()
        metrics['top_5_oi_pct'] = top_5_oi / total_put_oi
        metrics['max_strike_oi_pct'] = max_single_oi / total_put_oi
        
        # 7. Hedging activity indicators
        metrics['hedging_activity'] = 1 if metrics['put_vol_oi_ratio'] < 0.5 else 0
        metrics['deep_otm_hedging'] = 1 if metrics.get('deep_otm_vol_oi', 1) < 0.3 else 0
        metrics['institutional_hedging'] = 1 if metrics.get('long_dated_oi_pct', 0) > 0.3 else 0
        
        # 8. Support level analysis
        support_levels = [spy_price * 0.95, spy_price * 0.90, spy_price * 0.85, spy_price * 0.80]
        for i, level in enumerate(support_levels):
            level_puts = puts[abs(puts['strike'] - level) <= 5]
            if not level_puts.empty:
                metrics[f'support_{i+1}_oi'] = level_puts['oi_proxy'].sum()
                metrics[f'support_{i+1}_vol_oi'] = level_puts['volume'].sum() / (level_puts['oi_proxy'].sum() + 1e-6)
        
        return metrics
    
    def analyze_jan_feb_period(self) -> pd.DataFrame:
        """Analyze January/February 2025 hedging patterns"""
        print("📊 ANALYZING JANUARY/FEBRUARY 2025 PERIOD")
        print("=" * 50)
        
        start_date = pd.to_datetime('2025-01-01')
        end_date = pd.to_datetime('2025-02-28')
        
        period_data = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    metrics = self.calculate_hedging_metrics(df, spy_price)
                    
                    if metrics:
                        metrics['date'] = current_date
                        metrics['spy_price'] = spy_price
                        period_data.append(metrics)
            
            current_date += timedelta(days=1)
        
        if period_data:
            df_period = pd.DataFrame(period_data)
            print(f"✅ Analyzed {len(df_period)} trading days")
            return df_period
        else:
            print("❌ No data found for January/February 2025")
            return pd.DataFrame()
    
    def analyze_september_period(self) -> pd.DataFrame:
        """Analyze September 2025 hedging patterns"""
        print("\n📊 ANALYZING SEPTEMBER 2025 PERIOD")
        print("=" * 40)
        
        start_date = pd.to_datetime('2025-09-01')
        end_date = pd.to_datetime('2025-09-30')
        
        period_data = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    metrics = self.calculate_hedging_metrics(df, spy_price)
                    
                    if metrics:
                        metrics['date'] = current_date
                        metrics['spy_price'] = spy_price
                        period_data.append(metrics)
            
            current_date += timedelta(days=1)
        
        if period_data:
            df_period = pd.DataFrame(period_data)
            print(f"✅ Analyzed {len(df_period)} trading days")
            return df_period
        else:
            print("❌ No data found for September 2025")
            return pd.DataFrame()
    
    def compare_periods(self, jan_feb_df: pd.DataFrame, sept_df: pd.DataFrame) -> dict:
        """Compare hedging patterns between periods"""
        
        if jan_feb_df.empty or sept_df.empty:
            return {}
        
        print("\n🔍 COMPARING HEDGING PATTERNS")
        print("=" * 35)
        
        comparison = {}
        
        # Key metrics to compare
        key_metrics = [
            'deep_otm_oi', 'deep_otm_vol_oi', 'deep_otm_hedging',
            'put_vol_oi_ratio', 'hedging_activity', 'institutional_hedging',
            'pc_ratio_oi', 'long_dated_oi_pct', 'top_5_oi_pct',
            'max_strike_oi_pct', 'otm_5_10_oi_pct', 'otm_10_20_oi_pct',
            'deep_otm_oi_pct', 'near_money_oi_pct'
        ]
        
        for metric in key_metrics:
            if metric in jan_feb_df.columns and metric in sept_df.columns:
                jan_feb_avg = jan_feb_df[metric].mean()
                sept_avg = sept_df[metric].mean()
                
                # Calculate percentage change
                if jan_feb_avg != 0:
                    pct_change = ((sept_avg - jan_feb_avg) / jan_feb_avg) * 100
                else:
                    pct_change = 0
                
                comparison[metric] = {
                    'jan_feb_avg': jan_feb_avg,
                    'sept_avg': sept_avg,
                    'change': sept_avg - jan_feb_avg,
                    'pct_change': pct_change
                }
        
        return comparison
    
    def analyze_trends(self, jan_feb_df: pd.DataFrame, sept_df: pd.DataFrame) -> dict:
        """Analyze trends within each period"""
        
        trends = {}
        
        # January/February trends
        if not jan_feb_df.empty:
            jan_feb_trends = {}
            
            # Calculate trend for key metrics
            trend_metrics = ['deep_otm_oi', 'put_vol_oi_ratio', 'pc_ratio_oi', 'long_dated_oi_pct']
            
            for metric in trend_metrics:
                if metric in jan_feb_df.columns:
                    # Calculate linear trend (slope)
                    x = np.arange(len(jan_feb_df))
                    y = jan_feb_df[metric].values
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        jan_feb_trends[metric] = slope
            
            trends['jan_feb'] = jan_feb_trends
        
        # September trends
        if not sept_df.empty:
            sept_trends = {}
            
            for metric in trend_metrics:
                if metric in sept_df.columns:
                    x = np.arange(len(sept_df))
                    y = sept_df[metric].values
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        sept_trends[metric] = slope
            
            trends['september'] = sept_trends
        
        return trends
    
    def generate_comparison_report(self, jan_feb_df: pd.DataFrame, sept_df: pd.DataFrame, 
                                 comparison: dict, trends: dict) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("=" * 80)
        report.append("SPY HEDGING PATTERNS: JANUARY/FEBRUARY 2025 vs SEPTEMBER 2025")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Period summaries
        report.append("📊 PERIOD SUMMARIES:")
        report.append("-" * 20)
        report.append(f"January/February 2025: {len(jan_feb_df)} trading days")
        report.append(f"September 2025: {len(sept_df)} trading days")
        
        if not jan_feb_df.empty:
            jan_feb_spy_avg = jan_feb_df['spy_price'].mean()
            jan_feb_spy_range = f"{jan_feb_df['spy_price'].min():.2f} - {jan_feb_df['spy_price'].max():.2f}"
            report.append(f"Jan/Feb SPY: ${jan_feb_spy_avg:.2f} avg (${jan_feb_spy_range})")
        
        if not sept_df.empty:
            sept_spy_avg = sept_df['spy_price'].mean()
            sept_spy_range = f"{sept_df['spy_price'].min():.2f} - {sept_df['spy_price'].max():.2f}"
            report.append(f"September SPY: ${sept_spy_avg:.2f} avg (${sept_spy_range})")
        
        report.append("")
        
        # Key similarities and differences
        report.append("🔍 KEY SIMILARITIES & DIFFERENCES:")
        report.append("-" * 40)
        
        # Analyze similarities (changes < 20%)
        similarities = []
        differences = []
        
        for metric, data in comparison.items():
            if abs(data['pct_change']) < 20:
                similarities.append((metric, data['pct_change']))
            else:
                differences.append((metric, data['pct_change']))
        
        if similarities:
            report.append("✅ SIMILAR PATTERNS:")
            for metric, change in similarities[:5]:  # Top 5 similarities
                direction = "slightly higher" if change > 0 else "slightly lower"
                report.append(f"• {metric.replace('_', ' ').title()}: {direction} ({change:+.1f}%)")
        
        if differences:
            report.append("\n⚠️  DIFFERENT PATTERNS:")
            for metric, change in differences[:5]:  # Top 5 differences
                direction = "significantly higher" if change > 0 else "significantly lower"
                report.append(f"• {metric.replace('_', ' ').title()}: {direction} ({change:+.1f}%)")
        
        report.append("")
        
        # Detailed metrics comparison
        report.append("📈 DETAILED METRICS COMPARISON:")
        report.append("-" * 35)
        
        metrics_display = [
            ('deep_otm_oi', 'Deep OTM Put OI'),
            ('deep_otm_vol_oi', 'Deep OTM V/OI Ratio'),
            ('put_vol_oi_ratio', 'Overall Put V/OI Ratio'),
            ('pc_ratio_oi', 'Put/Call Ratio (OI)'),
            ('long_dated_oi_pct', 'Long-dated Options %'),
            ('top_5_oi_pct', 'Top 5 Strikes %'),
            ('max_strike_oi_pct', 'Max Single Strike %'),
            ('otm_5_10_oi_pct', '5-10% OTM %'),
            ('otm_10_20_oi_pct', '10-20% OTM %'),
            ('deep_otm_oi_pct', 'Deep OTM %'),
            ('near_money_oi_pct', 'Near-the-Money %')
        ]
        
        for metric_key, display_name in metrics_display:
            if metric_key in comparison:
                data = comparison[metric_key]
                jan_feb_val = data['jan_feb_avg']
                sept_val = data['sept_avg']
                pct_change = data['pct_change']
                
                report.append(f"\n{display_name}:")
                report.append(f"  Jan/Feb:  {jan_feb_val:.3f}")
                report.append(f"  September: {sept_val:.3f}")
                report.append(f"  Change:    {pct_change:+.1f}%")
        
        report.append("")
        
        # Trend analysis
        if 'jan_feb' in trends and 'september' in trends:
            report.append("📈 TREND ANALYSIS:")
            report.append("-" * 18)
            
            trend_metrics = ['deep_otm_oi', 'put_vol_oi_ratio', 'pc_ratio_oi', 'long_dated_oi_pct']
            
            for metric in trend_metrics:
                if metric in trends['jan_feb'] and metric in trends['september']:
                    jan_feb_trend = trends['jan_feb'][metric]
                    sept_trend = trends['september'][metric]
                    
                    jan_feb_direction = "rising" if jan_feb_trend > 0 else "falling"
                    sept_direction = "rising" if sept_trend > 0 else "falling"
                    
                    report.append(f"\n{metric.replace('_', ' ').title()}:")
                    report.append(f"  Jan/Feb trend: {jan_feb_direction}")
                    report.append(f"  September trend: {sept_direction}")
        
        report.append("")
        
        # Market context analysis
        report.append("🎯 MARKET CONTEXT ANALYSIS:")
        report.append("-" * 30)
        
        # Check for similar market conditions
        if not jan_feb_df.empty and not sept_df.empty:
            jan_feb_spy_vol = jan_feb_df['spy_price'].std()
            sept_spy_vol = sept_df['spy_price'].std()
            
            if abs(sept_spy_vol - jan_feb_spy_vol) / jan_feb_spy_vol < 0.2:
                report.append("• Similar volatility levels between periods")
            else:
                report.append("• Different volatility levels between periods")
        
        # Hedging activity comparison
        if 'hedging_activity' in comparison:
            jan_feb_hedging = comparison['hedging_activity']['jan_feb_avg']
            sept_hedging = comparison['hedging_activity']['sept_avg']
            
            if abs(sept_hedging - jan_feb_hedging) < 0.2:
                report.append("• Similar hedging activity levels")
            else:
                if sept_hedging > jan_feb_hedging:
                    report.append("• September shows HIGHER hedging activity")
                else:
                    report.append("• September shows LOWER hedging activity")
        
        # Institutional hedging comparison
        if 'institutional_hedging' in comparison:
            jan_feb_inst = comparison['institutional_hedging']['jan_feb_avg']
            sept_inst = comparison['institutional_hedging']['sept_avg']
            
            if abs(sept_inst - jan_feb_inst) < 0.2:
                report.append("• Similar institutional hedging levels")
            else:
                if sept_inst > jan_feb_inst:
                    report.append("• September shows HIGHER institutional hedging")
                else:
                    report.append("• September shows LOWER institutional hedging")
        
        report.append("")
        
        # What this means for market outlook
        report.append("💡 MARKET OUTLOOK IMPLICATIONS:")
        report.append("-" * 35)
        
        # Count significant changes
        significant_changes = len([m for m in comparison.values() if abs(m['pct_change']) > 20])
        
        if significant_changes > 5:
            report.append("🔴 MAJOR DIFFERENCES: September shows substantially different")
            report.append("   hedging patterns compared to January/February, suggesting")
            report.append("   a significant shift in market sentiment and risk appetite.")
        elif significant_changes > 2:
            report.append("🟡 MODERATE DIFFERENCES: September shows some notable changes")
            report.append("   in hedging patterns, indicating evolving market conditions.")
        else:
            report.append("🟢 SIMILAR PATTERNS: September shows similar hedging patterns")
            report.append("   to January/February, suggesting consistent market behavior.")
        
        # Specific implications
        if 'deep_otm_hedging' in comparison:
            jan_feb_deep = comparison['deep_otm_hedging']['jan_feb_avg']
            sept_deep = comparison['deep_otm_hedging']['sept_avg']
            
            if sept_deep > jan_feb_deep + 0.2:
                report.append("\n• Elevated deep OTM hedging suggests increased downside protection")
            elif sept_deep < jan_feb_deep - 0.2:
                report.append("\n• Reduced deep OTM hedging suggests decreased downside concerns")
        
        if 'institutional_hedging' in comparison:
            jan_feb_inst = comparison['institutional_hedging']['jan_feb_avg']
            sept_inst = comparison['institutional_hedging']['sept_avg']
            
            if sept_inst > jan_feb_inst + 0.2:
                report.append("• Higher institutional hedging indicates defensive positioning")
            elif sept_inst < jan_feb_inst - 0.2:
                report.append("• Lower institutional hedging indicates reduced defensive stance")
        
        report.append("")
        
        # Conclusion
        report.append("🎯 CONCLUSION:")
        report.append("-" * 15)
        
        if significant_changes > 5:
            report.append("September 2025 represents a SIGNIFICANTLY DIFFERENT market")
            report.append("environment compared to January/February 2025, with notable")
            report.append("changes in hedging behavior suggesting evolving risk perceptions.")
        elif significant_changes > 2:
            report.append("September 2025 shows MODERATE DIFFERENCES from January/February")
            report.append("2025, indicating some evolution in market conditions and")
            report.append("hedging strategies.")
        else:
            report.append("September 2025 shows SIMILAR patterns to January/February 2025,")
            report.append("suggesting consistent market behavior and risk management")
            report.append("approaches across these periods.")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    
    # Initialize comparator
    comparator = JanFebComparison()
    
    print("🔍 SPY HEDGING PATTERNS: JANUARY/FEBRUARY vs SEPTEMBER 2025")
    print("=" * 65)
    
    # Analyze January/February period
    jan_feb_df = comparator.analyze_jan_feb_period()
    
    # Analyze September period
    sept_df = comparator.analyze_september_period()
    
    if jan_feb_df.empty or sept_df.empty:
        print("❌ Insufficient data for comparison")
        return
    
    # Compare periods
    comparison = comparator.compare_periods(jan_feb_df, sept_df)
    
    # Analyze trends
    trends = comparator.analyze_trends(jan_feb_df, sept_df)
    
    # Generate report
    report = comparator.generate_comparison_report(jan_feb_df, sept_df, comparison, trends)
    print("\n" + report)
    
    # Save results
    with open('jan_feb_vs_september_comparison.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Comparison report saved to 'jan_feb_vs_september_comparison.txt'")


if __name__ == "__main__":
    main()
