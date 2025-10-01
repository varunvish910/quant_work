"""
SPY July/August 2025 vs September 2025 Comparison
================================================

This script compares hedging patterns between July/August 2025 and September 2025
to identify key differences that might explain different market behavior.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class JulyAugustComparison:
    """
    Compares hedging patterns between July/August and September 2025
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
        
        return metrics
    
    def analyze_july_august_period(self) -> pd.DataFrame:
        """Analyze July/August 2025 hedging patterns"""
        print("📊 ANALYZING JULY/AUGUST 2025 PERIOD")
        print("=" * 45)
        
        start_date = pd.to_datetime('2025-07-01')
        end_date = pd.to_datetime('2025-08-31')
        
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
            print("❌ No data found for July/August 2025")
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
    
    def compare_periods(self, july_aug_df: pd.DataFrame, sept_df: pd.DataFrame) -> dict:
        """Compare hedging patterns between periods"""
        
        if july_aug_df.empty or sept_df.empty:
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
            'deep_otm_oi_pct'
        ]
        
        for metric in key_metrics:
            if metric in july_aug_df.columns and metric in sept_df.columns:
                july_aug_avg = july_aug_df[metric].mean()
                sept_avg = sept_df[metric].mean()
                
                # Calculate percentage change
                if july_aug_avg != 0:
                    pct_change = ((sept_avg - july_aug_avg) / july_aug_avg) * 100
                else:
                    pct_change = 0
                
                comparison[metric] = {
                    'july_aug_avg': july_aug_avg,
                    'sept_avg': sept_avg,
                    'change': sept_avg - july_aug_avg,
                    'pct_change': pct_change
                }
        
        return comparison
    
    def generate_comparison_report(self, july_aug_df: pd.DataFrame, sept_df: pd.DataFrame, comparison: dict) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("=" * 80)
        report.append("SPY HEDGING PATTERNS: JULY/AUGUST 2025 vs SEPTEMBER 2025")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Period summaries
        report.append("📊 PERIOD SUMMARIES:")
        report.append("-" * 20)
        report.append(f"July/August 2025: {len(july_aug_df)} trading days")
        report.append(f"September 2025: {len(sept_df)} trading days")
        
        if not july_aug_df.empty:
            july_aug_spy_avg = july_aug_df['spy_price'].mean()
            july_aug_spy_range = f"{july_aug_df['spy_price'].min():.2f} - {july_aug_df['spy_price'].max():.2f}"
            report.append(f"July/August SPY: ${july_aug_spy_avg:.2f} avg (${july_aug_spy_range})")
        
        if not sept_df.empty:
            sept_spy_avg = sept_df['spy_price'].mean()
            sept_spy_range = f"{sept_df['spy_price'].min():.2f} - {sept_df['spy_price'].max():.2f}"
            report.append(f"September SPY: ${sept_spy_avg:.2f} avg (${sept_spy_range})")
        
        report.append("")
        
        # Key differences
        report.append("🔍 KEY DIFFERENCES:")
        report.append("-" * 20)
        
        # Deep OTM analysis
        if 'deep_otm_oi' in comparison:
            deep_otm_change = comparison['deep_otm_oi']['pct_change']
            if abs(deep_otm_change) > 10:
                direction = "INCREASED" if deep_otm_change > 0 else "DECREASED"
                report.append(f"• Deep OTM Put OI: {direction} by {abs(deep_otm_change):.1f}%")
        
        if 'deep_otm_vol_oi' in comparison:
            vol_oi_change = comparison['deep_otm_vol_oi']['pct_change']
            if abs(vol_oi_change) > 20:
                direction = "INCREASED" if vol_oi_change > 0 else "DECREASED"
                report.append(f"• Deep OTM V/OI Ratio: {direction} by {abs(vol_oi_change):.1f}%")
        
        # Hedging activity
        if 'hedging_activity' in comparison:
            july_aug_hedging = comparison['hedging_activity']['july_aug_avg']
            sept_hedging = comparison['hedging_activity']['sept_avg']
            if abs(sept_hedging - july_aug_hedging) > 0.2:
                if sept_hedging > july_aug_hedging:
                    report.append("• Hedging Activity: INCREASED significantly")
                else:
                    report.append("• Hedging Activity: DECREASED significantly")
        
        # Institutional hedging
        if 'institutional_hedging' in comparison:
            july_aug_inst = comparison['institutional_hedging']['july_aug_avg']
            sept_inst = comparison['institutional_hedging']['sept_avg']
            if abs(sept_inst - july_aug_inst) > 0.2:
                if sept_inst > july_aug_inst:
                    report.append("• Institutional Hedging: INCREASED significantly")
                else:
                    report.append("• Institutional Hedging: DECREASED significantly")
        
        # Put/Call ratio
        if 'pc_ratio_oi' in comparison:
            pc_change = comparison['pc_ratio_oi']['pct_change']
            if abs(pc_change) > 10:
                direction = "INCREASED" if pc_change > 0 else "DECREASED"
                report.append(f"• Put/Call Ratio: {direction} by {abs(pc_change):.1f}%")
        
        # Long-dated options
        if 'long_dated_oi_pct' in comparison:
            long_dated_change = comparison['long_dated_oi_pct']['pct_change']
            if abs(long_dated_change) > 10:
                direction = "INCREASED" if long_dated_change > 0 else "DECREASED"
                report.append(f"• Long-dated Options %: {direction} by {abs(long_dated_change):.1f}%")
        
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
            ('deep_otm_oi_pct', 'Deep OTM %')
        ]
        
        for metric_key, display_name in metrics_display:
            if metric_key in comparison:
                data = comparison[metric_key]
                july_aug_val = data['july_aug_avg']
                sept_val = data['sept_avg']
                pct_change = data['pct_change']
                
                report.append(f"\n{display_name}:")
                report.append(f"  July/August: {july_aug_val:.3f}")
                report.append(f"  September:   {sept_val:.3f}")
                report.append(f"  Change:      {pct_change:+.1f}%")
        
        report.append("")
        
        # What this means
        report.append("💡 WHAT THIS MEANS:")
        report.append("-" * 18)
        
        # Analyze the changes
        significant_changes = []
        
        for metric_key, data in comparison.items():
            if abs(data['pct_change']) > 20:  # Significant change threshold
                significant_changes.append((metric_key, data['pct_change']))
        
        if significant_changes:
            report.append("Significant changes detected:")
            for metric, change in significant_changes:
                direction = "increased" if change > 0 else "decreased"
                report.append(f"• {metric}: {direction} by {abs(change):.1f}%")
        else:
            report.append("No dramatic changes detected between periods")
        
        # Risk assessment
        report.append("\n⚠️  RISK ASSESSMENT:")
        report.append("-" * 20)
        
        # Check for concerning patterns
        concerns = []
        
        if 'deep_otm_oi' in comparison and comparison['deep_otm_oi']['pct_change'] > 20:
            concerns.append("Elevated deep OTM put activity")
        
        if 'hedging_activity' in comparison and comparison['hedging_activity']['sept_avg'] > 0.5:
            concerns.append("High hedging activity detected")
        
        if 'institutional_hedging' in comparison and comparison['institutional_hedging']['sept_avg'] > 0.5:
            concerns.append("Strong institutional hedging present")
        
        if 'pc_ratio_oi' in comparison and comparison['pc_ratio_oi']['sept_avg'] > 1.3:
            concerns.append("Elevated Put/Call ratio")
        
        if concerns:
            report.append("Current concerns in September:")
            for concern in concerns:
                report.append(f"• {concern}")
        else:
            report.append("No major concerns identified")
        
        # Conclusion
        report.append("\n🎯 CONCLUSION:")
        report.append("-" * 15)
        
        if len(significant_changes) > 3:
            report.append("September shows SIGNIFICANTLY different hedging patterns")
            report.append("compared to July/August, suggesting a shift in market sentiment")
        elif len(significant_changes) > 1:
            report.append("September shows MODERATE changes in hedging patterns")
            report.append("compared to July/August, indicating some shift in sentiment")
        else:
            report.append("September shows MINIMAL changes in hedging patterns")
            report.append("compared to July/August, suggesting similar market conditions")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    
    # Initialize comparator
    comparator = JulyAugustComparison()
    
    print("🔍 SPY HEDGING PATTERNS: JULY/AUGUST vs SEPTEMBER 2025")
    print("=" * 60)
    
    # Analyze July/August period
    july_aug_df = comparator.analyze_july_august_period()
    
    # Analyze September period
    sept_df = comparator.analyze_september_period()
    
    if july_aug_df.empty or sept_df.empty:
        print("❌ Insufficient data for comparison")
        return
    
    # Compare periods
    comparison = comparator.compare_periods(july_aug_df, sept_df)
    
    # Generate report
    report = comparator.generate_comparison_report(july_aug_df, sept_df, comparison)
    print("\n" + report)
    
    # Save results
    with open('july_august_vs_september_comparison.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Comparison report saved to 'july_august_vs_september_comparison.txt'")


if __name__ == "__main__":
    main()
