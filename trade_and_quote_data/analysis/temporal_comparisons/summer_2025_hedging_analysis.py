"""
Summer 2025 Hedging Analysis
============================

Analyzes hedging activity during June, July, and August 2025
to identify any rises in hedging patterns that might indicate
increased risk concerns.

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

class Summer2025HedgingAnalyzer:
    """
    Analyzes hedging patterns during summer 2025 months
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
    
    def build_summer_analysis(self) -> pd.DataFrame:
        """Build analysis for June, July, August 2025"""
        
        print("📊 SUMMER 2025 HEDGING ANALYSIS")
        print("Analyzing June, July, August 2025")
        print("=" * 50)
        
        # Define summer months
        months = {
            'June': '2025-06-01',
            'July': '2025-07-01', 
            'August': '2025-08-01'
        }
        
        all_data = []
        
        for month_name, start_date in months.items():
            print(f"\n🔍 Analyzing {month_name} 2025...")
            
            start = pd.to_datetime(start_date)
            if month_name == 'June':
                end = pd.to_datetime('2025-06-30')
            elif month_name == 'July':
                end = pd.to_datetime('2025-07-31')
            else:  # August
                end = pd.to_datetime('2025-08-31')
            
            month_data = []
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
                            analysis['month'] = month_name
                            month_data.append(analysis)
                
                current_date += timedelta(days=1)
            
            if month_data:
                df_month = pd.DataFrame(month_data)
                print(f"   ✅ {month_name}: {len(df_month)} trading days")
                all_data.append(df_month)
            else:
                print(f"   ❌ {month_name}: No data available")
        
        if all_data:
            df_summer = pd.concat(all_data, ignore_index=True)
            print(f"\n✅ Built summer analysis with {len(df_summer)} total trading days")
            return df_summer
        else:
            print("❌ No summer data available")
            return pd.DataFrame()
    
    def analyze_hedging_rises(self, df: pd.DataFrame) -> dict:
        """Analyze rises in hedging activity during summer 2025"""
        
        if df.empty:
            return {}
        
        analysis = {}
        
        # Calculate monthly averages
        monthly_stats = df.groupby('month').agg({
            'hedging_intensity': ['mean', 'std', 'max'],
            'pc_ratio': ['mean', 'std', 'max'],
            'institutional_pct': ['mean', 'std', 'max'],
            'defensive_positioning': ['mean', 'std', 'max'],
            'total_put_oi': ['mean', 'std', 'max'],
            'vol_oi_ratio': ['mean', 'std', 'max']
        }).round(3)
        
        analysis['monthly_stats'] = monthly_stats
        
        # Identify significant rises
        rises = {}
        
        for month in ['June', 'July', 'August']:
            month_data = df[df['month'] == month]
            if len(month_data) > 5:  # Need enough data points
                
                # Calculate trends (first half vs second half)
                mid_point = len(month_data) // 2
                first_half = month_data.iloc[:mid_point]
                second_half = month_data.iloc[mid_point:]
                
                month_rises = {}
                
                # Hedging intensity rise
                first_avg = first_half['hedging_intensity'].mean()
                second_avg = second_half['hedging_intensity'].mean()
                hedging_rise = second_avg - first_avg
                month_rises['hedging_intensity_rise'] = hedging_rise
                
                # PC ratio rise
                first_pc = first_half['pc_ratio'].mean()
                second_pc = second_half['pc_ratio'].mean()
                pc_rise = second_pc - first_pc
                month_rises['pc_ratio_rise'] = pc_rise
                
                # Institutional positioning rise
                first_inst = first_half['institutional_pct'].mean()
                second_inst = second_half['institutional_pct'].mean()
                inst_rise = second_inst - first_inst
                month_rises['institutional_rise'] = inst_rise
                
                # Defensive positioning rise
                first_def = first_half['defensive_positioning'].mean()
                second_def = second_half['defensive_positioning'].mean()
                def_rise = second_def - first_def
                month_rises['defensive_rise'] = def_rise
                
                # Total put OI rise
                first_oi = first_half['total_put_oi'].mean()
                second_oi = second_half['total_put_oi'].mean()
                oi_rise_pct = (second_oi - first_oi) / first_oi * 100
                month_rises['put_oi_rise_pct'] = oi_rise_pct
                
                rises[month] = month_rises
        
        analysis['monthly_rises'] = rises
        
        # Identify significant spikes (daily)
        spikes = {}
        
        for month in ['June', 'July', 'August']:
            month_data = df[df['month'] == month]
            if len(month_data) > 0:
                
                # Calculate thresholds (mean + 2*std)
                hedging_threshold = month_data['hedging_intensity'].mean() + 2 * month_data['hedging_intensity'].std()
                pc_threshold = month_data['pc_ratio'].mean() + 2 * month_data['pc_ratio'].std()
                inst_threshold = month_data['institutional_pct'].mean() + 2 * month_data['institutional_pct'].std()
                def_threshold = month_data['defensive_positioning'].mean() + 2 * month_data['defensive_positioning'].std()
                
                # Find spikes
                hedging_spikes = month_data[month_data['hedging_intensity'] > hedging_threshold]
                pc_spikes = month_data[month_data['pc_ratio'] > pc_threshold]
                inst_spikes = month_data[month_data['institutional_pct'] > inst_threshold]
                def_spikes = month_data[month_data['defensive_positioning'] > def_threshold]
                
                spikes[month] = {
                    'hedging_spikes': len(hedging_spikes),
                    'pc_spikes': len(pc_spikes),
                    'institutional_spikes': len(inst_spikes),
                    'defensive_spikes': len(def_spikes),
                    'hedging_spike_dates': hedging_spikes['date'].tolist(),
                    'pc_spike_dates': pc_spikes['date'].tolist(),
                    'institutional_spike_dates': inst_spikes['date'].tolist(),
                    'defensive_spike_dates': def_spikes['date'].tolist()
                }
        
        analysis['monthly_spikes'] = spikes
        
        return analysis
    
    def generate_summer_report(self, df: pd.DataFrame, analysis: dict) -> str:
        """Generate comprehensive summer 2025 hedging report"""
        
        report = []
        report.append("📊 SUMMER 2025 HEDGING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        if df.empty:
            report.append("❌ No summer 2025 data available")
            return "\n".join(report)
        
        # Overall summary
        total_days = len(df)
        june_days = len(df[df['month'] == 'June'])
        july_days = len(df[df['month'] == 'July'])
        august_days = len(df[df['month'] == 'August'])
        
        report.append("📈 OVERALL SUMMARY:")
        report.append("-" * 18)
        report.append(f"• Total Trading Days: {total_days}")
        report.append(f"• June 2025: {june_days} days")
        report.append(f"• July 2025: {july_days} days")
        report.append(f"• August 2025: {august_days} days")
        report.append("")
        
        # Monthly statistics
        if 'monthly_stats' in analysis:
            report.append("📊 MONTHLY HEDGING STATISTICS:")
            report.append("-" * 30)
            
            for month in ['June', 'July', 'August']:
                if month in analysis['monthly_stats'].index:
                    stats = analysis['monthly_stats'].loc[month]
                    report.append(f"\n{month} 2025:")
                    report.append(f"  • Avg Hedging Intensity: {stats[('hedging_intensity', 'mean')]:.3f}")
                    report.append(f"  • Max Hedging Intensity: {stats[('hedging_intensity', 'max')]:.3f}")
                    report.append(f"  • Avg P/C Ratio: {stats[('pc_ratio', 'mean')]:.3f}")
                    report.append(f"  • Max P/C Ratio: {stats[('pc_ratio', 'max')]:.3f}")
                    report.append(f"  • Avg Institutional %: {stats[('institutional_pct', 'mean')]:.1%}")
                    report.append(f"  • Avg Defensive Positioning: {stats[('defensive_positioning', 'mean')]:.3f}")
                    report.append(f"  • Avg Put OI: {stats[('total_put_oi', 'mean')]:,.0f}")
        
        report.append("")
        
        # Monthly rises analysis
        if 'monthly_rises' in analysis:
            report.append("📈 MONTHLY RISES ANALYSIS:")
            report.append("-" * 25)
            report.append("(First half vs Second half of each month)")
            report.append("")
            
            for month in ['June', 'July', 'August']:
                if month in analysis['monthly_rises']:
                    rises = analysis['monthly_rises'][month]
                    report.append(f"{month} 2025 Rises:")
                    report.append(f"  • Hedging Intensity: {rises['hedging_intensity_rise']:+.3f}")
                    report.append(f"  • P/C Ratio: {rises['pc_ratio_rise']:+.3f}")
                    report.append(f"  • Institutional %: {rises['institutional_rise']:+.1%}")
                    report.append(f"  • Defensive Positioning: {rises['defensive_rise']:+.3f}")
                    report.append(f"  • Put OI Change: {rises['put_oi_rise_pct']:+.1f}%")
                    report.append("")
        
        # Significant spikes
        if 'monthly_spikes' in analysis:
            report.append("🚨 SIGNIFICANT SPIKES:")
            report.append("-" * 20)
            
            for month in ['June', 'July', 'August']:
                if month in analysis['monthly_spikes']:
                    spikes = analysis['monthly_spikes'][month]
                    report.append(f"\n{month} 2025 Spikes:")
                    report.append(f"  • Hedging Intensity Spikes: {spikes['hedging_spikes']}")
                    report.append(f"  • P/C Ratio Spikes: {spikes['pc_spikes']}")
                    report.append(f"  • Institutional Spikes: {spikes['institutional_spikes']}")
                    report.append(f"  • Defensive Spikes: {spikes['defensive_spikes']}")
                    
                    # Show spike dates
                    if spikes['hedging_spike_dates']:
                        dates_str = [d.strftime('%m/%d') for d in spikes['hedging_spike_dates']]
                        report.append(f"    Hedging spike dates: {', '.join(dates_str)}")
                    
                    if spikes['pc_spike_dates']:
                        dates_str = [d.strftime('%m/%d') for d in spikes['pc_spike_dates']]
                        report.append(f"    P/C spike dates: {', '.join(dates_str)}")
        
        report.append("")
        
        # Key findings
        report.append("🎯 KEY FINDINGS:")
        report.append("-" * 15)
        
        # Check for significant rises
        significant_rises = []
        if 'monthly_rises' in analysis:
            for month, rises in analysis['monthly_rises'].items():
                if rises['hedging_intensity_rise'] > 0.1:
                    significant_rises.append(f"{month}: Hedging Intensity +{rises['hedging_intensity_rise']:.3f}")
                if rises['pc_ratio_rise'] > 0.2:
                    significant_rises.append(f"{month}: P/C Ratio +{rises['pc_ratio_rise']:.3f}")
                if rises['institutional_rise'] > 0.05:
                    significant_rises.append(f"{month}: Institutional % +{rises['institutional_rise']:.1%}")
                if rises['defensive_rise'] > 0.1:
                    significant_rises.append(f"{month}: Defensive Positioning +{rises['defensive_rise']:.3f}")
        
        if significant_rises:
            report.append("✅ SIGNIFICANT RISES DETECTED:")
            for rise in significant_rises:
                report.append(f"  • {rise}")
        else:
            report.append("ℹ️  NO SIGNIFICANT RISES DETECTED")
            report.append("  • Hedging activity remained relatively stable")
            report.append("  • No major spikes in defensive positioning")
            report.append("  • Normal institutional hedging patterns")
        
        report.append("")
        
        # Conclusion
        report.append("💡 CONCLUSION:")
        report.append("-" * 12)
        
        if significant_rises:
            report.append("⚠️  HEDGING ACTIVITY INCREASED DURING SUMMER 2025")
            report.append("• Options market showed elevated risk concerns")
            report.append("• May indicate building defensive positioning")
            report.append("• Monitor for continuation into fall")
        else:
            report.append("✅ HEDGING ACTIVITY REMAINED STABLE DURING SUMMER 2025")
            report.append("• No significant rises in defensive positioning")
            report.append("• Normal institutional hedging behavior")
            report.append("• No indication of elevated risk concerns")
        
        return "\n".join(report)


def main():
    """Main summer 2025 hedging analysis"""
    
    # Initialize analyzer
    analyzer = Summer2025HedgingAnalyzer()
    
    print("📊 SUMMER 2025 HEDGING ANALYSIS")
    print("Checking for rises in June, July, August 2025")
    print("=" * 60)
    
    # Build summer analysis
    df_summer = analyzer.build_summer_analysis()
    
    if df_summer.empty:
        print("❌ No summer 2025 data available")
        return
    
    # Analyze hedging rises
    analysis = analyzer.analyze_hedging_rises(df_summer)
    
    # Generate report
    report = analyzer.generate_summer_report(df_summer, analysis)
    print(report)
    
    # Save analysis
    df_summer.to_csv('summer_2025_hedging_data.csv', index=False)
    with open('summer_2025_hedging_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Analysis saved to files")


if __name__ == "__main__":
    main()
