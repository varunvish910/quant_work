"""
Hedging Buildup Analysis: July 16, 2024 vs April 2024 vs Today
==============================================================

This script analyzes hedging activity buildup before key market events:
1. July 16, 2024 (market stress period)
2. April 2024 (baseline period)
3. September 30, 2025 (today)

Focus: How did institutional positioning change during stress vs normal periods?

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

class HedgingBuildupAnalyzer:
    """
    Analyzes hedging buildup patterns before key market events
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
    
    def analyze_hedging_positioning(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze hedging positioning patterns"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return {}
        
        analysis = {}
        
        # 1. Strike Distribution Analysis
        analysis['spy_price'] = spy_price
        
        # Key support levels
        support_levels = {
            '5%_below': spy_price * 0.95,
            '10%_below': spy_price * 0.90,
            '15%_below': spy_price * 0.85,
            '20%_below': spy_price * 0.80
        }
        
        for level_name, level_price in support_levels.items():
            level_puts = puts[abs(puts['strike'] - level_price) <= 5]
            if not level_puts.empty:
                analysis[f'{level_name}_oi'] = level_puts['oi_proxy'].sum()
                analysis[f'{level_name}_vol'] = level_puts['volume'].sum()
                analysis[f'{level_name}_vol_oi'] = analysis[f'{level_name}_vol'] / (analysis[f'{level_name}_oi'] + 1e-6)
            else:
                analysis[f'{level_name}_oi'] = 0
                analysis[f'{level_name}_vol'] = 0
                analysis[f'{level_name}_vol_oi'] = 0
        
        # 2. Deep OTM Put Analysis (crash protection)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.85]
        if not deep_otm_puts.empty:
            analysis['deep_otm_oi'] = deep_otm_puts['oi_proxy'].sum()
            analysis['deep_otm_vol'] = deep_otm_puts['volume'].sum()
            analysis['deep_otm_vol_oi'] = analysis['deep_otm_vol'] / (analysis['deep_otm_oi'] + 1e-6)
        else:
            analysis['deep_otm_oi'] = 0
            analysis['deep_otm_vol'] = 0
            analysis['deep_otm_vol_oi'] = 0
        
        # 3. Institutional vs Retail Analysis
        if 'dte' in df.columns:
            institutional_puts = puts[puts['dte'] > 7]  # >7 DTE = institutional
            retail_puts = puts[puts['dte'] <= 7]        # <=7 DTE = retail/speculative
            
            analysis['institutional_put_oi'] = institutional_puts['oi_proxy'].sum()
            analysis['retail_put_oi'] = retail_puts['oi_proxy'].sum()
            analysis['institutional_put_vol'] = institutional_puts['volume'].sum()
            analysis['retail_put_vol'] = retail_puts['volume'].sum()
            
            total_put_oi = puts['oi_proxy'].sum()
            analysis['institutional_pct'] = analysis['institutional_put_oi'] / (total_put_oi + 1e-6)
            analysis['retail_pct'] = analysis['retail_put_oi'] / (total_put_oi + 1e-6)
        else:
            analysis['institutional_put_oi'] = 0
            analysis['retail_put_oi'] = 0
            analysis['institutional_put_vol'] = 0
            analysis['retail_put_vol'] = 0
            analysis['institutional_pct'] = 0
            analysis['retail_pct'] = 0
        
        # 4. Put/Call Ratio Analysis
        if not calls.empty:
            total_put_oi = puts['oi_proxy'].sum()
            total_call_oi = calls['oi_proxy'].sum()
            analysis['pc_ratio_oi'] = total_put_oi / (total_call_oi + 1e-6)
            
            total_put_vol = puts['volume'].sum()
            total_call_vol = calls['volume'].sum()
            analysis['pc_ratio_vol'] = total_put_vol / (total_call_vol + 1e-6)
        else:
            analysis['pc_ratio_oi'] = 1.0
            analysis['pc_ratio_vol'] = 1.0
        
        # 5. Volume/OI Ratio Analysis (hedging vs speculation)
        total_put_oi = puts['oi_proxy'].sum()
        total_put_vol = puts['volume'].sum()
        analysis['put_vol_oi_ratio'] = total_put_vol / (total_put_oi + 1e-6)
        
        # 6. Strike Concentration Analysis
        strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
        top_5_oi = strike_oi.head(5).sum()
        analysis['strike_concentration'] = top_5_oi / (total_put_oi + 1e-6)
        
        # 7. ATM vs OTM Analysis
        atm_puts = puts[abs(puts['strike'] - spy_price) <= 10]
        otm_puts = puts[puts['strike'] < spy_price * 0.95]
        
        if not atm_puts.empty:
            analysis['atm_put_oi'] = atm_puts['oi_proxy'].sum()
            analysis['atm_put_vol'] = atm_puts['volume'].sum()
        else:
            analysis['atm_put_oi'] = 0
            analysis['atm_put_vol'] = 0
        
        if not otm_puts.empty:
            analysis['otm_put_oi'] = otm_puts['oi_proxy'].sum()
            analysis['otm_put_vol'] = otm_puts['volume'].sum()
        else:
            analysis['otm_put_oi'] = 0
            analysis['otm_put_vol'] = 0
        
        return analysis
    
    def build_period_analysis(self, start_date: str, end_date: str, period_name: str) -> pd.DataFrame:
        """Build analysis for a specific period"""
        
        print(f"📊 ANALYZING {period_name.upper()}")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
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
                    analysis = self.analyze_hedging_positioning(df, spy_price)
                    
                    if analysis:
                        analysis['date'] = current_date
                        analysis['period'] = period_name
                        period_data.append(analysis)
            
            current_date += timedelta(days=1)
        
        if period_data:
            df_period = pd.DataFrame(period_data)
            print(f"✅ Analyzed {len(df_period)} trading days for {period_name}")
            return df_period
        else:
            print(f"❌ No data found for {period_name}")
            return pd.DataFrame()
    
    def compare_periods(self, july_data: pd.DataFrame, april_data: pd.DataFrame, today_data: pd.DataFrame) -> str:
        """Compare hedging patterns across periods"""
        
        report = []
        report.append("🔍 HEDGING POSITIONING COMPARISON")
        report.append("=" * 50)
        report.append("")
        
        # Key metrics to compare
        metrics = [
            'institutional_pct',
            'pc_ratio_oi',
            'put_vol_oi_ratio',
            'strike_concentration',
            'deep_otm_oi',
            '5%_below_oi',
            '10%_below_oi',
            '15%_below_oi',
            '20%_below_oi'
        ]
        
        for metric in metrics:
            if metric in july_data.columns and metric in april_data.columns and metric in today_data.columns:
                july_avg = july_data[metric].mean()
                april_avg = april_data[metric].mean()
                today_avg = today_data[metric].mean()
                
                report.append(f"📊 {metric.replace('_', ' ').title()}:")
                report.append(f"  • April 2024: {april_avg:.2f}")
                report.append(f"  • July 2024: {july_avg:.2f}")
                report.append(f"  • Today (Sep 30, 2025): {today_avg:.2f}")
                
                # Calculate changes
                july_vs_april = ((july_avg - april_avg) / april_avg * 100) if april_avg != 0 else 0
                today_vs_july = ((today_avg - july_avg) / july_avg * 100) if july_avg != 0 else 0
                today_vs_april = ((today_avg - april_avg) / april_avg * 100) if april_avg != 0 else 0
                
                report.append(f"  • July vs April: {july_vs_april:+.1f}%")
                report.append(f"  • Today vs July: {today_vs_july:+.1f}%")
                report.append(f"  • Today vs April: {today_vs_april:+.1f}%")
                report.append("")
        
        # Key insights
        report.append("🎯 KEY INSIGHTS:")
        report.append("-" * 20)
        
        # Institutional positioning
        if 'institutional_pct' in july_data.columns:
            july_inst = july_data['institutional_pct'].mean()
            april_inst = april_data['institutional_pct'].mean()
            today_inst = today_data['institutional_pct'].mean()
            
            if july_inst > april_inst:
                report.append("• July 2024: INSTITUTIONAL HEDGING INCREASED vs April")
            else:
                report.append("• July 2024: Institutional hedging decreased vs April")
            
            if today_inst > july_inst:
                report.append("• Today: INSTITUTIONAL HEDGING HIGHER than July 2024")
            else:
                report.append("• Today: Institutional hedging lower than July 2024")
        
        # Put/Call ratios
        if 'pc_ratio_oi' in july_data.columns:
            july_pc = july_data['pc_ratio_oi'].mean()
            april_pc = april_data['pc_ratio_oi'].mean()
            today_pc = today_data['pc_ratio_oi'].mean()
            
            if july_pc > april_pc:
                report.append("• July 2024: BEARISH POSITIONING INCREASED vs April")
            else:
                report.append("• July 2024: Bearish positioning decreased vs April")
            
            if today_pc > july_pc:
                report.append("• Today: MORE BEARISH than July 2024")
            else:
                report.append("• Today: Less bearish than July 2024")
        
        # Deep OTM protection
        if 'deep_otm_oi' in july_data.columns:
            july_deep = july_data['deep_otm_oi'].mean()
            april_deep = april_data['deep_otm_oi'].mean()
            today_deep = today_data['deep_otm_oi'].mean()
            
            if july_deep > april_deep:
                report.append("• July 2024: CRASH PROTECTION INCREASED vs April")
            else:
                report.append("• July 2024: Crash protection decreased vs April")
            
            if today_deep > july_deep:
                report.append("• Today: MORE CRASH PROTECTION than July 2024")
            else:
                report.append("• Today: Less crash protection than July 2024")
        
        return "\n".join(report)
    
    def create_comparison_chart(self, july_data: pd.DataFrame, april_data: pd.DataFrame, 
                              today_data: pd.DataFrame, save_path: str = None):
        """Create comparison visualization"""
        
        if july_data.empty or april_data.empty or today_data.empty:
            print("Insufficient data for comparison chart")
            return
        
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hedging Positioning Comparison: April 2024 vs July 2024 vs Today', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Institutional vs Retail Positioning
        periods = ['April 2024', 'July 2024', 'Today']
        institutional_pct = [
            april_data['institutional_pct'].mean(),
            july_data['institutional_pct'].mean(),
            today_data['institutional_pct'].mean()
        ]
        retail_pct = [
            april_data['retail_pct'].mean(),
            july_data['retail_pct'].mean(),
            today_data['retail_pct'].mean()
        ]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax1.bar(x - width/2, institutional_pct, width, label='Institutional', color='blue', alpha=0.7)
        ax1.bar(x + width/2, retail_pct, width, label='Retail', color='red', alpha=0.7)
        
        ax1.set_title('Institutional vs Retail Positioning')
        ax1.set_ylabel('Percentage of Total Put OI')
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Put/Call Ratios
        pc_ratios = [
            april_data['pc_ratio_oi'].mean(),
            july_data['pc_ratio_oi'].mean(),
            today_data['pc_ratio_oi'].mean()
        ]
        
        ax2.bar(periods, pc_ratios, color=['green', 'orange', 'red'], alpha=0.7)
        ax2.set_title('Put/Call Ratio (OI)')
        ax2.set_ylabel('P/C Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Support Level OI
        support_levels = ['5%_below_oi', '10%_below_oi', '15%_below_oi', '20%_below_oi']
        support_labels = ['5% Below', '10% Below', '15% Below', '20% Below']
        
        april_support = [april_data[level].mean() for level in support_levels]
        july_support = [july_data[level].mean() for level in support_levels]
        today_support = [today_data[level].mean() for level in support_levels]
        
        x_support = np.arange(len(support_labels))
        width_support = 0.25
        
        ax3.bar(x_support - width_support, april_support, width_support, label='April 2024', color='green', alpha=0.7)
        ax3.bar(x_support, july_support, width_support, label='July 2024', color='orange', alpha=0.7)
        ax3.bar(x_support + width_support, today_support, width_support, label='Today', color='red', alpha=0.7)
        
        ax3.set_title('Support Level Put OI')
        ax3.set_ylabel('Put OI')
        ax3.set_xticks(x_support)
        ax3.set_xticklabels(support_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volume/OI Ratios
        vol_oi_ratios = [
            april_data['put_vol_oi_ratio'].mean(),
            july_data['put_vol_oi_ratio'].mean(),
            today_data['put_vol_oi_ratio'].mean()
        ]
        
        ax4.bar(periods, vol_oi_ratios, color=['green', 'orange', 'red'], alpha=0.7)
        ax4.set_title('Put Volume/OI Ratio (Hedging vs Speculation)')
        ax4.set_ylabel('V/OI Ratio')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()


def main():
    """Main hedging buildup analysis"""
    
    # Initialize analyzer
    analyzer = HedgingBuildupAnalyzer()
    
    print("🔍 HEDGING BUILDUP ANALYSIS")
    print("July 16, 2024 vs April 2024 vs Today (Sep 30, 2025)")
    print("=" * 60)
    
    # Analyze each period
    print("\n1. ANALYZING APRIL 2024 (Baseline Period)")
    april_data = analyzer.build_period_analysis('2024-04-01', '2024-04-30', 'April 2024')
    
    print("\n2. ANALYZING JULY 2024 (Pre-Stress Period)")
    july_data = analyzer.build_period_analysis('2024-07-01', '2024-07-16', 'July 2024')
    
    print("\n3. ANALYZING TODAY (September 30, 2025)")
    today_data = analyzer.build_period_analysis('2025-09-30', '2025-09-30', 'Today')
    
    if april_data.empty or july_data.empty or today_data.empty:
        print("❌ Insufficient data for comparison")
        return
    
    # Compare periods
    print("\n4. COMPARING PERIODS")
    comparison = analyzer.compare_periods(july_data, april_data, today_data)
    print(comparison)
    
    # Create visualization
    print("\n5. CREATING COMPARISON CHART")
    analyzer.create_comparison_chart(july_data, april_data, today_data, 'hedging_buildup_comparison.png')
    
    # Save data
    april_data.to_csv('april_2024_hedging_data.csv', index=False)
    july_data.to_csv('july_2024_hedging_data.csv', index=False)
    today_data.to_csv('today_hedging_data.csv', index=False)
    
    with open('hedging_buildup_analysis.txt', 'w') as f:
        f.write(comparison)
    
    print(f"\n💾 Data and analysis saved")


if __name__ == "__main__":
    main()
