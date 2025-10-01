"""
Institutional Hedge Rolling Analysis
===================================

Analyzes if institutions are systematically rolling 10% hedges
as SPY moves higher, maintaining consistent downside protection.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

class InstitutionalHedgeRollingAnalyzer:
    """
    Analyzes institutional hedge rolling patterns
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
    
    def analyze_hedge_levels(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze hedge levels relative to current price"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        analysis = {}
        analysis['spy_price'] = spy_price
        analysis['date'] = df['date'].iloc[0] if 'date' in df.columns else pd.Timestamp.now()
        
        # Calculate hedge levels as percentages below current price
        hedge_levels = {
            '5%_below': spy_price * 0.95,
            '10%_below': spy_price * 0.90,
            '15%_below': spy_price * 0.85,
            '20%_below': spy_price * 0.80,
            '25%_below': spy_price * 0.75
        }
        
        for level_name, level_price in hedge_levels.items():
            # Find puts within 2.5% of the target level
            level_puts = puts[abs(puts['strike'] - level_price) <= spy_price * 0.025]
            
            if not level_puts.empty:
                analysis[f'{level_name}_oi'] = level_puts['oi_proxy'].sum()
                analysis[f'{level_name}_vol'] = level_puts['volume'].sum()
                analysis[f'{level_name}_vol_oi'] = analysis[f'{level_name}_vol'] / (analysis[f'{level_name}_oi'] + 1e-6)
                analysis[f'{level_name}_strike_avg'] = level_puts['strike'].mean()
            else:
                analysis[f'{level_name}_oi'] = 0
                analysis[f'{level_name}_vol'] = 0
                analysis[f'{level_name}_vol_oi'] = 0
                analysis[f'{level_name}_strike_avg'] = level_price
        
        # Calculate total put OI for normalization
        total_put_oi = puts['oi_proxy'].sum()
        analysis['total_put_oi'] = total_put_oi
        
        # Calculate percentage of OI at each hedge level
        for level_name in hedge_levels.keys():
            oi_key = f'{level_name}_oi'
            if oi_key in analysis:
                analysis[f'{level_name}_pct'] = analysis[oi_key] / (total_put_oi + 1e-6)
        
        return analysis
    
    def build_rolling_analysis(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build rolling hedge analysis"""
        
        print(f"📊 BUILDING ROLLING HEDGE ANALYSIS")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        rolling_data = []
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    analysis = self.analyze_hedge_levels(df, spy_price)
                    
                    if analysis:
                        rolling_data.append(analysis)
            
            current_date += timedelta(days=1)
        
        if rolling_data:
            df_rolling = pd.DataFrame(rolling_data)
            
            # Calculate rolling averages
            for level in ['5%_below', '10%_below', '15%_below', '20%_below', '25%_below']:
                df_rolling[f'{level}_oi_ma_10'] = df_rolling[f'{level}_oi'].rolling(window=10, min_periods=1).mean()
                df_rolling[f'{level}_pct_ma_10'] = df_rolling[f'{level}_pct'].rolling(window=10, min_periods=1).mean()
            
            print(f"✅ Built rolling analysis for {len(df_rolling)} trading days")
            return df_rolling
        else:
            print("❌ No data found for the specified period")
            return pd.DataFrame()
    
    def analyze_rolling_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze rolling hedge patterns"""
        
        if df.empty:
            return {}
        
        patterns = {}
        
        # 1. Correlation between SPY price and hedge levels
        spy_price = df['spy_price']
        
        for level in ['5%_below', '10%_below', '15%_below', '20%_below', '25%_below']:
            oi_col = f'{level}_oi'
            pct_col = f'{level}_pct'
            
            if oi_col in df.columns and pct_col in df.columns:
                # Correlation with SPY price
                oi_corr = spy_price.corr(df[oi_col])
                pct_corr = spy_price.corr(df[pct_col])
                
                patterns[f'{level}_oi_correlation'] = oi_corr
                patterns[f'{level}_pct_correlation'] = pct_corr
                
                # Average OI and percentage
                patterns[f'{level}_avg_oi'] = df[oi_col].mean()
                patterns[f'{level}_avg_pct'] = df[pct_col].mean()
        
        # 2. Identify dominant hedge level
        hedge_levels = ['5%_below', '10%_below', '15%_below', '20%_below', '25%_below']
        avg_oi_by_level = {}
        
        for level in hedge_levels:
            oi_col = f'{level}_oi'
            if oi_col in df.columns:
                avg_oi_by_level[level] = df[oi_col].mean()
        
        if avg_oi_by_level:
            dominant_level = max(avg_oi_by_level, key=avg_oi_by_level.get)
            patterns['dominant_hedge_level'] = dominant_level
            patterns['dominant_hedge_oi'] = avg_oi_by_level[dominant_level]
        
        # 3. Rolling behavior analysis
        # Check if 10% below maintains consistent OI as SPY moves
        if '10%_below_oi' in df.columns:
            # Calculate rolling correlation between SPY price and 10% below OI
            spy_ma_10 = spy_price.rolling(window=10, min_periods=1).mean()
            oi_ma_10 = df['10%_below_oi'].rolling(window=10, min_periods=1).mean()
            
            rolling_corr = spy_ma_10.corr(oi_ma_10)
            patterns['10%_below_rolling_correlation'] = rolling_corr
            
            # Check if OI stays relatively constant
            oi_std = df['10%_below_oi'].std()
            oi_mean = df['10%_below_oi'].mean()
            oi_cv = oi_std / (oi_mean + 1e-6)  # Coefficient of variation
            
            patterns['10%_below_oi_consistency'] = 1 - oi_cv  # Higher = more consistent
        
        return patterns
    
    def create_rolling_chart(self, df: pd.DataFrame, save_path: str = None):
        """Create rolling hedge analysis chart"""
        
        if df.empty:
            print("No data to plot")
            return
        
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Institutional Hedge Rolling Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: SPY Price vs 10% Below OI
        ax1_twin = ax1.twinx()
        
        ax1.plot(df['date'], df['spy_price'], label='SPY Price', linewidth=2, color='blue', alpha=0.8)
        ax1_twin.plot(df['date'], df['10%_below_oi'], label='10% Below OI', linewidth=2, color='red', alpha=0.8)
        
        ax1.set_title('SPY Price vs 10% Below Put OI')
        ax1.set_ylabel('SPY Price ($)', color='blue')
        ax1_twin.set_ylabel('10% Below Put OI', color='red')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hedge Level OI Distribution
        hedge_levels = ['5%_below', '10%_below', '15%_below', '20%_below', '25%_below']
        hedge_labels = ['5% Below', '10% Below', '15% Below', '20% Below', '25% Below']
        
        avg_oi = [df[f'{level}_oi'].mean() for level in hedge_levels]
        
        ax2.bar(hedge_labels, avg_oi, color=['green', 'red', 'orange', 'purple', 'brown'], alpha=0.7)
        ax2.set_title('Average Put OI by Hedge Level')
        ax2.set_ylabel('Average Put OI')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hedge Level Percentages
        avg_pct = [df[f'{level}_pct'].mean() for level in hedge_levels]
        
        ax3.bar(hedge_labels, avg_pct, color=['green', 'red', 'orange', 'purple', 'brown'], alpha=0.7)
        ax3.set_title('Average Put OI Percentage by Hedge Level')
        ax3.set_ylabel('Percentage of Total Put OI')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rolling Correlation Analysis
        if '10%_below_oi_ma_10' in df.columns:
            spy_ma_10 = df['spy_price'].rolling(window=10, min_periods=1).mean()
            oi_ma_10 = df['10%_below_oi'].rolling(window=10, min_periods=1).mean()
            
            ax4.plot(df['date'], spy_ma_10, label='SPY 10-day MA', linewidth=2, color='blue', alpha=0.8)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(df['date'], oi_ma_10, label='10% Below OI 10-day MA', linewidth=2, color='red', alpha=0.8)
            
            ax4.set_title('Rolling Correlation: SPY vs 10% Below OI')
            ax4.set_ylabel('SPY 10-day MA', color='blue')
            ax4_twin.set_ylabel('10% Below OI 10-day MA', color='red')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def generate_rolling_analysis(self, patterns: dict) -> str:
        """Generate rolling hedge analysis report"""
        
        report = []
        report.append("🔄 INSTITUTIONAL HEDGE ROLLING ANALYSIS")
        report.append("=" * 50)
        report.append("")
        
        # Dominant hedge level
        if 'dominant_hedge_level' in patterns:
            dominant = patterns['dominant_hedge_level']
            avg_oi = patterns['dominant_hedge_oi']
            report.append(f"🎯 DOMINANT HEDGE LEVEL: {dominant.replace('_', ' ').title()}")
            report.append(f"   Average OI: {avg_oi:,.0f}")
            report.append("")
        
        # 10% below analysis
        if '10%_below_rolling_correlation' in patterns:
            corr = patterns['10%_below_rolling_correlation']
            consistency = patterns.get('10%_below_oi_consistency', 0)
            
            report.append("📊 10% BELOW HEDGE ANALYSIS:")
            report.append(f"   Rolling Correlation with SPY: {corr:.3f}")
            report.append(f"   OI Consistency Score: {consistency:.3f}")
            
            if abs(corr) < 0.3:
                report.append("   ✅ LOW CORRELATION = Rolling hedge behavior")
            else:
                report.append("   ⚠️  HIGH CORRELATION = Price-following behavior")
            
            if consistency > 0.7:
                report.append("   ✅ HIGH CONSISTENCY = Systematic rolling")
            else:
                report.append("   ⚠️  LOW CONSISTENCY = Irregular positioning")
            
            report.append("")
        
        # Hedge level correlations
        report.append("📈 HEDGE LEVEL CORRELATIONS WITH SPY:")
        report.append("-" * 40)
        
        for level in ['5%_below', '10%_below', '15%_below', '20%_below', '25%_below']:
            oi_corr_key = f'{level}_oi_correlation'
            pct_corr_key = f'{level}_pct_correlation'
            
            if oi_corr_key in patterns and pct_corr_key in patterns:
                oi_corr = patterns[oi_corr_key]
                pct_corr = patterns[pct_corr_key]
                avg_oi = patterns.get(f'{level}_avg_oi', 0)
                avg_pct = patterns.get(f'{level}_avg_pct', 0)
                
                report.append(f"{level.replace('_', ' ').title()}:")
                report.append(f"  OI Correlation: {oi_corr:.3f}")
                report.append(f"  % Correlation: {pct_corr:.3f}")
                report.append(f"  Avg OI: {avg_oi:,.0f}")
                report.append(f"  Avg %: {avg_pct:.1%}")
                report.append("")
        
        # Rolling behavior conclusion
        report.append("🎯 ROLLING BEHAVIOR CONCLUSION:")
        report.append("-" * 30)
        
        if '10%_below_rolling_correlation' in patterns:
            corr = patterns['10%_below_rolling_correlation']
            consistency = patterns.get('10%_below_oi_consistency', 0)
            
            if abs(corr) < 0.3 and consistency > 0.7:
                report.append("✅ INSTITUTIONS ARE ROLLING 10% HEDGES")
                report.append("   - Low correlation with SPY price")
                report.append("   - High consistency in OI levels")
                report.append("   - Systematic downside protection")
            elif abs(corr) < 0.5:
                report.append("⚠️  PARTIAL ROLLING BEHAVIOR")
                report.append("   - Some correlation with SPY price")
                report.append("   - Mixed consistency")
            else:
                report.append("❌ NOT ROLLING 10% HEDGES")
                report.append("   - High correlation with SPY price")
                report.append("   - Price-following behavior")
        
        return "\n".join(report)


def main():
    """Main rolling hedge analysis"""
    
    # Initialize analyzer
    analyzer = InstitutionalHedgeRollingAnalyzer()
    
    print("🔄 INSTITUTIONAL HEDGE ROLLING ANALYSIS")
    print("Are institutions rolling 10% hedges as SPY moves?")
    print("=" * 60)
    
    # Build rolling analysis for 2024-2025
    df_rolling = analyzer.build_rolling_analysis('2024-01-01', '2025-09-30')
    
    if df_rolling.empty:
        print("❌ No data available")
        return
    
    # Analyze rolling patterns
    patterns = analyzer.analyze_rolling_patterns(df_rolling)
    
    # Generate analysis
    analysis = analyzer.generate_rolling_analysis(patterns)
    print(analysis)
    
    # Create visualization
    analyzer.create_rolling_chart(df_rolling, 'institutional_hedge_rolling_chart.png')
    
    # Save data
    df_rolling.to_csv('institutional_hedge_rolling_data.csv', index=False)
    with open('institutional_hedge_rolling_analysis.txt', 'w') as f:
        f.write(analysis)
    
    print(f"\n💾 Data and analysis saved")


if __name__ == "__main__":
    main()
