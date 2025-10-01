"""
SPY Hedging Buildup Analyzer
============================

This script specifically analyzes hedging activity buildup patterns that may
precede market pullbacks. It focuses on:

1. Put option accumulation at key support levels
2. Volume/OI ratios indicating institutional hedging
3. Strike concentration patterns
4. Temporal buildup detection
5. Risk assessment and early warning signals

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HedgingBuildupAnalyzer:
    """
    Analyzes hedging buildup patterns in SPY options data
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        
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
                # Convert date column to datetime if it exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                else:
                    df['date'] = pd.to_datetime(date)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading data for {date}: {e}")
            return pd.DataFrame()
    
    def analyze_put_accumulation(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze put option accumulation patterns"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        analysis = {}
        
        # 1. Deep OTM puts (potential hedging)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.90]
        if not deep_otm_puts.empty:
            analysis['deep_otm_oi'] = deep_otm_puts['oi_proxy'].sum()
            analysis['deep_otm_volume'] = deep_otm_puts['volume'].sum()
            analysis['deep_otm_vol_oi'] = deep_otm_puts['volume'].sum() / (deep_otm_puts['oi_proxy'].sum() + 1e-6)
            analysis['deep_otm_contracts'] = len(deep_otm_puts)
        
        # 2. Key support levels analysis
        support_levels = [
            spy_price * 0.95,  # 5% below current
            spy_price * 0.90,  # 10% below current
            spy_price * 0.85,  # 15% below current
            spy_price * 0.80,  # 20% below current
        ]
        
        for i, level in enumerate(support_levels):
            level_puts = puts[abs(puts['strike'] - level) <= 5]  # Within $5 of level
            if not level_puts.empty:
                analysis[f'support_{i+1}_oi'] = level_puts['oi_proxy'].sum()
                analysis[f'support_{i+1}_volume'] = level_puts['volume'].sum()
                analysis[f'support_{i+1}_vol_oi'] = level_puts['volume'].sum() / (level_puts['oi_proxy'].sum() + 1e-6)
                analysis[f'support_{i+1}_level'] = level
        
        # 3. Strike concentration analysis
        total_put_oi = puts['oi_proxy'].sum()
        if total_put_oi > 0:
            # Top 5 strikes by OI
            top_strikes = puts.nlargest(5, 'oi_proxy')
            analysis['top_5_oi_pct'] = top_strikes['oi_proxy'].sum() / total_put_oi
            
            # Single strike concentration
            max_single_oi = puts['oi_proxy'].max()
            analysis['max_strike_oi_pct'] = max_single_oi / total_put_oi
            
            # Check for unusual concentration (>20% at single strike)
            analysis['unusual_concentration'] = 1 if analysis['max_strike_oi_pct'] > 0.20 else 0
        
        # 4. Time to expiration analysis (hedging often uses longer-dated options)
        if 'dte' in puts.columns:
            analysis['avg_put_dte'] = puts['dte'].mean()
            long_dated_puts = puts[puts['dte'] > 60]
            if not long_dated_puts.empty:
                analysis['long_dated_oi_pct'] = long_dated_puts['oi_proxy'].sum() / total_put_oi
            else:
                analysis['long_dated_oi_pct'] = 0
        
        # 5. Volume/OI ratio analysis (low ratio = accumulation/hedging)
        analysis['put_vol_oi_ratio'] = puts['volume'].sum() / (total_put_oi + 1e-6)
        analysis['hedging_like_activity'] = 1 if analysis['put_vol_oi_ratio'] < 0.3 else 0
        
        return analysis
    
    def calculate_buildup_score(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate hedging buildup score"""
        if df.empty:
            return {}
        
        analysis = self.analyze_put_accumulation(df, spy_price)
        if not analysis:
            return {}
        
        # Calculate buildup score (0-100)
        score = 0
        factors = []
        
        # Factor 1: Deep OTM put activity (0-25 points)
        if 'deep_otm_oi' in analysis and analysis['deep_otm_oi'] > 0:
            deep_otm_score = min(25, analysis['deep_otm_oi'] / 100000)  # Scale by 100k OI
            score += deep_otm_score
            factors.append(f"Deep OTM puts: {deep_otm_score:.1f}/25")
        
        # Factor 2: Strike concentration (0-25 points)
        if 'max_strike_oi_pct' in analysis:
            concentration_score = min(25, analysis['max_strike_oi_pct'] * 100)
            score += concentration_score
            factors.append(f"Strike concentration: {concentration_score:.1f}/25")
        
        # Factor 3: Low V/OI ratio (0-25 points)
        if 'put_vol_oi_ratio' in analysis:
            vol_oi_score = max(0, 25 - analysis['put_vol_oi_ratio'] * 50)  # Lower ratio = higher score
            score += vol_oi_score
            factors.append(f"Low V/OI ratio: {vol_oi_score:.1f}/25")
        
        # Factor 4: Long-dated options (0-25 points)
        if 'long_dated_oi_pct' in analysis:
            long_dated_score = analysis['long_dated_oi_pct'] * 25
            score += long_dated_score
            factors.append(f"Long-dated options: {long_dated_score:.1f}/25")
        
        analysis['buildup_score'] = min(100, score)
        analysis['buildup_factors'] = factors
        
        # Risk level assessment
        if score >= 75:
            analysis['risk_level'] = 'HIGH'
        elif score >= 50:
            analysis['risk_level'] = 'MEDIUM'
        elif score >= 25:
            analysis['risk_level'] = 'LOW'
        else:
            analysis['risk_level'] = 'MINIMAL'
        
        return analysis
    
    def analyze_risk_concentration(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze where hedging risk is concentrated by strike levels"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        # Group by strike ranges to find concentration
        strike_ranges = [
            (spy_price * 0.95, spy_price * 1.05, "Near-the-Money"),
            (spy_price * 0.90, spy_price * 0.95, "5-10% OTM"),
            (spy_price * 0.80, spy_price * 0.90, "10-20% OTM"),
            (0, spy_price * 0.80, "Deep OTM (>20%)")
        ]
        
        concentration_analysis = {}
        total_put_oi = puts['oi_proxy'].sum()
        
        for min_strike, max_strike, range_name in strike_ranges:
            range_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] < max_strike)]
            
            if not range_puts.empty:
                range_oi = range_puts['oi_proxy'].sum()
                range_volume = range_puts['volume'].sum()
                range_vol_oi = range_volume / (range_oi + 1e-6)
                
                concentration_analysis[f'{range_name}_oi'] = range_oi
                concentration_analysis[f'{range_name}_oi_pct'] = range_oi / total_put_oi
                concentration_analysis[f'{range_name}_volume'] = range_volume
                concentration_analysis[f'{range_name}_vol_oi'] = range_vol_oi
                concentration_analysis[f'{range_name}_contracts'] = len(range_puts)
        
        # Find top risk strikes (highest OI)
        top_strikes = puts.nlargest(10, 'oi_proxy')[['strike', 'oi_proxy', 'volume']].copy()
        top_strikes['vol_oi_ratio'] = top_strikes['volume'] / (top_strikes['oi_proxy'] + 1e-6)
        top_strikes['distance_from_price'] = abs(top_strikes['strike'] - spy_price)
        top_strikes['distance_pct'] = (top_strikes['distance_from_price'] / spy_price) * 100
        
        concentration_analysis['top_risk_strikes'] = top_strikes.to_dict('records')
        
        return concentration_analysis
    
    def analyze_recent_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Analyze hedging buildup over recent period"""
        print(f"Analyzing hedging buildup from {start_date} to {end_date}")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        daily_analysis = []
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_date.weekday() < 5:
                df = self.load_daily_data(date_str)
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    analysis = self.calculate_buildup_score(df, spy_price)
                    
                    # Add risk concentration analysis
                    concentration = self.analyze_risk_concentration(df, spy_price)
                    analysis.update(concentration)
                    
                    if analysis:
                        analysis['date'] = current_date
                        analysis['spy_price'] = spy_price
                        daily_analysis.append(analysis)
            
            current_date += timedelta(days=1)
        
        if not daily_analysis:
            print("No data found for the specified period")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_analysis = pd.DataFrame(daily_analysis)
        df_analysis = df_analysis.sort_values('date').reset_index(drop=True)
        
        # Calculate trends
        if 'buildup_score' in df_analysis.columns:
            df_analysis['buildup_trend'] = df_analysis['buildup_score'].diff()
            df_analysis['buildup_ma_5'] = df_analysis['buildup_score'].rolling(window=5, min_periods=1).mean()
        
        return df_analysis
    
    def check_new_activity(self, target_date: str, comparison_days: int = 5) -> dict:
        """Check for new hedging activity on a specific date compared to recent days"""
        target_df = self.load_daily_data(target_date)
        if target_df.empty:
            return {}
        
        spy_price = target_df['underlying_price'].iloc[0]
        target_analysis = self.calculate_buildup_score(target_df, spy_price)
        target_concentration = self.analyze_risk_concentration(target_df, spy_price)
        target_analysis.update(target_concentration)
        
        # Get comparison data from recent days
        target_dt = pd.to_datetime(target_date)
        comparison_data = []
        
        for i in range(1, comparison_days + 1):
            comp_date = target_dt - timedelta(days=i)
            if comp_date.weekday() < 5:  # Skip weekends
                comp_date_str = comp_date.strftime('%Y-%m-%d')
                comp_df = self.load_daily_data(comp_date_str)
                if not comp_df.empty:
                    comp_spy_price = comp_df['underlying_price'].iloc[0]
                    comp_analysis = self.calculate_buildup_score(comp_df, comp_spy_price)
                    comparison_data.append(comp_analysis)
        
        if not comparison_data:
            return target_analysis
        
        # Calculate changes
        changes = {}
        for key in target_analysis:
            if key in ['date', 'spy_price', 'buildup_factors', 'top_risk_strikes']:
                continue
            
            if isinstance(target_analysis[key], (int, float)):
                recent_avg = np.mean([comp.get(key, 0) for comp in comparison_data])
                changes[f'{key}_change'] = target_analysis[key] - recent_avg
                changes[f'{key}_change_pct'] = ((target_analysis[key] - recent_avg) / (recent_avg + 1e-6)) * 100
        
        target_analysis['changes'] = changes
        return target_analysis
    
    def generate_buildup_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive hedging buildup report"""
        if df.empty:
            return "No data available for analysis"
        
        report = []
        report.append("=" * 70)
        report.append("SPY HEDGING BUILDUP ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        report.append(f"Total Trading Days: {len(df)}")
        report.append("")
        
        # Current status
        latest = df.iloc[-1]
        report.append("🔍 CURRENT HEDGING BUILDUP STATUS:")
        report.append("-" * 40)
        
        if 'buildup_score' in latest:
            score = latest['buildup_score']
            risk_level = latest.get('risk_level', 'UNKNOWN')
            report.append(f"Hedging Buildup Score: {score:.1f}/100 ({risk_level} RISK)")
            
            if 'buildup_factors' in latest and isinstance(latest['buildup_factors'], list):
                report.append("")
                report.append("Key Factors:")
                for factor in latest['buildup_factors']:
                    report.append(f"  • {factor}")
        
        if 'spy_price' in latest:
            report.append(f"Current SPY Price: ${latest['spy_price']:.2f}")
        
        report.append("")
        
        # Recent trends
        report.append("📈 RECENT BUILDUP TRENDS:")
        report.append("-" * 30)
        
        if 'buildup_score' in df.columns:
            recent_score = df['buildup_score'].tail(3).mean()
            historical_avg = df['buildup_score'].mean()
            
            if recent_score > historical_avg * 1.2:
                trend = "ACCELERATING BUILDUP"
            elif recent_score > historical_avg:
                trend = "RISING BUILDUP"
            elif recent_score < historical_avg * 0.8:
                trend = "DECLINING BUILDUP"
            else:
                trend = "STABLE"
            
            report.append(f"Buildup Trend: {trend}")
            report.append(f"Recent Score: {recent_score:.1f} (Avg: {historical_avg:.1f})")
        
        # Risk concentration analysis
        report.append("")
        report.append("🎯 RISK CONCENTRATION ANALYSIS:")
        report.append("-" * 35)
        
        # Strike range concentrations
        range_names = ["Near-the-Money", "5-10% OTM", "10-20% OTM", "Deep OTM (>20%)"]
        for range_name in range_names:
            oi_key = f'{range_name}_oi'
            oi_pct_key = f'{range_name}_oi_pct'
            vol_oi_key = f'{range_name}_vol_oi'
            
            if oi_key in latest:
                oi = latest.get(oi_key, 0)
                oi_pct = latest.get(oi_pct_key, 0) * 100
                vol_oi = latest.get(vol_oi_key, 0)
                
                if oi > 0:
                    report.append(f"{range_name}: {oi:,.0f} OI ({oi_pct:.1f}% of total, V/OI: {vol_oi:.2f})")
        
        # Top risk strikes
        if 'top_risk_strikes' in latest and isinstance(latest['top_risk_strikes'], list):
            report.append("")
            report.append("🔴 TOP RISK STRIKES:")
            report.append("-" * 20)
            for i, strike_data in enumerate(latest['top_risk_strikes'][:5]):
                strike = strike_data['strike']
                oi = strike_data['oi_proxy']
                vol_oi = strike_data['vol_oi_ratio']
                distance_pct = strike_data['distance_pct']
                
                report.append(f"{i+1}. ${strike:.0f}: {oi:,.0f} OI (V/OI: {vol_oi:.2f}, {distance_pct:.1f}% from price)")
        
        # Deep OTM analysis
        if 'deep_otm_oi' in latest:
            deep_otm_oi = latest['deep_otm_oi']
            avg_deep_otm = df['deep_otm_oi'].mean()
            
            report.append("")
            report.append("🛡️ DEEP OTM PUT ANALYSIS:")
            report.append("-" * 30)
            report.append(f"Deep OTM Put OI: {deep_otm_oi:,.0f} (Avg: {avg_deep_otm:,.0f})")
            
            if deep_otm_oi > avg_deep_otm * 1.5:
                report.append("⚠️  HIGH deep OTM put activity - potential hedging buildup")
            elif deep_otm_oi > avg_deep_otm:
                report.append("📈 Elevated deep OTM put activity")
            else:
                report.append("✅ Normal deep OTM put activity")
        
        # Risk assessment
        report.append("")
        report.append("⚠️  RISK ASSESSMENT:")
        report.append("-" * 20)
        
        if 'buildup_score' in latest:
            score = latest['buildup_score']
            if score >= 75:
                report.append("🔴 HIGH RISK - Significant hedging buildup detected")
                report.append("   • Monitor for potential market weakness")
                report.append("   • Consider defensive positioning")
            elif score >= 50:
                report.append("🟡 MEDIUM RISK - Moderate hedging activity")
                report.append("   • Watch for acceleration in hedging")
                report.append("   • Prepare for potential volatility")
            elif score >= 25:
                report.append("🟢 LOW RISK - Minimal hedging buildup")
                report.append("   • Normal market conditions")
                report.append("   • Continue monitoring")
            else:
                report.append("✅ MINIMAL RISK - Very low hedging activity")
                report.append("   • Market appears stable")
                report.append("   • No immediate concerns")
        
        return "\n".join(report)
    
    def plot_buildup_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Create hedging buildup analysis plots"""
        if df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SPY Hedging Buildup Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Buildup Score over time
        if 'buildup_score' in df.columns:
            axes[0, 0].plot(df['date'], df['buildup_score'], label='Buildup Score', linewidth=2, color='red')
            if 'buildup_ma_5' in df.columns:
                axes[0, 0].plot(df['date'], df['buildup_ma_5'], 
                               label='5-day MA', alpha=0.7, linestyle='--', color='orange')
            
            # Add risk level zones
            axes[0, 0].axhspan(0, 25, alpha=0.1, color='green', label='Minimal Risk')
            axes[0, 0].axhspan(25, 50, alpha=0.1, color='yellow', label='Low Risk')
            axes[0, 0].axhspan(50, 75, alpha=0.1, color='orange', label='Medium Risk')
            axes[0, 0].axhspan(75, 100, alpha=0.1, color='red', label='High Risk')
            
            axes[0, 0].set_title('Hedging Buildup Score Over Time')
            axes[0, 0].set_ylabel('Buildup Score (0-100)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 100)
        
        # Plot 2: Deep OTM Put Activity
        if 'deep_otm_oi' in df.columns:
            axes[0, 1].plot(df['date'], df['deep_otm_oi'], label='Deep OTM Put OI', 
                           linewidth=2, color='purple')
            axes[0, 1].set_title('Deep OTM Put Open Interest')
            axes[0, 1].set_ylabel('Open Interest')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Strike Concentration
        if 'max_strike_oi_pct' in df.columns:
            axes[1, 0].plot(df['date'], df['max_strike_oi_pct'] * 100, 
                           label='Max Strike Concentration %', linewidth=2, color='blue')
            axes[1, 0].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='High Concentration (20%)')
            axes[1, 0].set_title('Put Strike Concentration')
            axes[1, 0].set_ylabel('Concentration %')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Volume/OI Ratio
        if 'put_vol_oi_ratio' in df.columns:
            axes[1, 1].plot(df['date'], df['put_vol_oi_ratio'], 
                           label='Put V/OI Ratio', linewidth=2, color='green')
            axes[1, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, 
                             label='Hedging Threshold (0.3)')
            axes[1, 1].set_title('Put Volume/Open Interest Ratio')
            axes[1, 1].set_ylabel('V/OI Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """Main analysis function"""
    
    # Initialize analyzer
    analyzer = HedgingBuildupAnalyzer()
    
    # Analyze all of 2025
    end_date = '2025-09-30'  # Latest available data
    start_date = '2025-01-01'  # Full year
    
    print("🔍 Analyzing SPY Hedging Buildup...")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 50)
    
    # Run analysis
    df = analyzer.analyze_recent_period(start_date, end_date)
    
    if df.empty:
        print("❌ No data found for analysis")
        return
    
    # Generate report
    report = analyzer.generate_buildup_report(df)
    print(report)
    
    # Create plots
    analyzer.plot_buildup_analysis(df, 'hedging_buildup_analysis.png')
    
    # Check for new activity on September 30th
    print("\n" + "="*50)
    print("🔍 CHECKING FOR NEW ACTIVITY ON SEPTEMBER 30TH")
    print("="*50)
    
    new_activity = analyzer.check_new_activity('2025-09-30')
    if new_activity:
        print(f"September 30th Analysis:")
        print(f"Buildup Score: {new_activity.get('buildup_score', 0):.1f}/100")
        print(f"Risk Level: {new_activity.get('risk_level', 'UNKNOWN')}")
        
        if 'changes' in new_activity:
            print("\nKey Changes vs Recent Average:")
            changes = new_activity['changes']
            for key, value in changes.items():
                if 'change_pct' in key and abs(value) > 10:  # Only show significant changes
                    print(f"  {key}: {value:.1f}%")
        
        if 'top_risk_strikes' in new_activity:
            print("\nTop Risk Strikes on Sep 30:")
            for i, strike_data in enumerate(new_activity['top_risk_strikes'][:3]):
                strike = strike_data['strike']
                oi = strike_data['oi_proxy']
                print(f"  {i+1}. ${strike:.0f}: {oi:,.0f} OI")
    
    # Save results
    df.to_csv('hedging_buildup_results.csv', index=False)
    print(f"\n💾 Results saved to 'hedging_buildup_results.csv'")


if __name__ == "__main__":
    main()
