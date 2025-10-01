"""
Sustained vs Snapback Analysis
==============================

This analysis determines if current hedging patterns indicate:
1. SUSTAINED DRAWDOWN (similar to other major pullbacks)
2. QUICK SNAPBACK (temporary dip)

Key question: What makes this look similar to other pullback instances?

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

class SustainedVsSnapbackAnalyzer:
    """
    Analyzes whether current patterns indicate sustained drawdown or quick snapback
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
    
    def analyze_sustained_vs_snapback_signals(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze signals that distinguish sustained drawdowns from quick snapbacks"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return {}
        
        analysis = {}
        analysis['spy_price'] = spy_price
        analysis['date'] = df['date'].iloc[0] if 'date' in df.columns else pd.Timestamp.now()
        
        # 1. SUSTAINED DRAWDOWN SIGNALS
        
        # A. Deep OTM Put Accumulation (crash protection)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.80]  # 20%+ below
        if not deep_otm_puts.empty:
            deep_otm_oi = deep_otm_puts['oi_proxy'].sum()
            deep_otm_vol = deep_otm_puts['volume'].sum()
            analysis['deep_otm_oi'] = deep_otm_oi
            analysis['deep_otm_vol'] = deep_otm_vol
            analysis['deep_otm_vol_oi'] = deep_otm_vol / (deep_otm_oi + 1e-6)
        else:
            analysis['deep_otm_oi'] = 0
            analysis['deep_otm_vol'] = 0
            analysis['deep_otm_vol_oi'] = 0
        
        # B. Long-dated Put Accumulation (institutional conviction)
        if 'dte' in df.columns:
            long_dated_puts = puts[puts['dte'] > 90]  # 3+ months
            if not long_dated_puts.empty:
                long_dated_oi = long_dated_puts['oi_proxy'].sum()
                long_dated_vol = long_dated_puts['volume'].sum()
                analysis['long_dated_oi'] = long_dated_oi
                analysis['long_dated_vol'] = long_dated_vol
                analysis['long_dated_vol_oi'] = long_dated_vol / (long_dated_oi + 1e-6)
            else:
                analysis['long_dated_oi'] = 0
                analysis['long_dated_vol'] = 0
                analysis['long_dated_vol_oi'] = 0
        else:
            analysis['long_dated_oi'] = 0
            analysis['long_dated_vol'] = 0
            analysis['long_dated_vol_oi'] = 0
        
        # C. Strike Concentration (focused downside targeting)
        strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
        top_5_oi = strike_oi.head(5).sum()
        total_put_oi = puts['oi_proxy'].sum()
        concentration = top_5_oi / (total_put_oi + 1e-6)
        
        analysis['strike_concentration'] = concentration
        analysis['top_5_oi'] = top_5_oi
        
        # D. Put/Call Ratio (bearish sentiment)
        if not calls.empty:
            pc_ratio = total_put_oi / calls['oi_proxy'].sum()
            analysis['pc_ratio'] = pc_ratio
        else:
            analysis['pc_ratio'] = 1.0
        
        # E. Volume/OI Ratio (accumulation vs speculation)
        total_put_vol = puts['volume'].sum()
        vol_oi_ratio = total_put_vol / (total_put_oi + 1e-6)
        analysis['vol_oi_ratio'] = vol_oi_ratio
        
        # 2. QUICK SNAPBACK SIGNALS
        
        # A. ATM Put Activity (short-term hedging)
        atm_puts = puts[abs(puts['strike'] - spy_price) <= 10]
        if not atm_puts.empty:
            atm_oi = atm_puts['oi_proxy'].sum()
            atm_vol = atm_puts['volume'].sum()
            analysis['atm_oi'] = atm_oi
            analysis['atm_vol'] = atm_vol
            analysis['atm_vol_oi'] = atm_vol / (atm_oi + 1e-6)
        else:
            analysis['atm_oi'] = 0
            analysis['atm_vol'] = 0
            analysis['atm_vol_oi'] = 0
        
        # B. Short-dated Put Activity (immediate hedging)
        if 'dte' in df.columns:
            short_dated_puts = puts[puts['dte'] <= 30]  # 1 month or less
            if not short_dated_puts.empty:
                short_dated_oi = short_dated_puts['oi_proxy'].sum()
                short_dated_vol = short_dated_puts['volume'].sum()
                analysis['short_dated_oi'] = short_dated_oi
                analysis['short_dated_vol'] = short_dated_vol
                analysis['short_dated_vol_oi'] = short_dated_vol / (short_dated_oi + 1e-6)
            else:
                analysis['short_dated_oi'] = 0
                analysis['short_dated_vol'] = 0
                analysis['short_dated_vol_oi'] = 0
        else:
            analysis['short_dated_oi'] = 0
            analysis['short_dated_vol'] = 0
            analysis['short_dated_vol_oi'] = 0
        
        # C. High Volume/OI Ratio (speculative activity)
        analysis['high_vol_oi'] = 1 if vol_oi_ratio > 1.5 else 0
        
        # 3. COMBINED SIGNAL ANALYSIS
        
        # Sustained drawdown signals
        sustained_signals = [
            1 if analysis['deep_otm_oi'] > 500000 else 0,  # High deep OTM OI
            1 if analysis['long_dated_oi'] > 300000 else 0,  # High long-dated OI
            1 if analysis['strike_concentration'] > 0.6 else 0,  # High concentration
            1 if analysis['pc_ratio'] > 1.5 else 0,  # High PC ratio
            1 if analysis['vol_oi_ratio'] < 0.5 else 0  # Low V/OI (accumulation)
        ]
        
        # Quick snapback signals
        snapback_signals = [
            1 if analysis['atm_oi'] > 200000 else 0,  # High ATM OI
            1 if analysis['short_dated_oi'] > 300000 else 0,  # High short-dated OI
            1 if analysis['high_vol_oi'] else 0,  # High V/OI (speculation)
            1 if analysis['pc_ratio'] < 1.2 else 0,  # Low PC ratio
            1 if analysis['vol_oi_ratio'] > 1.0 else 0  # High V/OI (speculation)
        ]
        
        analysis['sustained_signal_count'] = sum(sustained_signals)
        analysis['snapback_signal_count'] = sum(snapback_signals)
        analysis['is_sustained_signal'] = 1 if sum(sustained_signals) >= 3 else 0
        analysis['is_snapback_signal'] = 1 if sum(snapback_signals) >= 3 else 0
        
        return analysis
    
    def build_sustained_vs_snapback_analysis(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build sustained vs snapback analysis"""
        
        print(f"📊 BUILDING SUSTAINED VS SNAPBACK ANALYSIS")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        analysis_data = []
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    analysis = self.analyze_sustained_vs_snapback_signals(df, spy_price)
                    
                    if analysis:
                        analysis_data.append(analysis)
            
            current_date += timedelta(days=1)
        
        if analysis_data:
            df_analysis = pd.DataFrame(analysis_data)
            
            # Calculate rolling averages
            df_analysis['sustained_signal_ma_10'] = df_analysis['is_sustained_signal'].rolling(window=10, min_periods=1).mean()
            df_analysis['snapback_signal_ma_10'] = df_analysis['is_snapback_signal'].rolling(window=10, min_periods=1).mean()
            
            print(f"✅ Built analysis for {len(df_analysis)} trading days")
            return df_analysis
        else:
            print("❌ No data found for the specified period")
            return pd.DataFrame()
    
    def compare_to_historical_pullbacks(self, df: pd.DataFrame) -> dict:
        """Compare current patterns to historical pullback instances"""
        
        if df.empty:
            return {}
        
        comparison = {}
        
        # Define historical pullback periods (you can adjust these dates)
        historical_periods = {
            'COVID_Crash': ['2020-02-15', '2020-03-31'],
            '2022_Bear': ['2022-01-01', '2022-10-31'],
            '2018_Vol': ['2018-09-01', '2018-12-31'],
            '2016_Election': ['2016-10-01', '2016-12-31']
        }
        
        # Current period (last 30 days)
        current_period = df.tail(30)
        
        if current_period.empty:
            return comparison
        
        # Calculate current averages
        current_metrics = {
            'deep_otm_oi': current_period['deep_otm_oi'].mean(),
            'long_dated_oi': current_period['long_dated_oi'].mean(),
            'strike_concentration': current_period['strike_concentration'].mean(),
            'pc_ratio': current_period['pc_ratio'].mean(),
            'vol_oi_ratio': current_period['vol_oi_ratio'].mean(),
            'sustained_signal_count': current_period['sustained_signal_count'].mean(),
            'snapback_signal_count': current_period['snapback_signal_count'].mean()
        }
        
        comparison['current_metrics'] = current_metrics
        
        # Compare to historical periods
        historical_comparisons = {}
        
        for period_name, (start_date, end_date) in historical_periods:
            period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if not period_data.empty:
                period_metrics = {
                    'deep_otm_oi': period_data['deep_otm_oi'].mean(),
                    'long_dated_oi': period_data['long_dated_oi'].mean(),
                    'strike_concentration': period_data['strike_concentration'].mean(),
                    'pc_ratio': period_data['pc_ratio'].mean(),
                    'vol_oi_ratio': period_data['vol_oi_ratio'].mean(),
                    'sustained_signal_count': period_data['sustained_signal_count'].mean(),
                    'snapback_signal_count': period_data['snapback_signal_count'].mean()
                }
                
                # Calculate similarity scores
                similarity_scores = {}
                for metric in current_metrics:
                    if metric in period_metrics:
                        current_val = current_metrics[metric]
                        period_val = period_metrics[metric]
                        
                        if period_val != 0:
                            similarity = 1 - abs(current_val - period_val) / period_val
                            similarity_scores[metric] = max(0, similarity)
                        else:
                            similarity_scores[metric] = 0
                
                historical_comparisons[period_name] = {
                    'metrics': period_metrics,
                    'similarity_scores': similarity_scores,
                    'avg_similarity': np.mean(list(similarity_scores.values()))
                }
        
        comparison['historical_comparisons'] = historical_comparisons
        
        # Find most similar historical period
        if historical_comparisons:
            most_similar = max(historical_comparisons, 
                             key=lambda x: historical_comparisons[x]['avg_similarity'])
            comparison['most_similar_period'] = most_similar
            comparison['similarity_score'] = historical_comparisons[most_similar]['avg_similarity']
        
        return comparison
    
    def generate_sustained_vs_snapback_report(self, df: pd.DataFrame, comparison: dict) -> str:
        """Generate sustained vs snapback analysis report"""
        
        report = []
        report.append("🎯 SUSTAINED DRAWDOWN vs QUICK SNAPBACK ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        # Current signal analysis
        if not df.empty:
            recent_data = df.tail(30)  # Last 30 days
            
            sustained_days = recent_data['is_sustained_signal'].sum()
            snapback_days = recent_data['is_snapback_signal'].sum()
            total_days = len(recent_data)
            
            report.append("📊 CURRENT SIGNAL ANALYSIS (Last 30 Days):")
            report.append(f"  • Sustained drawdown signals: {sustained_days} days ({sustained_days/total_days:.1%})")
            report.append(f"  • Quick snapback signals: {snapback_days} days ({snapback_days/total_days:.1%})")
            report.append("")
            
            # Recent metrics
            report.append("📈 RECENT METRICS:")
            report.append(f"  • Deep OTM Put OI: {recent_data['deep_otm_oi'].mean():,.0f}")
            report.append(f"  • Long-dated Put OI: {recent_data['long_dated_oi'].mean():,.0f}")
            report.append(f"  • Strike Concentration: {recent_data['strike_concentration'].mean():.2f}")
            report.append(f"  • Put/Call Ratio: {recent_data['pc_ratio'].mean():.2f}")
            report.append(f"  • Volume/OI Ratio: {recent_data['vol_oi_ratio'].mean():.2f}")
            report.append("")
        
        # Historical comparison
        if 'historical_comparisons' in comparison:
            report.append("🔍 HISTORICAL COMPARISON:")
            report.append("-" * 30)
            
            for period_name, period_data in comparison['historical_comparisons'].items():
                similarity = period_data['avg_similarity']
                metrics = period_data['metrics']
                
                report.append(f"\n{period_name.replace('_', ' ').title()}:")
                report.append(f"  Similarity Score: {similarity:.2f}")
                report.append(f"  Deep OTM OI: {metrics['deep_otm_oi']:,.0f}")
                report.append(f"  Long-dated OI: {metrics['long_dated_oi']:,.0f}")
                report.append(f"  Strike Concentration: {metrics['strike_concentration']:.2f}")
                report.append(f"  Put/Call Ratio: {metrics['pc_ratio']:.2f}")
                report.append(f"  Volume/OI Ratio: {metrics['vol_oi_ratio']:.2f}")
        
        # Most similar period
        if 'most_similar_period' in comparison:
            most_similar = comparison['most_similar_period']
            similarity_score = comparison['similarity_score']
            
            report.append(f"\n🎯 MOST SIMILAR HISTORICAL PERIOD: {most_similar.replace('_', ' ').title()}")
            report.append(f"   Similarity Score: {similarity_score:.2f}")
            report.append("")
        
        # Signal interpretation
        report.append("🔍 SIGNAL INTERPRETATION:")
        report.append("-" * 25)
        
        if not df.empty:
            recent_data = df.tail(30)
            avg_sustained = recent_data['sustained_signal_count'].mean()
            avg_snapback = recent_data['snapback_signal_count'].mean()
            
            if avg_sustained > avg_snapback:
                report.append("✅ SUSTAINED DRAWDOWN SIGNALS DOMINATE")
                report.append("")
                report.append("What this means:")
                report.append("• Deep OTM put accumulation (crash protection)")
                report.append("• Long-dated institutional positioning")
                report.append("• High strike concentration (focused targeting)")
                report.append("• Low V/OI ratio (accumulation, not speculation)")
                report.append("• High Put/Call ratio (bearish sentiment)")
                report.append("")
                report.append("Expected outcome:")
                report.append("• Sustained pullback similar to historical periods")
                report.append("• Multiple support level tests")
                report.append("• Extended time to recovery")
                
            elif avg_snapback > avg_sustained:
                report.append("✅ QUICK SNAPBACK SIGNALS DOMINATE")
                report.append("")
                report.append("What this means:")
                report.append("• High ATM put activity (short-term hedging)")
                report.append("• Short-dated positioning (immediate protection)")
                report.append("• High V/OI ratio (speculative activity)")
                report.append("• Lower Put/Call ratio (less bearish)")
                report.append("")
                report.append("Expected outcome:")
                report.append("• Quick dip followed by snapback")
                report.append("• V-shaped recovery")
                report.append("• Short-lived pullback")
                
            else:
                report.append("⚠️  MIXED SIGNALS")
                report.append("")
                report.append("What this means:")
                report.append("• Both sustained and snapback signals present")
                report.append("• Unclear direction")
                report.append("• Wait for clearer signals")
        
        # Historical context
        if 'most_similar_period' in comparison:
            most_similar = comparison['most_similar_period']
            report.append(f"\n📚 HISTORICAL CONTEXT:")
            report.append(f"Current patterns most similar to: {most_similar.replace('_', ' ').title()}")
            
            if 'COVID' in most_similar:
                report.append("• This was a sustained crash period")
                report.append("• Multiple support level breaks")
                report.append("• Extended recovery time")
            elif '2022' in most_similar:
                report.append("• This was a sustained bear market")
                report.append("• Gradual decline over months")
                report.append("• Multiple bounce attempts")
            elif '2018' in most_similar:
                report.append("• This was a volatility spike period")
                report.append("• Sharp but short-lived decline")
                report.append("• Quick recovery")
            elif '2016' in most_similar:
                report.append("• This was an election uncertainty period")
                report.append("• Moderate pullback")
                report.append("• Post-election recovery")
        
        return "\n".join(report)


def main():
    """Main sustained vs snapback analysis"""
    
    # Initialize analyzer
    analyzer = SustainedVsSnapbackAnalyzer()
    
    print("🎯 SUSTAINED DRAWDOWN vs QUICK SNAPBACK ANALYSIS")
    print("What type of pullback are we looking at?")
    print("=" * 60)
    
    # Build analysis for 2024-2025
    df_analysis = analyzer.build_sustained_vs_snapback_analysis('2024-01-01', '2025-09-30')
    
    if df_analysis.empty:
        print("❌ No data available")
        return
    
    # Compare to historical pullbacks
    comparison = analyzer.compare_to_historical_pullbacks(df_analysis)
    
    # Generate report
    report = analyzer.generate_sustained_vs_snapback_report(df_analysis, comparison)
    print(report)
    
    # Save data
    df_analysis.to_csv('sustained_vs_snapback_data.csv', index=False)
    with open('sustained_vs_snapback_analysis.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Data and analysis saved")


if __name__ == "__main__":
    main()
