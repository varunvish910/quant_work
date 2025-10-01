"""
Hedging Aggression Analysis
==========================

This analysis determines if current hedging activity is:
1. MORE AGGRESSIVE than normal (anomaly)
2. FOLLOWING NORMAL PATTERNS (not an anomaly)

Key question: Are institutions being more aggressive with hedging,
or are we just seeing normal 10% rolling hedge patterns?

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

class HedgingAggressionAnalyzer:
    """
    Analyzes whether current hedging is more aggressive than normal
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
    
    def analyze_hedging_aggression(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze hedging aggression levels"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return {}
        
        analysis = {}
        analysis['spy_price'] = spy_price
        analysis['date'] = df['date'].iloc[0] if 'date' in df.columns else pd.Timestamp.now()
        
        # 1. HEDGING INTENSITY ANALYSIS
        
        # A. Total Put OI (absolute hedging level)
        total_put_oi = puts['oi_proxy'].sum()
        analysis['total_put_oi'] = total_put_oi
        
        # B. Put OI as % of total options OI
        total_call_oi = calls['oi_proxy'].sum() if not calls.empty else 0
        total_options_oi = total_put_oi + total_call_oi
        put_oi_pct = total_put_oi / (total_options_oi + 1e-6)
        analysis['put_oi_pct'] = put_oi_pct
        
        # C. Put/Call Ratio (relative hedging)
        pc_ratio = total_put_oi / (total_call_oi + 1e-6)
        analysis['pc_ratio'] = pc_ratio
        
        # 2. HEDGING DEPTH ANALYSIS (How far out are they hedging?)
        
        # A. 10% below hedging (normal rolling hedge)
        ten_pct_below = spy_price * 0.90
        ten_pct_puts = puts[abs(puts['strike'] - ten_pct_below) <= 5]
        if not ten_pct_puts.empty:
            ten_pct_oi = ten_pct_puts['oi_proxy'].sum()
            analysis['10%_below_oi'] = ten_pct_oi
            analysis['10%_below_pct'] = ten_pct_oi / (total_put_oi + 1e-6)
        else:
            analysis['10%_below_oi'] = 0
            analysis['10%_below_pct'] = 0
        
        # B. 15% below hedging (deeper protection)
        fifteen_pct_below = spy_price * 0.85
        fifteen_pct_puts = puts[abs(puts['strike'] - fifteen_pct_below) <= 5]
        if not fifteen_pct_puts.empty:
            fifteen_pct_oi = fifteen_pct_puts['oi_proxy'].sum()
            analysis['15%_below_oi'] = fifteen_pct_oi
            analysis['15%_below_pct'] = fifteen_pct_oi / (total_put_oi + 1e-6)
        else:
            analysis['15%_below_oi'] = 0
            analysis['15%_below_pct'] = 0
        
        # C. 20% below hedging (crash protection)
        twenty_pct_below = spy_price * 0.80
        twenty_pct_puts = puts[abs(puts['strike'] - twenty_pct_below) <= 5]
        if not twenty_pct_puts.empty:
            twenty_pct_oi = twenty_pct_puts['oi_proxy'].sum()
            analysis['20%_below_oi'] = twenty_pct_oi
            analysis['20%_below_pct'] = twenty_pct_oi / (total_put_oi + 1e-6)
        else:
            analysis['20%_below_oi'] = 0
            analysis['20%_below_pct'] = 0
        
        # D. Deep OTM hedging (25%+ below)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.75]
        if not deep_otm_puts.empty:
            deep_otm_oi = deep_otm_puts['oi_proxy'].sum()
            analysis['deep_otm_oi'] = deep_otm_oi
            analysis['deep_otm_pct'] = deep_otm_oi / (total_put_oi + 1e-6)
        else:
            analysis['deep_otm_oi'] = 0
            analysis['deep_otm_pct'] = 0
        
        # 3. HEDGING DURATION ANALYSIS (How long are they hedging for?)
        
        if 'dte' in df.columns:
            # A. Short-term hedging (0-30 DTE)
            short_term_puts = puts[puts['dte'] <= 30]
            if not short_term_puts.empty:
                short_term_oi = short_term_puts['oi_proxy'].sum()
                analysis['short_term_oi'] = short_term_oi
                analysis['short_term_pct'] = short_term_oi / (total_put_oi + 1e-6)
            else:
                analysis['short_term_oi'] = 0
                analysis['short_term_pct'] = 0
            
            # B. Medium-term hedging (31-90 DTE)
            medium_term_puts = puts[(puts['dte'] > 30) & (puts['dte'] <= 90)]
            if not medium_term_puts.empty:
                medium_term_oi = medium_term_puts['oi_proxy'].sum()
                analysis['medium_term_oi'] = medium_term_oi
                analysis['medium_term_pct'] = medium_term_oi / (total_put_oi + 1e-6)
            else:
                analysis['medium_term_oi'] = 0
                analysis['medium_term_pct'] = 0
            
            # C. Long-term hedging (90+ DTE)
            long_term_puts = puts[puts['dte'] > 90]
            if not long_term_puts.empty:
                long_term_oi = long_term_puts['oi_proxy'].sum()
                analysis['long_term_oi'] = long_term_oi
                analysis['long_term_pct'] = long_term_oi / (total_put_oi + 1e-6)
            else:
                analysis['long_term_oi'] = 0
                analysis['long_term_pct'] = 0
        else:
            analysis['short_term_oi'] = 0
            analysis['short_term_pct'] = 0
            analysis['medium_term_oi'] = 0
            analysis['medium_term_pct'] = 0
            analysis['long_term_oi'] = 0
            analysis['long_term_pct'] = 0
        
        # 4. HEDGING CONCENTRATION ANALYSIS (Are they focused or spread out?)
        
        # A. Strike concentration (focused vs spread out)
        strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
        top_5_oi = strike_oi.head(5).sum()
        concentration = top_5_oi / (total_put_oi + 1e-6)
        analysis['strike_concentration'] = concentration
        
        # B. Volume/OI ratio (accumulation vs speculation)
        total_put_vol = puts['volume'].sum()
        vol_oi_ratio = total_put_vol / (total_put_oi + 1e-6)
        analysis['vol_oi_ratio'] = vol_oi_ratio
        
        # 5. AGGRESSION SCORING
        
        # Calculate aggression score (0-100)
        aggression_factors = []
        
        # Factor 1: High total OI (0-25 points)
        if total_put_oi > 2000000:  # Above 2M OI
            aggression_factors.append(25)
        elif total_put_oi > 1500000:  # Above 1.5M OI
            aggression_factors.append(20)
        elif total_put_oi > 1000000:  # Above 1M OI
            aggression_factors.append(15)
        else:
            aggression_factors.append(10)
        
        # Factor 2: High Put/Call ratio (0-20 points)
        if pc_ratio > 2.0:
            aggression_factors.append(20)
        elif pc_ratio > 1.5:
            aggression_factors.append(15)
        elif pc_ratio > 1.2:
            aggression_factors.append(10)
        else:
            aggression_factors.append(5)
        
        # Factor 3: Deep hedging (0-20 points)
        deep_hedging_score = 0
        if analysis['15%_below_pct'] > 0.1:  # 10%+ at 15% below
            deep_hedging_score += 10
        if analysis['20%_below_pct'] > 0.05:  # 5%+ at 20% below
            deep_hedging_score += 10
        aggression_factors.append(deep_hedging_score)
        
        # Factor 4: Long-term positioning (0-15 points)
        if 'long_term_pct' in analysis and analysis['long_term_pct'] > 0.3:  # 30%+ long-term
            aggression_factors.append(15)
        elif 'long_term_pct' in analysis and analysis['long_term_pct'] > 0.2:  # 20%+ long-term
            aggression_factors.append(10)
        else:
            aggression_factors.append(5)
        
        # Factor 5: High concentration (0-10 points)
        if concentration > 0.6:  # 60%+ in top 5 strikes
            aggression_factors.append(10)
        elif concentration > 0.4:  # 40%+ in top 5 strikes
            aggression_factors.append(5)
        else:
            aggression_factors.append(0)
        
        # Factor 6: Low V/OI ratio (accumulation) (0-10 points)
        if vol_oi_ratio < 0.5:  # Low V/OI = accumulation
            aggression_factors.append(10)
        elif vol_oi_ratio < 1.0:  # Moderate V/OI
            aggression_factors.append(5)
        else:
            aggression_factors.append(0)
        
        total_aggression = sum(aggression_factors)
        analysis['aggression_score'] = min(100, total_aggression)
        
        # Determine aggression level
        if total_aggression >= 80:
            analysis['aggression_level'] = 'EXTREME'
        elif total_aggression >= 60:
            analysis['aggression_level'] = 'HIGH'
        elif total_aggression >= 40:
            analysis['aggression_level'] = 'MODERATE'
        elif total_aggression >= 20:
            analysis['aggression_level'] = 'LOW'
        else:
            analysis['aggression_level'] = 'NORMAL'
        
        return analysis
    
    def build_aggression_analysis(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build hedging aggression analysis"""
        
        print(f"📊 BUILDING HEDGING AGGRESSION ANALYSIS")
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
                    analysis = self.analyze_hedging_aggression(df, spy_price)
                    
                    if analysis:
                        analysis_data.append(analysis)
            
            current_date += timedelta(days=1)
        
        if analysis_data:
            df_analysis = pd.DataFrame(analysis_data)
            
            # Calculate rolling averages
            df_analysis['aggression_ma_10'] = df_analysis['aggression_score'].rolling(window=10, min_periods=1).mean()
            df_analysis['aggression_ma_30'] = df_analysis['aggression_score'].rolling(window=30, min_periods=1).mean()
            
            print(f"✅ Built aggression analysis for {len(df_analysis)} trading days")
            return df_analysis
        else:
            print("❌ No data found for the specified period")
            return pd.DataFrame()
    
    def analyze_aggression_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze aggression patterns to determine if current activity is normal"""
        
        if df.empty:
            return {}
        
        patterns = {}
        
        # 1. Current vs Historical Aggression
        recent_data = df.tail(30)  # Last 30 days
        historical_data = df.iloc[:-30]  # All but last 30 days
        
        if not historical_data.empty:
            current_avg_aggression = recent_data['aggression_score'].mean()
            historical_avg_aggression = historical_data['aggression_score'].mean()
            
            patterns['current_avg_aggression'] = current_avg_aggression
            patterns['historical_avg_aggression'] = historical_avg_aggression
            patterns['aggression_difference'] = current_avg_aggression - historical_avg_aggression
            patterns['aggression_ratio'] = current_avg_aggression / (historical_avg_aggression + 1e-6)
        
        # 2. Aggression Level Distribution
        aggression_levels = df['aggression_level'].value_counts()
        patterns['aggression_distribution'] = aggression_levels.to_dict()
        
        # 3. Recent Aggression Trend
        if len(recent_data) >= 10:
            recent_trend = recent_data['aggression_score'].rolling(window=5, min_periods=1).mean()
            patterns['recent_trend'] = recent_trend.iloc[-1] - recent_trend.iloc[0]
        
        # 4. Normal vs Anomaly Classification
        if 'aggression_ratio' in patterns:
            ratio = patterns['aggression_ratio']
            if ratio > 1.5:
                patterns['classification'] = 'MORE_AGGRESSIVE'
            elif ratio > 1.2:
                patterns['classification'] = 'SLIGHTLY_MORE_AGGRESSIVE'
            elif ratio > 0.8:
                patterns['classification'] = 'NORMAL'
            else:
                patterns['classification'] = 'LESS_AGGRESSIVE'
        
        return patterns
    
    def generate_aggression_report(self, df: pd.DataFrame, patterns: dict) -> str:
        """Generate hedging aggression analysis report"""
        
        report = []
        report.append("🎯 HEDGING AGGRESSION ANALYSIS")
        report.append("=" * 50)
        report.append("")
        
        # Key question
        report.append("❓ KEY QUESTION:")
        report.append("Are institutions being more aggressive with hedging,")
        report.append("or are we just seeing normal 10% rolling hedge patterns?")
        report.append("")
        
        # Current vs Historical Analysis
        if 'current_avg_aggression' in patterns and 'historical_avg_aggression' in patterns:
            current = patterns['current_avg_aggression']
            historical = patterns['historical_avg_aggression']
            difference = patterns['aggression_difference']
            ratio = patterns['aggression_ratio']
            
            report.append("📊 CURRENT vs HISTORICAL AGGRESSION:")
            report.append(f"  • Current Average: {current:.1f}")
            report.append(f"  • Historical Average: {historical:.1f}")
            report.append(f"  • Difference: {difference:+.1f}")
            report.append(f"  • Ratio: {ratio:.2f}x")
            report.append("")
            
            # Classification
            if 'classification' in patterns:
                classification = patterns['classification']
                report.append(f"🎯 CLASSIFICATION: {classification.replace('_', ' ').title()}")
                report.append("")
                
                if classification == 'MORE_AGGRESSIVE':
                    report.append("✅ INSTITUTIONS ARE BEING MORE AGGRESSIVE")
                    report.append("   → This IS an anomaly")
                    report.append("   → Hedging activity exceeds normal patterns")
                    report.append("   → May indicate elevated risk concerns")
                elif classification == 'SLIGHTLY_MORE_AGGRESSIVE':
                    report.append("⚠️  INSTITUTIONS ARE SLIGHTLY MORE AGGRESSIVE")
                    report.append("   → This is a minor anomaly")
                    report.append("   → Hedging activity moderately above normal")
                    report.append("   → Worth monitoring but not extreme")
                elif classification == 'NORMAL':
                    report.append("✅ INSTITUTIONS ARE FOLLOWING NORMAL PATTERNS")
                    report.append("   → This is NOT an anomaly")
                    report.append("   → Hedging activity within normal ranges")
                    report.append("   → Standard 10% rolling hedge behavior")
                else:
                    report.append("ℹ️  INSTITUTIONS ARE LESS AGGRESSIVE")
                    report.append("   → Hedging activity below normal")
                    report.append("   → May indicate reduced risk concerns")
        
        # Recent Trend Analysis
        if 'recent_trend' in patterns:
            trend = patterns['recent_trend']
            report.append("📈 RECENT TREND (Last 30 Days):")
            if trend > 5:
                report.append(f"  • Aggression increasing: +{trend:.1f}")
                report.append("  • Institutions becoming more defensive")
            elif trend < -5:
                report.append(f"  • Aggression decreasing: {trend:.1f}")
                report.append("  • Institutions becoming less defensive")
            else:
                report.append(f"  • Aggression stable: {trend:+.1f}")
                report.append("  • No significant change in hedging behavior")
            report.append("")
        
        # Aggression Level Distribution
        if 'aggression_distribution' in patterns:
            distribution = patterns['aggression_distribution']
            report.append("📊 AGGRESSION LEVEL DISTRIBUTION:")
            for level, count in distribution.items():
                pct = count / sum(distribution.values()) * 100
                report.append(f"  • {level}: {count} days ({pct:.1f}%)")
            report.append("")
        
        # Detailed Analysis
        if not df.empty:
            recent_data = df.tail(30)
            
            report.append("🔍 DETAILED ANALYSIS (Last 30 Days):")
            report.append("-" * 35)
            
            # Total OI
            avg_total_oi = recent_data['total_put_oi'].mean()
            report.append(f"• Average Total Put OI: {avg_total_oi:,.0f}")
            
            # Put/Call Ratio
            avg_pc_ratio = recent_data['pc_ratio'].mean()
            report.append(f"• Average Put/Call Ratio: {avg_pc_ratio:.2f}")
            
            # Hedging Depth
            avg_10_pct = recent_data['10%_below_pct'].mean()
            avg_15_pct = recent_data['15%_below_pct'].mean()
            avg_20_pct = recent_data['20%_below_pct'].mean()
            
            report.append(f"• 10% Below Hedging: {avg_10_pct:.1%}")
            report.append(f"• 15% Below Hedging: {avg_15_pct:.1%}")
            report.append(f"• 20% Below Hedging: {avg_20_pct:.1%}")
            
            # Duration Analysis
            if 'long_term_pct' in recent_data.columns:
                avg_long_term = recent_data['long_term_pct'].mean()
                report.append(f"• Long-term Hedging: {avg_long_term:.1%}")
            
            # Concentration
            avg_concentration = recent_data['strike_concentration'].mean()
            report.append(f"• Strike Concentration: {avg_concentration:.1%}")
            
            # V/OI Ratio
            avg_vol_oi = recent_data['vol_oi_ratio'].mean()
            report.append(f"• Volume/OI Ratio: {avg_vol_oi:.2f}")
            report.append("")
        
        # Conclusion
        report.append("🎯 CONCLUSION:")
        report.append("-" * 15)
        
        if 'classification' in patterns:
            classification = patterns['classification']
            
            if classification == 'NORMAL':
                report.append("✅ CURRENT HEDGING IS NORMAL")
                report.append("")
                report.append("What this means:")
                report.append("• Institutions are following standard patterns")
                report.append("• 10% rolling hedge behavior is normal")
                report.append("• No anomaly detected")
                report.append("• This is NOT a signal for market pullback")
                report.append("")
                report.append("Trading implication:")
                report.append("• Continue normal market analysis")
                report.append("• Don't overweight hedging data")
                report.append("• Look for other signals")
                
            elif classification in ['MORE_AGGRESSIVE', 'SLIGHTLY_MORE_AGGRESSIVE']:
                report.append("⚠️  CURRENT HEDGING IS MORE AGGRESSIVE")
                report.append("")
                report.append("What this means:")
                report.append("• Institutions are hedging more than usual")
                report.append("• This IS an anomaly")
                report.append("• May indicate elevated risk concerns")
                report.append("• Could be a signal for market stress")
                report.append("")
                report.append("Trading implication:")
                report.append("• Monitor for additional stress signals")
                report.append("• Consider defensive positioning")
                report.append("• Watch for confirmation from other indicators")
        
        return "\n".join(report)


def main():
    """Main hedging aggression analysis"""
    
    # Initialize analyzer
    analyzer = HedgingAggressionAnalyzer()
    
    print("🎯 HEDGING AGGRESSION ANALYSIS")
    print("Are institutions being more aggressive or following normal patterns?")
    print("=" * 70)
    
    # Build analysis for 2024-2025
    df_analysis = analyzer.build_aggression_analysis('2024-01-01', '2025-09-30')
    
    if df_analysis.empty:
        print("❌ No data available")
        return
    
    # Analyze patterns
    patterns = analyzer.analyze_aggression_patterns(df_analysis)
    
    # Generate report
    report = analyzer.generate_aggression_report(df_analysis, patterns)
    print(report)
    
    # Save data
    df_analysis.to_csv('hedging_aggression_data.csv', index=False)
    with open('hedging_aggression_analysis.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Data and analysis saved")


if __name__ == "__main__":
    main()
