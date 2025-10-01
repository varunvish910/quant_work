"""
Pre-Correction Signal Analyzer
==============================

Analyzes hedging patterns 2-4 weeks BEFORE corrections start to determine:
1. Is the options market signaling coming weakness?
2. Or is it just setting defensive floors?

Focus: Lead-up period analysis, not the correction itself.

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

class PreCorrectionSignalAnalyzer:
    """
    Analyzes hedging patterns before corrections to distinguish
    between weakness signaling vs defensive floor setting
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
    
    def identify_correction_start_dates(self, start_date: str, end_date: str) -> list:
        """Identify when corrections actually STARTED (not when they completed)"""
        
        print(f"🔍 IDENTIFYING CORRECTION START DATES")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        # Build price data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        price_data = []
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    price_data.append({
                        'date': current_date,
                        'spy_price': spy_price
                    })
            
            current_date += timedelta(days=1)
        
        if not price_data:
            return []
        
        df_prices = pd.DataFrame(price_data)
        df_prices = df_prices.sort_values('date').reset_index(drop=True)
        
        # Find correction start dates (when decline began)
        correction_starts = []
        
        for i in range(20, len(df_prices) - 10):  # Look ahead 10 days for confirmation
            current_price = df_prices.iloc[i]['spy_price']
            current_date = df_prices.iloc[i]['date']
            
            # Look ahead 10 days to see if correction occurs
            future_prices = df_prices.iloc[i+1:i+11]['spy_price']
            if len(future_prices) >= 5:  # Need at least 5 days ahead
                min_future_price = future_prices.min()
                max_decline = (current_price - min_future_price) / current_price
                
                # Check if this is a 5-10% correction
                if 0.05 <= max_decline <= 0.10:
                    # Find the actual start of decline (when price peaked)
                    lookback_prices = df_prices.iloc[i-20:i]['spy_price']
                    peak_price = lookback_prices.max()
                    peak_idx = df_prices[df_prices['spy_price'] == peak_price].index[-1]
                    peak_date = df_prices.iloc[peak_idx]['date']
                    
                    # Only add if we haven't already identified this correction
                    if not any(abs((peak_date - existing['start_date']).days) < 5 
                              for existing in correction_starts):
                        correction_starts.append({
                            'start_date': peak_date,
                            'peak_price': peak_price,
                            'trough_price': min_future_price,
                            'decline_pct': max_decline,
                            'duration_days': (df_prices.iloc[i+10]['date'] - peak_date).days
                        })
        
        print(f"✅ Found {len(correction_starts)} correction start dates")
        return correction_starts
    
    def analyze_pre_correction_period(self, correction_start: dict, lookback_weeks: int = 4) -> dict:
        """Analyze hedging patterns 2-4 weeks BEFORE correction starts"""
        
        start_date = correction_start['start_date']
        pre_start = start_date - timedelta(weeks=lookback_weeks)
        
        print(f"📊 Analyzing {lookback_weeks} weeks before correction starting {start_date.strftime('%Y-%m-%d')}")
        
        # Build pre-correction data
        pre_correction_data = []
        current_date = pre_start
        
        while current_date < start_date:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    analysis = self.analyze_hedging_signals(df, spy_price)
                    if analysis:
                        analysis['date'] = current_date
                        analysis['days_before_correction'] = (start_date - current_date).days
                        pre_correction_data.append(analysis)
            
            current_date += timedelta(days=1)
        
        if not pre_correction_data:
            return {}
        
        df_pre = pd.DataFrame(pre_correction_data)
        
        # Analyze patterns
        patterns = {
            'correction_start': start_date,
            'decline_pct': correction_start['decline_pct'],
            'lookback_weeks': lookback_weeks,
            'data_points': len(df_pre)
        }
        
        # Signal vs Floor Analysis
        patterns['signal_indicators'] = self.analyze_signal_indicators(df_pre)
        patterns['floor_indicators'] = self.analyze_floor_indicators(df_pre)
        patterns['signal_vs_floor_score'] = self.calculate_signal_vs_floor_score(patterns)
        
        return patterns
    
    def analyze_hedging_signals(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze hedging signals that distinguish weakness signaling from floor setting"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return {}
        
        analysis = {}
        
        # Basic metrics
        total_put_oi = puts['oi_proxy'].sum()
        total_call_oi = calls['oi_proxy'].sum() if not calls.empty else 0
        total_put_vol = puts['volume'].sum()
        
        analysis['total_put_oi'] = total_put_oi
        analysis['pc_ratio'] = total_put_oi / (total_call_oi + 1e-6)
        analysis['vol_oi_ratio'] = total_put_vol / (total_put_oi + 1e-6)
        
        # Institutional vs Retail
        if 'dte' in df.columns:
            institutional_puts = puts[puts['dte'] > 7]
            retail_puts = puts[puts['dte'] <= 7]
            
            analysis['institutional_pct'] = institutional_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
            analysis['retail_pct'] = retail_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        else:
            analysis['institutional_pct'] = 0
            analysis['retail_pct'] = 0
        
        # Hedging depth analysis
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.80]  # 20%+ below
        medium_otm_puts = puts[(puts['strike'] < spy_price * 0.95) & (puts['strike'] >= spy_price * 0.80)]
        atm_puts = puts[abs(puts['strike'] - spy_price) <= 10]
        
        analysis['deep_otm_pct'] = deep_otm_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        analysis['medium_otm_pct'] = medium_otm_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        analysis['atm_pct'] = atm_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        
        # Duration analysis
        if 'dte' in df.columns:
            short_term_puts = puts[puts['dte'] <= 30]
            medium_term_puts = puts[(puts['dte'] > 30) & (puts['dte'] <= 90)]
            long_term_puts = puts[puts['dte'] > 90]
            
            analysis['short_term_pct'] = short_term_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
            analysis['medium_term_pct'] = medium_term_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
            analysis['long_term_pct'] = long_term_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        else:
            analysis['short_term_pct'] = 0
            analysis['medium_term_pct'] = 0
            analysis['long_term_pct'] = 0
        
        # Strike concentration
        strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
        top_5_oi = strike_oi.head(5).sum()
        analysis['strike_concentration'] = top_5_oi / (total_put_oi + 1e-6)
        
        return analysis
    
    def analyze_signal_indicators(self, df: pd.DataFrame) -> dict:
        """Analyze indicators that suggest weakness signaling (not just floor setting)"""
        
        indicators = {}
        
        # 1. Rising hedging intensity over time (weakness signal)
        if 'pc_ratio' in df.columns and len(df) > 5:
            pc_trend = df['pc_ratio'].iloc[-1] - df['pc_ratio'].iloc[0]
            indicators['rising_pc_ratio'] = 1 if pc_trend > 0.1 else 0
        
        # 2. Increasing institutional positioning (smart money positioning)
        if 'institutional_pct' in df.columns and len(df) > 5:
            inst_trend = df['institutional_pct'].iloc[-1] - df['institutional_pct'].iloc[0]
            indicators['rising_institutional'] = 1 if inst_trend > 0.05 else 0
        
        # 3. Deep OTM accumulation (crash protection, not just floor)
        if 'deep_otm_pct' in df.columns:
            avg_deep_otm = df['deep_otm_pct'].mean()
            indicators['high_deep_otm'] = 1 if avg_deep_otm > 0.15 else 0
        
        # 4. Long-term positioning (conviction, not just short-term hedging)
        if 'long_term_pct' in df.columns:
            avg_long_term = df['long_term_pct'].mean()
            indicators['high_long_term'] = 1 if avg_long_term > 0.3 else 0
        
        # 5. Low V/OI ratio (accumulation, not speculation)
        if 'vol_oi_ratio' in df.columns:
            avg_vol_oi = df['vol_oi_ratio'].mean()
            indicators['low_vol_oi'] = 1 if avg_vol_oi < 0.8 else 0
        
        # 6. Strike concentration (focused targeting, not broad hedging)
        if 'strike_concentration' in df.columns:
            avg_concentration = df['strike_concentration'].mean()
            indicators['high_concentration'] = 1 if avg_concentration > 0.5 else 0
        
        return indicators
    
    def analyze_floor_indicators(self, df: pd.DataFrame) -> dict:
        """Analyze indicators that suggest defensive floor setting (not weakness signaling)"""
        
        indicators = {}
        
        # 1. Stable hedging levels (maintaining floors, not increasing)
        if 'pc_ratio' in df.columns and len(df) > 5:
            pc_std = df['pc_ratio'].std()
            indicators['stable_pc_ratio'] = 1 if pc_std < 0.1 else 0
        
        # 2. Medium-term positioning (rolling hedges, not long-term conviction)
        if 'medium_term_pct' in df.columns:
            avg_medium_term = df['medium_term_pct'].mean()
            indicators['high_medium_term'] = 1 if avg_medium_term > 0.4 else 0
        
        # 3. ATM/OTM balance (defensive positioning, not crash protection)
        if 'atm_pct' in df.columns and 'medium_otm_pct' in df.columns:
            atm_otm_balance = abs(df['atm_pct'].mean() - df['medium_otm_pct'].mean())
            indicators['balanced_positioning'] = 1 if atm_otm_balance < 0.1 else 0
        
        # 4. High V/OI ratio (active hedging, not accumulation)
        if 'vol_oi_ratio' in df.columns:
            avg_vol_oi = df['vol_oi_ratio'].mean()
            indicators['high_vol_oi'] = 1 if avg_vol_oi > 1.2 else 0
        
        # 5. Low concentration (broad hedging, not focused targeting)
        if 'strike_concentration' in df.columns:
            avg_concentration = df['strike_concentration'].mean()
            indicators['low_concentration'] = 1 if avg_concentration < 0.3 else 0
        
        return indicators
    
    def calculate_signal_vs_floor_score(self, patterns: dict) -> dict:
        """Calculate score indicating signal vs floor behavior"""
        
        signal_indicators = patterns['signal_indicators']
        floor_indicators = patterns['floor_indicators']
        
        # Count signal indicators
        signal_count = sum(signal_indicators.values())
        total_signal_indicators = len(signal_indicators)
        
        # Count floor indicators
        floor_count = sum(floor_indicators.values())
        total_floor_indicators = len(floor_indicators)
        
        # Calculate scores
        signal_score = signal_count / total_signal_indicators if total_signal_indicators > 0 else 0
        floor_score = floor_count / total_floor_indicators if total_floor_indicators > 0 else 0
        
        # Determine behavior
        if signal_score > floor_score + 0.2:
            behavior = 'WEAKNESS_SIGNALING'
            confidence = 'HIGH' if signal_score > 0.7 else 'MEDIUM'
        elif floor_score > signal_score + 0.2:
            behavior = 'FLOOR_SETTING'
            confidence = 'HIGH' if floor_score > 0.7 else 'MEDIUM'
        else:
            behavior = 'MIXED'
            confidence = 'LOW'
        
        return {
            'signal_score': signal_score,
            'floor_score': floor_score,
            'behavior': behavior,
            'confidence': confidence,
            'signal_count': signal_count,
            'floor_count': floor_count
        }
    
    def build_pre_correction_analysis(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build comprehensive pre-correction analysis"""
        
        print(f"📊 BUILDING PRE-CORRECTION SIGNAL ANALYSIS")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        # Identify correction start dates
        correction_starts = self.identify_correction_start_dates(start_date, end_date)
        
        if not correction_starts:
            print("❌ No correction starts found")
            return pd.DataFrame()
        
        # Analyze pre-correction periods
        all_patterns = []
        
        for i, correction in enumerate(correction_starts):
            print(f"\n🔍 Analyzing pre-correction period {i+1}/{len(correction_starts)}")
            patterns = self.analyze_pre_correction_period(correction, lookback_weeks=4)
            
            if patterns:
                patterns['correction_id'] = i + 1
                all_patterns.append(patterns)
        
        if all_patterns:
            df_patterns = pd.DataFrame(all_patterns)
            print(f"\n✅ Built pre-correction analysis with {len(df_patterns)} periods")
            return df_patterns
        else:
            print("❌ No pre-correction patterns found")
            return pd.DataFrame()
    
    def generate_signal_analysis_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive signal vs floor analysis report"""
        
        report = []
        report.append("🎯 PRE-CORRECTION SIGNAL vs FLOOR ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        if df.empty:
            report.append("❌ No data available for analysis")
            return "\n".join(report)
        
        # Overall statistics
        total_periods = len(df)
        weakness_signaling = len(df[df['signal_vs_floor_score'].apply(lambda x: x['behavior'] == 'WEAKNESS_SIGNALING')])
        floor_setting = len(df[df['signal_vs_floor_score'].apply(lambda x: x['behavior'] == 'FLOOR_SETTING')])
        mixed_behavior = len(df[df['signal_vs_floor_score'].apply(lambda x: x['behavior'] == 'MIXED')])
        
        report.append("📊 OVERALL STATISTICS:")
        report.append("-" * 20)
        report.append(f"• Total Pre-Correction Periods: {total_periods}")
        report.append(f"• Weakness Signaling: {weakness_signaling} ({weakness_signaling/total_periods:.1%})")
        report.append(f"• Floor Setting: {floor_setting} ({floor_setting/total_periods:.1%})")
        report.append(f"• Mixed Behavior: {mixed_behavior} ({mixed_behavior/total_periods:.1%})")
        report.append("")
        
        # Signal vs Floor Analysis
        report.append("🔍 SIGNAL vs FLOOR ANALYSIS:")
        report.append("-" * 30)
        
        # Calculate average scores
        avg_signal_score = df['signal_vs_floor_score'].apply(lambda x: x['signal_score']).mean()
        avg_floor_score = df['signal_vs_floor_score'].apply(lambda x: x['floor_score']).mean()
        
        report.append(f"• Average Signal Score: {avg_signal_score:.2f}")
        report.append(f"• Average Floor Score: {avg_floor_score:.2f}")
        report.append("")
        
        if avg_signal_score > avg_floor_score:
            report.append("✅ OPTIONS MARKET IS SIGNALING WEAKNESS")
            report.append("   → Hedging patterns suggest coming corrections")
            report.append("   → Smart money positioning for downside")
            report.append("   → Not just defensive floor setting")
        elif avg_floor_score > avg_signal_score:
            report.append("✅ OPTIONS MARKET IS SETTING FLOORS")
            report.append("   → Hedging patterns suggest defensive positioning")
            report.append("   → Risk management, not weakness signaling")
            report.append("   → Normal institutional hedging behavior")
        else:
            report.append("⚠️  MIXED BEHAVIOR DETECTED")
            report.append("   → Some periods show weakness signaling")
            report.append("   → Other periods show floor setting")
            report.append("   → Context-dependent behavior")
        
        report.append("")
        
        # Key Findings
        report.append("🎯 KEY FINDINGS:")
        report.append("-" * 15)
        
        # Most common behavior
        behavior_counts = df['signal_vs_floor_score'].apply(lambda x: x['behavior']).value_counts()
        most_common = behavior_counts.index[0]
        
        report.append(f"• Most Common Behavior: {most_common.replace('_', ' ').title()}")
        report.append(f"• Frequency: {behavior_counts.iloc[0]}/{total_periods} periods")
        report.append("")
        
        # Confidence analysis
        high_conf_signals = len(df[df['signal_vs_floor_score'].apply(lambda x: x['confidence'] == 'HIGH')])
        report.append(f"• High Confidence Classifications: {high_conf_signals}/{total_periods}")
        report.append("")
        
        # Trading Implications
        report.append("💡 TRADING IMPLICATIONS:")
        report.append("-" * 20)
        
        if most_common == 'WEAKNESS_SIGNALING':
            report.append("🚨 WEAKNESS SIGNALING DOMINATES")
            report.append("• Options market is predicting corrections")
            report.append("• Hedging patterns are leading indicators")
            report.append("• Use options data for market timing")
            report.append("• High predictive value for pullbacks")
        elif most_common == 'FLOOR_SETTING':
            report.append("🛡️  FLOOR SETTING DOMINATES")
            report.append("• Options market is defensive positioning")
            report.append("• Hedging patterns are risk management")
            report.append("• Use options data for support levels")
            report.append("• Low predictive value for timing")
        else:
            report.append("⚠️  MIXED SIGNALS")
            report.append("• Options market behavior is context-dependent")
            report.append("• Need additional analysis for each period")
            report.append("• Combine with other indicators")
            report.append("• Moderate predictive value")
        
        report.append("")
        
        # Recent Analysis (if available)
        recent_data = df.tail(5)
        if not recent_data.empty:
            report.append("📈 RECENT PERIODS ANALYSIS:")
            report.append("-" * 25)
            
            for _, row in recent_data.iterrows():
                behavior = row['signal_vs_floor_score']['behavior']
                confidence = row['signal_vs_floor_score']['confidence']
                correction_start = row['correction_start']
                
                report.append(f"• {correction_start.strftime('%Y-%m-%d')}: {behavior.replace('_', ' ').title()} ({confidence})")
        
        return "\n".join(report)


def main():
    """Main pre-correction signal analysis"""
    
    # Initialize analyzer
    analyzer = PreCorrectionSignalAnalyzer()
    
    print("🎯 PRE-CORRECTION SIGNAL vs FLOOR ANALYSIS")
    print("Is options market signaling weakness or just setting floors?")
    print("=" * 70)
    
    # Build analysis
    df_analysis = analyzer.build_pre_correction_analysis('2020-01-01', '2025-09-30')
    
    if df_analysis.empty:
        print("❌ No analysis data available")
        return
    
    # Generate report
    report = analyzer.generate_signal_analysis_report(df_analysis)
    print(report)
    
    # Save analysis
    df_analysis.to_csv('pre_correction_signal_analysis.csv', index=False)
    with open('pre_correction_signal_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Analysis saved to files")


if __name__ == "__main__":
    main()
