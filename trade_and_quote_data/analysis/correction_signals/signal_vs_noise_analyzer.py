"""
SPY Signal vs Noise Analyzer
============================

This script analyzes what makes current hedging patterns a signal rather than noise
by examining key distinguishing factors and historical precedents.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SignalVsNoiseAnalyzer:
    """
    Analyzes what makes hedging patterns a signal vs noise
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
    
    def calculate_signal_strength(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate signal strength indicators"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        signals = {}
        
        # 1. Deep OTM put accumulation (institutional hedging signal)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.90]
        if not deep_otm_puts.empty:
            deep_otm_oi = deep_otm_puts['oi_proxy'].sum()
            deep_otm_vol = deep_otm_puts['volume'].sum()
            deep_otm_vol_oi = deep_otm_vol / (deep_otm_oi + 1e-6)
            
            signals['deep_otm_oi'] = deep_otm_oi
            signals['deep_otm_vol_oi'] = deep_otm_vol_oi
            signals['deep_otm_accumulation'] = 1 if deep_otm_vol_oi < 0.3 else 0
        
        # 2. Long-dated options (institutional hedging)
        total_put_oi = puts['oi_proxy'].sum()
        if 'dte' in puts.columns:
            long_dated_puts = puts[puts['dte'] > 60]
            if not long_dated_puts.empty:
                long_dated_oi_pct = long_dated_puts['oi_proxy'].sum() / total_put_oi
                signals['long_dated_oi_pct'] = long_dated_oi_pct
                signals['institutional_hedging'] = 1 if long_dated_oi_pct > 0.3 else 0
            else:
                signals['long_dated_oi_pct'] = 0
                signals['institutional_hedging'] = 0
        
        # 3. Strike concentration (defensive positioning)
        top_5_oi = puts.nlargest(5, 'oi_proxy')['oi_proxy'].sum()
        max_single_oi = puts['oi_proxy'].max()
        signals['strike_concentration'] = top_5_oi / total_put_oi
        signals['max_strike_concentration'] = max_single_oi / total_put_oi
        signals['defensive_positioning'] = 1 if signals['strike_concentration'] > 0.1 else 0
        
        # 4. Volume/OI ratio (hedging vs speculation)
        total_put_vol = puts['volume'].sum()
        signals['put_vol_oi_ratio'] = total_put_vol / (total_put_oi + 1e-6)
        signals['hedging_activity'] = 1 if signals['put_vol_oi_ratio'] < 0.5 else 0
        
        # 5. Put/Call ratio
        calls = df[df['option_type'] == 'C']
        if not calls.empty:
            signals['pc_ratio_oi'] = total_put_oi / calls['oi_proxy'].sum()
        else:
            signals['pc_ratio_oi'] = 1.0
        
        # 6. Support level analysis
        support_levels = [spy_price * 0.95, spy_price * 0.90, spy_price * 0.85, spy_price * 0.80]
        support_oi_total = 0
        
        for level in support_levels:
            level_puts = puts[abs(puts['strike'] - level) <= 5]
            if not level_puts.empty:
                support_oi_total += level_puts['oi_proxy'].sum()
        
        signals['support_level_oi'] = support_oi_total
        signals['support_level_pct'] = support_oi_total / total_put_oi
        
        return signals
    
    def analyze_historical_noise_periods(self) -> dict:
        """Analyze periods that were just noise (no significant moves)"""
        
        # Define noise periods (stable markets with no major moves)
        noise_periods = {
            '2024_Stable_Q2': {
                'start': '2024-04-01',
                'end': '2024-06-30',
                'description': 'Stable Q2 2024 - no major moves'
            },
            '2024_Stable_Q3': {
                'start': '2024-07-01',
                'end': '2024-09-30',
                'description': 'Stable Q3 2024 - no major moves'
            },
            '2025_Stable_Jan': {
                'start': '2025-01-01',
                'end': '2025-01-31',
                'description': 'Stable January 2025'
            }
        }
        
        print("📊 ANALYZING HISTORICAL NOISE PERIODS")
        print("=" * 45)
        
        noise_analysis = {}
        
        for period_name, period_info in noise_periods.items():
            print(f"\n📈 Analyzing {period_name}: {period_info['description']}")
            
            start_date = pd.to_datetime(period_info['start'])
            end_date = pd.to_datetime(period_info['end'])
            
            period_signals = []
            current_date = start_date
            
            # Sample every 5 trading days
            sample_count = 0
            while current_date <= end_date:
                if current_date.weekday() < 5:
                    if sample_count % 5 == 0:
                        date_str = current_date.strftime('%Y-%m-%d')
                        df = self.load_daily_data(date_str)
                        
                        if not df.empty and 'underlying_price' in df.columns:
                            spy_price = df['underlying_price'].iloc[0]
                            signals = self.calculate_signal_strength(df, spy_price)
                            
                            if signals:
                                signals['date'] = current_date
                                signals['spy_price'] = spy_price
                                period_signals.append(signals)
                    
                    sample_count += 1
                
                current_date += timedelta(days=1)
            
            if period_signals:
                period_df = pd.DataFrame(period_signals)
                noise_analysis[period_name] = {
                    'signals': period_df,
                    'summary': self._summarize_signals(period_df),
                    'description': period_info['description']
                }
                print(f"✅ Analyzed {len(period_df)} data points")
            else:
                print(f"❌ No data found for {period_name}")
        
        return noise_analysis
    
    def analyze_historical_signal_periods(self) -> dict:
        """Analyze periods that were signals (preceded significant moves)"""
        
        # Define signal periods (preceded significant moves)
        signal_periods = {
            'COVID_Pre_Crash': {
                'start': '2020-02-01',
                'end': '2020-02-28',
                'description': 'Pre-COVID crash - hedging buildup before 35% decline'
            },
            '2022_Pre_Bear': {
                'start': '2022-01-01',
                'end': '2022-02-28',
                'description': 'Pre-2022 bear market - hedging before 20% decline'
            },
            '2018_Pre_Volatility': {
                'start': '2018-09-01',
                'end': '2018-10-31',
                'description': 'Pre-2018 Q4 volatility - hedging before 20% decline'
            }
        }
        
        print("\n📊 ANALYZING HISTORICAL SIGNAL PERIODS")
        print("=" * 45)
        
        signal_analysis = {}
        
        for period_name, period_info in signal_periods.items():
            print(f"\n📈 Analyzing {period_name}: {period_info['description']}")
            
            start_date = pd.to_datetime(period_info['start'])
            end_date = pd.to_datetime(period_info['end'])
            
            period_signals = []
            current_date = start_date
            
            # Sample every 5 trading days
            sample_count = 0
            while current_date <= end_date:
                if current_date.weekday() < 5:
                    if sample_count % 5 == 0:
                        date_str = current_date.strftime('%Y-%m-%d')
                        df = self.load_daily_data(date_str)
                        
                        if not df.empty and 'underlying_price' in df.columns:
                            spy_price = df['underlying_price'].iloc[0]
                            signals = self.calculate_signal_strength(df, spy_price)
                            
                            if signals:
                                signals['date'] = current_date
                                signals['spy_price'] = spy_price
                                period_signals.append(signals)
                    
                    sample_count += 1
                
                current_date += timedelta(days=1)
            
            if period_signals:
                period_df = pd.DataFrame(period_signals)
                signal_analysis[period_name] = {
                    'signals': period_df,
                    'summary': self._summarize_signals(period_df),
                    'description': period_info['description']
                }
                print(f"✅ Analyzed {len(period_df)} data points")
            else:
                print(f"❌ No data found for {period_name}")
        
        return signal_analysis
    
    def _summarize_signals(self, df: pd.DataFrame) -> dict:
        """Summarize signal strength for a period"""
        summary = {}
        
        # Calculate averages for key signals
        signal_cols = [
            'deep_otm_accumulation', 'institutional_hedging', 'defensive_positioning',
            'hedging_activity', 'pc_ratio_oi', 'put_vol_oi_ratio',
            'long_dated_oi_pct', 'strike_concentration', 'support_level_pct'
        ]
        
        for col in signal_cols:
            if col in df.columns:
                summary[f'{col}_avg'] = df[col].mean()
                summary[f'{col}_max'] = df[col].max()
                summary[f'{col}_min'] = df[col].min()
        
        # Calculate composite signal score
        signal_components = []
        if 'deep_otm_accumulation' in df.columns:
            signal_components.append(df['deep_otm_accumulation'].mean())
        if 'institutional_hedging' in df.columns:
            signal_components.append(df['institutional_hedging'].mean())
        if 'defensive_positioning' in df.columns:
            signal_components.append(df['defensive_positioning'].mean())
        if 'hedging_activity' in df.columns:
            signal_components.append(df['hedging_activity'].mean())
        
        if signal_components:
            summary['composite_signal_score'] = np.mean(signal_components)
        
        return summary
    
    def compare_current_to_historical(self, current_signals: dict, noise_analysis: dict, signal_analysis: dict) -> dict:
        """Compare current signals to historical noise and signal periods"""
        
        print("\n🔍 COMPARING CURRENT TO HISTORICAL PATTERNS")
        print("=" * 50)
        
        comparison = {
            'noise_similarity': {},
            'signal_similarity': {},
            'signal_strength': 0,
            'noise_likelihood': 0,
            'signal_likelihood': 0
        }
        
        # Compare to noise periods
        noise_similarities = []
        for period_name, period_data in noise_analysis.items():
            summary = period_data['summary']
            similarity = self._calculate_similarity(current_signals, summary)
            comparison['noise_similarity'][period_name] = similarity
            noise_similarities.append(similarity)
        
        # Compare to signal periods
        signal_similarities = []
        for period_name, period_data in signal_analysis.items():
            summary = period_data['summary']
            similarity = self._calculate_similarity(current_signals, summary)
            comparison['signal_similarity'][period_name] = similarity
            signal_similarities.append(similarity)
        
        # Calculate overall likelihoods
        if noise_similarities:
            comparison['noise_likelihood'] = np.mean(noise_similarities)
        if signal_similarities:
            comparison['signal_likelihood'] = np.mean(signal_similarities)
        
        # Calculate signal strength
        signal_components = [
            current_signals.get('deep_otm_accumulation', 0),
            current_signals.get('institutional_hedging', 0),
            current_signals.get('defensive_positioning', 0),
            current_signals.get('hedging_activity', 0)
        ]
        comparison['signal_strength'] = np.mean(signal_components)
        
        return comparison
    
    def _calculate_similarity(self, current_signals: dict, historical_summary: dict) -> float:
        """Calculate similarity between current signals and historical period"""
        
        similarity_score = 0
        comparisons = 0
        
        # Compare key metrics
        key_metrics = [
            'deep_otm_accumulation', 'institutional_hedging', 'defensive_positioning',
            'hedging_activity', 'pc_ratio_oi', 'put_vol_oi_ratio',
            'long_dated_oi_pct', 'strike_concentration'
        ]
        
        for metric in key_metrics:
            if metric in current_signals and f'{metric}_avg' in historical_summary:
                current_val = current_signals[metric]
                historical_avg = historical_summary[f'{metric}_avg']
                
                # Calculate similarity (closer values = higher similarity)
                if historical_avg > 0:
                    similarity = 1 - abs(current_val - historical_avg) / max(current_val, historical_avg)
                else:
                    similarity = 1 if current_val == historical_avg else 0
                
                similarity_score += similarity
                comparisons += 1
        
        return similarity_score / comparisons if comparisons > 0 else 0
    
    def generate_signal_vs_noise_report(self, current_signals: dict, comparison: dict, 
                                      noise_analysis: dict, signal_analysis: dict) -> str:
        """Generate comprehensive signal vs noise analysis report"""
        
        report = []
        report.append("=" * 80)
        report.append("SPY HEDGING: SIGNAL vs NOISE ANALYSIS")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current signal analysis
        report.append("🔍 CURRENT HEDGING SIGNALS:")
        report.append("-" * 30)
        report.append(f"Deep OTM Accumulation: {'YES' if current_signals.get('deep_otm_accumulation', 0) == 1 else 'NO'}")
        report.append(f"Institutional Hedging: {'YES' if current_signals.get('institutional_hedging', 0) == 1 else 'NO'}")
        report.append(f"Defensive Positioning: {'YES' if current_signals.get('defensive_positioning', 0) == 1 else 'NO'}")
        report.append(f"Hedging Activity: {'YES' if current_signals.get('hedging_activity', 0) == 1 else 'NO'}")
        report.append(f"Put/Call Ratio: {current_signals.get('pc_ratio_oi', 0):.2f}")
        report.append(f"Put V/OI Ratio: {current_signals.get('put_vol_oi_ratio', 0):.2f}")
        report.append(f"Long-dated Options %: {current_signals.get('long_dated_oi_pct', 0):.1%}")
        report.append(f"Support Level %: {current_signals.get('support_level_pct', 0):.1%}")
        report.append("")
        
        # Signal strength assessment
        report.append("📊 SIGNAL STRENGTH ASSESSMENT:")
        report.append("-" * 35)
        
        signal_strength = comparison['signal_strength']
        if signal_strength >= 0.75:
            strength_level = "STRONG SIGNAL"
            strength_color = "🔴"
        elif signal_strength >= 0.5:
            strength_level = "MODERATE SIGNAL"
            strength_color = "🟡"
        elif signal_strength >= 0.25:
            strength_level = "WEAK SIGNAL"
            strength_color = "🟠"
        else:
            strength_level = "NOISE"
            strength_color = "🟢"
        
        report.append(f"{strength_color} Signal Strength: {signal_strength:.2f}/1.0 ({strength_level})")
        report.append("")
        
        # Historical comparison
        report.append("📈 HISTORICAL COMPARISON:")
        report.append("-" * 30)
        
        noise_likelihood = comparison['noise_likelihood']
        signal_likelihood = comparison['signal_likelihood']
        
        report.append(f"Noise Likelihood: {noise_likelihood:.2f}/1.0")
        report.append(f"Signal Likelihood: {signal_likelihood:.2f}/1.0")
        
        if signal_likelihood > noise_likelihood + 0.2:
            report.append("🎯 CONCLUSION: More similar to HISTORICAL SIGNAL periods")
        elif noise_likelihood > signal_likelihood + 0.2:
            report.append("🎯 CONCLUSION: More similar to HISTORICAL NOISE periods")
        else:
            report.append("🎯 CONCLUSION: Mixed signals - not clearly noise or signal")
        
        report.append("")
        
        # What makes this a signal vs noise
        report.append("💡 WHAT MAKES THIS A SIGNAL vs NOISE:")
        report.append("-" * 45)
        
        # Analyze individual components
        signal_components = []
        noise_components = []
        
        if current_signals.get('deep_otm_accumulation', 0) == 1:
            signal_components.append("Deep OTM put accumulation (institutional hedging)")
        else:
            noise_components.append("No deep OTM put accumulation")
        
        if current_signals.get('institutional_hedging', 0) == 1:
            signal_components.append("High long-dated options % (institutional positioning)")
        else:
            noise_components.append("Low long-dated options %")
        
        if current_signals.get('defensive_positioning', 0) == 1:
            signal_components.append("Concentrated strike positioning (defensive)")
        else:
            noise_components.append("Distributed strike positioning")
        
        if current_signals.get('hedging_activity', 0) == 1:
            signal_components.append("Low V/OI ratios (accumulation vs speculation)")
        else:
            noise_components.append("High V/OI ratios (active trading)")
        
        if current_signals.get('pc_ratio_oi', 0) > 1.2:
            signal_components.append("Elevated Put/Call ratio (bearish sentiment)")
        else:
            noise_components.append("Normal Put/Call ratio")
        
        if current_signals.get('support_level_pct', 0) > 0.3:
            signal_components.append("High support level concentration")
        else:
            noise_components.append("Normal support level distribution")
        
        if signal_components:
            report.append("🔴 SIGNAL INDICATORS:")
            for component in signal_components:
                report.append(f"  • {component}")
        
        if noise_components:
            report.append("\n🟢 NOISE INDICATORS:")
            for component in noise_components:
                report.append(f"  • {component}")
        
        report.append("")
        
        # Key distinguishing factors
        report.append("🎯 KEY DISTINGUISHING FACTORS:")
        report.append("-" * 35)
        
        # Count signal vs noise indicators
        signal_count = len(signal_components)
        noise_count = len(noise_components)
        
        if signal_count > noise_count + 2:
            report.append("✅ CLEAR SIGNAL: Multiple hedging indicators present")
            report.append("   • Institutional players are positioning defensively")
            report.append("   • Pattern suggests preparation for potential weakness")
            report.append("   • Not typical of normal market noise")
        elif signal_count > noise_count:
            report.append("🟡 MIXED SIGNALS: Some hedging indicators present")
            report.append("   • Some defensive positioning detected")
            report.append("   • Pattern could indicate either signal or noise")
            report.append("   • Monitor for pattern acceleration")
        else:
            report.append("🟢 LIKELY NOISE: Few hedging indicators present")
            report.append("   • Normal market activity patterns")
            report.append("   • No significant defensive positioning")
            report.append("   • Current weakness likely temporary")
        
        report.append("")
        
        # Specific analysis of current patterns
        report.append("🔍 SPECIFIC PATTERN ANALYSIS:")
        report.append("-" * 30)
        
        # Deep OTM analysis
        deep_otm_oi = current_signals.get('deep_otm_oi', 0)
        deep_otm_vol_oi = current_signals.get('deep_otm_vol_oi', 0)
        
        if deep_otm_vol_oi < 0.3:
            report.append("• Deep OTM puts show LOW V/OI ratio - indicates ACCUMULATION")
            report.append("  This is a SIGNAL of institutional hedging, not noise")
        else:
            report.append("• Deep OTM puts show HIGH V/OI ratio - indicates ACTIVE TRADING")
            report.append("  This suggests noise rather than hedging signal")
        
        # Long-dated options analysis
        long_dated_pct = current_signals.get('long_dated_oi_pct', 0)
        if long_dated_pct > 0.3:
            report.append("• High % of long-dated options - indicates INSTITUTIONAL HEDGING")
            report.append("  This is a SIGNAL of defensive positioning, not noise")
        else:
            report.append("• Low % of long-dated options - indicates RETAIL/SPECULATIVE activity")
            report.append("  This suggests noise rather than institutional signal")
        
        # PC ratio analysis
        pc_ratio = current_signals.get('pc_ratio_oi', 0)
        if pc_ratio > 1.2:
            report.append("• Elevated Put/Call ratio - indicates BEARISH SENTIMENT")
            report.append("  This could be a SIGNAL of market concern, not just noise")
        else:
            report.append("• Normal Put/Call ratio - indicates BALANCED SENTIMENT")
            report.append("  This suggests normal market conditions")
        
        report.append("")
        
        # Final verdict
        report.append("🎯 FINAL VERDICT:")
        report.append("-" * 15)
        
        if signal_strength >= 0.5 and signal_likelihood > noise_likelihood:
            report.append("🔴 THIS IS A SIGNAL, NOT NOISE")
            report.append("   • Multiple hedging indicators present")
            report.append("   • Pattern similar to historical signal periods")
            report.append("   • Institutional players positioning defensively")
            report.append("   • Current weakness likely has substance")
        elif signal_strength >= 0.25:
            report.append("🟡 MIXED SIGNALS - MONITOR CLOSELY")
            report.append("   • Some hedging indicators present")
            report.append("   • Pattern unclear - could be signal or noise")
            report.append("   • Watch for pattern acceleration")
            report.append("   • Prepare for either scenario")
        else:
            report.append("🟢 LIKELY NOISE - NORMAL MARKET CONDITIONS")
            report.append("   • Few hedging indicators present")
            report.append("   • Pattern similar to historical noise periods")
            report.append("   • Normal market activity")
            report.append("   • Current weakness likely temporary")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    
    # Initialize analyzer
    analyzer = SignalVsNoiseAnalyzer()
    
    print("🔍 SPY HEDGING: SIGNAL vs NOISE ANALYSIS")
    print("=" * 50)
    
    # Load current signals (September 30, 2025)
    print("Loading current hedging signals...")
    current_df = analyzer.load_daily_data('2025-09-30')
    
    if current_df.empty:
        print("❌ No current data found")
        return
    
    spy_price = current_df['underlying_price'].iloc[0]
    current_signals = analyzer.calculate_signal_strength(current_df, spy_price)
    
    print(f"✅ Current SPY Price: ${spy_price:.2f}")
    print(f"✅ Current signals calculated")
    
    # Analyze historical noise periods
    print("\nAnalyzing historical noise periods...")
    noise_analysis = analyzer.analyze_historical_noise_periods()
    
    # Analyze historical signal periods
    print("\nAnalyzing historical signal periods...")
    signal_analysis = analyzer.analyze_historical_signal_periods()
    
    # Compare current to historical
    print("\nComparing current patterns to historical data...")
    comparison = analyzer.compare_current_to_historical(current_signals, noise_analysis, signal_analysis)
    
    # Generate report
    report = analyzer.generate_signal_vs_noise_report(current_signals, comparison, noise_analysis, signal_analysis)
    print("\n" + report)
    
    # Save results
    with open('signal_vs_noise_analysis.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Signal vs noise analysis saved to 'signal_vs_noise_analysis.txt'")


if __name__ == "__main__":
    main()
