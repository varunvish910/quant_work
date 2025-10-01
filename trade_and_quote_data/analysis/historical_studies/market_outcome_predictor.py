"""
SPY Market Outcome Predictor
============================

This script analyzes current hedging patterns to predict likely market outcomes:
1. Bounce at 20SMA/50SMA levels
2. Larger correction scenarios
3. Noise vs signal analysis

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketOutcomePredictor:
    """
    Predicts market outcomes based on hedging patterns
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
    
    def calculate_hedging_signals(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate hedging signals that predict market outcomes"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        signals = {}
        
        # 1. Deep OTM put accumulation (hedging signal)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.90]
        if not deep_otm_puts.empty:
            deep_otm_oi = deep_otm_puts['oi_proxy'].sum()
            deep_otm_vol = deep_otm_puts['volume'].sum()
            deep_otm_vol_oi = deep_otm_vol / (deep_otm_oi + 1e-6)
            
            signals['deep_otm_oi'] = deep_otm_oi
            signals['deep_otm_vol_oi'] = deep_otm_vol_oi
            signals['deep_otm_hedging'] = 1 if deep_otm_vol_oi < 0.3 else 0
        
        # 2. Support level analysis (key strikes)
        support_levels = [spy_price * 0.95, spy_price * 0.90, spy_price * 0.85, spy_price * 0.80]
        support_oi = []
        
        for level in support_levels:
            level_puts = puts[abs(puts['strike'] - level) <= 5]
            if not level_puts.empty:
                support_oi.append(level_puts['oi_proxy'].sum())
            else:
                support_oi.append(0)
        
        signals['support_oi_95'] = support_oi[0]  # 5% below current
        signals['support_oi_90'] = support_oi[1]  # 10% below current
        signals['support_oi_85'] = support_oi[2]  # 15% below current
        signals['support_oi_80'] = support_oi[3]  # 20% below current
        
        # 3. Put/Call ratio analysis
        calls = df[df['option_type'] == 'C']
        if not calls.empty:
            put_oi = puts['oi_proxy'].sum()
            call_oi = calls['oi_proxy'].sum()
            signals['pc_ratio_oi'] = put_oi / call_oi
        else:
            signals['pc_ratio_oi'] = 1.0
        
        # 4. Volume/OI ratio (hedging vs speculation)
        total_put_oi = puts['oi_proxy'].sum()
        total_put_vol = puts['volume'].sum()
        signals['put_vol_oi_ratio'] = total_put_vol / (total_put_oi + 1e-6)
        signals['hedging_activity'] = 1 if signals['put_vol_oi_ratio'] < 0.5 else 0
        
        # 5. Long-dated options (institutional hedging)
        if 'dte' in puts.columns:
            long_dated_puts = puts[puts['dte'] > 60]
            if not long_dated_puts.empty:
                long_dated_oi_pct = long_dated_puts['oi_proxy'].sum() / total_put_oi
                signals['long_dated_oi_pct'] = long_dated_oi_pct
                signals['institutional_hedging'] = 1 if long_dated_oi_pct > 0.3 else 0
            else:
                signals['long_dated_oi_pct'] = 0
                signals['institutional_hedging'] = 0
        
        # 6. Strike concentration (defensive positioning)
        top_5_oi = puts.nlargest(5, 'oi_proxy')['oi_proxy'].sum()
        signals['strike_concentration'] = top_5_oi / total_put_oi
        signals['defensive_positioning'] = 1 if signals['strike_concentration'] > 0.1 else 0
        
        return signals
    
    def analyze_historical_outcomes(self) -> dict:
        """Analyze historical outcomes following similar hedging patterns"""
        
        # Define key periods with known outcomes
        historical_periods = {
            'COVID_Crash_2020': {
                'start': '2020-02-15',
                'end': '2020-03-15',
                'outcome': 'MAJOR_CRASH',
                'description': 'COVID crash - 35% decline'
            },
            '2022_Bear_Start': {
                'start': '2022-01-01',
                'end': '2022-03-31',
                'outcome': 'BEAR_MARKET',
                'description': '2022 bear market start - 20% decline'
            },
            '2018_Q4_Volatility': {
                'start': '2018-10-01',
                'end': '2018-12-31',
                'outcome': 'VOLATILITY_SPIKE',
                'description': '2018 Q4 volatility - 20% decline then recovery'
            },
            '2016_Election_Volatility': {
                'start': '2016-10-01',
                'end': '2016-12-31',
                'outcome': 'VOLATILITY_THEN_RALLY',
                'description': '2016 election - volatility then strong rally'
            },
            '2024_Normal_Periods': {
                'start': '2024-01-01',
                'end': '2024-12-31',
                'outcome': 'NORMAL_MARKET',
                'description': '2024 normal market conditions'
            }
        }
        
        print("🔍 ANALYZING HISTORICAL OUTCOMES")
        print("=" * 40)
        
        outcomes = {}
        
        for period_name, period_info in historical_periods.items():
            print(f"\n📊 Analyzing {period_name}: {period_info['description']}")
            
            start_date = pd.to_datetime(period_info['start'])
            end_date = pd.to_datetime(period_info['end'])
            
            period_signals = []
            current_date = start_date
            
            # Sample every 3 trading days
            sample_count = 0
            while current_date <= end_date:
                if current_date.weekday() < 5:
                    if sample_count % 3 == 0:
                        date_str = current_date.strftime('%Y-%m-%d')
                        df = self.load_daily_data(date_str)
                        
                        if not df.empty and 'underlying_price' in df.columns:
                            spy_price = df['underlying_price'].iloc[0]
                            signals = self.calculate_hedging_signals(df, spy_price)
                            
                            if signals:
                                signals['date'] = current_date
                                signals['spy_price'] = spy_price
                                period_signals.append(signals)
                    
                    sample_count += 1
                
                current_date += timedelta(days=1)
            
            if period_signals:
                period_df = pd.DataFrame(period_signals)
                outcomes[period_name] = {
                    'signals': period_df,
                    'outcome': period_info['outcome'],
                    'description': period_info['description'],
                    'summary': self._summarize_signals(period_df)
                }
                print(f"✅ Analyzed {len(period_df)} data points")
            else:
                print(f"❌ No data found for {period_name}")
        
        return outcomes
    
    def _summarize_signals(self, df: pd.DataFrame) -> dict:
        """Summarize hedging signals for a period"""
        summary = {}
        
        # Calculate averages for key signals
        signal_cols = [
            'deep_otm_hedging', 'hedging_activity', 'institutional_hedging',
            'defensive_positioning', 'pc_ratio_oi', 'put_vol_oi_ratio',
            'long_dated_oi_pct', 'strike_concentration'
        ]
        
        for col in signal_cols:
            if col in df.columns:
                summary[f'{col}_avg'] = df[col].mean()
                summary[f'{col}_max'] = df[col].max()
                summary[f'{col}_min'] = df[col].min()
        
        # Calculate composite hedging score
        hedging_components = []
        if 'deep_otm_hedging' in df.columns:
            hedging_components.append(df['deep_otm_hedging'].mean())
        if 'hedging_activity' in df.columns:
            hedging_components.append(df['hedging_activity'].mean())
        if 'institutional_hedging' in df.columns:
            hedging_components.append(df['institutional_hedging'].mean())
        if 'defensive_positioning' in df.columns:
            hedging_components.append(df['defensive_positioning'].mean())
        
        if hedging_components:
            summary['composite_hedging_score'] = np.mean(hedging_components)
        
        return summary
    
    def predict_current_outcome(self, current_signals: dict, historical_outcomes: dict) -> dict:
        """Predict current market outcome based on hedging signals"""
        
        print("\n🎯 PREDICTING CURRENT MARKET OUTCOME")
        print("=" * 45)
        
        # Find most similar historical periods
        similarities = {}
        
        for period_name, period_data in historical_outcomes.items():
            summary = period_data['summary']
            similarity_score = 0
            comparisons = 0
            
            # Compare key signals
            signal_comparisons = [
                'deep_otm_hedging', 'hedging_activity', 'institutional_hedging',
                'defensive_positioning', 'pc_ratio_oi', 'put_vol_oi_ratio'
            ]
            
            for signal in signal_comparisons:
                if signal in current_signals and f'{signal}_avg' in summary:
                    current_val = current_signals[signal]
                    historical_avg = summary[f'{signal}_avg']
                    
                    # Calculate similarity (closer values = higher similarity)
                    if historical_avg > 0:
                        similarity = 1 - abs(current_val - historical_avg) / max(current_val, historical_avg)
                    else:
                        similarity = 1 if current_val == historical_avg else 0
                    
                    similarity_score += similarity
                    comparisons += 1
            
            if comparisons > 0:
                similarities[period_name] = {
                    'similarity': similarity_score / comparisons,
                    'outcome': period_data['outcome'],
                    'description': period_data['description']
                }
        
        # Sort by similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1]['similarity'], reverse=True)
        
        # Generate prediction
        prediction = {
            'most_similar_period': sorted_similarities[0][0] if sorted_similarities else 'UNKNOWN',
            'similarity_score': sorted_similarities[0][1]['similarity'] if sorted_similarities else 0,
            'predicted_outcome': sorted_similarities[0][1]['outcome'] if sorted_similarities else 'UNKNOWN',
            'confidence': self._calculate_confidence(current_signals, historical_outcomes),
            'scenario_probabilities': self._calculate_scenario_probabilities(current_signals, historical_outcomes)
        }
        
        return prediction
    
    def _calculate_confidence(self, current_signals: dict, historical_outcomes: dict) -> str:
        """Calculate confidence level in prediction"""
        
        # Check signal strength
        strong_signals = 0
        total_signals = 0
        
        if current_signals.get('deep_otm_hedging', 0) == 1:
            strong_signals += 1
        total_signals += 1
        
        if current_signals.get('hedging_activity', 0) == 1:
            strong_signals += 1
        total_signals += 1
        
        if current_signals.get('institutional_hedging', 0) == 1:
            strong_signals += 1
        total_signals += 1
        
        if current_signals.get('defensive_positioning', 0) == 1:
            strong_signals += 1
        total_signals += 1
        
        signal_strength = strong_signals / total_signals if total_signals > 0 else 0
        
        if signal_strength >= 0.75:
            return 'HIGH'
        elif signal_strength >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_scenario_probabilities(self, current_signals: dict, historical_outcomes: dict) -> dict:
        """Calculate probabilities for different market scenarios"""
        
        scenarios = {
            'BOUNCE_20SMA': 0.0,
            'BOUNCE_50SMA': 0.0,
            'LARGER_CORRECTION': 0.0,
            'NOISE_CONTINUATION': 0.0
        }
        
        # Analyze current signal patterns
        hedging_score = 0
        if current_signals.get('deep_otm_hedging', 0) == 1:
            hedging_score += 25
        if current_signals.get('hedging_activity', 0) == 1:
            hedging_score += 25
        if current_signals.get('institutional_hedging', 0) == 1:
            hedging_score += 25
        if current_signals.get('defensive_positioning', 0) == 1:
            hedging_score += 25
        
        # PC ratio analysis
        pc_ratio = current_signals.get('pc_ratio_oi', 1.0)
        
        # V/OI ratio analysis
        vol_oi_ratio = current_signals.get('put_vol_oi_ratio', 1.0)
        
        # Scenario probability calculation
        if hedging_score < 25 and pc_ratio < 1.2 and vol_oi_ratio > 0.8:
            # Low hedging, normal ratios = likely noise
            scenarios['NOISE_CONTINUATION'] = 0.6
            scenarios['BOUNCE_20SMA'] = 0.3
            scenarios['BOUNCE_50SMA'] = 0.1
        elif hedging_score < 50 and pc_ratio < 1.5:
            # Moderate hedging = potential bounce scenarios
            scenarios['BOUNCE_20SMA'] = 0.4
            scenarios['BOUNCE_50SMA'] = 0.3
            scenarios['NOISE_CONTINUATION'] = 0.2
            scenarios['LARGER_CORRECTION'] = 0.1
        elif hedging_score >= 50:
            # High hedging = potential larger correction
            scenarios['LARGER_CORRECTION'] = 0.4
            scenarios['BOUNCE_50SMA'] = 0.3
            scenarios['BOUNCE_20SMA'] = 0.2
            scenarios['NOISE_CONTINUATION'] = 0.1
        else:
            # Default probabilities
            scenarios['BOUNCE_20SMA'] = 0.3
            scenarios['BOUNCE_50SMA'] = 0.3
            scenarios['LARGER_CORRECTION'] = 0.2
            scenarios['NOISE_CONTINUATION'] = 0.2
        
        return scenarios
    
    def generate_outcome_report(self, current_signals: dict, prediction: dict) -> str:
        """Generate comprehensive outcome prediction report"""
        
        report = []
        report.append("=" * 70)
        report.append("SPY MARKET OUTCOME PREDICTION REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current signal analysis
        report.append("🔍 CURRENT HEDGING SIGNALS:")
        report.append("-" * 30)
        report.append(f"Deep OTM Hedging: {'YES' if current_signals.get('deep_otm_hedging', 0) == 1 else 'NO'}")
        report.append(f"Hedging Activity: {'YES' if current_signals.get('hedging_activity', 0) == 1 else 'NO'}")
        report.append(f"Institutional Hedging: {'YES' if current_signals.get('institutional_hedging', 0) == 1 else 'NO'}")
        report.append(f"Defensive Positioning: {'YES' if current_signals.get('defensive_positioning', 0) == 1 else 'NO'}")
        report.append(f"Put/Call Ratio: {current_signals.get('pc_ratio_oi', 0):.2f}")
        report.append(f"Put V/OI Ratio: {current_signals.get('put_vol_oi_ratio', 0):.2f}")
        report.append("")
        
        # Prediction summary
        report.append("🎯 MARKET OUTCOME PREDICTION:")
        report.append("-" * 35)
        report.append(f"Most Similar Period: {prediction['most_similar_period']}")
        report.append(f"Similarity Score: {prediction['similarity_score']:.2f}/1.0")
        report.append(f"Predicted Outcome: {prediction['predicted_outcome']}")
        report.append(f"Confidence Level: {prediction['confidence']}")
        report.append("")
        
        # Scenario probabilities
        report.append("📊 SCENARIO PROBABILITIES:")
        report.append("-" * 30)
        scenarios = prediction['scenario_probabilities']
        for scenario, probability in scenarios.items():
            percentage = probability * 100
            if percentage >= 30:
                report.append(f"🔴 {scenario.replace('_', ' ').title()}: {percentage:.0f}%")
            elif percentage >= 20:
                report.append(f"🟡 {scenario.replace('_', ' ').title()}: {percentage:.0f}%")
            else:
                report.append(f"🟢 {scenario.replace('_', ' ').title()}: {percentage:.0f}%")
        
        report.append("")
        
        # Key insights
        report.append("💡 KEY INSIGHTS:")
        report.append("-" * 15)
        
        # Analyze current patterns
        hedging_score = 0
        if current_signals.get('deep_otm_hedging', 0) == 1:
            hedging_score += 25
        if current_signals.get('hedging_activity', 0) == 1:
            hedging_score += 25
        if current_signals.get('institutional_hedging', 0) == 1:
            hedging_score += 25
        if current_signals.get('defensive_positioning', 0) == 1:
            hedging_score += 25
        
        if hedging_score >= 75:
            report.append("• HIGH hedging activity detected - monitor for potential weakness")
            report.append("• Institutional players appear defensive")
            report.append("• Risk of larger correction elevated")
        elif hedging_score >= 50:
            report.append("• MODERATE hedging activity - mixed signals")
            report.append("• Some defensive positioning present")
            report.append("• Watch for pattern acceleration")
        else:
            report.append("• LOW hedging activity - normal market conditions")
            report.append("• Limited defensive positioning")
            report.append("• Current weakness likely noise")
        
        # PC ratio insights
        pc_ratio = current_signals.get('pc_ratio_oi', 1.0)
        if pc_ratio > 1.5:
            report.append("• Elevated Put/Call ratio suggests bearish sentiment")
        elif pc_ratio < 0.8:
            report.append("• Low Put/Call ratio suggests bullish sentiment")
        else:
            report.append("• Normal Put/Call ratio - balanced sentiment")
        
        # V/OI ratio insights
        vol_oi_ratio = current_signals.get('put_vol_oi_ratio', 1.0)
        if vol_oi_ratio < 0.3:
            report.append("• Low V/OI ratio indicates accumulation/hedging")
        elif vol_oi_ratio > 1.0:
            report.append("• High V/OI ratio indicates active trading/speculation")
        else:
            report.append("• Normal V/OI ratio - mixed activity")
        
        report.append("")
        
        # Recommendations
        report.append("📋 RECOMMENDATIONS:")
        report.append("-" * 20)
        
        if scenarios['LARGER_CORRECTION'] > 0.3:
            report.append("• Consider defensive positioning")
            report.append("• Monitor key support levels closely")
            report.append("• Prepare for potential volatility")
        elif scenarios['BOUNCE_50SMA'] > 0.3 or scenarios['BOUNCE_20SMA'] > 0.3:
            report.append("• Watch for bounce at key moving averages")
            report.append("• Consider buying dips near support")
            report.append("• Monitor for confirmation signals")
        else:
            report.append("• Current weakness appears to be noise")
            report.append("• Continue normal monitoring")
            report.append("• No immediate action required")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    
    # Initialize predictor
    predictor = MarketOutcomePredictor()
    
    print("🔍 SPY MARKET OUTCOME PREDICTION ANALYSIS")
    print("=" * 50)
    
    # Load current signals (September 30, 2025)
    print("Loading current hedging signals...")
    current_df = predictor.load_daily_data('2025-09-30')
    
    if current_df.empty:
        print("❌ No current data found")
        return
    
    spy_price = current_df['underlying_price'].iloc[0]
    current_signals = predictor.calculate_hedging_signals(current_df, spy_price)
    
    print(f"✅ Current SPY Price: ${spy_price:.2f}")
    print(f"✅ Current signals calculated")
    
    # Analyze historical outcomes
    print("\nAnalyzing historical outcomes...")
    historical_outcomes = predictor.analyze_historical_outcomes()
    
    # Predict current outcome
    print("\nPredicting current market outcome...")
    prediction = predictor.predict_current_outcome(current_signals, historical_outcomes)
    
    # Generate report
    report = predictor.generate_outcome_report(current_signals, prediction)
    print("\n" + report)
    
    # Save results
    with open('market_outcome_prediction.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Outcome prediction report saved to 'market_outcome_prediction.txt'")


if __name__ == "__main__":
    main()
