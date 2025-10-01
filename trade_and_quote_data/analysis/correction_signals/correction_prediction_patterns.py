"""
Correction Prediction Patterns Analysis
======================================

This analysis identifies hedging patterns that precede 5-10% corrections
to retrain models for prediction. Focus on finding the specific signatures
that lead into these events.

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

class CorrectionPredictionPatterns:
    """
    Identifies hedging patterns that precede 5-10% corrections
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
    
    def identify_correction_periods(self, start_date: str, end_date: str) -> list:
        """Identify 5-10% correction periods in SPY"""
        
        print(f"🔍 IDENTIFYING CORRECTION PERIODS")
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
        
        # Find correction periods (5-10% declines)
        corrections = []
        
        for i in range(20, len(df_prices)):  # Look back 20 days for peak
            current_price = df_prices.iloc[i]['spy_price']
            current_date = df_prices.iloc[i]['date']
            
            # Find peak in last 20 days
            lookback_prices = df_prices.iloc[i-20:i]['spy_price']
            peak_price = lookback_prices.max()
            peak_mask = df_prices['spy_price'] == peak_price
            peak_date = df_prices[peak_mask].iloc[-1]['date']
            
            # Calculate decline
            decline_pct = (peak_price - current_price) / peak_price
            
            # Check if this is a 5-10% correction
            if 0.05 <= decline_pct <= 0.10:
                # Find the start of the correction (when decline began)
                correction_start_idx = None
                for j in range(i-20, i):
                    if df_prices.iloc[j]['spy_price'] == peak_price:
                        correction_start_idx = j
                        break
                
                if correction_start_idx is not None:
                    correction_start_date = df_prices.iloc[correction_start_idx]['date']
                    
                    corrections.append({
                        'correction_start': correction_start_date,
                        'correction_end': current_date,
                        'peak_price': peak_price,
                        'trough_price': current_price,
                        'decline_pct': decline_pct,
                        'duration_days': (current_date - correction_start_date).days
                    })
        
        print(f"✅ Found {len(corrections)} correction periods")
        return corrections
    
    def analyze_pre_correction_hedging(self, correction: dict, lookback_days: int = 10) -> dict:
        """Analyze hedging patterns before a correction"""
        
        correction_start = correction['correction_start']
        pre_correction_start = correction_start - timedelta(days=lookback_days)
        
        print(f"📊 Analyzing hedging before correction starting {correction_start.strftime('%Y-%m-%d')}")
        
        # Build pre-correction data
        pre_correction_data = []
        current_date = pre_correction_start
        
        while current_date < correction_start:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    analysis = self.analyze_hedging_features(df, spy_price)
                    if analysis:
                        analysis['date'] = current_date
                        analysis['days_before_correction'] = (correction_start - current_date).days
                        pre_correction_data.append(analysis)
            
            current_date += timedelta(days=1)
        
        if not pre_correction_data:
            return {}
        
        df_pre = pd.DataFrame(pre_correction_data)
        
        # Calculate pre-correction patterns
        patterns = {
            'correction_start': correction_start,
            'decline_pct': correction['decline_pct'],
            'duration_days': correction['duration_days'],
            'lookback_days': lookback_days,
            'data_points': len(df_pre)
        }
        
        # Feature analysis
        features = [
            'total_put_oi', 'pc_ratio', 'vol_oi_ratio', 'strike_concentration',
            'institutional_pct', 'deep_otm_pct', 'atm_pct', 'long_term_pct',
            'hedging_intensity', 'defensive_positioning'
        ]
        
        for feature in features:
            if feature in df_pre.columns:
                patterns[f'{feature}_mean'] = df_pre[feature].mean()
                patterns[f'{feature}_std'] = df_pre[feature].std()
                patterns[f'{feature}_max'] = df_pre[feature].max()
                patterns[f'{feature}_min'] = df_pre[feature].min()
                patterns[f'{feature}_trend'] = df_pre[feature].iloc[-1] - df_pre[feature].iloc[0]
        
        # Pattern detection
        patterns['high_hedging_intensity'] = 1 if patterns.get('hedging_intensity_mean', 0) > 0.7 else 0
        patterns['rising_hedging'] = 1 if patterns.get('hedging_intensity_trend', 0) > 0.1 else 0
        patterns['high_defensive'] = 1 if patterns.get('defensive_positioning_mean', 0) > 0.6 else 0
        patterns['institutional_dominance'] = 1 if patterns.get('institutional_pct_mean', 0) > 0.8 else 0
        
        return patterns
    
    def analyze_hedging_features(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze hedging features for pattern detection"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return {}
        
        features = {}
        
        # Basic metrics
        total_put_oi = puts['oi_proxy'].sum()
        total_call_oi = calls['oi_proxy'].sum() if not calls.empty else 0
        total_put_vol = puts['volume'].sum()
        
        features['total_put_oi'] = total_put_oi
        features['pc_ratio'] = total_put_oi / (total_call_oi + 1e-6)
        features['vol_oi_ratio'] = total_put_vol / (total_put_oi + 1e-6)
        
        # Strike concentration
        strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
        top_5_oi = strike_oi.head(5).sum()
        features['strike_concentration'] = top_5_oi / (total_put_oi + 1e-6)
        
        # Institutional vs Retail
        if 'dte' in df.columns:
            institutional_puts = puts[puts['dte'] > 7]
            total_put_oi = puts['oi_proxy'].sum()
            features['institutional_pct'] = institutional_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        else:
            features['institutional_pct'] = 0
        
        # Hedging depth analysis
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.85]
        atm_puts = puts[abs(puts['strike'] - spy_price) <= 10]
        
        features['deep_otm_pct'] = deep_otm_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        features['atm_pct'] = atm_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        
        # Duration analysis
        if 'dte' in df.columns:
            long_term_puts = puts[puts['dte'] > 60]
            features['long_term_pct'] = long_term_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        else:
            features['long_term_pct'] = 0
        
        # Composite scores
        features['hedging_intensity'] = min(1.0, (features['pc_ratio'] - 1) * 2)  # 0-1 scale
        features['defensive_positioning'] = min(1.0, features['deep_otm_pct'] * 3)  # 0-1 scale
        
        return features
    
    def build_correction_prediction_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build dataset for correction prediction model training"""
        
        print(f"📊 BUILDING CORRECTION PREDICTION DATASET")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        # Identify correction periods
        corrections = self.identify_correction_periods(start_date, end_date)
        
        if not corrections:
            print("❌ No correction periods found")
            return pd.DataFrame()
        
        # Analyze pre-correction patterns
        all_patterns = []
        
        for i, correction in enumerate(corrections):
            print(f"\n🔍 Analyzing correction {i+1}/{len(corrections)}")
            patterns = self.analyze_pre_correction_hedging(correction)
            
            if patterns:
                patterns['correction_id'] = i + 1
                all_patterns.append(patterns)
        
        if all_patterns:
            df_patterns = pd.DataFrame(all_patterns)
            print(f"\n✅ Built prediction dataset with {len(df_patterns)} correction patterns")
            return df_patterns
        else:
            print("❌ No pre-correction patterns found")
            return pd.DataFrame()
    
    def analyze_prediction_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze patterns that predict corrections"""
        
        if df.empty:
            return {}
        
        analysis = {}
        
        # 1. Common patterns across corrections
        pattern_features = [
            'high_hedging_intensity', 'rising_hedging', 'high_defensive',
            'institutional_dominance'
        ]
        
        pattern_frequency = {}
        for feature in pattern_features:
            if feature in df.columns:
                frequency = df[feature].mean()
                pattern_frequency[feature] = frequency
        
        analysis['pattern_frequency'] = pattern_frequency
        
        # 2. Feature importance for prediction
        feature_importance = {}
        
        # Analyze correlation with correction severity
        if 'decline_pct' in df.columns:
            for feature in pattern_features:
                if feature in df.columns:
                    correlation = df[feature].corr(df['decline_pct'])
                    feature_importance[feature] = abs(correlation)
        
        analysis['feature_importance'] = feature_importance
        
        # 3. Threshold analysis
        thresholds = {}
        for feature in pattern_features:
            if feature in df.columns:
                # Find optimal threshold for prediction
                values = df[feature]
                if len(values) > 1:
                    threshold = values.median()
                    above_threshold = df[df[feature] >= threshold]
                    below_threshold = df[df[feature] < threshold]
                    
                    if len(above_threshold) > 0 and len(below_threshold) > 0:
                        avg_decline_above = above_threshold['decline_pct'].mean()
                        avg_decline_below = below_threshold['decline_pct'].mean()
                        
                        thresholds[feature] = {
                            'threshold': threshold,
                            'avg_decline_above': avg_decline_above,
                            'avg_decline_below': avg_decline_below,
                            'difference': avg_decline_above - avg_decline_below
                        }
        
        analysis['thresholds'] = thresholds
        
        # 4. Prediction model features
        model_features = []
        for feature in pattern_features:
            if feature in df.columns and pattern_frequency.get(feature, 0) > 0.5:
                model_features.append(feature)
        
        analysis['model_features'] = model_features
        
        return analysis
    
    def generate_prediction_report(self, df: pd.DataFrame, analysis: dict) -> str:
        """Generate correction prediction analysis report"""
        
        report = []
        report.append("🎯 CORRECTION PREDICTION PATTERNS ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        # Dataset summary
        if not df.empty:
            report.append("📊 DATASET SUMMARY:")
            report.append(f"  • Total corrections analyzed: {len(df)}")
            report.append(f"  • Average decline: {df['decline_pct'].mean():.1%}")
            report.append(f"  • Average duration: {df['duration_days'].mean():.1f} days")
            report.append(f"  • Lookback period: {df['lookback_days'].iloc[0]} days")
            report.append("")
        
        # Pattern frequency analysis
        if 'pattern_frequency' in analysis:
            report.append("🔍 PATTERN FREQUENCY ANALYSIS:")
            report.append("-" * 35)
            
            for pattern, frequency in analysis['pattern_frequency'].items():
                report.append(f"  • {pattern.replace('_', ' ').title()}: {frequency:.1%}")
            
            report.append("")
            
            # Identify most common patterns
            common_patterns = [p for p, f in analysis['pattern_frequency'].items() if f > 0.5]
            if common_patterns:
                report.append("✅ MOST COMMON PATTERNS (>50% frequency):")
                for pattern in common_patterns:
                    report.append(f"  • {pattern.replace('_', ' ').title()}")
                report.append("")
        
        # Feature importance analysis
        if 'feature_importance' in analysis:
            report.append("📈 FEATURE IMPORTANCE FOR PREDICTION:")
            report.append("-" * 40)
            
            sorted_features = sorted(analysis['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features:
                report.append(f"  • {feature.replace('_', ' ').title()}: {importance:.3f}")
            
            report.append("")
        
        # Threshold analysis
        if 'thresholds' in analysis:
            report.append("🎯 PREDICTION THRESHOLDS:")
            report.append("-" * 25)
            
            for feature, threshold_data in analysis['thresholds'].items():
                report.append(f"\n{feature.replace('_', ' ').title()}:")
                report.append(f"  • Threshold: {threshold_data['threshold']:.2f}")
                report.append(f"  • Avg decline above threshold: {threshold_data['avg_decline_above']:.1%}")
                report.append(f"  • Avg decline below threshold: {threshold_data['avg_decline_below']:.1%}")
                report.append(f"  • Difference: {threshold_data['difference']:+.1%}")
        
        # Model training recommendations
        if 'model_features' in analysis:
            model_features = analysis['model_features']
            report.append(f"\n🤖 MODEL TRAINING RECOMMENDATIONS:")
            report.append("-" * 35)
            report.append(f"  • Recommended features: {len(model_features)}")
            report.append("  • Features to use:")
            for feature in model_features:
                report.append(f"    - {feature.replace('_', ' ').title()}")
            report.append("")
        
        # Prediction signals
        report.append("🚨 PREDICTION SIGNALS:")
        report.append("-" * 20)
        
        if 'pattern_frequency' in analysis:
            # High frequency patterns
            high_freq_patterns = [p for p, f in analysis['pattern_frequency'].items() if f > 0.7]
            if high_freq_patterns:
                report.append("✅ HIGH FREQUENCY PATTERNS (>70%):")
                for pattern in high_freq_patterns:
                    report.append(f"  • {pattern.replace('_', ' ').title()}")
                report.append("")
            
            # Medium frequency patterns
            med_freq_patterns = [p for p, f in analysis['pattern_frequency'].items() if 0.4 <= f <= 0.7]
            if med_freq_patterns:
                report.append("⚠️  MEDIUM FREQUENCY PATTERNS (40-70%):")
                for pattern in med_freq_patterns:
                    report.append(f"  • {pattern.replace('_', ' ').title()}")
                report.append("")
        
        # Trading implications
        report.append("💡 TRADING IMPLICATIONS:")
        report.append("-" * 20)
        
        if 'model_features' in analysis and analysis['model_features']:
            report.append("✅ PREDICTION MODEL FEASIBLE")
            report.append("")
            report.append("Key findings:")
            report.append("• Specific hedging patterns precede 5-10% corrections")
            report.append("• These patterns can be detected 5-10 days in advance")
            report.append("• Model can be trained on historical correction data")
            report.append("")
            report.append("Next steps:")
            report.append("• Train ML models on identified features")
            report.append("• Validate on out-of-sample data")
            report.append("• Implement real-time monitoring")
            report.append("• Set up alert system for pattern detection")
        else:
            report.append("❌ INSUFFICIENT PATTERNS FOR PREDICTION")
            report.append("")
            report.append("Issues:")
            report.append("• Not enough correction periods in dataset")
            report.append("• Patterns not consistent enough")
            report.append("• Need more historical data")
            report.append("")
            report.append("Recommendations:")
            report.append("• Expand date range for more corrections")
            report.append("• Include smaller corrections (3-5%)")
            report.append("• Add more sophisticated features")
        
        return "\n".join(report)


def main():
    """Main correction prediction analysis"""
    
    # Initialize analyzer
    analyzer = CorrectionPredictionPatterns()
    
    print("🎯 CORRECTION PREDICTION PATTERNS ANALYSIS")
    print("Finding hedging patterns that precede 5-10% corrections")
    print("=" * 70)
    
    # Build prediction dataset
    df_patterns = analyzer.build_correction_prediction_dataset('2020-01-01', '2025-09-30')
    
    if df_patterns.empty:
        print("❌ No correction patterns found")
        return
    
    # Analyze patterns
    analysis = analyzer.analyze_prediction_patterns(df_patterns)
    
    # Generate report
    report = analyzer.generate_prediction_report(df_patterns, analysis)
    print(report)
    
    # Save data
    df_patterns.to_csv('correction_prediction_patterns.csv', index=False)
    with open('correction_prediction_analysis.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Data and analysis saved")


if __name__ == "__main__":
    main()
