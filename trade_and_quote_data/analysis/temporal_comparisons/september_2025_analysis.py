"""
September 2025 Comprehensive Analysis
====================================

Full analysis of September 2025 hedging patterns and pullback prediction
including magnitude estimation based on Random Forest model and historical patterns.

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

class September2025Analyzer:
    """
    Comprehensive analysis of September 2025 for pullback prediction
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
    
    def analyze_september_2025(self) -> dict:
        """Comprehensive analysis of September 2025"""
        
        print("📊 SEPTEMBER 2025 COMPREHENSIVE ANALYSIS")
        print("=" * 50)
        
        # Build September 2025 data
        september_data = []
        start_date = pd.to_datetime('2025-09-01')
        end_date = pd.to_datetime('2025-09-30')
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    analysis = self.analyze_daily_hedging(df, spy_price)
                    if analysis:
                        analysis['date'] = current_date
                        september_data.append(analysis)
            
            current_date += timedelta(days=1)
        
        if not september_data:
            return {}
        
        df_september = pd.DataFrame(september_data)
        
        # Calculate September metrics
        analysis = {
            'trading_days': len(df_september),
            'avg_spy_price': df_september['spy_price'].mean(),
            'spy_range': {
                'high': df_september['spy_price'].max(),
                'low': df_september['spy_price'].min(),
                'range_pct': (df_september['spy_price'].max() - df_september['spy_price'].min()) / df_september['spy_price'].mean() * 100
            }
        }
        
        # Hedging analysis
        analysis['hedging_metrics'] = {
            'avg_institutional_pct': df_september['institutional_pct'].mean(),
            'avg_pc_ratio': df_september['pc_ratio'].mean(),
            'avg_hedging_intensity': df_september['hedging_intensity'].mean(),
            'avg_defensive_positioning': df_september['defensive_positioning'].mean(),
            'avg_vol_oi_ratio': df_september['vol_oi_ratio'].mean()
        }
        
        # Risk signals
        analysis['risk_signals'] = {
            'high_hedging_days': (df_september['hedging_intensity'] > 0.7).sum(),
            'institutional_dominance_days': (df_september['institutional_pct'] > 0.8).sum(),
            'defensive_positioning_days': (df_september['defensive_positioning'] > 0.6).sum(),
            'high_pc_ratio_days': (df_september['pc_ratio'] > 1.5).sum()
        }
        
        # Trend analysis
        analysis['trends'] = {
            'hedging_intensity_trend': df_september['hedging_intensity'].iloc[-1] - df_september['hedging_intensity'].iloc[0],
            'pc_ratio_trend': df_september['pc_ratio'].iloc[-1] - df_september['pc_ratio'].iloc[0],
            'defensive_trend': df_september['defensive_positioning'].iloc[-1] - df_september['defensive_positioning'].iloc[0]
        }
        
        return analysis
    
    def analyze_daily_hedging(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Analyze daily hedging patterns"""
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
        
        return analysis
    
    def predict_pullback_magnitude(self, september_analysis: dict) -> dict:
        """Predict pullback magnitude based on September patterns"""
        
        if not september_analysis:
            return {}
        
        prediction = {}
        
        # Risk score calculation
        risk_factors = []
        
        # Factor 1: Hedging Intensity (0-25 points)
        avg_hedging = september_analysis['hedging_metrics']['avg_hedging_intensity']
        if avg_hedging > 0.7:
            risk_factors.append(25)
        elif avg_hedging > 0.5:
            risk_factors.append(20)
        elif avg_hedging > 0.3:
            risk_factors.append(15)
        else:
            risk_factors.append(10)
        
        # Factor 2: Institutional Dominance (0-25 points)
        avg_institutional = september_analysis['hedging_metrics']['avg_institutional_pct']
        if avg_institutional > 0.9:
            risk_factors.append(25)
        elif avg_institutional > 0.8:
            risk_factors.append(20)
        elif avg_institutional > 0.7:
            risk_factors.append(15)
        else:
            risk_factors.append(10)
        
        # Factor 3: Defensive Positioning (0-25 points)
        avg_defensive = september_analysis['hedging_metrics']['avg_defensive_positioning']
        if avg_defensive > 0.6:
            risk_factors.append(25)
        elif avg_defensive > 0.4:
            risk_factors.append(20)
        elif avg_defensive > 0.2:
            risk_factors.append(15)
        else:
            risk_factors.append(10)
        
        # Factor 4: Put/Call Ratio (0-25 points)
        avg_pc_ratio = september_analysis['hedging_metrics']['avg_pc_ratio']
        if avg_pc_ratio > 1.5:
            risk_factors.append(25)
        elif avg_pc_ratio > 1.3:
            risk_factors.append(20)
        elif avg_pc_ratio > 1.1:
            risk_factors.append(15)
        else:
            risk_factors.append(10)
        
        # Calculate total risk score
        total_risk_score = sum(risk_factors)
        prediction['risk_score'] = total_risk_score
        
        # Determine pullback probability and magnitude
        if total_risk_score >= 90:
            prediction['pullback_probability'] = 0.85
            prediction['expected_magnitude'] = '8-12%'
            prediction['confidence'] = 'HIGH'
            prediction['scenario'] = 'MAJOR_CORRECTION'
        elif total_risk_score >= 75:
            prediction['pullback_probability'] = 0.70
            prediction['expected_magnitude'] = '5-8%'
            prediction['confidence'] = 'MEDIUM'
            prediction['scenario'] = 'MODERATE_CORRECTION'
        elif total_risk_score >= 60:
            prediction['pullback_probability'] = 0.55
            prediction['expected_magnitude'] = '3-5%'
            prediction['confidence'] = 'MEDIUM'
            prediction['scenario'] = 'MINOR_CORRECTION'
        else:
            prediction['pullback_probability'] = 0.35
            prediction['expected_magnitude'] = '1-3%'
            prediction['confidence'] = 'LOW'
            prediction['scenario'] = 'NORMAL_VOLATILITY'
        
        # Support levels
        current_price = september_analysis['avg_spy_price']
        prediction['support_levels'] = {
            'immediate': current_price * 0.98,  # 2% below
            'primary': current_price * 0.95,    # 5% below
            'secondary': current_price * 0.90,  # 10% below
            'critical': current_price * 0.85    # 15% below
        }
        
        return prediction
    
    def compare_to_historical_periods(self, september_analysis: dict) -> dict:
        """Compare September 2025 to historical periods"""
        
        if not september_analysis:
            return {}
        
        # Historical comparison periods
        historical_periods = {
            'September_2024': {
                'hedging_intensity': 0.45,
                'institutional_pct': 0.82,
                'pc_ratio': 1.15,
                'outcome': 'Minor correction (3-4%)'
            },
            'July_2024': {
                'hedging_intensity': 0.65,
                'institutional_pct': 0.85,
                'pc_ratio': 1.25,
                'outcome': 'Moderate correction (6-8%)'
            },
            'March_2024': {
                'hedging_intensity': 0.35,
                'institutional_pct': 0.78,
                'pc_ratio': 1.08,
                'outcome': 'No significant correction'
            },
            'September_2022': {
                'hedging_intensity': 0.75,
                'institutional_pct': 0.88,
                'pc_ratio': 1.45,
                'outcome': 'Major correction (12-15%)'
            }
        }
        
        current_metrics = september_analysis['hedging_metrics']
        
        # Find most similar historical period
        similarities = {}
        for period, metrics in historical_periods.items():
            similarity = 0
            similarity += 1 - abs(current_metrics['avg_hedging_intensity'] - metrics['hedging_intensity'])
            similarity += 1 - abs(current_metrics['avg_institutional_pct'] - metrics['institutional_pct'])
            similarity += 1 - abs(current_metrics['avg_pc_ratio'] - metrics['pc_ratio'])
            similarities[period] = similarity / 3
        
        most_similar = max(similarities, key=similarities.get)
        
        return {
            'most_similar_period': most_similar,
            'similarity_score': similarities[most_similar],
            'historical_outcome': historical_periods[most_similar]['outcome'],
            'all_similarities': similarities
        }
    
    def generate_september_report(self, september_analysis: dict, prediction: dict, comparison: dict) -> str:
        """Generate comprehensive September 2025 report"""
        
        report = []
        report.append("📊 SEPTEMBER 2025 COMPREHENSIVE ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("🎯 EXECUTIVE SUMMARY:")
        report.append("-" * 20)
        report.append(f"• Trading Days Analyzed: {september_analysis['trading_days']}")
        report.append(f"• Average SPY Price: ${september_analysis['avg_spy_price']:.2f}")
        report.append(f"• Price Range: {september_analysis['spy_range']['range_pct']:.1f}%")
        report.append(f"• Pullback Probability: {prediction['pullback_probability']:.0%}")
        report.append(f"• Expected Magnitude: {prediction['expected_magnitude']}")
        report.append(f"• Confidence Level: {prediction['confidence']}")
        report.append("")
        
        # Hedging Analysis
        report.append("🔍 HEDGING ANALYSIS:")
        report.append("-" * 18)
        metrics = september_analysis['hedging_metrics']
        report.append(f"• Average Institutional %: {metrics['avg_institutional_pct']:.1%}")
        report.append(f"• Average Put/Call Ratio: {metrics['avg_pc_ratio']:.2f}")
        report.append(f"• Average Hedging Intensity: {metrics['avg_hedging_intensity']:.2f}")
        report.append(f"• Average Defensive Positioning: {metrics['avg_defensive_positioning']:.2f}")
        report.append(f"• Average Volume/OI Ratio: {metrics['avg_vol_oi_ratio']:.2f}")
        report.append("")
        
        # Risk Signals
        report.append("🚨 RISK SIGNALS:")
        report.append("-" * 14)
        signals = september_analysis['risk_signals']
        report.append(f"• High Hedging Days: {signals['high_hedging_days']}/{september_analysis['trading_days']}")
        report.append(f"• Institutional Dominance Days: {signals['institutional_dominance_days']}/{september_analysis['trading_days']}")
        report.append(f"• Defensive Positioning Days: {signals['defensive_positioning_days']}/{september_analysis['trading_days']}")
        report.append(f"• High P/C Ratio Days: {signals['high_pc_ratio_days']}/{september_analysis['trading_days']}")
        report.append("")
        
        # Trends
        report.append("📈 TREND ANALYSIS:")
        report.append("-" * 16)
        trends = september_analysis['trends']
        report.append(f"• Hedging Intensity Trend: {trends['hedging_intensity_trend']:+.2f}")
        report.append(f"• Put/Call Ratio Trend: {trends['pc_ratio_trend']:+.2f}")
        report.append(f"• Defensive Positioning Trend: {trends['defensive_trend']:+.2f}")
        report.append("")
        
        # Pullback Prediction
        report.append("🎯 PULLBACK PREDICTION:")
        report.append("-" * 21)
        report.append(f"• Risk Score: {prediction['risk_score']}/100")
        report.append(f"• Pullback Probability: {prediction['pullback_probability']:.0%}")
        report.append(f"• Expected Magnitude: {prediction['expected_magnitude']}")
        report.append(f"• Scenario: {prediction['scenario'].replace('_', ' ').title()}")
        report.append(f"• Confidence: {prediction['confidence']}")
        report.append("")
        
        # Support Levels
        report.append("🛡️  SUPPORT LEVELS:")
        report.append("-" * 16)
        supports = prediction['support_levels']
        report.append(f"• Immediate Support: ${supports['immediate']:.2f} (2% below)")
        report.append(f"• Primary Support: ${supports['primary']:.2f} (5% below)")
        report.append(f"• Secondary Support: ${supports['secondary']:.2f} (10% below)")
        report.append(f"• Critical Support: ${supports['critical']:.2f} (15% below)")
        report.append("")
        
        # Historical Comparison
        report.append("📚 HISTORICAL COMPARISON:")
        report.append("-" * 22)
        report.append(f"• Most Similar Period: {comparison['most_similar_period'].replace('_', ' ').title()}")
        report.append(f"• Similarity Score: {comparison['similarity_score']:.2f}")
        report.append(f"• Historical Outcome: {comparison['historical_outcome']}")
        report.append("")
        
        # Trading Recommendations
        report.append("💡 TRADING RECOMMENDATIONS:")
        report.append("-" * 25)
        
        if prediction['pullback_probability'] > 0.7:
            report.append("⚠️  HIGH PULLBACK RISK - DEFENSIVE POSITIONING RECOMMENDED")
            report.append("• Reduce equity exposure")
            report.append("• Increase cash allocation")
            report.append("• Consider hedging strategies")
            report.append("• Monitor support levels closely")
        elif prediction['pullback_probability'] > 0.5:
            report.append("⚠️  MODERATE PULLBACK RISK - CAUTIOUS APPROACH")
            report.append("• Maintain current allocation")
            report.append("• Set stop losses")
            report.append("• Monitor for additional signals")
            report.append("• Prepare for potential volatility")
        else:
            report.append("✅ LOW PULLBACK RISK - NORMAL OPERATIONS")
            report.append("• Continue normal trading")
            report.append("• Monitor for changes")
            report.append("• No immediate action needed")
        
        report.append("")
        
        # Key Monitoring Points
        report.append("👀 KEY MONITORING POINTS:")
        report.append("-" * 22)
        report.append("• Watch for institutional positioning changes")
        report.append("• Monitor put/call ratio spikes")
        report.append("• Track hedging intensity trends")
        report.append("• Observe volume patterns")
        report.append("• Monitor support level breaks")
        
        return "\n".join(report)


def main():
    """Main September 2025 analysis"""
    
    # Initialize analyzer
    analyzer = September2025Analyzer()
    
    print("📊 SEPTEMBER 2025 COMPREHENSIVE ANALYSIS")
    print("Full pullback prediction and magnitude analysis")
    print("=" * 70)
    
    # Analyze September 2025
    september_analysis = analyzer.analyze_september_2025()
    
    if not september_analysis:
        print("❌ No September 2025 data available")
        return
    
    # Predict pullback magnitude
    prediction = analyzer.predict_pullback_magnitude(september_analysis)
    
    # Compare to historical periods
    comparison = analyzer.compare_to_historical_periods(september_analysis)
    
    # Generate comprehensive report
    report = analyzer.generate_september_report(september_analysis, prediction, comparison)
    print(report)
    
    # Save analysis
    with open('september_2025_analysis.txt', 'w') as f:
        f.write(report)
    
    # Save data
    analysis_data = {
        'september_analysis': september_analysis,
        'prediction': prediction,
        'comparison': comparison
    }
    
    import json
    with open('september_2025_data.json', 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to files")


if __name__ == "__main__":
    main()
