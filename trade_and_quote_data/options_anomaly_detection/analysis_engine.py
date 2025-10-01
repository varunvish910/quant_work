"""
SPY Options Anomaly Detection - Analysis and Reporting Engine
============================================================

This module provides comprehensive analysis and reporting capabilities:
- Historical analysis and backtesting
- Performance metrics and validation
- Signal quality assessment
- Risk analysis and monitoring
- Report generation

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsAnalysisEngine:
    """
    Comprehensive analysis engine for SPY options anomaly detection
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.results_cache = {}
        
    def load_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical options data for analysis
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_data = []
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_date.weekday() < 5:
                df = self._load_daily_data(date_str)
                if df is not None and len(df) > 0:
                    all_data.append(df)
            
            current_date += timedelta(days=1)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _load_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load daily data"""
        try:
            year = date[:4]
            month = date[5:7]
            
            file_path = self.data_dir / year / month / f"SPY_options_snapshot_{date}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(date)
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {date}: {e}")
            return None
    
    def calculate_daily_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate daily aggregate metrics
        """
        if df is None or len(df) == 0:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['date'] = df['date'].iloc[0] if 'date' in df.columns else None
        metrics['total_contracts'] = len(df)
        metrics['total_volume'] = df['volume'].sum() if 'volume' in df.columns else 0
        metrics['total_oi'] = df['oi_proxy'].sum() if 'oi_proxy' in df.columns else 0
        metrics['avg_oi'] = df['oi_proxy'].mean() if 'oi_proxy' in df.columns else 0
        metrics['median_oi'] = df['oi_proxy'].median() if 'oi_proxy' in df.columns else 0
        
        # Put/Call ratios
        if 'option_type' in df.columns:
            calls = df[df['option_type'] == 'C']
            puts = df[df['option_type'] == 'P']
            
            if len(calls) > 0 and len(puts) > 0:
                metrics['pc_ratio_volume'] = puts['volume'].sum() / calls['volume'].sum() if 'volume' in df.columns else 1.0
                metrics['pc_ratio_oi'] = puts['oi_proxy'].sum() / calls['oi_proxy'].sum() if 'oi_proxy' in df.columns else 1.0
                metrics['pc_ratio_contracts'] = len(puts) / len(calls)
            else:
                metrics['pc_ratio_volume'] = 1.0
                metrics['pc_ratio_oi'] = 1.0
                metrics['pc_ratio_contracts'] = 1.0
        else:
            metrics['pc_ratio_volume'] = 1.0
            metrics['pc_ratio_oi'] = 1.0
            metrics['pc_ratio_contracts'] = 1.0
        
        # Price level
        metrics['underlying_price'] = df['underlying_price'].iloc[0] if 'underlying_price' in df.columns else 0.0
        
        return metrics
    
    def analyze_historical_patterns(self, start_date: str, end_date: str) -> Dict:
        """
        Analyze historical patterns and trends
        """
        logger.info(f"Analyzing historical patterns from {start_date} to {end_date}")
        
        # Load historical data
        df = self.load_historical_data(start_date, end_date)
        
        if df.empty:
            logger.error("No historical data found")
            return {}
        
        # Calculate daily metrics
        daily_metrics = []
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            metrics = self.calculate_daily_metrics(day_data)
            if metrics:
                daily_metrics.append(metrics)
        
        if not daily_metrics:
            return {}
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(daily_metrics)
        metrics_df['date'] = pd.to_datetime(metrics_df['date'])
        metrics_df = metrics_df.sort_values('date')
        
        # Calculate trends and patterns
        patterns = {}
        
        # 1. Volume trends
        if 'total_volume' in metrics_df.columns:
            patterns['volume_trend'] = self._calculate_trend(metrics_df['total_volume'])
            patterns['volume_volatility'] = metrics_df['total_volume'].std()
            patterns['avg_daily_volume'] = metrics_df['total_volume'].mean()
        
        # 2. OI trends
        if 'total_oi' in metrics_df.columns:
            patterns['oi_trend'] = self._calculate_trend(metrics_df['total_oi'])
            patterns['oi_volatility'] = metrics_df['total_oi'].std()
            patterns['avg_daily_oi'] = metrics_df['total_oi'].mean()
        
        # 3. Put/Call ratio trends
        if 'pc_ratio_oi' in metrics_df.columns:
            patterns['pc_ratio_trend'] = self._calculate_trend(metrics_df['pc_ratio_oi'])
            patterns['pc_ratio_volatility'] = metrics_df['pc_ratio_oi'].std()
            patterns['avg_pc_ratio'] = metrics_df['pc_ratio_oi'].mean()
        
        # 4. Seasonal patterns
        patterns['seasonal_patterns'] = self._analyze_seasonality(metrics_df)
        
        # 5. Correlation analysis
        patterns['correlations'] = self._analyze_correlations(metrics_df)
        
        return {
            'patterns': patterns,
            'daily_metrics': metrics_df,
            'summary_stats': self._calculate_summary_stats(metrics_df)
        }
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend coefficient"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns"""
        if 'date' not in df.columns:
            return {}
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        seasonality = {}
        
        # Day of week patterns
        if 'total_volume' in df.columns:
            dow_volume = df.groupby('day_of_week')['total_volume'].mean()
            seasonality['day_of_week_volume'] = dow_volume.to_dict()
        
        # Monthly patterns
        if 'total_volume' in df.columns:
            monthly_volume = df.groupby('month')['total_volume'].mean()
            seasonality['monthly_volume'] = monthly_volume.to_dict()
        
        return seasonality
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between metrics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': self._find_strong_correlations(correlation_matrix)
        }
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations above threshold"""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    strong_corrs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return strong_corrs
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        stats_dict = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'date':
                stats_dict[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                }
        
        return stats_dict
    
    def backtest_anomaly_detection(self, start_date: str, end_date: str, 
                                  detector) -> Dict:
        """
        Backtest anomaly detection performance
        """
        logger.info(f"Backtesting anomaly detection from {start_date} to {end_date}")
        
        # Load historical data
        df = self.load_historical_data(start_date, end_date)
        
        if df.empty:
            logger.error("No historical data found for backtesting")
            return {}
        
        # Process each day
        backtest_results = []
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                # Detect anomalies
                features = detector.prepare_features(day_data)
                if len(features) > 0:
                    anomaly_results = detector.ensemble_detection(features)
                    metrics = detector.calculate_anomaly_metrics(day_data, anomaly_results)
                    signals = detector.generate_signals(day_data, anomaly_results)
                    
                    backtest_results.append({
                        'date': date_str,
                        'metrics': metrics,
                        'signals': signals
                    })
            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")
                continue
        
        if not backtest_results:
            return {}
        
        # Analyze backtest results
        return self._analyze_backtest_results(backtest_results)
    
    def _analyze_backtest_results(self, results: List[Dict]) -> Dict:
        """Analyze backtest results"""
        if not results:
            return {}
        
        # Extract metrics
        dates = [r['date'] for r in results]
        anomaly_rates = [r['metrics'].get('ensemble_anomaly_rate', 0) for r in results]
        signal_qualities = [r['signals'].get('quality', 'low') for r in results]
        
        # Calculate performance metrics
        analysis = {
            'total_days': len(results),
            'avg_anomaly_rate': np.mean(anomaly_rates),
            'anomaly_rate_std': np.std(anomaly_rates),
            'high_quality_signals': sum(1 for q in signal_qualities if q == 'high'),
            'medium_quality_signals': sum(1 for q in signal_qualities if q == 'medium'),
            'low_quality_signals': sum(1 for q in signal_qualities if q == 'low'),
            'signal_quality_distribution': {
                'high': sum(1 for q in signal_qualities if q == 'high') / len(signal_qualities),
                'medium': sum(1 for q in signal_qualities if q == 'medium') / len(signal_qualities),
                'low': sum(1 for q in signal_qualities if q == 'low') / len(signal_qualities)
            }
        }
        
        return analysis
    
    def generate_report(self, analysis_results: Dict, output_file: str = "anomaly_analysis_report.html"):
        """
        Generate comprehensive HTML report
        """
        html_content = self._create_html_report(analysis_results)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {output_file}")
    
    def _create_html_report(self, results: Dict) -> str:
        """Create HTML report content"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SPY Options Anomaly Detection Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }
                .high { color: #28a745; }
                .medium { color: #ffc107; }
                .low { color: #dc3545; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SPY Options Anomaly Detection Analysis Report</h1>
                <p>Generated on: {}</p>
            </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add sections based on available results
        if 'patterns' in results:
            html += self._add_patterns_section(results['patterns'])
        
        if 'summary_stats' in results:
            html += self._add_summary_section(results['summary_stats'])
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _add_patterns_section(self, patterns: Dict) -> str:
        """Add patterns section to HTML report"""
        html = """
        <div class="section">
            <h2>Historical Patterns Analysis</h2>
        """
        
        if 'volume_trend' in patterns:
            html += f"<div class='metric'>Volume Trend: {patterns['volume_trend']:.4f}</div>"
        
        if 'oi_trend' in patterns:
            html += f"<div class='metric'>OI Trend: {patterns['oi_trend']:.4f}</div>"
        
        if 'pc_ratio_trend' in patterns:
            html += f"<div class='metric'>Put/Call Ratio Trend: {patterns['pc_ratio_trend']:.4f}</div>"
        
        html += "</div>"
        return html
    
    def _add_summary_section(self, summary_stats: Dict) -> str:
        """Add summary statistics section to HTML report"""
        html = """
        <div class="section">
            <h2>Summary Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
        """
        
        for metric, stats in summary_stats.items():
            html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{stats['mean']:.2f}</td>
                    <td>{stats['std']:.2f}</td>
                    <td>{stats['min']:.2f}</td>
                    <td>{stats['max']:.2f}</td>
                </tr>
            """
        
        html += "</table></div>"
        return html


def main():
    """
    Example usage of the analysis engine
    """
    # Initialize analysis engine
    analyzer = OptionsAnalysisEngine()
    
    # Analyze historical patterns
    patterns = analyzer.analyze_historical_patterns("2024-01-01", "2024-12-31")
    
    if patterns:
        print("Historical patterns analysis completed")
        print(f"Patterns: {patterns['patterns']}")
        
        # Generate report
        analyzer.generate_report(patterns)


if __name__ == "__main__":
    main()
