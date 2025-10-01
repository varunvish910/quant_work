"""
SPY Hedging Activity Analyzer
============================

This script analyzes hedging activity patterns in SPY options data to identify
buildup that may precede market pullbacks. It focuses on:

1. Put option accumulation patterns
2. Volume/OI ratios indicating hedging vs speculation
3. Strike concentration at key support levels
4. Temporal patterns in hedging activity
5. Correlation with subsequent market moves

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

class HedgingActivityAnalyzer:
    """
    Analyzes hedging activity patterns in SPY options data
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.spy_price_cache = {}
        
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
    
    def get_spy_price(self, date: str) -> float:
        """Get SPY price for a given date from the data"""
        try:
            df = self.load_daily_data(date)
            if not df.empty and 'underlying_price' in df.columns:
                return df['underlying_price'].iloc[0]
            return 600  # Fallback
        except:
            return 600  # Fallback
    
    def calculate_hedging_metrics(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate hedging-specific metrics"""
        if df.empty:
            return {}
        
        metrics = {}
        
        # Separate puts and calls
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return metrics
        
        # 1. Put concentration at key support levels
        support_levels = [spy_price * 0.95, spy_price * 0.90, spy_price * 0.85, spy_price * 0.80]
        
        for level in support_levels:
            level_puts = puts[abs(puts['strike'] - level) <= 5]  # Within $5 of level
            if not level_puts.empty:
                level_oi = level_puts['oi_proxy'].sum()
                level_volume = level_puts['volume'].sum()
                level_vol_oi = level_volume / (level_oi + 1e-6)
                
                metrics[f'put_oi_level_{int(level)}'] = level_oi
                metrics[f'put_volume_level_{int(level)}'] = level_volume
                metrics[f'put_vol_oi_level_{int(level)}'] = level_vol_oi
        
        # 2. Deep OTM put activity (potential hedging)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.90]
        if not deep_otm_puts.empty:
            metrics['deep_otm_put_oi'] = deep_otm_puts['oi_proxy'].sum()
            metrics['deep_otm_put_volume'] = deep_otm_puts['volume'].sum()
            metrics['deep_otm_put_vol_oi'] = deep_otm_puts['volume'].sum() / (deep_otm_puts['oi_proxy'].sum() + 1e-6)
        
        # 3. Put/Call ratio metrics
        total_put_oi = puts['oi_proxy'].sum()
        total_call_oi = calls['oi_proxy'].sum() if not calls.empty else 0
        total_put_volume = puts['volume'].sum()
        total_call_volume = calls['volume'].sum() if not calls.empty else 0
        
        metrics['pc_ratio_oi'] = total_put_oi / (total_call_oi + 1e-6)
        metrics['pc_ratio_volume'] = total_put_volume / (total_call_volume + 1e-6)
        
        # 4. Hedging activity score (low V/OI ratio indicates accumulation/hedging)
        put_vol_oi = total_put_volume / (total_put_oi + 1e-6)
        metrics['hedging_activity_score'] = 1 / (put_vol_oi + 0.1)  # Higher = more hedging-like
        
        # 5. Strike concentration
        if not puts.empty:
            # Find strikes with highest OI
            top_strikes = puts.nlargest(5, 'oi_proxy')
            top_oi_pct = top_strikes['oi_proxy'].sum() / total_put_oi
            metrics['put_strike_concentration'] = top_oi_pct
            
            # Check for unusual concentration at specific strikes
            max_single_strike_oi = puts['oi_proxy'].max()
            metrics['max_single_strike_oi_pct'] = max_single_strike_oi / total_put_oi
        
        # 6. Time to expiration patterns (hedging often uses longer-dated options)
        if 'dte' in puts.columns:
            avg_put_dte = puts['dte'].mean()
            long_dated_puts = puts[puts['dte'] > 60]
            long_dated_oi_pct = long_dated_puts['oi_proxy'].sum() / total_put_oi if not long_dated_puts.empty else 0
            
            metrics['avg_put_dte'] = avg_put_dte
            metrics['long_dated_put_oi_pct'] = long_dated_oi_pct
        
        return metrics
    
    def analyze_hedging_buildup(self, start_date: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
        """Analyze hedging activity buildup over time"""
        print(f"Analyzing hedging activity from {start_date} to {end_date}")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        daily_metrics = []
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_date.weekday() < 5:
                df = self.load_daily_data(date_str)
                if not df.empty:
                    spy_price = self.get_spy_price(date_str)
                    metrics = self.calculate_hedging_metrics(df, spy_price)
                    
                    if metrics:
                        metrics['date'] = current_date
                        metrics['spy_price'] = spy_price
                        daily_metrics.append(metrics)
            
            current_date += timedelta(days=1)
        
        if not daily_metrics:
            print("No data found for the specified period")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_metrics = pd.DataFrame(daily_metrics)
        df_metrics = df_metrics.sort_values('date').reset_index(drop=True)
        
        # Calculate rolling averages and trends
        df_metrics = self._calculate_rolling_metrics(df_metrics, lookback_days)
        
        return df_metrics
    
    def _calculate_rolling_metrics(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling averages and trend indicators"""
        
        # Rolling averages for key metrics
        rolling_cols = ['pc_ratio_oi', 'hedging_activity_score', 'put_strike_concentration']
        
        for col in rolling_cols:
            if col in df.columns:
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_trend'] = df[col] - df[f'{col}_ma_{window}']
        
        # Calculate hedging buildup score
        if 'hedging_activity_score' in df.columns:
            df['hedging_buildup'] = df['hedging_activity_score'].rolling(window=5, min_periods=1).mean()
            df['hedging_acceleration'] = df['hedging_activity_score'].diff().rolling(window=3, min_periods=1).mean()
        
        # Calculate volatility of hedging metrics
        if 'pc_ratio_oi' in df.columns:
            df['pc_ratio_volatility'] = df['pc_ratio_oi'].rolling(window=10, min_periods=1).std()
        
        return df
    
    def identify_hedging_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify potential hedging signals that may precede pullbacks"""
        
        signals = df.copy()
        
        # Signal 1: High hedging activity score
        if 'hedging_activity_score' in df.columns:
            signals['high_hedging'] = (df['hedging_activity_score'] > df['hedging_activity_score'].quantile(0.8)).astype(int)
        
        # Signal 2: Rising hedging activity
        if 'hedging_acceleration' in df.columns:
            signals['rising_hedging'] = (df['hedging_acceleration'] > 0).astype(int)
        
        # Signal 3: High put/call ratio
        if 'pc_ratio_oi' in df.columns:
            signals['high_pc_ratio'] = (df['pc_ratio_oi'] > df['pc_ratio_oi'].quantile(0.8)).astype(int)
        
        # Signal 4: Strike concentration
        if 'put_strike_concentration' in df.columns:
            signals['high_concentration'] = (df['put_strike_concentration'] > df['put_strike_concentration'].quantile(0.8)).astype(int)
        
        # Signal 5: Deep OTM put activity
        if 'deep_otm_put_vol_oi' in df.columns:
            signals['deep_otm_activity'] = (df['deep_otm_put_vol_oi'] < 0.3).astype(int)  # Low V/OI = accumulation
        
        # Combined hedging signal
        signal_cols = ['high_hedging', 'rising_hedging', 'high_pc_ratio', 'high_concentration', 'deep_otm_activity']
        available_signals = [col for col in signal_cols if col in signals.columns]
        
        if available_signals:
            signals['hedging_signal_strength'] = signals[available_signals].sum(axis=1)
            signals['hedging_signal'] = (signals['hedging_signal_strength'] >= 3).astype(int)
        
        return signals
    
    def analyze_pullback_correlation(self, df: pd.DataFrame, forward_days: int = 5) -> dict:
        """Analyze correlation between hedging signals and subsequent pullbacks"""
        
        if 'spy_price' not in df.columns:
            return {}
        
        # Calculate forward returns
        df['forward_return'] = df['spy_price'].pct_change(forward_days).shift(-forward_days)
        df['pullback'] = (df['forward_return'] < -0.02).astype(int)  # 2%+ pullback
        
        # Calculate correlations
        correlations = {}
        
        if 'hedging_signal' in df.columns:
            correlations['hedging_signal_pullback_corr'] = df['hedging_signal'].corr(df['pullback'])
        
        if 'hedging_activity_score' in df.columns:
            correlations['hedging_score_pullback_corr'] = df['hedging_activity_score'].corr(df['pullback'])
        
        if 'pc_ratio_oi' in df.columns:
            correlations['pc_ratio_pullback_corr'] = df['pc_ratio_oi'].corr(df['pullback'])
        
        # Calculate signal effectiveness
        if 'hedging_signal' in df.columns:
            signal_days = df[df['hedging_signal'] == 1]
            if not signal_days.empty:
                pullback_rate = signal_days['pullback'].mean()
                correlations['hedging_signal_pullback_rate'] = pullback_rate
        
        return correlations
    
    def generate_hedging_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive hedging activity report"""
        
        if df.empty:
            return "No data available for analysis"
        
        report = []
        report.append("=" * 60)
        report.append("SPY HEDGING ACTIVITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        report.append(f"Total Trading Days: {len(df)}")
        report.append("")
        
        # Current hedging status
        latest = df.iloc[-1]
        report.append("🔍 CURRENT HEDGING STATUS:")
        report.append("-" * 30)
        
        if 'hedging_activity_score' in latest:
            score = latest['hedging_activity_score']
            if score > df['hedging_activity_score'].quantile(0.8):
                status = "HIGH HEDGING ACTIVITY"
            elif score > df['hedging_activity_score'].quantile(0.6):
                status = "ELEVATED HEDGING"
            else:
                status = "NORMAL HEDGING"
            report.append(f"Hedging Activity Score: {score:.2f} ({status})")
        
        if 'pc_ratio_oi' in latest:
            pc_ratio = latest['pc_ratio_oi']
            report.append(f"Put/Call Ratio (OI): {pc_ratio:.2f}")
        
        if 'hedging_signal' in latest:
            signal_strength = latest.get('hedging_signal_strength', 0)
            report.append(f"Hedging Signal Strength: {signal_strength}/5")
        
        report.append("")
        
        # Recent trends
        report.append("📈 RECENT TRENDS:")
        report.append("-" * 20)
        
        if 'hedging_activity_score' in df.columns:
            recent_trend = df['hedging_activity_score'].tail(5).mean()
            historical_avg = df['hedging_activity_score'].mean()
            trend_direction = "RISING" if recent_trend > historical_avg else "FALLING"
            report.append(f"Hedging Activity: {trend_direction} (Recent: {recent_trend:.2f} vs Avg: {historical_avg:.2f})")
        
        if 'pc_ratio_oi' in df.columns:
            recent_pc = df['pc_ratio_oi'].tail(5).mean()
            historical_pc = df['pc_ratio_oi'].mean()
            pc_trend = "RISING" if recent_pc > historical_pc else "FALLING"
            report.append(f"Put/Call Ratio: {pc_trend} (Recent: {recent_pc:.2f} vs Avg: {historical_pc:.2f})")
        
        report.append("")
        
        # Key levels analysis
        report.append("🎯 KEY SUPPORT LEVELS:")
        report.append("-" * 25)
        
        support_cols = [col for col in df.columns if 'put_oi_level_' in col]
        for col in support_cols:
            level = col.split('_')[-1]
            latest_oi = latest.get(col, 0)
            avg_oi = df[col].mean()
            report.append(f"Level ${level}: {latest_oi:,.0f} OI (Avg: {avg_oi:,.0f})")
        
        report.append("")
        
        # Risk assessment
        report.append("⚠️  RISK ASSESSMENT:")
        report.append("-" * 20)
        
        if 'hedging_signal' in df.columns:
            recent_signals = df['hedging_signal'].tail(10).sum()
            if recent_signals >= 5:
                risk_level = "HIGH - Multiple hedging signals detected"
            elif recent_signals >= 3:
                risk_level = "MEDIUM - Some hedging signals present"
            else:
                risk_level = "LOW - Normal hedging activity"
            report.append(f"Risk Level: {risk_level}")
        
        return "\n".join(report)
    
    def plot_hedging_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Create comprehensive hedging analysis plots"""
        
        if df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SPY Hedging Activity Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Hedging Activity Score over time
        if 'hedging_activity_score' in df.columns:
            axes[0, 0].plot(df['date'], df['hedging_activity_score'], label='Hedging Score', linewidth=2)
            if 'hedging_activity_score_ma_30' in df.columns:
                axes[0, 0].plot(df['date'], df['hedging_activity_score_ma_30'], 
                               label='30-day MA', alpha=0.7, linestyle='--')
            axes[0, 0].set_title('Hedging Activity Score')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Put/Call Ratio
        if 'pc_ratio_oi' in df.columns:
            axes[0, 1].plot(df['date'], df['pc_ratio_oi'], label='P/C Ratio', color='red', linewidth=2)
            if 'pc_ratio_oi_ma_30' in df.columns:
                axes[0, 1].plot(df['date'], df['pc_ratio_oi_ma_30'], 
                               label='30-day MA', alpha=0.7, linestyle='--')
            axes[0, 1].set_title('Put/Call Ratio (Open Interest)')
            axes[0, 1].set_ylabel('Ratio')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Strike Concentration
        if 'put_strike_concentration' in df.columns:
            axes[1, 0].plot(df['date'], df['put_strike_concentration'], 
                           label='Strike Concentration', color='orange', linewidth=2)
            axes[1, 0].set_title('Put Strike Concentration')
            axes[1, 0].set_ylabel('Concentration %')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Hedging Signals
        if 'hedging_signal' in df.columns:
            signal_dates = df[df['hedging_signal'] == 1]['date']
            axes[1, 1].scatter(signal_dates, [1] * len(signal_dates), 
                             color='red', s=50, alpha=0.7, label='Hedging Signals')
            axes[1, 1].set_title('Hedging Signals Over Time')
            axes[1, 1].set_ylabel('Signal')
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
    analyzer = HedgingActivityAnalyzer()
    
    # Analyze recent period (last 2 weeks of available data)
    end_date = '2025-09-30'  # Latest available data
    start_date = '2025-09-17'  # 2 weeks back
    
    print("🔍 Analyzing SPY Hedging Activity...")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 50)
    
    # Run analysis
    df = analyzer.analyze_hedging_buildup(start_date, end_date)
    
    if df.empty:
        print("❌ No data found for analysis")
        return
    
    # Identify signals
    df_signals = analyzer.identify_hedging_signals(df)
    
    # Generate report
    report = analyzer.generate_hedging_report(df_signals)
    print(report)
    
    # Analyze correlations
    correlations = analyzer.analyze_pullback_correlation(df_signals)
    if correlations:
        print("\n📊 CORRELATION ANALYSIS:")
        print("-" * 25)
        for metric, corr in correlations.items():
            print(f"{metric}: {corr:.3f}")
    
    # Create plots
    analyzer.plot_hedging_analysis(df_signals, 'hedging_analysis.png')
    
    # Save results
    df_signals.to_csv('hedging_analysis_results.csv', index=False)
    print(f"\n💾 Results saved to 'hedging_analysis_results.csv'")


if __name__ == "__main__":
    main()
