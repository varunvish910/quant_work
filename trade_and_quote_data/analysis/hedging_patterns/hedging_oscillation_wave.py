"""
SPY Hedging Oscillation Wave Indicator
=====================================

This script creates an oscillation wave indicator that combines multiple hedging
signals into a "watch out" alert system. The wave oscillates between 0-100
with different zones indicating risk levels.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HedgingOscillationWave:
    """
    Creates an oscillation wave indicator for hedging activity
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.wave_data = []
        
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
    
    def calculate_hedging_components(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate individual hedging components for the wave"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        if puts.empty:
            return {}
        
        components = {}
        
        # 1. Deep OTM Put Accumulation (0-25 points)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.90]
        if not deep_otm_puts.empty:
            deep_otm_oi = deep_otm_puts['oi_proxy'].sum()
            deep_otm_vol = deep_otm_puts['volume'].sum()
            deep_otm_vol_oi = deep_otm_vol / (deep_otm_oi + 1e-6)
            
            # Score based on OI size and low V/OI ratio
            oi_score = min(15, deep_otm_oi / 100000)  # Scale by 100k OI
            vol_oi_score = max(0, 10 - deep_otm_vol_oi * 20)  # Lower V/OI = higher score
            components['deep_otm_score'] = oi_score + vol_oi_score
        else:
            components['deep_otm_score'] = 0
        
        # 2. Institutional Hedging (0-25 points)
        total_put_oi = puts['oi_proxy'].sum()
        if 'dte' in puts.columns:
            long_dated_puts = puts[puts['dte'] > 60]
            if not long_dated_puts.empty:
                long_dated_oi_pct = long_dated_puts['oi_proxy'].sum() / total_put_oi
                # Score based on percentage of long-dated options
                components['institutional_score'] = min(25, long_dated_oi_pct * 100)
            else:
                components['institutional_score'] = 0
        else:
            components['institutional_score'] = 0
        
        # 3. Strike Concentration (0-20 points)
        top_5_oi = puts.nlargest(5, 'oi_proxy')['oi_proxy'].sum()
        max_single_oi = puts['oi_proxy'].max()
        top_5_pct = top_5_oi / total_put_oi
        max_strike_pct = max_single_oi / total_put_oi
        
        # Score based on concentration (defensive positioning)
        concentration_score = min(20, (top_5_pct + max_strike_pct) * 200)
        components['concentration_score'] = concentration_score
        
        # 4. Volume/OI Ratio (0-15 points)
        total_put_vol = puts['volume'].sum()
        vol_oi_ratio = total_put_vol / (total_put_oi + 1e-6)
        
        # Lower V/OI ratio = more hedging-like activity
        vol_oi_score = max(0, 15 - vol_oi_ratio * 15)
        components['vol_oi_score'] = vol_oi_score
        
        # 5. Put/Call Ratio (0-15 points)
        calls = df[df['option_type'] == 'C']
        if not calls.empty:
            pc_ratio = total_put_oi / calls['oi_proxy'].sum()
            # Higher PC ratio = more bearish sentiment
            pc_score = min(15, max(0, (pc_ratio - 1) * 15))
            components['pc_ratio_score'] = pc_score
        else:
            components['pc_ratio_score'] = 0
        
        return components
    
    def calculate_oscillation_wave(self, components: dict) -> dict:
        """Calculate the oscillation wave value and zones"""
        
        # Calculate total wave value (0-100)
        total_score = sum(components.values())
        wave_value = min(100, total_score)
        
        # Determine wave zone
        if wave_value >= 80:
            zone = "CRITICAL"
            color = "red"
            alert = "IMMEDIATE ATTENTION"
        elif wave_value >= 60:
            zone = "HIGH"
            color = "orange"
            alert = "WATCH OUT"
        elif wave_value >= 40:
            zone = "ELEVATED"
            color = "yellow"
            alert = "MONITOR CLOSELY"
        elif wave_value >= 20:
            zone = "NORMAL"
            color = "green"
            alert = "STABLE"
        else:
            zone = "LOW"
            color = "blue"
            alert = "QUIET"
        
        # Calculate wave momentum (rate of change)
        momentum = 0  # Will be calculated when we have historical data
        
        return {
            'wave_value': wave_value,
            'zone': zone,
            'color': color,
            'alert': alert,
            'components': components,
            'momentum': momentum
        }
    
    def build_historical_wave(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build historical oscillation wave data"""
        
        print(f"🌊 BUILDING HEDGING OSCILLATION WAVE")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        wave_data = []
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    components = self.calculate_hedging_components(df, spy_price)
                    
                    if components:
                        wave_info = self.calculate_oscillation_wave(components)
                        wave_info['date'] = current_date
                        wave_info['spy_price'] = spy_price
                        wave_data.append(wave_info)
            
            current_date += timedelta(days=1)
        
        if wave_data:
            df_wave = pd.DataFrame(wave_data)
            
            # Calculate momentum (rate of change)
            df_wave['momentum'] = df_wave['wave_value'].diff()
            df_wave['momentum_ma'] = df_wave['momentum'].rolling(window=5, min_periods=1).mean()
            
            # Calculate moving averages
            df_wave['wave_ma_5'] = df_wave['wave_value'].rolling(window=5, min_periods=1).mean()
            df_wave['wave_ma_10'] = df_wave['wave_value'].rolling(window=10, min_periods=1).mean()
            df_wave['wave_ma_20'] = df_wave['wave_value'].rolling(window=20, min_periods=1).mean()
            
            print(f"✅ Built wave data for {len(df_wave)} trading days")
            return df_wave
        else:
            print("❌ No data found for the specified period")
            return pd.DataFrame()
    
    def generate_wave_alerts(self, df_wave: pd.DataFrame) -> list:
        """Generate alerts based on wave patterns"""
        
        alerts = []
        
        if df_wave.empty:
            return alerts
        
        # Get latest data
        latest = df_wave.iloc[-1]
        
        # Current alert
        current_alert = {
            'date': latest['date'],
            'wave_value': latest['wave_value'],
            'zone': latest['zone'],
            'alert': latest['alert'],
            'spy_price': latest['spy_price']
        }
        alerts.append(current_alert)
        
        # Trend alerts
        if len(df_wave) >= 5:
            recent_avg = df_wave['wave_value'].tail(5).mean()
            historical_avg = df_wave['wave_value'].mean()
            
            if recent_avg > historical_avg * 1.3:
                trend_alert = {
                    'type': 'TREND',
                    'message': f"Wave RISING - Recent avg {recent_avg:.1f} vs historical {historical_avg:.1f}",
                    'severity': 'HIGH'
                }
                alerts.append(trend_alert)
            elif recent_avg < historical_avg * 0.7:
                trend_alert = {
                    'type': 'TREND',
                    'message': f"Wave FALLING - Recent avg {recent_avg:.1f} vs historical {historical_avg:.1f}",
                    'severity': 'LOW'
                }
                alerts.append(trend_alert)
        
        # Momentum alerts
        if 'momentum' in df_wave.columns:
            recent_momentum = df_wave['momentum'].tail(3).mean()
            if recent_momentum > 5:
                momentum_alert = {
                    'type': 'MOMENTUM',
                    'message': f"Wave ACCELERATING - Momentum: {recent_momentum:.1f}",
                    'severity': 'HIGH'
                }
                alerts.append(momentum_alert)
            elif recent_momentum < -5:
                momentum_alert = {
                    'type': 'MOMENTUM',
                    'message': f"Wave DECELERATING - Momentum: {recent_momentum:.1f}",
                    'severity': 'MEDIUM'
                }
                alerts.append(momentum_alert)
        
        # Zone change alerts
        if len(df_wave) >= 2:
            current_zone = latest['zone']
            previous_zone = df_wave.iloc[-2]['zone']
            
            if current_zone != previous_zone:
                zone_alert = {
                    'type': 'ZONE_CHANGE',
                    'message': f"Zone changed from {previous_zone} to {current_zone}",
                    'severity': 'HIGH' if current_zone in ['CRITICAL', 'HIGH'] else 'MEDIUM'
                }
                alerts.append(zone_alert)
        
        return alerts
    
    def plot_oscillation_wave(self, df_wave: pd.DataFrame, save_path: str = None):
        """Create oscillation wave visualization"""
        
        if df_wave.empty:
            print("No data to plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('SPY Hedging Oscillation Wave Indicator', fontsize=16, fontweight='bold')
        
        # Plot 1: Main oscillation wave
        ax1.plot(df_wave['date'], df_wave['wave_value'], label='Wave Value', linewidth=2, color='purple')
        
        if 'wave_ma_5' in df_wave.columns:
            ax1.plot(df_wave['date'], df_wave['wave_ma_5'], label='5-day MA', alpha=0.7, linestyle='--')
        if 'wave_ma_10' in df_wave.columns:
            ax1.plot(df_wave['date'], df_wave['wave_ma_10'], label='10-day MA', alpha=0.7, linestyle='--')
        
        # Add zone bands
        ax1.axhspan(0, 20, alpha=0.1, color='blue', label='LOW (0-20)')
        ax1.axhspan(20, 40, alpha=0.1, color='green', label='NORMAL (20-40)')
        ax1.axhspan(40, 60, alpha=0.1, color='yellow', label='ELEVATED (40-60)')
        ax1.axhspan(60, 80, alpha=0.1, color='orange', label='HIGH (60-80)')
        ax1.axhspan(80, 100, alpha=0.1, color='red', label='CRITICAL (80-100)')
        
        ax1.set_title('Hedging Oscillation Wave')
        ax1.set_ylabel('Wave Value (0-100)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Wave momentum
        if 'momentum' in df_wave.columns:
            ax2.plot(df_wave['date'], df_wave['momentum'], label='Momentum', linewidth=2, color='red')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title('Wave Momentum (Rate of Change)')
            ax2.set_ylabel('Momentum')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Component breakdown
        component_cols = ['deep_otm_score', 'institutional_score', 'concentration_score', 
                         'vol_oi_score', 'pc_ratio_score']
        available_components = [col for col in component_cols if col in df_wave.columns]
        
        if available_components:
            for col in available_components:
                ax3.plot(df_wave['date'], df_wave[col], label=col.replace('_score', '').replace('_', ' ').title(), linewidth=1.5)
            
            ax3.set_title('Wave Components Breakdown')
            ax3.set_ylabel('Component Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wave plot saved to {save_path}")
        
        plt.show()
    
    def generate_wave_report(self, df_wave: pd.DataFrame, alerts: list) -> str:
        """Generate comprehensive wave report"""
        
        if df_wave.empty:
            return "No wave data available"
        
        report = []
        report.append("=" * 70)
        report.append("SPY HEDGING OSCILLATION WAVE REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current wave status
        latest = df_wave.iloc[-1]
        report.append("🌊 CURRENT WAVE STATUS:")
        report.append("-" * 25)
        report.append(f"Wave Value: {latest['wave_value']:.1f}/100")
        report.append(f"Zone: {latest['zone']}")
        report.append(f"Alert Level: {latest['alert']}")
        report.append(f"SPY Price: ${latest['spy_price']:.2f}")
        report.append("")
        
        # Wave components
        if 'components' in latest:
            components = latest['components']
            report.append("📊 WAVE COMPONENTS:")
            report.append("-" * 20)
            report.append(f"Deep OTM Score: {components.get('deep_otm_score', 0):.1f}/25")
            report.append(f"Institutional Score: {components.get('institutional_score', 0):.1f}/25")
            report.append(f"Concentration Score: {components.get('concentration_score', 0):.1f}/20")
            report.append(f"Volume/OI Score: {components.get('vol_oi_score', 0):.1f}/15")
            report.append(f"Put/Call Score: {components.get('pc_ratio_score', 0):.1f}/15")
            report.append("")
        
        # Historical context
        report.append("📈 HISTORICAL CONTEXT:")
        report.append("-" * 25)
        
        avg_wave = df_wave['wave_value'].mean()
        max_wave = df_wave['wave_value'].max()
        min_wave = df_wave['wave_value'].min()
        
        report.append(f"Average Wave Value: {avg_wave:.1f}")
        report.append(f"Maximum Wave Value: {max_wave:.1f}")
        report.append(f"Minimum Wave Value: {min_wave:.1f}")
        
        # Recent trend
        if len(df_wave) >= 10:
            recent_avg = df_wave['wave_value'].tail(10).mean()
            if recent_avg > avg_wave * 1.2:
                report.append(f"Recent Trend: RISING (Recent: {recent_avg:.1f} vs Avg: {avg_wave:.1f})")
            elif recent_avg < avg_wave * 0.8:
                report.append(f"Recent Trend: FALLING (Recent: {recent_avg:.1f} vs Avg: {avg_wave:.1f})")
            else:
                report.append(f"Recent Trend: STABLE (Recent: {recent_avg:.1f} vs Avg: {avg_wave:.1f})")
        
        report.append("")
        
        # Alerts
        if alerts:
            report.append("🚨 ACTIVE ALERTS:")
            report.append("-" * 18)
            
            for alert in alerts:
                if isinstance(alert, dict):
                    if 'type' in alert:
                        report.append(f"• {alert['type']}: {alert['message']}")
                    else:
                        report.append(f"• Current: {alert['alert']} (Wave: {alert['wave_value']:.1f})")
            
            report.append("")
        
        # Wave interpretation
        report.append("💡 WAVE INTERPRETATION:")
        report.append("-" * 25)
        
        current_value = latest['wave_value']
        
        if current_value >= 80:
            report.append("🔴 CRITICAL ZONE (80-100):")
            report.append("   • Extreme hedging activity detected")
            report.append("   • High probability of significant market move")
            report.append("   • Immediate defensive action recommended")
        elif current_value >= 60:
            report.append("🟠 HIGH ZONE (60-80):")
            report.append("   • Elevated hedging activity")
            report.append("   • Increased risk of market weakness")
            report.append("   • Monitor closely, consider defensive positioning")
        elif current_value >= 40:
            report.append("🟡 ELEVATED ZONE (40-60):")
            report.append("   • Moderate hedging activity")
            report.append("   • Some defensive positioning present")
            report.append("   • Watch for pattern acceleration")
        elif current_value >= 20:
            report.append("🟢 NORMAL ZONE (20-40):")
            report.append("   • Normal hedging activity levels")
            report.append("   • Balanced market conditions")
            report.append("   • Continue regular monitoring")
        else:
            report.append("🔵 LOW ZONE (0-20):")
            report.append("   • Very low hedging activity")
            report.append("   • Quiet market conditions")
            report.append("   • Minimal defensive positioning")
        
        report.append("")
        
        # Recommendations
        report.append("📋 RECOMMENDATIONS:")
        report.append("-" * 20)
        
        if current_value >= 60:
            report.append("• Consider defensive positioning")
            report.append("• Monitor for pattern acceleration")
            report.append("• Prepare for potential volatility")
            report.append("• Watch key support levels")
        elif current_value >= 40:
            report.append("• Monitor hedging activity closely")
            report.append("• Watch for trend changes")
            report.append("• Prepare for potential shifts")
        else:
            report.append("• Continue normal monitoring")
            report.append("• Watch for wave acceleration")
            report.append("• No immediate action required")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    
    # Initialize wave indicator
    wave_indicator = HedgingOscillationWave()
    
    print("🌊 SPY HEDGING OSCILLATION WAVE INDICATOR")
    print("=" * 50)
    
    # Build wave for 2025
    df_wave = wave_indicator.build_historical_wave('2025-01-01', '2025-09-30')
    
    if df_wave.empty:
        print("❌ No wave data available")
        return
    
    # Generate alerts
    alerts = wave_indicator.generate_wave_alerts(df_wave)
    
    # Generate report
    report = wave_indicator.generate_wave_report(df_wave, alerts)
    print("\n" + report)
    
    # Create visualization
    wave_indicator.plot_oscillation_wave(df_wave, 'hedging_oscillation_wave.png')
    
    # Save results
    df_wave.to_csv('hedging_oscillation_wave.csv', index=False)
    with open('hedging_oscillation_wave_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n💾 Wave data saved to 'hedging_oscillation_wave.csv'")
    print(f"💾 Wave report saved to 'hedging_oscillation_wave_report.txt'")


if __name__ == "__main__":
    main()
