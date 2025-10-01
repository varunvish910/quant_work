"""
Combined Macro Signals System
=============================

This system combines hedging activity and yen carry trade signals to provide
a comprehensive macro risk assessment. It integrates:

1. Hedging Oscillation Wave (0-100)
2. Yen Carry Trade Signals (0-100)
3. Combined Macro Risk Score (0-100)
4. Alert levels and recommendations

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

class CombinedMacroSignals:
    """
    Combines hedging and yen carry trade signals for comprehensive macro risk assessment
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
    
    def calculate_hedging_components(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate hedging wave components"""
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
            
            oi_score = min(15, deep_otm_oi / 100000)
            vol_oi_score = max(0, 10 - deep_otm_vol_oi * 20)
            components['deep_otm_score'] = oi_score + vol_oi_score
        else:
            components['deep_otm_score'] = 0
        
        # 2. Institutional Hedging (0-25 points)
        total_put_oi = puts['oi_proxy'].sum()
        if 'dte' in puts.columns:
            long_dated_puts = puts[puts['dte'] > 60]
            if not long_dated_puts.empty:
                long_dated_oi_pct = long_dated_puts['oi_proxy'].sum() / total_put_oi
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
        
        concentration_score = min(20, (top_5_pct + max_strike_pct) * 200)
        components['concentration_score'] = concentration_score
        
        # 4. Volume/OI Ratio (0-15 points)
        total_put_vol = puts['volume'].sum()
        vol_oi_ratio = total_put_vol / (total_put_oi + 1e-6)
        vol_oi_score = max(0, 15 - vol_oi_ratio * 15)
        components['vol_oi_score'] = vol_oi_score
        
        # 5. Put/Call Ratio (0-15 points)
        calls = df[df['option_type'] == 'C']
        if not calls.empty:
            pc_ratio = total_put_oi / calls['oi_proxy'].sum()
            pc_score = min(15, max(0, (pc_ratio - 1) * 15))
            components['pc_ratio_score'] = pc_score
        else:
            components['pc_ratio_score'] = 0
        
        return components
    
    def calculate_yen_carry_components(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate yen carry trade components"""
        if df.empty:
            return {}
        
        signals = {}
        
        # 1. Volatility Skew Analysis
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if not puts.empty and not calls.empty:
            atm_strike = spy_price
            atm_puts = puts[abs(puts['strike'] - atm_strike) <= 5]
            atm_calls = calls[abs(calls['strike'] - atm_strike) <= 5]
            
            if not atm_puts.empty and not atm_calls.empty:
                put_vol_proxy = atm_puts['volume'].sum()
                call_vol_proxy = atm_calls['volume'].sum()
                vol_skew = put_vol_proxy / (call_vol_proxy + 1e-6)
                
                signals['vol_skew'] = vol_skew
                signals['vol_skew_extreme'] = 1 if vol_skew > 2.0 else 0
            else:
                signals['vol_skew'] = 1.0
                signals['vol_skew_extreme'] = 0
        
        # 2. Macro Hedging Activity
        if not puts.empty and not calls.empty:
            deep_otm_puts = puts[puts['strike'] < spy_price * 0.85]
            deep_otm_calls = calls[calls['strike'] > spy_price * 1.15]
            
            if not deep_otm_puts.empty and not deep_otm_calls.empty:
                deep_put_oi = deep_otm_puts['oi_proxy'].sum()
                deep_call_oi = deep_otm_calls['oi_proxy'].sum()
                deep_pc_ratio = deep_put_oi / (deep_call_oi + 1e-6)
                
                signals['deep_pc_ratio'] = deep_pc_ratio
                signals['macro_hedging'] = 1 if deep_pc_ratio > 3.0 else 0
            else:
                signals['deep_pc_ratio'] = 1.0
                signals['macro_hedging'] = 0
        
        # 3. Liquidity Stress Indicators
        if not puts.empty:
            total_put_oi = puts['oi_proxy'].sum()
            total_put_vol = puts['volume'].sum()
            vol_oi_ratio = total_put_vol / (total_put_oi + 1e-6)
            
            signals['put_vol_oi_ratio'] = vol_oi_ratio
            signals['liquidity_stress'] = 1 if vol_oi_ratio > 2.0 or vol_oi_ratio < 0.1 else 0
        
        # 4. Term Structure Analysis
        if 'dte' in df.columns:
            short_term_puts = puts[puts['dte'] <= 30]
            long_term_puts = puts[puts['dte'] > 60]
            
            if not short_term_puts.empty and not long_term_puts.empty:
                short_term_oi = short_term_puts['oi_proxy'].sum()
                long_term_oi = long_term_puts['oi_proxy'].sum()
                term_ratio = short_term_oi / (long_term_oi + 1e-6)
                
                signals['term_ratio'] = term_ratio
                signals['term_flattening'] = 1 if term_ratio > 2.0 else 0
            else:
                signals['term_ratio'] = 1.0
                signals['term_flattening'] = 0
        
        # 5. Currency Hedge Activity
        if not puts.empty:
            strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
            top_5_oi = strike_oi.head(5).sum()
            total_oi = strike_oi.sum()
            concentration = top_5_oi / (total_oi + 1e-6)
            
            signals['strike_concentration'] = concentration
            signals['currency_hedge'] = 1 if concentration > 0.6 else 0
        
        # 6. Risk-Off Positioning
        if not puts.empty and not calls.empty:
            total_put_oi = puts['oi_proxy'].sum()
            total_call_oi = calls['oi_proxy'].sum()
            pc_ratio = total_put_oi / (total_call_oi + 1e-6)
            
            signals['pc_ratio'] = pc_ratio
            signals['risk_off'] = 1 if pc_ratio > 1.5 else 0
        
        return signals
    
    def calculate_combined_score(self, hedging_components: dict, yen_carry_components: dict) -> dict:
        """Calculate combined macro risk score"""
        
        # Calculate individual scores
        hedging_score = sum(hedging_components.values()) if hedging_components else 0
        hedging_score = min(100, hedging_score)
        
        # Calculate yen carry score
        yen_carry_weights = {
            'vol_skew_extreme': 0.20,
            'macro_hedging': 0.20,
            'liquidity_stress': 0.15,
            'term_flattening': 0.15,
            'currency_hedge': 0.15,
            'risk_off': 0.15
        }
        
        yen_carry_score = 0
        max_yen_carry = 0
        
        for signal, weight in yen_carry_weights.items():
            if signal in yen_carry_components:
                yen_carry_score += yen_carry_components[signal] * weight * 100
                max_yen_carry += weight * 100
        
        if max_yen_carry > 0:
            yen_carry_score = (yen_carry_score / max_yen_carry) * 100
        else:
            yen_carry_score = 0
        
        yen_carry_score = min(100, yen_carry_score)
        
        # Calculate combined score (weighted average)
        combined_score = (hedging_score * 0.6) + (yen_carry_score * 0.4)
        combined_score = min(100, combined_score)
        
        # Determine alert levels
        if combined_score >= 80:
            level = 'CRITICAL'
            alert = 'MACRO RISK CRITICAL'
            color = 'red'
        elif combined_score >= 60:
            level = 'HIGH'
            alert = 'MACRO RISK HIGH'
            color = 'orange'
        elif combined_score >= 40:
            level = 'ELEVATED'
            alert = 'MACRO RISK ELEVATED'
            color = 'yellow'
        elif combined_score >= 20:
            level = 'NORMAL'
            alert = 'MACRO RISK NORMAL'
            color = 'green'
        else:
            level = 'LOW'
            alert = 'MACRO RISK LOW'
            color = 'blue'
        
        return {
            'hedging_score': hedging_score,
            'yen_carry_score': yen_carry_score,
            'combined_score': combined_score,
            'level': level,
            'alert': alert,
            'color': color
        }
    
    def build_combined_timeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build combined macro signals timeline"""
        
        print(f"🌍 BUILDING COMBINED MACRO SIGNALS TIMELINE")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        timeline_data = []
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    
                    # Calculate components
                    hedging_components = self.calculate_hedging_components(df, spy_price)
                    yen_carry_components = self.calculate_yen_carry_components(df, spy_price)
                    
                    if hedging_components or yen_carry_components:
                        combined_score = self.calculate_combined_score(hedging_components, yen_carry_components)
                        
                        # Combine data
                        row_data = {
                            'date': current_date,
                            'spy_price': spy_price,
                            'hedging_score': combined_score['hedging_score'],
                            'yen_carry_score': combined_score['yen_carry_score'],
                            'combined_score': combined_score['combined_score'],
                            'risk_level': combined_score['level'],
                            'risk_alert': combined_score['alert']
                        }
                        
                        # Add individual components
                        for key, value in hedging_components.items():
                            row_data[f'hedging_{key}'] = value
                        
                        for key, value in yen_carry_components.items():
                            row_data[f'yen_carry_{key}'] = value
                        
                        timeline_data.append(row_data)
            
            current_date += timedelta(days=1)
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            # Calculate moving averages
            df_timeline['hedging_ma_10'] = df_timeline['hedging_score'].rolling(window=10, min_periods=1).mean()
            df_timeline['yen_carry_ma_10'] = df_timeline['yen_carry_score'].rolling(window=10, min_periods=1).mean()
            df_timeline['combined_ma_10'] = df_timeline['combined_score'].rolling(window=10, min_periods=1).mean()
            
            # Calculate momentum
            df_timeline['hedging_momentum'] = df_timeline['hedging_score'].diff()
            df_timeline['yen_carry_momentum'] = df_timeline['yen_carry_score'].diff()
            df_timeline['combined_momentum'] = df_timeline['combined_score'].diff()
            
            print(f"✅ Built combined timeline for {len(df_timeline)} trading days")
            return df_timeline
        else:
            print("❌ No data found for the specified period")
            return pd.DataFrame()
    
    def identify_macro_events(self, df: pd.DataFrame) -> list:
        """Identify significant macro events"""
        
        events = []
        
        if df.empty or len(df) < 10:
            return events
        
        # Method 1: High combined scores (above 60)
        high_score_days = df[df['combined_score'] >= 60]
        if not high_score_days.empty:
            for _, row in high_score_days.iterrows():
                events.append({
                    'date': row['date'],
                    'type': 'HIGH_MACRO_RISK',
                    'value': row['combined_score'],
                    'description': f"Macro risk: {row['combined_score']:.1f}"
                })
        
        # Method 2: Combined momentum spikes
        if 'combined_momentum' in df.columns:
            momentum_spikes = df[df['combined_momentum'] > 15]
            if not momentum_spikes.empty:
                for _, row in momentum_spikes.iterrows():
                    events.append({
                        'date': row['date'],
                        'type': 'MOMENTUM_SPIKE',
                        'value': row['combined_momentum'],
                        'description': f"Momentum spike: {row['combined_momentum']:.1f}"
                    })
        
        # Method 3: Risk level changes
        level_changes = []
        for i in range(1, len(df)):
            current_level = df.iloc[i]['risk_level']
            previous_level = df.iloc[i-1]['risk_level']
            
            if current_level in ['HIGH', 'CRITICAL'] and previous_level not in ['HIGH', 'CRITICAL']:
                level_changes.append({
                    'date': df.iloc[i]['date'],
                    'type': 'RISK_LEVEL_CHANGE',
                    'value': df.iloc[i]['combined_score'],
                    'description': f"Risk level change to {current_level}"
                })
        
        events.extend(level_changes)
        
        # Remove duplicates and sort by date
        events = list({event['date']: event for event in events}.values())
        events.sort(key=lambda x: x['date'])
        
        return events
    
    def create_combined_chart(self, df: pd.DataFrame, events: list, save_path: str = None):
        """Create combined macro signals visualization"""
        
        if df.empty:
            print("No data to plot")
            return
        
        # Set up the plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 16), 
                                                gridspec_kw={'height_ratios': [3, 2, 2, 1]})
        fig.suptitle('SPY Price vs Combined Macro Signals (2024-Today)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: SPY Price
        ax1.plot(df['date'], df['spy_price'], label='SPY Price', linewidth=2, color='blue', alpha=0.8)
        
        # Add vertical lines for macro events
        for event in events:
            ax1.axvline(x=event['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax1.text(event['date'], ax1.get_ylim()[1] * 0.95, 
                    f"{event['date'].strftime('%m/%d')}\n{event['type']}", 
                    rotation=90, fontsize=8, ha='right', va='top')
        
        ax1.set_title('SPY Price with Macro Risk Events', fontsize=14, fontweight='bold')
        ax1.set_ylabel('SPY Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hedging Score
        ax2.plot(df['date'], df['hedging_score'], label='Hedging Score', linewidth=2, color='purple')
        
        if 'hedging_ma_10' in df.columns:
            ax2.plot(df['date'], df['hedging_ma_10'], label='10-day MA', linewidth=1, color='orange', alpha=0.7)
        
        # Add zone bands
        ax2.axhspan(0, 20, alpha=0.1, color='blue', label='LOW (0-20)')
        ax2.axhspan(20, 40, alpha=0.1, color='green', label='NORMAL (20-40)')
        ax2.axhspan(40, 60, alpha=0.1, color='yellow', label='ELEVATED (40-60)')
        ax2.axhspan(60, 80, alpha=0.1, color='orange', label='HIGH (60-80)')
        ax2.axhspan(80, 100, alpha=0.1, color='red', label='CRITICAL (80-100)')
        
        # Add vertical lines for macro events
        for event in events:
            ax2.axvline(x=event['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax2.set_title('Hedging Oscillation Wave (0-100)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Hedging Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Yen Carry Score
        ax3.plot(df['date'], df['yen_carry_score'], label='Yen Carry Score', linewidth=2, color='green')
        
        if 'yen_carry_ma_10' in df.columns:
            ax3.plot(df['date'], df['yen_carry_ma_10'], label='10-day MA', linewidth=1, color='orange', alpha=0.7)
        
        # Add zone bands
        ax3.axhspan(0, 20, alpha=0.1, color='blue', label='LOW (0-20)')
        ax3.axhspan(20, 40, alpha=0.1, color='green', label='NORMAL (20-40)')
        ax3.axhspan(40, 60, alpha=0.1, color='yellow', label='ELEVATED (40-60)')
        ax3.axhspan(60, 80, alpha=0.1, color='orange', label='HIGH (60-80)')
        ax3.axhspan(80, 100, alpha=0.1, color='red', label='CRITICAL (80-100)')
        
        # Add vertical lines for macro events
        for event in events:
            ax3.axvline(x=event['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax3.set_title('Yen Carry Trade Score (0-100)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Yen Carry Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Plot 4: Combined Score
        ax4.plot(df['date'], df['combined_score'], label='Combined Macro Score', linewidth=3, color='red')
        
        if 'combined_ma_10' in df.columns:
            ax4.plot(df['date'], df['combined_ma_10'], label='10-day MA', linewidth=1, color='orange', alpha=0.7)
        
        # Add zone bands
        ax4.axhspan(0, 20, alpha=0.1, color='blue', label='LOW (0-20)')
        ax4.axhspan(20, 40, alpha=0.1, color='green', label='NORMAL (20-40)')
        ax4.axhspan(40, 60, alpha=0.1, color='yellow', label='ELEVATED (40-60)')
        ax4.axhspan(60, 80, alpha=0.1, color='orange', label='HIGH (60-80)')
        ax4.axhspan(80, 100, alpha=0.1, color='red', label='CRITICAL (80-100)')
        
        # Add vertical lines for macro events
        for event in events:
            ax4.axvline(x=event['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax4.set_title('Combined Macro Risk Score (0-100)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Combined Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def generate_macro_analysis(self, events: list) -> str:
        """Generate combined macro analysis"""
        
        if not events:
            return "No significant macro events identified"
        
        report = []
        report.append("🌍 COMBINED MACRO SIGNALS ANALYSIS")
        report.append("=" * 40)
        report.append(f"Total Events: {len(events)}")
        report.append("")
        
        # Group by type
        by_type = {}
        for event in events:
            event_type = event['type']
            if event_type not in by_type:
                by_type[event_type] = []
            by_type[event_type].append(event)
        
        for event_type, events_list in by_type.items():
            report.append(f"📊 {event_type.replace('_', ' ').title()}: {len(events_list)} events")
            for event in events_list[-5:]:  # Show last 5 of each type
                report.append(f"  • {event['date'].strftime('%Y-%m-%d')}: {event['description']}")
            report.append("")
        
        # Recent activity
        recent_events = [e for e in events if e['date'] >= pd.Timestamp('2025-01-01')]
        if recent_events:
            report.append("📈 RECENT MACRO ACTIVITY (2025):")
            report.append("-" * 30)
            for event in recent_events:
                report.append(f"• {event['date'].strftime('%Y-%m-%d')}: {event['description']}")
        
        return "\n".join(report)


def main():
    """Main combined macro signals analysis"""
    
    # Initialize system
    system = CombinedMacroSignals()
    
    print("🌍 COMBINED MACRO SIGNALS SYSTEM")
    print("=" * 50)
    
    # Build combined timeline for 2024-today
    df_timeline = system.build_combined_timeline('2024-01-01', '2025-09-30')
    
    if df_timeline.empty:
        print("❌ No data available")
        return
    
    # Identify macro events
    events = system.identify_macro_events(df_timeline)
    
    print(f"\n🎯 Identified {len(events)} macro events")
    
    # Generate analysis
    analysis = system.generate_macro_analysis(events)
    print("\n" + analysis)
    
    # Create visualization
    system.create_combined_chart(df_timeline, events, 'combined_macro_signals_chart.png')
    
    # Save data
    df_timeline.to_csv('combined_macro_signals_data.csv', index=False)
    with open('combined_macro_analysis.txt', 'w') as f:
        f.write(analysis)
    
    print(f"\n💾 Data saved to 'combined_macro_signals_data.csv'")
    print(f"💾 Analysis saved to 'combined_macro_analysis.txt'")


if __name__ == "__main__":
    main()
