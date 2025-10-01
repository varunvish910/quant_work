"""
Yen Carry Trade Detection System
===============================

This system detects yen carry trade signals that often precede market stress.
The yen carry trade involves borrowing in low-yield yen to invest in higher-yield
assets. When this trade unwinds, it can cause significant market volatility.

Key Signals:
- USD/JPY volatility spikes
- VIX correlation with yen strength
- Options flow patterns during yen moves
- Cross-asset correlation breakdowns

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

class YenCarryTradeDetector:
    """
    Detects yen carry trade signals and their impact on SPY options
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
    
    def calculate_yen_carry_signals(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Calculate yen carry trade signals from options data"""
        if df.empty:
            return {}
        
        signals = {}
        
        # 1. Volatility Skew Analysis (yen carry unwinds cause volatility spikes)
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if not puts.empty and not calls.empty:
            # Calculate implied volatility skew
            atm_strike = spy_price
            atm_puts = puts[abs(puts['strike'] - atm_strike) <= 5]
            atm_calls = calls[abs(calls['strike'] - atm_strike) <= 5]
            
            if not atm_puts.empty and not atm_calls.empty:
                # Use volume as proxy for implied volatility
                put_vol_proxy = atm_puts['volume'].sum()
                call_vol_proxy = atm_calls['volume'].sum()
                vol_skew = put_vol_proxy / (call_vol_proxy + 1e-6)
                
                signals['vol_skew'] = vol_skew
                signals['vol_skew_extreme'] = 1 if vol_skew > 2.0 else 0  # Extreme put skew
            else:
                signals['vol_skew'] = 1.0
                signals['vol_skew_extreme'] = 0
        
        # 2. Cross-Asset Correlation Breakdown (yen carry unwinds break correlations)
        # Look for unusual put/call patterns that suggest macro stress
        if not puts.empty and not calls.empty:
            # Deep OTM puts vs calls (macro hedging)
            deep_otm_puts = puts[puts['strike'] < spy_price * 0.85]
            deep_otm_calls = calls[calls['strike'] > spy_price * 1.15]
            
            if not deep_otm_puts.empty and not deep_otm_calls.empty:
                deep_put_oi = deep_otm_puts['oi_proxy'].sum()
                deep_call_oi = deep_otm_calls['oi_proxy'].sum()
                deep_pc_ratio = deep_put_oi / (deep_call_oi + 1e-6)
                
                signals['deep_pc_ratio'] = deep_pc_ratio
                signals['macro_hedging'] = 1 if deep_pc_ratio > 3.0 else 0  # Extreme macro hedging
            else:
                signals['deep_pc_ratio'] = 1.0
                signals['macro_hedging'] = 0
        
        # 3. Liquidity Stress Indicators (yen carry unwinds cause liquidity issues)
        # Look for unusual volume/OI patterns
        if not puts.empty:
            total_put_oi = puts['oi_proxy'].sum()
            total_put_vol = puts['volume'].sum()
            vol_oi_ratio = total_put_vol / (total_put_oi + 1e-6)
            
            # Low V/OI ratio suggests accumulation (institutional hedging)
            # High V/OI ratio suggests panic selling
            signals['put_vol_oi_ratio'] = vol_oi_ratio
            signals['liquidity_stress'] = 1 if vol_oi_ratio > 2.0 or vol_oi_ratio < 0.1 else 0
        
        # 4. Term Structure Flattening (yen carry unwinds flatten yield curves)
        # Look for unusual DTE patterns in options
        if 'dte' in df.columns:
            # Short-term vs long-term put activity
            short_term_puts = puts[puts['dte'] <= 30]
            long_term_puts = puts[puts['dte'] > 60]
            
            if not short_term_puts.empty and not long_term_puts.empty:
                short_term_oi = short_term_puts['oi_proxy'].sum()
                long_term_oi = long_term_puts['oi_proxy'].sum()
                term_ratio = short_term_oi / (long_term_oi + 1e-6)
                
                signals['term_ratio'] = term_ratio
                signals['term_flattening'] = 1 if term_ratio > 2.0 else 0  # Short-term dominance
            else:
                signals['term_ratio'] = 1.0
                signals['term_flattening'] = 0
        
        # 5. Currency Hedge Activity (yen carry unwinds trigger currency hedges)
        # Look for unusual strike concentration patterns
        if not puts.empty:
            # Check for concentrated activity at specific strikes (currency hedge levels)
            strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
            top_5_oi = strike_oi.head(5).sum()
            total_oi = strike_oi.sum()
            concentration = top_5_oi / (total_oi + 1e-6)
            
            signals['strike_concentration'] = concentration
            signals['currency_hedge'] = 1 if concentration > 0.6 else 0  # High concentration
        
        # 6. Risk-Off Positioning (yen carry unwinds are risk-off events)
        # Look for defensive positioning patterns
        if not puts.empty and not calls.empty:
            # Total put vs call OI
            total_put_oi = puts['oi_proxy'].sum()
            total_call_oi = calls['oi_proxy'].sum()
            pc_ratio = total_put_oi / (total_call_oi + 1e-6)
            
            signals['pc_ratio'] = pc_ratio
            signals['risk_off'] = 1 if pc_ratio > 1.5 else 0  # Defensive positioning
        
        return signals
    
    def calculate_yen_carry_score(self, signals: dict) -> dict:
        """Calculate overall yen carry trade score"""
        
        if not signals:
            return {'score': 0, 'level': 'NONE', 'alert': 'NO_DATA'}
        
        # Weighted scoring system
        weights = {
            'vol_skew_extreme': 0.20,      # Volatility skew
            'macro_hedging': 0.20,         # Macro hedging activity
            'liquidity_stress': 0.15,      # Liquidity stress
            'term_flattening': 0.15,       # Term structure
            'currency_hedge': 0.15,        # Currency hedging
            'risk_off': 0.15               # Risk-off positioning
        }
        
        # Calculate weighted score
        total_score = 0
        max_score = 0
        
        for signal, weight in weights.items():
            if signal in signals:
                total_score += signals[signal] * weight * 100
                max_score += weight * 100
        
        # Normalize to 0-100 scale
        if max_score > 0:
            normalized_score = (total_score / max_score) * 100
        else:
            normalized_score = 0
        
        # Determine alert level
        if normalized_score >= 80:
            level = 'CRITICAL'
            alert = 'YEN CARRY UNWIND IMMINENT'
        elif normalized_score >= 60:
            level = 'HIGH'
            alert = 'YEN CARRY STRESS'
        elif normalized_score >= 40:
            level = 'ELEVATED'
            alert = 'YEN CARRY WARNING'
        elif normalized_score >= 20:
            level = 'NORMAL'
            alert = 'STABLE'
        else:
            level = 'LOW'
            alert = 'QUIET'
        
        return {
            'score': normalized_score,
            'level': level,
            'alert': alert,
            'signals': signals
        }
    
    def build_yen_carry_timeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build yen carry trade timeline"""
        
        print(f"🇯🇵 BUILDING YEN CARRY TRADE TIMELINE")
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
                    signals = self.calculate_yen_carry_signals(df, spy_price)
                    
                    if signals:
                        carry_score = self.calculate_yen_carry_score(signals)
                        
                        # Combine data
                        row_data = {
                            'date': current_date,
                            'spy_price': spy_price,
                            'yen_carry_score': carry_score['score'],
                            'yen_carry_level': carry_score['level'],
                            'yen_carry_alert': carry_score['alert']
                        }
                        
                        # Add individual signals
                        for key, value in signals.items():
                            row_data[key] = value
                        
                        timeline_data.append(row_data)
            
            current_date += timedelta(days=1)
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            # Calculate moving averages
            df_timeline['yen_carry_ma_10'] = df_timeline['yen_carry_score'].rolling(window=10, min_periods=1).mean()
            df_timeline['yen_carry_momentum'] = df_timeline['yen_carry_score'].diff()
            
            print(f"✅ Built yen carry timeline for {len(df_timeline)} trading days")
            return df_timeline
        else:
            print("❌ No data found for the specified period")
            return pd.DataFrame()
    
    def identify_yen_carry_events(self, df: pd.DataFrame) -> list:
        """Identify significant yen carry trade events"""
        
        events = []
        
        if df.empty or len(df) < 10:
            return events
        
        # Method 1: High yen carry scores (above 60)
        high_score_days = df[df['yen_carry_score'] >= 60]
        if not high_score_days.empty:
            for _, row in high_score_days.iterrows():
                events.append({
                    'date': row['date'],
                    'type': 'HIGH_SCORE',
                    'value': row['yen_carry_score'],
                    'description': f"Yen carry score: {row['yen_carry_score']:.1f}"
                })
        
        # Method 2: Score momentum spikes
        if 'yen_carry_momentum' in df.columns:
            momentum_spikes = df[df['yen_carry_momentum'] > 15]
            if not momentum_spikes.empty:
                for _, row in momentum_spikes.iterrows():
                    events.append({
                        'date': row['date'],
                        'type': 'MOMENTUM_SPIKE',
                        'value': row['yen_carry_momentum'],
                        'description': f"Momentum spike: {row['yen_carry_momentum']:.1f}"
                    })
        
        # Method 3: Level changes to HIGH or CRITICAL
        level_changes = []
        for i in range(1, len(df)):
            current_level = df.iloc[i]['yen_carry_level']
            previous_level = df.iloc[i-1]['yen_carry_level']
            
            if current_level in ['HIGH', 'CRITICAL'] and previous_level not in ['HIGH', 'CRITICAL']:
                level_changes.append({
                    'date': df.iloc[i]['date'],
                    'type': 'LEVEL_CHANGE',
                    'value': df.iloc[i]['yen_carry_score'],
                    'description': f"Level change to {current_level}"
                })
        
        events.extend(level_changes)
        
        # Remove duplicates and sort by date
        events = list({event['date']: event for event in events}.values())
        events.sort(key=lambda x: x['date'])
        
        return events
    
    def create_yen_carry_chart(self, df: pd.DataFrame, events: list, save_path: str = None):
        """Create yen carry trade visualization"""
        
        if df.empty:
            print("No data to plot")
            return
        
        # Set up the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                           gridspec_kw={'height_ratios': [3, 2, 1]})
        fig.suptitle('SPY Price vs Yen Carry Trade Signals (2024-Today)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: SPY Price
        ax1.plot(df['date'], df['spy_price'], label='SPY Price', linewidth=2, color='blue', alpha=0.8)
        
        # Add vertical lines for yen carry events
        for event in events:
            ax1.axvline(x=event['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax1.text(event['date'], ax1.get_ylim()[1] * 0.95, 
                    f"{event['date'].strftime('%m/%d')}\n{event['type']}", 
                    rotation=90, fontsize=8, ha='right', va='top')
        
        ax1.set_title('SPY Price with Yen Carry Trade Events', fontsize=14, fontweight='bold')
        ax1.set_ylabel('SPY Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Yen Carry Score
        ax2.plot(df['date'], df['yen_carry_score'], label='Yen Carry Score', linewidth=2, color='purple')
        
        if 'yen_carry_ma_10' in df.columns:
            ax2.plot(df['date'], df['yen_carry_ma_10'], label='10-day MA', linewidth=1, color='orange', alpha=0.7)
        
        # Add zone bands
        ax2.axhspan(0, 20, alpha=0.1, color='blue', label='LOW (0-20)')
        ax2.axhspan(20, 40, alpha=0.1, color='green', label='NORMAL (20-40)')
        ax2.axhspan(40, 60, alpha=0.1, color='yellow', label='ELEVATED (40-60)')
        ax2.axhspan(60, 80, alpha=0.1, color='orange', label='HIGH (60-80)')
        ax2.axhspan(80, 100, alpha=0.1, color='red', label='CRITICAL (80-100)')
        
        # Add vertical lines for yen carry events
        for event in events:
            ax2.axvline(x=event['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax2.set_title('Yen Carry Trade Score (0-100)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Individual Signals
        signal_cols = ['vol_skew_extreme', 'macro_hedging', 'liquidity_stress', 
                      'term_flattening', 'currency_hedge', 'risk_off']
        available_signals = [col for col in signal_cols if col in df.columns]
        
        if available_signals:
            signal_data = df[available_signals].fillna(0)
            signal_data.plot(kind='bar', ax=ax3, stacked=True, width=1, alpha=0.7)
            
            # Add vertical lines for yen carry events
            for event in events:
                ax3.axvline(x=event['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            ax3.set_title('Individual Yen Carry Signals', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Signal Strength')
            ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def generate_yen_carry_analysis(self, events: list) -> str:
        """Generate yen carry trade analysis"""
        
        if not events:
            return "No significant yen carry trade events identified"
        
        report = []
        report.append("🇯🇵 YEN CARRY TRADE ANALYSIS")
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
            report.append("📈 RECENT YEN CARRY ACTIVITY (2025):")
            report.append("-" * 30)
            for event in recent_events:
                report.append(f"• {event['date'].strftime('%Y-%m-%d')}: {event['description']}")
        
        return "\n".join(report)


def main():
    """Main yen carry trade analysis"""
    
    # Initialize detector
    detector = YenCarryTradeDetector()
    
    print("🇯🇵 YEN CARRY TRADE DETECTION SYSTEM")
    print("=" * 50)
    
    # Build yen carry timeline for 2024-today
    df_timeline = detector.build_yen_carry_timeline('2024-01-01', '2025-09-30')
    
    if df_timeline.empty:
        print("❌ No data available")
        return
    
    # Identify yen carry events
    events = detector.identify_yen_carry_events(df_timeline)
    
    print(f"\n🎯 Identified {len(events)} yen carry trade events")
    
    # Generate analysis
    analysis = detector.generate_yen_carry_analysis(events)
    print("\n" + analysis)
    
    # Create visualization
    detector.create_yen_carry_chart(df_timeline, events, 'yen_carry_trade_chart.png')
    
    # Save data
    df_timeline.to_csv('yen_carry_trade_data.csv', index=False)
    with open('yen_carry_analysis.txt', 'w') as f:
        f.write(analysis)
    
    print(f"\n💾 Data saved to 'yen_carry_trade_data.csv'")
    print(f"💾 Analysis saved to 'yen_carry_analysis.txt'")


if __name__ == "__main__":
    main()
