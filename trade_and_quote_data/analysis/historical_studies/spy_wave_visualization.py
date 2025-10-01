"""
SPY Price + Hedging Wave Visualization
=====================================

This script creates a comprehensive visualization showing SPY price with the
hedging oscillation wave below it, including vertical lines marking when
hedging activity picked up.

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

class SPYWaveVisualization:
    """
    Creates SPY price + hedging wave visualization
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
        
        return {
            'wave_value': wave_value,
            'zone': zone,
            'color': color,
            'alert': alert,
            'components': components
        }
    
    def build_combined_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build combined SPY price and hedging wave data"""
        
        print(f"📊 BUILDING SPY + HEDGING WAVE DATA")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        combined_data = []
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
                        
                        # Combine data
                        row_data = {
                            'date': current_date,
                            'spy_price': spy_price,
                            'wave_value': wave_info['wave_value'],
                            'zone': wave_info['zone'],
                            'alert': wave_info['alert']
                        }
                        
                        # Add component scores
                        for key, value in components.items():
                            row_data[key] = value
                        
                        combined_data.append(row_data)
            
            current_date += timedelta(days=1)
        
        if combined_data:
            df_combined = pd.DataFrame(combined_data)
            
            # Calculate moving averages
            df_combined['spy_ma_20'] = df_combined['spy_price'].rolling(window=20, min_periods=1).mean()
            df_combined['spy_ma_50'] = df_combined['spy_price'].rolling(window=50, min_periods=1).mean()
            df_combined['wave_ma_10'] = df_combined['wave_value'].rolling(window=10, min_periods=1).mean()
            
            # Calculate wave momentum
            df_combined['wave_momentum'] = df_combined['wave_value'].diff()
            
            print(f"✅ Built combined data for {len(df_combined)} trading days")
            return df_combined
        else:
            print("❌ No data found for the specified period")
            return pd.DataFrame()
    
    def identify_hedging_pickup_points(self, df: pd.DataFrame) -> list:
        """Identify when hedging activity picked up significantly"""
        
        pickup_points = []
        
        if df.empty or len(df) < 10:
            return pickup_points
        
        # Method 1: Wave value spikes (above 60)
        high_wave_days = df[df['wave_value'] >= 60]
        if not high_wave_days.empty:
            for _, row in high_wave_days.iterrows():
                pickup_points.append({
                    'date': row['date'],
                    'type': 'HIGH_WAVE',
                    'value': row['wave_value'],
                    'description': f"Wave spike to {row['wave_value']:.1f}"
                })
        
        # Method 2: Significant wave momentum (rate of change > 10)
        if 'wave_momentum' in df.columns:
            momentum_spikes = df[df['wave_momentum'] > 10]
            if not momentum_spikes.empty:
                for _, row in momentum_spikes.iterrows():
                    pickup_points.append({
                        'date': row['date'],
                        'type': 'MOMENTUM_SPIKE',
                        'value': row['wave_momentum'],
                        'description': f"Momentum spike: {row['wave_momentum']:.1f}"
                    })
        
        # Method 3: Zone changes to HIGH or CRITICAL
        zone_changes = []
        for i in range(1, len(df)):
            current_zone = df.iloc[i]['zone']
            previous_zone = df.iloc[i-1]['zone']
            
            if current_zone in ['HIGH', 'CRITICAL'] and previous_zone not in ['HIGH', 'CRITICAL']:
                zone_changes.append({
                    'date': df.iloc[i]['date'],
                    'type': 'ZONE_CHANGE',
                    'value': df.iloc[i]['wave_value'],
                    'description': f"Zone change to {current_zone}"
                })
        
        pickup_points.extend(zone_changes)
        
        # Remove duplicates and sort by date
        pickup_points = list({point['date']: point for point in pickup_points}.values())
        pickup_points.sort(key=lambda x: x['date'])
        
        return pickup_points
    
    def create_spy_wave_chart(self, df: pd.DataFrame, pickup_points: list, save_path: str = None):
        """Create the main SPY + Wave visualization chart"""
        
        if df.empty:
            print("No data to plot")
            return
        
        # Set up the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                           gridspec_kw={'height_ratios': [3, 2, 1]})
        fig.suptitle('SPY Price vs Hedging Oscillation Wave (2024-Today)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: SPY Price
        ax1.plot(df['date'], df['spy_price'], label='SPY Price', linewidth=2, color='blue', alpha=0.8)
        
        if 'spy_ma_20' in df.columns:
            ax1.plot(df['date'], df['spy_ma_20'], label='20-day MA', linewidth=1, color='orange', alpha=0.7)
        if 'spy_ma_50' in df.columns:
            ax1.plot(df['date'], df['spy_ma_50'], label='50-day MA', linewidth=1, color='red', alpha=0.7)
        
        # Add vertical lines for hedging pickup points
        for point in pickup_points:
            ax1.axvline(x=point['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax1.text(point['date'], ax1.get_ylim()[1] * 0.95, 
                    f"{point['date'].strftime('%m/%d')}\n{point['type']}", 
                    rotation=90, fontsize=8, ha='right', va='top')
        
        ax1.set_title('SPY Price with Hedging Activity Pickup Points', fontsize=14, fontweight='bold')
        ax1.set_ylabel('SPY Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hedging Oscillation Wave
        ax2.plot(df['date'], df['wave_value'], label='Hedging Wave', linewidth=2, color='purple')
        
        if 'wave_ma_10' in df.columns:
            ax2.plot(df['date'], df['wave_ma_10'], label='10-day MA', linewidth=1, color='orange', alpha=0.7)
        
        # Add zone bands
        ax2.axhspan(0, 20, alpha=0.1, color='blue', label='LOW (0-20)')
        ax2.axhspan(20, 40, alpha=0.1, color='green', label='NORMAL (20-40)')
        ax2.axhspan(40, 60, alpha=0.1, color='yellow', label='ELEVATED (40-60)')
        ax2.axhspan(60, 80, alpha=0.1, color='orange', label='HIGH (60-80)')
        ax2.axhspan(80, 100, alpha=0.1, color='red', label='CRITICAL (80-100)')
        
        # Add vertical lines for hedging pickup points
        for point in pickup_points:
            ax2.axvline(x=point['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax2.set_title('Hedging Oscillation Wave (0-100)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Wave Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Wave Momentum
        if 'wave_momentum' in df.columns:
            colors = ['red' if x > 0 else 'green' for x in df['wave_momentum']]
            ax3.bar(df['date'], df['wave_momentum'], color=colors, alpha=0.7, width=1)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add vertical lines for hedging pickup points
            for point in pickup_points:
                ax3.axvline(x=point['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            ax3.set_title('Wave Momentum (Rate of Change)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Momentum')
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
    
    def generate_pickup_analysis(self, pickup_points: list) -> str:
        """Generate analysis of hedging pickup points"""
        
        if not pickup_points:
            return "No significant hedging pickup points identified"
        
        report = []
        report.append("🚨 HEDGING ACTIVITY PICKUP ANALYSIS")
        report.append("=" * 40)
        report.append(f"Total Pickup Points: {len(pickup_points)}")
        report.append("")
        
        # Group by type
        by_type = {}
        for point in pickup_points:
            point_type = point['type']
            if point_type not in by_type:
                by_type[point_type] = []
            by_type[point_type].append(point)
        
        for point_type, points in by_type.items():
            report.append(f"📊 {point_type.replace('_', ' ').title()}: {len(points)} events")
            for point in points[-5:]:  # Show last 5 of each type
                report.append(f"  • {point['date'].strftime('%Y-%m-%d')}: {point['description']}")
            report.append("")
        
        # Recent activity
        recent_points = [p for p in pickup_points if p['date'] >= pd.Timestamp('2025-01-01')]
        if recent_points:
            report.append("📈 RECENT ACTIVITY (2025):")
            report.append("-" * 30)
            for point in recent_points:
                report.append(f"• {point['date'].strftime('%Y-%m-%d')}: {point['description']}")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    
    # Initialize visualizer
    visualizer = SPYWaveVisualization()
    
    print("📊 SPY PRICE + HEDGING WAVE VISUALIZATION")
    print("=" * 50)
    
    # Build combined data for 2024-today
    df_combined = visualizer.build_combined_data('2024-01-01', '2025-09-30')
    
    if df_combined.empty:
        print("❌ No data available")
        return
    
    # Identify hedging pickup points
    pickup_points = visualizer.identify_hedging_pickup_points(df_combined)
    
    print(f"\n🎯 Identified {len(pickup_points)} hedging pickup points")
    
    # Generate pickup analysis
    pickup_analysis = visualizer.generate_pickup_analysis(pickup_points)
    print("\n" + pickup_analysis)
    
    # Create visualization
    visualizer.create_spy_wave_chart(df_combined, pickup_points, 'spy_hedging_wave_chart.png')
    
    # Save data
    df_combined.to_csv('spy_hedging_wave_data.csv', index=False)
    with open('hedging_pickup_analysis.txt', 'w') as f:
        f.write(pickup_analysis)
    
    print(f"\n💾 Data saved to 'spy_hedging_wave_data.csv'")
    print(f"💾 Pickup analysis saved to 'hedging_pickup_analysis.txt'")


if __name__ == "__main__":
    main()
