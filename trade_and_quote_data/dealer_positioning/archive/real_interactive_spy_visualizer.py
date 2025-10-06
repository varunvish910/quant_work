#!/usr/bin/env python3
"""
Real Interactive SPY Options Visualizer  
Creates interactive charts using real SPY options data with strike range and time analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta, date
from pathlib import Path
import json
import yfinance as yf


class RealInteractiveSPYVisualizer:
    """Creates interactive visualizations using real SPY options data"""
    
    def __init__(self, data_dir: str = "outputs/real_spy_analysis"):
        self.data_dir = Path(data_dir) 
        self.output_dir = Path("outputs/real_interactive_spy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_real_analysis_data(self) -> dict:
        """Load the real analysis results"""
        
        print("📊 Loading real SPY analysis data...")
        
        analysis_files = list(self.data_dir.glob("*_analysis.json"))
        greeks_files = list(self.data_dir.glob("*_greeks.parquet"))
        
        if not analysis_files:
            raise FileNotFoundError(f"No analysis files found in {self.data_dir}")
        
        real_data = {}
        
        for analysis_file in analysis_files:
            group_name = analysis_file.stem.replace('_analysis', '')
            
            # Load analysis JSON
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            # Load corresponding Greeks data
            greeks_file = self.data_dir / f"{group_name}_greeks.parquet"
            if greeks_file.exists():
                greeks_df = pd.read_parquet(greeks_file)
            else:
                print(f"⚠️  Greeks file not found for {group_name}")
                continue
            
            real_data[group_name] = {
                'analysis': analysis_data,
                'greeks': greeks_df,
                'expiry_list': analysis_data['expiry_list'],
                'spot_price': analysis_data['spot_price'],
                'trades_count': analysis_data['trades_count'],
                'strikes_count': analysis_data['strikes_count']
            }
            
            print(f"   ✓ Loaded {group_name}: {len(greeks_df)} strikes")
        
        print(f"✅ Loaded {len(real_data)} expiry groups")
        return real_data
    
    def create_strike_range_comparison(self, real_data: dict) -> str:
        """Create interactive comparison across different strike ranges"""
        
        print("📈 Creating strike range comparison chart...")
        
        # Use the All_Expiries data as base
        base_data = real_data.get('All_Expiries', list(real_data.values())[0])
        greeks_df = base_data['greeks']
        spot_price = base_data['spot_price']
        
        # Define different strike ranges for analysis
        strike_ranges = {
            "ATM_Tight": (spot_price - 20, spot_price + 20),  # ±$20 around spot
            "ATM_Wide": (spot_price - 50, spot_price + 50),   # ±$50 around spot  
            "OTM_Calls": (spot_price + 10, spot_price + 80),  # OTM calls
            "OTM_Puts": (spot_price - 80, spot_price - 10),   # OTM puts
            "Full_Range": (greeks_df['strike'].min(), greeks_df['strike'].max())  # All strikes
        }
        
        fig = go.Figure()
        
        # Create traces for each strike range
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (range_name, (min_strike, max_strike)) in enumerate(strike_ranges.items()):
            # Filter data for this strike range
            range_data = greeks_df[
                (greeks_df['strike'] >= min_strike) & 
                (greeks_df['strike'] <= max_strike)
            ].copy()
            
            if len(range_data) == 0:
                continue
            
            # Add gamma trace
            fig.add_trace(
                go.Scatter(
                    x=range_data['strike'],
                    y=range_data['dealer_gamma'],
                    mode='lines+markers',
                    name=f'{range_name} - Gamma',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    visible=True if i == 0 else False,
                    hovertemplate='Strike: $%{x}<br>Dealer Gamma: %{y}<extra></extra>'
                )
            )
            
            # Add delta trace
            fig.add_trace(
                go.Scatter(
                    x=range_data['strike'],
                    y=range_data['dealer_delta'],
                    mode='lines+markers',
                    name=f'{range_name} - Delta',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    marker=dict(size=4),
                    yaxis='y2',
                    visible=True if i == 0 else False,
                    hovertemplate='Strike: $%{x}<br>Dealer Delta: %{y}<extra></extra>'
                )
            )
        
        # Create dropdown for strike range selection
        dropdown_buttons = []
        
        for i, (range_name, (min_strike, max_strike)) in enumerate(strike_ranges.items()):
            # Create visibility array
            visibility = [False] * (len(strike_ranges) * 2)
            visibility[i*2] = True      # Gamma trace
            visibility[i*2 + 1] = True  # Delta trace
            
            dropdown_buttons.append(
                dict(
                    label=f"{range_name} (${min_strike:.0f}-${max_strike:.0f})",
                    method="update",
                    args=[{"visible": visibility},
                          {"title": f"Real SPY Options Positioning - {range_name}"}]
                )
            )
        
        # Add "All Ranges" option
        all_visibility = [True] * (len(strike_ranges) * 2)
        dropdown_buttons.insert(0,
            dict(
                label="All Strike Ranges",
                method="update",
                args=[{"visible": all_visibility},
                      {"title": "Real SPY Options Positioning - All Strike Ranges"}]
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Real SPY Options Positioning - Strike Range Analysis",
            xaxis_title="Strike Price ($)",
            yaxis_title="Dealer Gamma",
            yaxis2=dict(
                title="Dealer Delta",
                overlaying="y",
                side="right"
            ),
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        # Add current SPY price line
        fig.add_vline(x=spot_price, line_dash="solid", line_color="white", 
                     annotation_text=f"SPY: ${spot_price:.2f}")
        
        # Save chart
        chart_file = self.output_dir / "strike_range_analysis.html"
        fig.write_html(chart_file)
        
        print(f"✅ Strike range analysis saved to {chart_file}")
        return str(chart_file)
    
    def create_greeks_breakdown_chart(self, real_data: dict) -> str:
        """Create interactive breakdown of all Greeks"""
        
        print("🔢 Creating Greeks breakdown chart...")
        
        base_data = real_data.get('All_Expiries', list(real_data.values())[0])
        greeks_df = base_data['greeks']
        spot_price = base_data['spot_price']
        
        # Create subplots for different Greeks
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Dealer Gamma', 'Dealer Delta', 'Dealer Vega', 'Dealer Theta'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gamma
        fig.add_trace(
            go.Scatter(x=greeks_df['strike'], y=greeks_df['dealer_gamma'],
                      mode='lines+markers', name='Gamma',
                      line=dict(color='blue', width=2)), 
            row=1, col=1
        )
        
        # Delta
        fig.add_trace(
            go.Scatter(x=greeks_df['strike'], y=greeks_df['dealer_delta'],
                      mode='lines+markers', name='Delta',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # Vega
        fig.add_trace(
            go.Scatter(x=greeks_df['strike'], y=greeks_df['dealer_vega'],
                      mode='lines+markers', name='Vega',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # Theta
        fig.add_trace(
            go.Scatter(x=greeks_df['strike'], y=greeks_df['dealer_theta'],
                      mode='lines+markers', name='Theta',
                      line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        # Add SPY price lines to all subplots
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(x=spot_price, line_dash="dot", line_color="white",
                             row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=f"Real SPY Options Greeks Breakdown - Expiry: {base_data['expiry_list'][0]}",
            height=700,
            template='plotly_dark',
            showlegend=False
        )
        
        # Update axes labels
        for row in [1, 2]:
            for col in [1, 2]:
                fig.update_xaxes(title_text="Strike Price ($)", row=row, col=col)
        
        # Save chart
        chart_file = self.output_dir / "greeks_breakdown.html"
        fig.write_html(chart_file)
        
        print(f"✅ Greeks breakdown saved to {chart_file}")
        return str(chart_file)
    
    def create_positioning_heatmap(self, real_data: dict) -> str:
        """Create heatmap showing positioning intensity"""
        
        print("🔥 Creating positioning heatmap...")
        
        base_data = real_data.get('All_Expiries', list(real_data.values())[0])
        greeks_df = base_data['greeks']
        spot_price = base_data['spot_price']
        
        # Create positioning intensity matrix
        # Combine different Greeks into intensity measures
        greeks_df = greeks_df.copy()
        greeks_df['gamma_intensity'] = np.abs(greeks_df['dealer_gamma'])
        greeks_df['delta_intensity'] = np.abs(greeks_df['dealer_delta'])
        greeks_df['vega_intensity'] = np.abs(greeks_df['dealer_vega'])
        greeks_df['total_intensity'] = (
            greeks_df['gamma_intensity'] + 
            greeks_df['delta_intensity'] * 0.1 +  # Scale delta
            greeks_df['vega_intensity'] * 0.01     # Scale vega
        )
        
        # Create strike buckets for heatmap
        min_strike = int(greeks_df['strike'].min() / 10) * 10
        max_strike = int(greeks_df['strike'].max() / 10) * 10 + 10
        strike_buckets = list(range(min_strike, max_strike, 10))
        
        # Group by strike buckets
        def assign_bucket(strike):
            for i, bucket in enumerate(strike_buckets[:-1]):
                if bucket <= strike < strike_buckets[i+1]:
                    return f"${bucket}-${strike_buckets[i+1]}"
            return f"${strike_buckets[-1]}+"
        
        greeks_df['strike_bucket'] = greeks_df['strike'].apply(assign_bucket)
        
        # Aggregate by bucket
        bucket_data = greeks_df.groupby('strike_bucket').agg({
            'gamma_intensity': 'sum',
            'delta_intensity': 'sum', 
            'vega_intensity': 'sum',
            'total_intensity': 'sum',
            'dealer_gamma': 'sum',
            'dealer_delta': 'sum'
        }).reset_index()
        
        # Create heatmap
        fig = go.Figure()
        
        # Add heatmap trace
        fig.add_trace(
            go.Bar(
                x=bucket_data['strike_bucket'],
                y=bucket_data['total_intensity'],
                marker=dict(
                    color=bucket_data['total_intensity'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Positioning Intensity")
                ),
                text=bucket_data['total_intensity'].round(1),
                textposition='auto',
                hovertemplate='Strike Range: %{x}<br>Total Intensity: %{y}<br>Gamma: %{customdata[0]}<br>Delta: %{customdata[1]}<extra></extra>',
                customdata=np.column_stack((bucket_data['dealer_gamma'], bucket_data['dealer_delta']))
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Real SPY Options Positioning Intensity Heatmap",
            xaxis_title="Strike Range",
            yaxis_title="Positioning Intensity",
            template='plotly_dark',
            height=500
        )
        
        # Save chart
        chart_file = self.output_dir / "positioning_heatmap.html"
        fig.write_html(chart_file)
        
        print(f"✅ Positioning heatmap saved to {chart_file}")
        return str(chart_file)
    
    def create_comprehensive_dashboard(self, real_data: dict) -> str:
        """Create comprehensive dashboard with all real data analysis"""
        
        print("🏗️ Creating comprehensive real data dashboard...")
        
        # Create all individual charts
        strike_chart = self.create_strike_range_comparison(real_data)
        greeks_chart = self.create_greeks_breakdown_chart(real_data)
        heatmap_chart = self.create_positioning_heatmap(real_data)
        
        # Get summary statistics
        base_data = real_data.get('All_Expiries', list(real_data.values())[0])
        analysis = base_data['analysis']['analysis']
        
        # Create dashboard HTML
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Real SPY Options Positioning Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: white; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .chart-container {{ margin: 20px 0; border: 1px solid #333; border-radius: 10px; overflow: hidden; }}
        .chart-title {{ background-color: #333; padding: 15px; margin: 0; font-size: 18px; font-weight: bold; }}
        .chart-frame {{ width: 100%; height: 650px; border: none; }}
        .summary {{ background-color: #2a2a2a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .metric {{ background-color: #333; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #ccc; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Real SPY Options Dealer Positioning Dashboard</h1>
        <p>Analysis of actual SPY options trades and dealer positioning</p>
        <p><strong>Data Source:</strong> Real SPY options trades</p>
        <p><strong>Expiry Date:</strong> {base_data['expiry_list'][0]}</p>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    
    <div class="summary">
        <h2>📈 Real Data Summary</h2>
        <div class="grid">
            <div class="metric">
                <div class="metric-value">{base_data['trades_count']:,}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{base_data['strikes_count']}</div>
                <div class="metric-label">Strike Prices</div>
            </div>
            <div class="metric">
                <div class="metric-value">${base_data['spot_price']:.2f}</div>
                <div class="metric-label">SPY Price</div>
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <strong>Market Regime:</strong> {analysis.get('market_regime', {}).get('regime_type', 'Unknown')} 
            ({analysis.get('market_regime', {}).get('confidence', 0):.1%} confidence)<br>
            <strong>Gamma Centroid:</strong> ${analysis.get('key_levels', {}).get('gamma_centroid', 0):.2f}<br>
            <strong>Total Dealer Gamma:</strong> {analysis.get('greeks_summary', {}).get('total_gamma', 0):.0f}<br>
            <strong>Total Dealer Delta:</strong> {analysis.get('greeks_summary', {}).get('total_delta', 0):.0f}
        </div>
    </div>
    
    <div class="chart-container">
        <h2 class="chart-title">📈 Strike Range Analysis (Interactive)</h2>
        <iframe class="chart-frame" src="strike_range_analysis.html"></iframe>
    </div>
    
    <div class="chart-container">
        <h2 class="chart-title">🔢 Complete Greeks Breakdown</h2>
        <iframe class="chart-frame" src="greeks_breakdown.html"></iframe>
    </div>
    
    <div class="chart-container">
        <h2 class="chart-title">🔥 Positioning Intensity Heatmap</h2>
        <iframe class="chart-frame" src="positioning_heatmap.html"></iframe>
    </div>
    
    <div class="summary">
        <h2>📋 Analysis Features</h2>
        <p><strong>Strike Range Selection:</strong> Use the dropdown in the first chart to focus on different strike ranges (ATM, OTM calls/puts, full range)</p>
        <p><strong>Greeks Breakdown:</strong> View all four major Greeks (Gamma, Delta, Vega, Theta) simultaneously</p>
        <p><strong>Intensity Heatmap:</strong> See where dealer positioning is most concentrated</p>
        <p><strong>Real Data Advantage:</strong> All analysis based on actual SPY options trades, not simulated data</p>
        
        <h3>🎯 Key Insights from Real Data:</h3>
        <ul>
            <li>This is actual dealer positioning from real SPY options trades</li>
            <li>Market shows <strong>{analysis.get('market_regime', {}).get('regime_type', 'Unknown')}</strong> regime characteristics</li>
            <li>Gamma centroid at ${analysis.get('key_levels', {}).get('gamma_centroid', 0):.2f} vs SPY at ${base_data['spot_price']:.2f}</li>
            <li>Interactive charts let you explore different strike ranges and Greeks</li>
        </ul>
    </div>
</body>
</html>
"""
        
        # Save dashboard
        dashboard_file = self.output_dir / "real_dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        print(f"✅ Real data dashboard created at {dashboard_file}")
        return str(dashboard_file)


def main():
    """Main execution"""
    
    print("🚀 Creating Real Interactive SPY Visualizations...")
    
    try:
        visualizer = RealInteractiveSPYVisualizer()
        
        # Load real analysis data
        real_data = visualizer.load_real_analysis_data()
        
        if not real_data:
            print("❌ No real data available")
            return
        
        # Create comprehensive dashboard
        dashboard_file = visualizer.create_comprehensive_dashboard(real_data)
        
        print(f"\n🎉 Real SPY interactive dashboard created!")
        print(f"📁 Open this file in your browser: {dashboard_file}")
        print(f"\n📊 Dashboard Features:")
        print(f"   • Strike Range Analysis: Interactive dropdown to select different strike ranges")
        print(f"   • Greeks Breakdown: All four Greeks (Gamma, Delta, Vega, Theta) in one view")
        print(f"   • Positioning Heatmap: Visual intensity map showing concentration areas")
        print(f"   • Real Data: Based on actual SPY options trades, not simulated")
        
        print(f"\n💡 This shows real dealer positioning from actual SPY options data!")
        print(f"🔍 You can analyze different strike ranges to see how positioning varies")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()