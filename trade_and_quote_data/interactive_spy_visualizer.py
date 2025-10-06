#!/usr/bin/env python3
"""
Interactive SPY Options Positioning Visualizer
Creates interactive charts showing dealer positioning evolution over time with expiry selection capability
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta, date
from pathlib import Path
import json
from spy_trades_downloader import SPYTradesDownloader
from trade_classifier import TradeClassifier
from greeks_calculator import GreeksCalculator
from market_structure_analyzer import MarketStructureAnalyzer
import yfinance as yf


class InteractiveSPYVisualizer:
    """Creates interactive visualizations for SPY options positioning"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("outputs/interactive_spy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def simulate_historical_data(self, expiry_dates: list, num_days: int = 10) -> dict:
        """Simulate historical data by generating variations of the existing data"""
        print(f"📊 Simulating {num_days} days of historical SPY positioning data...")
        
        # Load the existing SPY data as a base
        base_file = self.data_dir / "spy_options" / "trades" / "2025-09-12_enriched_trades.parquet"
        
        if not base_file.exists():
            raise FileNotFoundError(f"Base SPY data not found at {base_file}")
        
        base_trades = pd.read_parquet(base_file)
        print(f"✓ Loaded base data: {len(base_trades):,} trades")
        
        # Generate historical scenarios
        historical_data = {}
        start_date = datetime(2025, 9, 5).date()  # Week before
        
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
                
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"   Generating data for {date_str}...")
            
            # Create variations of the base data
            daily_data = self._create_daily_variation(base_trades, expiry_dates, i, num_days)
            
            if daily_data is not None:
                historical_data[date_str] = daily_data
                print(f"   ✓ Generated {daily_data['strikes_count']} strikes for {date_str}")
        
        print(f"✅ Generated {len(historical_data)} days of historical data")
        return historical_data
    
    def _create_daily_variation(self, base_trades: pd.DataFrame, expiry_dates: list, 
                               day_index: int, total_days: int) -> dict:
        """Create a variation of base trades for a specific day"""
        
        try:
            # Create some realistic variations
            variation_factor = 0.8 + (day_index / total_days) * 0.4  # Progress from 0.8 to 1.2
            
            # Filter for target expiries
            target_expiries = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expiry_dates]
            filtered_trades = base_trades[base_trades['expiry'].isin(target_expiries)].copy()
            
            if len(filtered_trades) == 0:
                return None
            
            # Apply variations
            price_variation = 1.0 + (np.random.normal(0, 0.02))  # 2% daily price variation
            volume_variation = variation_factor * (1.0 + np.random.normal(0, 0.1))  # 10% volume variation
            
            # Modify prices and volumes
            filtered_trades['price'] = filtered_trades['price'] * price_variation
            filtered_trades['size'] = (filtered_trades['size'] * volume_variation).astype(int)
            
            # Vary spot price
            base_spot = 569.21
            spot_price = base_spot * price_variation
            
            # Process the modified data
            classifier = TradeClassifier()
            classified_df = classifier.classify_all_trades(filtered_trades)
            
            # Calculate Greeks
            calculator = GreeksCalculator(
                spot=spot_price,
                rate=0.05,
                dividend_yield=0.015
            )
            
            aggregated_greeks, trade_greeks = calculator.aggregate_dealer_greeks(classified_df)
            
            # Market structure analysis
            analyzer = MarketStructureAnalyzer(spot_price=spot_price)
            analysis = analyzer.analyze_full_structure(aggregated_greeks)
            
            return {
                'spot_price': spot_price,
                'aggregated_greeks': aggregated_greeks,
                'trade_greeks': trade_greeks,
                'analysis': analysis,
                'trades_count': len(classified_df),
                'strikes_count': len(aggregated_greeks)
            }
            
        except Exception as e:
            print(f"      Error creating variation: {e}")
            return None
    
    def create_time_series_charts(self, historical_data: dict, expiry_dates: list) -> str:
        """Create interactive time series charts showing positioning evolution"""
        
        print("📈 Creating time series charts...")
        
        # Prepare time series data
        dates = []
        spot_prices = []
        gamma_centroids = []
        total_gammas = []
        total_deltas = []
        market_regimes = []
        regime_confidences = []
        
        for date_str in sorted(historical_data.keys()):
            data = historical_data[date_str]
            dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            spot_prices.append(data['spot_price'])
            
            # Extract key metrics from analysis
            analysis = data['analysis']
            key_levels = analysis.get('key_levels', {})
            greeks_summary = analysis.get('greeks_summary', {})
            market_regime = analysis.get('market_regime', {})
            
            gamma_centroid = getattr(key_levels, 'gamma_centroid', data['spot_price']) if hasattr(key_levels, 'gamma_centroid') else key_levels.get('gamma_centroid', data['spot_price'])
            gamma_centroids.append(gamma_centroid)
            
            total_gammas.append(greeks_summary.get('total_gamma', 0))
            total_deltas.append(greeks_summary.get('total_delta', 0))
            
            regime = getattr(market_regime, 'regime_type', 'unknown') if hasattr(market_regime, 'regime_type') else market_regime.get('regime_type', 'unknown')
            confidence = getattr(market_regime, 'confidence', 0) if hasattr(market_regime, 'confidence') else market_regime.get('confidence', 0)
            
            market_regimes.append(regime)
            regime_confidences.append(confidence)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'SPY Price vs Gamma Centroid',
                'Total Dealer Gamma Exposure',
                'Total Dealer Delta Exposure', 
                'Market Regime Evolution'
            ],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]],
            vertical_spacing=0.08
        )
        
        # Price and Gamma Centroid
        fig.add_trace(
            go.Scatter(x=dates, y=spot_prices, name='SPY Price', 
                      line=dict(color='blue', width=2)), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=gamma_centroids, name='Gamma Centroid',
                      line=dict(color='red', width=2, dash='dash')), 
            row=1, col=1
        )
        
        # Total Gamma
        fig.add_trace(
            go.Scatter(x=dates, y=total_gammas, name='Total Dealer Gamma',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # Total Delta  
        fig.add_trace(
            go.Scatter(x=dates, y=total_deltas, name='Total Dealer Delta',
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        # Market Regime (as colored background)
        regime_colors = {'pinning': 'rgba(255,0,0,0.2)', 'directional': 'rgba(0,255,0,0.2)', 'unknown': 'rgba(128,128,128,0.1)'}
        
        for i, (date, regime) in enumerate(zip(dates, market_regimes)):
            if i < len(dates) - 1:
                fig.add_shape(
                    type="rect",
                    x0=date, x1=dates[i+1],
                    y0=0, y1=1,
                    fillcolor=regime_colors.get(regime, 'rgba(128,128,128,0.1)'),
                    layer="below",
                    line_width=0,
                    row=4, col=1
                )
        
        # Add regime confidence line
        fig.add_trace(
            go.Scatter(x=dates, y=regime_confidences, name='Regime Confidence',
                      line=dict(color='orange', width=2)),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'SPY Options Dealer Positioning Evolution - Expiries: {", ".join(expiry_dates)}',
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Gamma", row=2, col=1)
        fig.update_yaxes(title_text="Delta", row=3, col=1)
        fig.update_yaxes(title_text="Confidence", row=4, col=1)
        
        # Save chart
        chart_file = self.output_dir / "time_series_positioning.html"
        fig.write_html(chart_file)
        
        print(f"✅ Time series chart saved to {chart_file}")
        return str(chart_file)
    
    def create_expiry_comparison_charts(self, historical_data: dict, expiry_dates: list) -> str:
        """Create charts comparing positioning across different expiries"""
        
        print("📊 Creating expiry comparison charts...")
        
        # Get the most recent date's data for comparison
        latest_date = max(historical_data.keys())
        latest_data = historical_data[latest_date]
        
        # Create comparison by expiry
        aggregated_greeks = latest_data['aggregated_greeks']
        
        # Create dropdown for expiry selection
        fig = go.Figure()
        
        # Add traces for each expiry
        for expiry in expiry_dates:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            expiry_data = aggregated_greeks[aggregated_greeks['expiry'] == expiry_date]
            
            if len(expiry_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=expiry_data['strike'],
                        y=expiry_data['dealer_gamma'],
                        mode='lines+markers',
                        name=f'Gamma - {expiry}',
                        visible=True if expiry == expiry_dates[0] else False,
                        line=dict(width=3),
                        marker=dict(size=6)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=expiry_data['strike'],
                        y=expiry_data['dealer_delta'],
                        mode='lines+markers', 
                        name=f'Delta - {expiry}',
                        visible=True if expiry == expiry_dates[0] else False,
                        yaxis='y2',
                        line=dict(width=3, dash='dash'),
                        marker=dict(size=6)
                    )
                )
        
        # Create dropdown menu for expiry selection
        dropdown_buttons = []
        for i, expiry in enumerate(expiry_dates):
            # Create visibility array
            visibility = [False] * (len(expiry_dates) * 2)
            visibility[i*2] = True      # Gamma trace
            visibility[i*2 + 1] = True  # Delta trace
            
            dropdown_buttons.append(
                dict(
                    label=expiry,
                    method="update",
                    args=[{"visible": visibility},
                          {"title": f"SPY Options Positioning - {expiry}"}]
                )
            )
        
        # Add "All" option
        all_visibility = [True] * (len(expiry_dates) * 2)
        dropdown_buttons.insert(0, 
            dict(
                label="All Expiries",
                method="update", 
                args=[{"visible": all_visibility},
                      {"title": "SPY Options Positioning - All Expiries"}]
            )
        )
        
        # Update layout with dropdown
        fig.update_layout(
            title=f"SPY Options Positioning - {expiry_dates[0]}",
            xaxis_title="Strike Price",
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
            height=600
        )
        
        # Add current SPY price line
        current_price = latest_data['spot_price']
        fig.add_vline(x=current_price, line_dash="solid", line_color="red", 
                     annotation_text=f"SPY: ${current_price:.2f}")
        
        # Save chart
        chart_file = self.output_dir / "expiry_comparison.html"
        fig.write_html(chart_file)
        
        print(f"✅ Expiry comparison chart saved to {chart_file}")
        return str(chart_file)
    
    def create_gamma_profile_evolution(self, historical_data: dict) -> str:
        """Create animated gamma profile showing evolution over time"""
        
        print("🎬 Creating animated gamma profile...")
        
        # Prepare data for animation
        frames = []
        dates_list = sorted(historical_data.keys())
        
        for date_str in dates_list:
            data = historical_data[date_str]
            aggregated_greeks = data['aggregated_greeks']
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=aggregated_greeks['strike'],
                        y=aggregated_greeks['dealer_gamma'],
                        mode='lines+markers',
                        name='Dealer Gamma',
                        line=dict(color='green', width=3),
                        marker=dict(size=6)
                    ),
                    go.Scatter(
                        x=[data['spot_price'], data['spot_price']],
                        y=[aggregated_greeks['dealer_gamma'].min(), aggregated_greeks['dealer_gamma'].max()],
                        mode='lines',
                        name='SPY Price',
                        line=dict(color='red', width=2, dash='solid')
                    )
                ],
                name=date_str
            )
            frames.append(frame)
        
        # Initial frame
        initial_data = historical_data[dates_list[0]]
        initial_greeks = initial_data['aggregated_greeks']
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=initial_greeks['strike'],
                    y=initial_greeks['dealer_gamma'],
                    mode='lines+markers',
                    name='Dealer Gamma',
                    line=dict(color='green', width=3),
                    marker=dict(size=6)
                ),
                go.Scatter(
                    x=[initial_data['spot_price'], initial_data['spot_price']],
                    y=[initial_greeks['dealer_gamma'].min(), initial_greeks['dealer_gamma'].max()],
                    mode='lines',
                    name='SPY Price',
                    line=dict(color='red', width=2, dash='solid')
                )
            ],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="SPY Dealer Gamma Profile Evolution",
            xaxis_title="Strike Price",
            yaxis_title="Dealer Gamma",
            template='plotly_dark',
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 1000, "redraw": True},
                                         "fromcurrent": True}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}])
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            args=[[frame.name], {"frame": {"duration": 300, "redraw": True},
                                                 "mode": "immediate",
                                                 "transition": {"duration": 300}}],
                            label=frame.name,
                            method="animate"
                        ) for frame in frames
                    ],
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue={"prefix": "Date: "},
                    transition={"duration": 300, "easing": "cubic-in-out"},
                    len=0.9,
                    x=0.1,
                    y=0
                )
            ]
        )
        
        # Save chart
        chart_file = self.output_dir / "gamma_profile_evolution.html"
        fig.write_html(chart_file)
        
        print(f"✅ Animated gamma profile saved to {chart_file}")
        return str(chart_file)
    
    def create_dashboard(self, historical_data: dict, expiry_dates: list) -> str:
        """Create a comprehensive dashboard with all charts"""
        
        print("🏗️ Creating comprehensive dashboard...")
        
        # Create all individual charts
        time_series_file = self.create_time_series_charts(historical_data, expiry_dates)
        expiry_comparison_file = self.create_expiry_comparison_charts(historical_data, expiry_dates)
        animation_file = self.create_gamma_profile_evolution(historical_data)
        
        # Create HTML dashboard that embeds all charts
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SPY Options Positioning Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: white; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .chart-container {{ margin: 20px 0; border: 1px solid #333; border-radius: 10px; overflow: hidden; }}
        .chart-title {{ background-color: #333; padding: 15px; margin: 0; font-size: 18px; font-weight: bold; }}
        .chart-frame {{ width: 100%; height: 600px; border: none; }}
        .summary {{ background-color: #2a2a2a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 SPY Options Dealer Positioning Dashboard</h1>
        <p>Interactive analysis of dealer positioning evolution over time</p>
        <p><strong>Target Expiries:</strong> {', '.join(expiry_dates)}</p>
        <p><strong>Data Period:</strong> {min(historical_data.keys())} to {max(historical_data.keys())}</p>
    </div>
    
    <div class="summary">
        <h2>📈 Summary</h2>
        <div class="grid">
            <div>
                <strong>Trading Days Analyzed:</strong> {len(historical_data)}<br>
                <strong>Latest SPY Price:</strong> ${historical_data[max(historical_data.keys())]['spot_price']:.2f}<br>
                <strong>Latest Strikes:</strong> {historical_data[max(historical_data.keys())]['strikes_count']}
            </div>
            <div>
                <strong>How to Use:</strong><br>
                • Time Series: Shows evolution over time<br>
                • Expiry Comparison: Compare different expiry dates<br>
                • Animation: Watch gamma profile changes
            </div>
        </div>
    </div>
    
    <div class="chart-container">
        <h2 class="chart-title">⏰ Time Series Evolution</h2>
        <iframe class="chart-frame" src="time_series_positioning.html"></iframe>
    </div>
    
    <div class="chart-container">
        <h2 class="chart-title">📊 Expiry Comparison (Interactive)</h2>
        <iframe class="chart-frame" src="expiry_comparison.html"></iframe>
    </div>
    
    <div class="chart-container">
        <h2 class="chart-title">🎬 Gamma Profile Evolution (Animated)</h2>
        <iframe class="chart-frame" src="gamma_profile_evolution.html"></iframe>
    </div>
    
    <div class="summary">
        <h2>📋 Key Features</h2>
        <p><strong>Time Series Charts:</strong> Track how dealer positioning changes day by day</p>
        <p><strong>Expiry Selection:</strong> Use the dropdown to focus on specific expiry dates</p>
        <p><strong>Animation Controls:</strong> Play/pause to watch gamma profiles evolve</p>
        <p><strong>Interactive Elements:</strong> Zoom, pan, and hover for detailed information</p>
    </div>
</body>
</html>
"""
        
        # Save dashboard
        dashboard_file = self.output_dir / "dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        print(f"✅ Dashboard created at {dashboard_file}")
        return str(dashboard_file)


def main():
    """Main execution"""
    # Target expiry dates for demonstration
    expiry_dates = [
        "2025-10-06",  # Monday
        "2025-10-07",  # Tuesday  
        "2025-10-08",  # Wednesday
        "2025-10-09",  # Thursday
        "2025-10-10"   # Friday
    ]
    
    visualizer = InteractiveSPYVisualizer()
    
    print("🚀 Creating interactive SPY positioning visualizations...")
    print(f"Target expiries: {', '.join(expiry_dates)}")
    
    try:
        # Step 1: Generate simulated historical data
        historical_data = visualizer.simulate_historical_data(expiry_dates, num_days=10)
        
        if not historical_data:
            print("❌ No historical data generated")
            return
        
        # Step 2: Create comprehensive dashboard
        dashboard_file = visualizer.create_dashboard(historical_data, expiry_dates)
        
        print(f"\n🎉 Interactive dashboard created!")
        print(f"📁 Open this file in your browser: {dashboard_file}")
        print(f"\n📊 Available charts:")
        print(f"   • Time Series: Shows positioning evolution over time")
        print(f"   • Expiry Comparison: Interactive dropdown to select expiry dates")
        print(f"   • Animated Profile: Watch gamma profiles change day by day")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()