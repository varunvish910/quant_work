#!/usr/bin/env python3
"""
SPX Weekly Options Dealer Positioning Analysis - Phase 6: Visualization
Creates professional-grade charts showing dealer positioning and Greeks exposure
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json


class DealerPositioningVisualizer:
    """Creates visualizations for dealer positioning analysis"""
    
    def __init__(self, spot_price: float, theme: str = "dark"):
        self.spot_price = spot_price
        self.theme = theme
        self.setup_theme()
    
    def setup_theme(self):
        """Setup color theme and styling"""
        if self.theme == "dark":
            self.bg_color = "#1e1e1e"
            self.text_color = "#ffffff"
            self.grid_color = "#333333"
            self.positive_color = "#00ff88"
            self.negative_color = "#ff4444"
            self.neutral_color = "#888888"
            self.spot_color = "#ffaa00"
        else:
            self.bg_color = "#ffffff"
            self.text_color = "#000000"
            self.grid_color = "#cccccc"
            self.positive_color = "#00aa44"
            self.negative_color = "#cc0000"
            self.neutral_color = "#666666"
            self.spot_color = "#ff8800"
    
    def create_greeks_panel(self, greeks_df: pd.DataFrame, analysis: Dict, 
                           output_path: str = "dealer_positioning_chart.html") -> go.Figure:
        """Create multi-panel Greeks exposure chart similar to reference image"""
        
        # Sort by strike for plotting
        sorted_df = greeks_df.sort_values('strike')
        strikes = sorted_df['strike']
        
        # Create subplots - 6 panels like reference
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Charm Exposure", "Gamma Exposure", 
                          "Vanna Exposure", "Delta Exposure",
                          "Vega Exposure", "Vomma Exposure"),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Color mapping for positive/negative exposure
        def get_colors(values):
            return [self.positive_color if v >= 0 else self.negative_color for v in values]
        
        # 1. Charm Exposure (top left)
        if 'dealer_charm' in sorted_df.columns:
            charm_colors = get_colors(sorted_df['dealer_charm'])
            fig.add_trace(
                go.Bar(x=strikes, y=sorted_df['dealer_charm'], 
                      marker_color=charm_colors, name="Charm",
                      showlegend=False),
                row=1, col=1
            )
        
        # 2. Gamma Exposure (top right) - Main chart
        gamma_colors = get_colors(sorted_df['dealer_gamma'])
        fig.add_trace(
            go.Bar(x=strikes, y=sorted_df['dealer_gamma'], 
                  marker_color=gamma_colors, name="Gamma",
                  showlegend=False),
            row=1, col=2
        )
        
        # 3. Vanna Exposure (middle left)
        if 'dealer_vanna' in sorted_df.columns:
            vanna_colors = get_colors(sorted_df['dealer_vanna'])
            fig.add_trace(
                go.Bar(x=strikes, y=sorted_df['dealer_vanna'], 
                      marker_color=vanna_colors, name="Vanna",
                      showlegend=False),
                row=2, col=1
            )
        
        # 4. Delta Exposure (middle right)
        delta_colors = get_colors(sorted_df['dealer_delta'])
        fig.add_trace(
            go.Bar(x=strikes, y=sorted_df['dealer_delta'], 
                  marker_color=delta_colors, name="Delta",
                  showlegend=False),
            row=2, col=2
        )
        
        # 5. Vega Exposure (bottom left)
        if 'dealer_vega' in sorted_df.columns:
            vega_colors = get_colors(sorted_df['dealer_vega'])
            fig.add_trace(
                go.Bar(x=strikes, y=sorted_df['dealer_vega'], 
                      marker_color=vega_colors, name="Vega",
                      showlegend=False),
                row=3, col=1
            )
        
        # 6. Vomma Exposure (bottom right)
        if 'dealer_vomma' in sorted_df.columns:
            vomma_colors = get_colors(sorted_df['dealer_vomma'])
            fig.add_trace(
                go.Bar(x=strikes, y=sorted_df['dealer_vomma'], 
                      marker_color=vomma_colors, name="Vomma",
                      showlegend=False),
                row=3, col=2
            )
        
        # Add spot price lines to all panels
        for row in range(1, 4):
            for col in range(1, 3):
                fig.add_vline(x=self.spot_price, line_dash="dot", 
                            line_color=self.spot_color, line_width=2,
                            row=row, col=col)
        
        # Add key levels from analysis
        key_levels = analysis.get('key_levels', {})
        
        # Add pivot lines to gamma chart (main chart)
        if hasattr(key_levels, 'upside_pivot') and key_levels.upside_pivot:
            fig.add_vline(x=key_levels.upside_pivot, line_dash="dash",
                        line_color="#ff6600", line_width=2,
                        row=1, col=2)
            fig.add_annotation(x=key_levels.upside_pivot, y=sorted_df['dealer_gamma'].max(),
                             text="Upside Pivot", showarrow=True,
                             row=1, col=2)
        
        if hasattr(key_levels, 'downside_pivot') and key_levels.downside_pivot:
            fig.add_vline(x=key_levels.downside_pivot, line_dash="dash",
                        line_color="#ff6600", line_width=2,
                        row=1, col=2)
            fig.add_annotation(x=key_levels.downside_pivot, y=sorted_df['dealer_gamma'].min(),
                             text="Downside Pivot", showarrow=True,
                             row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"SPX Dealer Positioning Analysis - Spot: {self.spot_price:.0f}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.text_color}
            },
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font_color=self.text_color,
            height=900,
            width=1400,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Update all axes
        for row in range(1, 4):
            for col in range(1, 3):
                fig.update_xaxes(
                    gridcolor=self.grid_color,
                    title_text="Strike" if row == 3 else "",
                    row=row, col=col
                )
                fig.update_yaxes(
                    gridcolor=self.grid_color,
                    zeroline=True, zerolinecolor=self.neutral_color,
                    row=row, col=col
                )
        
        # Save chart
        fig.write_html(output_path)
        print(f"Greeks panel chart saved to {output_path}")
        
        return fig
    
    def create_gamma_profile_3d(self, greeks_df: pd.DataFrame, 
                               output_path: str = "gamma_3d_surface.html") -> go.Figure:
        """Create 3D surface plot of gamma over strikes and time"""
        
        # For now, create a 2D gamma profile since we only have one expiry
        sorted_df = greeks_df.sort_values('strike')
        
        fig = go.Figure()
        
        # Main gamma curve
        fig.add_trace(go.Scatter(
            x=sorted_df['strike'],
            y=sorted_df['dealer_gamma'],
            mode='lines+markers',
            line=dict(color=self.positive_color, width=3),
            marker=dict(size=8, color=sorted_df['dealer_gamma'],
                       colorscale=[[0, self.negative_color], [1, self.positive_color]],
                       showscale=True, colorbar=dict(title="Gamma Exposure")),
            name="Dealer Gamma",
            hovertemplate="Strike: %{x}<br>Gamma: %{y:.2f}<extra></extra>"
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color=self.neutral_color)
        
        # Add spot line
        fig.add_vline(x=self.spot_price, line_dash="dot", 
                     line_color=self.spot_color, line_width=2)
        
        # Add spot annotation
        fig.add_annotation(
            x=self.spot_price, y=sorted_df['dealer_gamma'].max(),
            text=f"Spot: {self.spot_price:.0f}",
            showarrow=True, arrowcolor=self.spot_color
        )
        
        fig.update_layout(
            title="SPX Dealer Gamma Profile",
            xaxis_title="Strike Price",
            yaxis_title="Dealer Gamma Exposure",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font_color=self.text_color,
            height=600,
            width=1000
        )
        
        fig.update_xaxes(gridcolor=self.grid_color)
        fig.update_yaxes(gridcolor=self.grid_color, zeroline=True, 
                        zerolinecolor=self.neutral_color)
        
        fig.write_html(output_path)
        print(f"Gamma profile chart saved to {output_path}")
        
        return fig
    
    def create_positioning_heatmap(self, trade_greeks_df: pd.DataFrame,
                                  output_path: str = "positioning_heatmap.png") -> plt.Figure:
        """Create heatmap of dealer positioning changes over time"""
        
        # Handle different timestamp column names
        timestamp_col = None
        for col in ['timestamp', 'sip_timestamp', 'trade_timestamp']:
            if col in trade_greeks_df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            print("Warning: No timestamp column found, skipping heatmap")
            return None
        
        # Aggregate by strike and hour
        trade_greeks_df['hour'] = pd.to_datetime(trade_greeks_df[timestamp_col], unit='ns').dt.hour
        
        heatmap_data = trade_greeks_df.groupby(['strike', 'hour']).agg({
            'dealer_gamma': 'sum',
            'dealer_delta': 'sum'
        }).reset_index()
        
        # Pivot for heatmap
        gamma_pivot = heatmap_data.pivot(index='strike', columns='hour', values='dealer_gamma')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Set style
        if self.theme == "dark":
            plt.style.use('dark_background')
            cmap = 'RdBu_r'
        else:
            cmap = 'RdBu'
        
        sns.heatmap(gamma_pivot, cmap=cmap, center=0, 
                   annot=False, fmt='.0f', cbar_kws={'label': 'Dealer Gamma'})
        
        ax.set_title('Dealer Gamma Evolution Throughout Trading Day', fontsize=16, pad=20)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Strike Price', fontsize=12)
        
        # Add spot price line
        if self.spot_price in gamma_pivot.index:
            spot_idx = gamma_pivot.index.get_loc(self.spot_price)
            ax.axhline(y=spot_idx, color='orange', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Positioning heatmap saved to {output_path}")
        
        return fig
    
    def create_volatility_smile(self, trade_greeks_df: pd.DataFrame,
                               output_path: str = "volatility_smile.html") -> go.Figure:
        """Create volatility smile chart with smoothing"""
        
        # Calculate moneyness and average IV by strike
        smile_data = trade_greeks_df.groupby('strike').agg({
            'implied_vol': 'mean',
            'volume': 'sum'
        }).reset_index()
        
        smile_data['moneyness'] = smile_data['strike'] / self.spot_price
        
        # Sort by moneyness for proper interpolation
        smile_data = smile_data.sort_values('moneyness')
        
        fig = go.Figure()
        
        # Original data points (smaller, semi-transparent)
        fig.add_trace(go.Scatter(
            x=smile_data['moneyness'],
            y=smile_data['implied_vol'] * 100,
            mode='markers',
            marker=dict(
                size=smile_data['volume'] / smile_data['volume'].max() * 15 + 3,
                color=smile_data['implied_vol'],
                colorscale='Viridis',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            name="Raw Data",
            hovertemplate="Moneyness: %{x:.3f}<br>IV: %{y:.1f}%<br>Volume: %{marker.size}<extra></extra>"
        ))
        
        # Create smooth interpolated curve
        if len(smile_data) >= 3:
            from scipy.interpolate import UnivariateSpline
            
            # Use spline interpolation for smoothing
            try:
                # Create more points for smooth curve
                x_smooth = np.linspace(smile_data['moneyness'].min(), 
                                     smile_data['moneyness'].max(), 100)
                
                # Apply spline smoothing (s parameter controls smoothness)
                spline = UnivariateSpline(smile_data['moneyness'], 
                                        smile_data['implied_vol'] * 100, 
                                        s=len(smile_data) * 0.1)  # Smoothing factor
                y_smooth = spline(x_smooth)
                
                # Add smooth curve
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    line=dict(color=self.positive_color, width=3),
                    name="Smooth Volatility Smile",
                    hovertemplate="Moneyness: %{x:.3f}<br>Smoothed IV: %{y:.1f}%<extra></extra>"
                ))
                
            except ImportError:
                # Fallback to moving average if scipy not available
                window = max(3, len(smile_data) // 5)
                smile_data['iv_smooth'] = smile_data['implied_vol'].rolling(
                    window=window, center=True, min_periods=1
                ).mean() * 100
                
                fig.add_trace(go.Scatter(
                    x=smile_data['moneyness'],
                    y=smile_data['iv_smooth'],
                    mode='lines',
                    line=dict(color=self.positive_color, width=3),
                    name="Smoothed Volatility Smile",
                    hovertemplate="Moneyness: %{x:.3f}<br>Smoothed IV: %{y:.1f}%<extra></extra>"
                ))
        
        # Add colorbar for the scatter points
        fig.update_traces(
            marker_colorbar=dict(title="Implied Vol"),
            selector=dict(mode="markers")
        )
        
        # Add ATM line
        fig.add_vline(x=1.0, line_dash="dot", line_color=self.spot_color, line_width=2)
        fig.add_annotation(x=1.0, y=smile_data['implied_vol'].max() * 100,
                          text="ATM", showarrow=True)
        
        fig.update_layout(
            title="SPX Options Volatility Smile",
            xaxis_title="Moneyness (Strike/Spot)",
            yaxis_title="Implied Volatility (%)",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font_color=self.text_color,
            height=600,
            width=1000
        )
        
        fig.update_xaxes(gridcolor=self.grid_color)
        fig.update_yaxes(gridcolor=self.grid_color)
        
        fig.write_html(output_path)
        print(f"Volatility smile chart saved to {output_path}")
        
        return fig
    
    def create_summary_dashboard(self, analysis: Dict, greeks_df: pd.DataFrame,
                               output_path: str = "dealer_positioning_dashboard.html") -> go.Figure:
        """Create comprehensive dashboard with all key metrics"""
        
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("Gamma Profile", "Key Metrics", "Risk Assessment",
                          "Position Summary", "Market Regime", "Trading Insights"),
            specs=[[{"type": "scatter"}, {"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}, {"type": "table"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Gamma Profile (top left)
        sorted_df = greeks_df.sort_values('strike')
        gamma_colors = [self.positive_color if g >= 0 else self.negative_color 
                       for g in sorted_df['dealer_gamma']]
        
        fig.add_trace(
            go.Bar(x=sorted_df['strike'], y=sorted_df['dealer_gamma'],
                  marker_color=gamma_colors, name="Gamma", showlegend=False),
            row=1, col=1
        )
        
        # Add spot line to gamma chart
        fig.add_vline(x=self.spot_price, line_dash="dot", 
                     line_color=self.spot_color, row=1, col=1)
        
        # 2. Key Metrics (top middle)
        key_levels = analysis.get('key_levels', {})
        if hasattr(key_levels, 'gamma_centroid'):
            centroid = key_levels.gamma_centroid
        else:
            centroid = self.spot_price
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=centroid,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Gamma Centroid"},
                gauge={'axis': {'range': [None, sorted_df['strike'].max()]},
                      'bar': {'color': self.positive_color},
                      'steps': [{'range': [sorted_df['strike'].min(), self.spot_price], 'color': self.grid_color}],
                      'threshold': {'line': {'color': self.spot_color, 'width': 4},
                                  'thickness': 0.75, 'value': self.spot_price}}
            ),
            row=1, col=2
        )
        
        # 3. Risk Assessment (top right)
        risks = analysis.get('risk_assessment', {})
        risk_levels = ['Low', 'Medium', 'High']
        risk_values = [
            1 if risks.get('gamma_risk', {}).get('level') == 'low' else 0,
            1 if risks.get('gamma_risk', {}).get('level') == 'medium' else 0,
            1 if risks.get('gamma_risk', {}).get('level') == 'high' else 0
        ]
        
        fig.add_trace(
            go.Bar(x=risk_levels, y=risk_values, 
                  marker_color=[self.positive_color, '#ffaa00', self.negative_color],
                  name="Risk Level", showlegend=False),
            row=1, col=3
        )
        
        # 4. Position Summary (bottom left) 
        greeks_summary = {
            'Gamma': sorted_df['dealer_gamma'].sum(),
            'Delta': sorted_df['dealer_delta'].sum(),
            'Vega': sorted_df.get('dealer_vega', pd.Series([0])).sum(),
            'Theta': sorted_df.get('dealer_theta', pd.Series([0])).sum()
        }
        
        greek_names = list(greeks_summary.keys())
        greek_values = list(greeks_summary.values())
        greek_colors = [self.positive_color if v >= 0 else self.negative_color for v in greek_values]
        
        fig.add_trace(
            go.Bar(x=greek_names, y=greek_values,
                  marker_color=greek_colors, name="Greeks", showlegend=False),
            row=2, col=1
        )
        
        # 5. Market Regime (bottom middle)
        regime = analysis.get('market_regime', {})
        if hasattr(regime, 'confidence'):
            confidence = regime.confidence
            regime_type = regime.regime_type
        else:
            confidence = 0.5
            regime_type = "unknown"
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Regime: {regime_type.title()}"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': self.positive_color},
                      'steps': [{'range': [0, 50], 'color': self.negative_color},
                               {'range': [50, 75], 'color': '#ffaa00'},
                               {'range': [75, 100], 'color': self.positive_color}]}
            ),
            row=2, col=2
        )
        
        # 6. Trading Insights (bottom right) - skip for now due to subplot compatibility
        # insights = analysis.get('trading_insights', ['No insights available'])
        # insights_text = "<br>".join([f"• {insight}" for insight in insights[:5]])
        
        # Add a simple bar chart instead
        insight_count = len(analysis.get('trading_insights', []))
        fig.add_trace(
            go.Bar(x=['Insights'], y=[insight_count],
                  marker_color=self.positive_color, 
                  name="Insights", showlegend=False),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"SPX Dealer Positioning Dashboard - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.text_color}
            },
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font_color=self.text_color,
            height=800,
            width=1600,
            margin=dict(l=60, r=60, t=100, b=60)
        )
        
        # Update axes styling
        fig.update_xaxes(gridcolor=self.grid_color)
        fig.update_yaxes(gridcolor=self.grid_color, zeroline=True, 
                        zerolinecolor=self.neutral_color)
        
        fig.write_html(output_path)
        print(f"Dashboard saved to {output_path}")
        
        return fig
    
    def create_all_charts(self, greeks_df: pd.DataFrame, trade_greeks_df: pd.DataFrame,
                         analysis: Dict, output_dir: str = "outputs/") -> Dict[str, str]:
        """Create all visualization charts"""
        
        chart_files = {}
        
        # Main Greeks panel
        main_chart = self.create_greeks_panel(greeks_df, analysis, 
                                            f"{output_dir}/greeks_panel.html")
        chart_files['greeks_panel'] = f"{output_dir}/greeks_panel.html"
        
        # Gamma profile
        gamma_chart = self.create_gamma_profile_3d(greeks_df, 
                                                 f"{output_dir}/gamma_profile.html")
        chart_files['gamma_profile'] = f"{output_dir}/gamma_profile.html"
        
        # Volatility smile
        if len(trade_greeks_df) > 0:
            vol_chart = self.create_volatility_smile(trade_greeks_df, 
                                                   f"{output_dir}/volatility_smile.html")
            chart_files['volatility_smile'] = f"{output_dir}/volatility_smile.html"
            
            # Positioning heatmap
            heatmap = self.create_positioning_heatmap(trade_greeks_df, 
                                                    f"{output_dir}/positioning_heatmap.png")
            if heatmap is not None:
                chart_files['positioning_heatmap'] = f"{output_dir}/positioning_heatmap.png"
        
        # Summary dashboard - skip for now due to subplot compatibility issues
        # dashboard = self.create_summary_dashboard(analysis, greeks_df, 
        #                                         f"{output_dir}/dashboard.html")
        # chart_files['dashboard'] = f"{output_dir}/dashboard.html"
        print("Note: Dashboard creation skipped due to subplot compatibility")
        
        print(f"\nAll charts created and saved to {output_dir}/")
        
        return chart_files


def main():
    """Example usage"""
    try:
        # Load data
        greeks_df = pd.read_parquet("data/spx_options/greeks/aggregated_greeks.parquet")
        trade_greeks_df = pd.read_parquet("data/spx_options/greeks/trade_level_greeks.parquet")
        
        # Load analysis
        with open("data/spx_options/market_structure_analysis.json", "r") as f:
            analysis = json.load(f)
        
        print(f"Loaded Greeks data: {len(greeks_df)} strikes, {len(trade_greeks_df)} trades")
        
        # Current SPX
        current_spx = 5800
        
        # Create visualizer
        visualizer = DealerPositioningVisualizer(spot_price=current_spx, theme="dark")
        
        # Create output directory
        import os
        os.makedirs("outputs/dealer_positioning", exist_ok=True)
        
        # Create all charts
        chart_files = visualizer.create_all_charts(
            greeks_df, trade_greeks_df, analysis, 
            output_dir="outputs/dealer_positioning"
        )
        
        print("\n=== Visualization Complete ===")
        for chart_type, file_path in chart_files.items():
            print(f"{chart_type}: {file_path}")
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Run the previous analysis steps first.")


if __name__ == "__main__":
    main()