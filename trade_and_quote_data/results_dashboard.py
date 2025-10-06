#!/usr/bin/env python3
"""
Results Dashboard and Comparison Framework
Phase 0 implementation from MODEL_IMPROVEMENT_ROADMAP.md

Comprehensive dashboard for comparing all target combinations and selecting
the optimal targets for ensemble modeling.

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import logging
from datetime import datetime
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsDashboard:
    """
    Comprehensive dashboard for analyzing and comparing multiple target results
    
    Features:
    - Target performance comparison
    - 2024-specific analysis
    - Feature importance analysis
    - Ensemble recommendation
    - Interactive visualizations
    """
    
    def __init__(self, output_dir: str = 'analysis/outputs/dashboard'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_data = {}
        self.target_analysis = None
        self.feature_importance = {}
        self.ensemble_recommendations = []
        
    def load_target_results(self, results_file: str) -> pd.DataFrame:
        """Load target analysis results from file"""
        logger.info(f"Loading target results from {results_file}")
        
        if results_file.endswith('.csv'):
            df = pd.read_csv(results_file)
        elif results_file.endswith('.json'):
            with open(results_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        self.target_analysis = df
        logger.info(f"Loaded {len(df)} target results")
        return df
    
    def create_performance_heatmap(self, metric: str = 'f1_at_optimal') -> go.Figure:
        """
        Create heatmap showing performance across magnitude vs horizon
        
        Args:
            metric: Performance metric to visualize
            
        Returns:
            Plotly figure object
        """
        if self.target_analysis is None:
            raise ValueError("No target analysis data loaded")
        
        # Parse target names to extract magnitude and horizon
        df = self.target_analysis.copy()
        df['magnitude'] = df['target_name'].str.extract(r'(\d+)pct').astype(int)
        df['horizon'] = df['target_name'].str.extract(r'(\d+)d').astype(int)
        
        # Create pivot table for heatmap
        heatmap_data = df.pivot(index='magnitude', columns='horizon', values=metric)
        
        # Create plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlBu_r',
            text=np.round(heatmap_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title=metric.replace('_', ' ').title())
        ))
        
        fig.update_layout(
            title=f'{metric.replace("_", " ").title()} by Magnitude and Horizon',
            xaxis_title='Horizon (days)',
            yaxis_title='Magnitude (%)',
            font=dict(size=12),
            width=800,
            height=500
        )
        
        return fig
    
    def create_metrics_comparison(self) -> go.Figure:
        """Create comprehensive metrics comparison chart"""
        if self.target_analysis is None:
            raise ValueError("No target analysis data loaded")
        
        df = self.target_analysis.copy()
        
        # Select key metrics for comparison
        metrics = ['roc_auc', 'f1_at_optimal', 'precision_at_optimal', 'recall_at_optimal', 'precision_at_80pct']
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*2 + [{"colspan": 1}]]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Sort by metric value
            sorted_df = df.nlargest(10, metric)
            
            fig.add_trace(
                go.Bar(
                    x=sorted_df[metric],
                    y=sorted_df['target_name'],
                    orientation='h',
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)
            fig.update_yaxes(title_text='Target', row=row, col=col)
        
        fig.update_layout(
            title='Top 10 Targets by Different Metrics',
            height=800,
            font=dict(size=10)
        )
        
        return fig
    
    def create_2024_performance_analysis(self, performance_2024_file: str = None) -> go.Figure:
        """Create specific analysis of 2024 performance"""
        
        if performance_2024_file and Path(performance_2024_file).exists():
            df_2024 = pd.read_csv(performance_2024_file)
        else:
            logger.warning("No 2024 performance file provided, creating dummy data")
            # Create dummy 2024 data for demonstration
            df_2024 = self.target_analysis.copy()
            df_2024['roc_auc_2024'] = np.random.uniform(0.4, 0.8, len(df_2024))
            df_2024['precision_high_conf_2024'] = np.random.uniform(0.3, 0.9, len(df_2024))
            df_2024['n_high_conf_signals'] = np.random.poisson(5, len(df_2024))
        
        # Create subplots for 2024 analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '2024 ROC AUC Performance',
                '2024 High Confidence Precision',
                '2024 Signal Frequency',
                'Overall vs 2024 Performance'
            ]
        )
        
        # ROC AUC in 2024
        top_2024_roc = df_2024.nlargest(10, 'roc_auc_2024')
        fig.add_trace(
            go.Bar(
                x=top_2024_roc['roc_auc_2024'],
                y=top_2024_roc['target_name'],
                orientation='h',
                name='2024 ROC AUC',
                marker_color='darkblue'
            ),
            row=1, col=1
        )
        
        # High confidence precision in 2024
        if 'precision_high_conf_2024' in df_2024.columns:
            top_2024_precision = df_2024.nlargest(10, 'precision_high_conf_2024')
            fig.add_trace(
                go.Bar(
                    x=top_2024_precision['precision_high_conf_2024'],
                    y=top_2024_precision['target_name'],
                    orientation='h',
                    name='2024 High Conf Precision',
                    marker_color='darkgreen'
                ),
                row=1, col=2
            )
        
        # Signal frequency in 2024
        if 'n_high_conf_signals' in df_2024.columns:
            fig.add_trace(
                go.Bar(
                    x=df_2024['n_high_conf_signals'],
                    y=df_2024['target_name'],
                    orientation='h',
                    name='2024 Signal Count',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # Overall vs 2024 performance scatter
        if 'roc_auc' in self.target_analysis.columns and 'roc_auc_2024' in df_2024.columns:
            merged_df = pd.merge(self.target_analysis, df_2024, on='target_name', how='inner')
            fig.add_trace(
                go.Scatter(
                    x=merged_df['roc_auc'],
                    y=merged_df['roc_auc_2024'],
                    mode='markers+text',
                    text=merged_df['target_name'].str.replace('pullback_', '').str.replace('pct_', '%_'),
                    textposition='top center',
                    name='Overall vs 2024',
                    marker=dict(size=8, color='red')
                ),
                row=2, col=2
            )
            
            # Add diagonal line for reference
            min_val = min(merged_df['roc_auc'].min(), merged_df['roc_auc_2024'].min())
            max_val = max(merged_df['roc_auc'].max(), merged_df['roc_auc_2024'].max())
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="gray", dash="dash"),
                row=2, col=2
            )
        
        fig.update_layout(
            title='2024 Performance Analysis',
            height=800,
            showlegend=False,
            font=dict(size=10)
        )
        
        return fig
    
    def create_target_selection_matrix(self) -> pd.DataFrame:
        """
        Create target selection matrix based on multiple criteria
        
        Returns:
            DataFrame with scoring for different use cases
        """
        if self.target_analysis is None:
            raise ValueError("No target analysis data loaded")
        
        df = self.target_analysis.copy()
        
        # Calculate composite scores for different use cases
        selection_matrix = pd.DataFrame()
        selection_matrix['target_name'] = df['target_name']
        
        # Trading Score: Balance of F1 and signal frequency
        if 'f1_at_optimal' in df.columns and 'positive_rate_test' in df.columns:
            selection_matrix['trading_score'] = (
                df['f1_at_optimal'] * 0.7 + 
                (1 - df['positive_rate_test']) * 0.3  # Prefer not too frequent
            )
        
        # Risk Management Score: High precision at high confidence
        if 'precision_at_80pct' in df.columns:
            selection_matrix['risk_mgmt_score'] = df['precision_at_80pct']
        
        # Event Detection Score: Recall and precision for major events
        if 'recall_at_optimal' in df.columns and 'precision_at_optimal' in df.columns:
            selection_matrix['event_detection_score'] = (
                df['recall_at_optimal'] * 0.6 +
                df['precision_at_optimal'] * 0.4
            )
        
        # Stability Score: ROC AUC (proxy for stability)
        if 'roc_auc' in df.columns:
            selection_matrix['stability_score'] = df['roc_auc']
        
        # Overall Score: Weighted combination
        score_cols = [col for col in selection_matrix.columns if col.endswith('_score')]
        if score_cols:
            selection_matrix['overall_score'] = selection_matrix[score_cols].mean(axis=1)
        
        # Add rank for each score
        for col in score_cols + ['overall_score']:
            if col in selection_matrix.columns:
                selection_matrix[f'{col}_rank'] = selection_matrix[col].rank(ascending=False)
        
        # Sort by overall score
        selection_matrix = selection_matrix.sort_values('overall_score', ascending=False)
        
        return selection_matrix
    
    def generate_ensemble_recommendations(self) -> Dict:
        """Generate recommendations for ensemble modeling"""
        
        selection_matrix = self.create_target_selection_matrix()
        
        recommendations = {
            'top_overall': [],
            'trading_ensemble': [],
            'risk_management_ensemble': [],
            'event_detection_ensemble': [],
            'diversified_ensemble': []
        }
        
        # Top 3 overall performers
        recommendations['top_overall'] = selection_matrix.nlargest(3, 'overall_score')['target_name'].tolist()
        
        # Best for trading (good balance)
        if 'trading_score' in selection_matrix.columns:
            recommendations['trading_ensemble'] = selection_matrix.nlargest(3, 'trading_score')['target_name'].tolist()
        
        # Best for risk management (high precision)
        if 'risk_mgmt_score' in selection_matrix.columns:
            recommendations['risk_management_ensemble'] = selection_matrix.nlargest(3, 'risk_mgmt_score')['target_name'].tolist()
        
        # Best for event detection
        if 'event_detection_score' in selection_matrix.columns:
            recommendations['event_detection_ensemble'] = selection_matrix.nlargest(3, 'event_detection_score')['target_name'].tolist()
        
        # Diversified ensemble (one from each magnitude)
        if self.target_analysis is not None:
            df = self.target_analysis.copy()
            df['magnitude'] = df['target_name'].str.extract(r'(\d+)pct').astype(int)
            
            diversified = []
            for magnitude in [2, 5, 10]:
                mag_targets = df[df['magnitude'] == magnitude]
                if len(mag_targets) > 0:
                    best_mag = mag_targets.loc[mag_targets['f1_at_optimal'].idxmax() if 'f1_at_optimal' in mag_targets.columns else mag_targets.index[0]]
                    diversified.append(best_mag['target_name'])
            
            recommendations['diversified_ensemble'] = diversified
        
        self.ensemble_recommendations = recommendations
        return recommendations
    
    def create_ensemble_comparison_chart(self) -> go.Figure:
        """Create chart comparing different ensemble strategies"""
        
        if not self.ensemble_recommendations:
            self.generate_ensemble_recommendations()
        
        # Create data for comparison
        ensemble_data = []
        
        for ensemble_type, targets in self.ensemble_recommendations.items():
            if targets and self.target_analysis is not None:
                # Get metrics for these targets
                target_metrics = self.target_analysis[self.target_analysis['target_name'].isin(targets)]
                
                if len(target_metrics) > 0:
                    ensemble_data.append({
                        'ensemble_type': ensemble_type.replace('_', ' ').title(),
                        'avg_roc_auc': target_metrics['roc_auc'].mean() if 'roc_auc' in target_metrics.columns else 0,
                        'avg_precision': target_metrics['precision_at_optimal'].mean() if 'precision_at_optimal' in target_metrics.columns else 0,
                        'avg_recall': target_metrics['recall_at_optimal'].mean() if 'recall_at_optimal' in target_metrics.columns else 0,
                        'avg_f1': target_metrics['f1_at_optimal'].mean() if 'f1_at_optimal' in target_metrics.columns else 0,
                        'n_targets': len(target_metrics)
                    })
        
        df_ensembles = pd.DataFrame(ensemble_data)
        
        if len(df_ensembles) == 0:
            logger.warning("No ensemble data available")
            return go.Figure()
        
        # Create radar chart
        metrics = ['avg_roc_auc', 'avg_precision', 'avg_recall', 'avg_f1']
        
        fig = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, row in df_ensembles.iterrows():
            values = [row[metric] for metric in metrics]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=row['ensemble_type'],
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Ensemble Strategy Comparison"
        )
        
        return fig
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive text report"""
        
        if self.target_analysis is None:
            return "No analysis data available"
        
        # Generate ensemble recommendations
        recommendations = self.generate_ensemble_recommendations()
        selection_matrix = self.create_target_selection_matrix()
        
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE TARGET ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY:")
        report.append(f"Analyzed {len(self.target_analysis)} target combinations")
        
        if 'roc_auc' in self.target_analysis.columns:
            avg_roc = self.target_analysis['roc_auc'].mean()
            best_roc = self.target_analysis['roc_auc'].max()
            best_target = self.target_analysis.loc[self.target_analysis['roc_auc'].idxmax(), 'target_name']
            
            report.append(f"Average ROC AUC: {avg_roc:.3f}")
            report.append(f"Best ROC AUC: {best_roc:.3f} ({best_target})")
        
        report.append("")
        
        # Top Performers
        report.append("TOP PERFORMING TARGETS:")
        top_targets = selection_matrix.head(5)
        for i, row in top_targets.iterrows():
            report.append(f"{i+1}. {row['target_name']}")
            report.append(f"   Overall Score: {row['overall_score']:.3f}")
            if 'trading_score' in row:
                report.append(f"   Trading Score: {row['trading_score']:.3f}")
            if 'risk_mgmt_score' in row:
                report.append(f"   Risk Mgmt Score: {row['risk_mgmt_score']:.3f}")
        report.append("")
        
        # Ensemble Recommendations
        report.append("ENSEMBLE RECOMMENDATIONS:")
        for ensemble_type, targets in recommendations.items():
            if targets:
                report.append(f"{ensemble_type.replace('_', ' ').title()}:")
                for target in targets:
                    report.append(f"  - {target}")
                report.append("")
        
        # Implementation Strategy
        report.append("IMPLEMENTATION STRATEGY:")
        report.append("1. Start with Top Overall ensemble for general use")
        report.append("2. Use Trading ensemble for active trading strategies")
        report.append("3. Use Risk Management ensemble for portfolio protection")
        report.append("4. Consider Diversified ensemble for robustness")
        report.append("")
        
        # Key Insights
        report.append("KEY INSIGHTS:")
        
        # Analyze patterns in the data
        df = self.target_analysis.copy()
        df['magnitude'] = df['target_name'].str.extract(r'(\d+)pct').astype(int)
        df['horizon'] = df['target_name'].str.extract(r'(\d+)d').astype(int)
        
        # Best magnitude
        if 'f1_at_optimal' in df.columns:
            best_magnitude = df.groupby('magnitude')['f1_at_optimal'].mean().idxmax()
            report.append(f"- Best magnitude: {best_magnitude}% pullbacks")
        
        # Best horizon
        if 'f1_at_optimal' in df.columns:
            best_horizon = df.groupby('horizon')['f1_at_optimal'].mean().idxmax()
            report.append(f"- Best horizon: {best_horizon} days")
        
        # Signal frequency insights
        if 'positive_rate_test' in df.columns:
            avg_signal_rate = df['positive_rate_test'].mean()
            report.append(f"- Average signal frequency: {avg_signal_rate:.3f}")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_all_visualizations(self):
        """Save all visualizations to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Performance heatmap
            fig1 = self.create_performance_heatmap()
            pyo.plot(fig1, filename=str(self.output_dir / f'performance_heatmap_{timestamp}.html'), auto_open=False)
            
            # Metrics comparison
            fig2 = self.create_metrics_comparison()
            pyo.plot(fig2, filename=str(self.output_dir / f'metrics_comparison_{timestamp}.html'), auto_open=False)
            
            # 2024 performance analysis
            fig3 = self.create_2024_performance_analysis()
            pyo.plot(fig3, filename=str(self.output_dir / f'2024_analysis_{timestamp}.html'), auto_open=False)
            
            # Ensemble comparison
            fig4 = self.create_ensemble_comparison_chart()
            pyo.plot(fig4, filename=str(self.output_dir / f'ensemble_comparison_{timestamp}.html'), auto_open=False)
            
            logger.info(f"Visualizations saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")
    
    def save_analysis_results(self):
        """Save analysis results and recommendations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save selection matrix
        selection_matrix = self.create_target_selection_matrix()
        selection_file = self.output_dir / f'target_selection_matrix_{timestamp}.csv'
        selection_matrix.to_csv(selection_file, index=False)
        
        # Save ensemble recommendations
        recommendations = self.generate_ensemble_recommendations()
        recommendations_file = self.output_dir / f'ensemble_recommendations_{timestamp}.json'
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        report_file = self.output_dir / f'comprehensive_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis results saved to {self.output_dir}")
        
        return {
            'selection_matrix_file': str(selection_file),
            'recommendations_file': str(recommendations_file),
            'report_file': str(report_file)
        }


def test_results_dashboard():
    """Test the results dashboard"""
    logger.info("Testing Results Dashboard")
    
    # Create dummy results data for testing
    target_combinations = []
    magnitudes = [2, 5, 10]
    horizons = [5, 10, 15, 20]
    
    for mag in magnitudes:
        for hor in horizons:
            target_combinations.append({
                'target_name': f'pullback_{mag}pct_{hor}d',
                'magnitude': mag / 100,
                'horizon': hor,
                'roc_auc': np.random.uniform(0.5, 0.8),
                'f1_at_optimal': np.random.uniform(0.3, 0.7),
                'precision_at_optimal': np.random.uniform(0.4, 0.8),
                'recall_at_optimal': np.random.uniform(0.3, 0.7),
                'precision_at_80pct': np.random.uniform(0.5, 0.9),
                'positive_rate_test': np.random.uniform(0.05, 0.25),
                'n_positive_test': np.random.randint(10, 100)
            })
    
    # Create test DataFrame
    test_results = pd.DataFrame(target_combinations)
    
    # Initialize dashboard
    dashboard = ResultsDashboard()
    dashboard.target_analysis = test_results
    
    print("Testing dashboard components...")
    
    # Test selection matrix
    selection_matrix = dashboard.create_target_selection_matrix()
    print(f"Selection matrix created: {len(selection_matrix)} targets")
    
    # Test ensemble recommendations
    recommendations = dashboard.generate_ensemble_recommendations()
    print(f"Ensemble recommendations: {len(recommendations)} strategies")
    
    # Test comprehensive report
    report = dashboard.generate_comprehensive_report()
    print(f"Report generated: {len(report.split('\\n'))} lines")
    
    # Test saving results
    saved_files = dashboard.save_analysis_results()
    print(f"Results saved: {len(saved_files)} files")
    
    print("\\nSample Selection Matrix:")
    print(selection_matrix[['target_name', 'overall_score', 'trading_score']].head())
    
    print("\\nSample Ensemble Recommendations:")
    for strategy, targets in recommendations.items():
        if targets:
            print(f"{strategy}: {targets[:2]}...")  # Show first 2 targets
    
    return True


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    success = test_results_dashboard()
    
    if success:
        print("\\nResults dashboard test completed successfully!")
    else:
        print("\\nResults dashboard test failed!")