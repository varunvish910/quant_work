#!/usr/bin/env python3
"""
Correction Analysis - Analyze patterns and characteristics of corrections
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

class CorrectionAnalyzer:
    """Analyze characteristics of historical corrections for insights"""
    
    def __init__(self, correction_events: List[Dict]):
        self.correction_events = correction_events
        
    def analyze_correction_patterns(self) -> Dict:
        """Analyze patterns in correction timing, magnitude, duration"""
        if not self.correction_events:
            return {"error": "No correction events to analyze"}
        
        analysis = {}
        
        # Basic statistics
        magnitudes = [c['magnitude'] for c in self.correction_events]
        durations = [c['duration_days'] for c in self.correction_events]
        
        analysis['basic_stats'] = {
            'total_corrections': len(self.correction_events),
            'magnitude_stats': {
                'mean': np.mean(magnitudes),
                'median': np.median(magnitudes),
                'std': np.std(magnitudes),
                'min': np.min(magnitudes),
                'max': np.max(magnitudes),
                'q25': np.percentile(magnitudes, 25),
                'q75': np.percentile(magnitudes, 75)
            },
            'duration_stats': {
                'mean': np.mean(durations),
                'median': np.median(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            }
        }
        
        # Magnitude distribution
        analysis['magnitude_distribution'] = {
            'minor_4_5pct': sum(1 for m in magnitudes if 0.04 <= m < 0.05),
            'moderate_5_8pct': sum(1 for m in magnitudes if 0.05 <= m < 0.08),
            'major_8_12pct': sum(1 for m in magnitudes if 0.08 <= m < 0.12),
            'severe_12pct_plus': sum(1 for m in magnitudes if m >= 0.12)
        }
        
        # Seasonal patterns
        peak_dates = [pd.to_datetime(c['peak_date']) for c in self.correction_events]
        peak_months = [d.month for d in peak_dates]
        peak_days_of_week = [d.dayofweek for d in peak_dates]
        
        analysis['seasonal_patterns'] = {
            'monthly_distribution': {month: peak_months.count(month) for month in range(1, 13)},
            'day_of_week_distribution': {dow: peak_days_of_week.count(dow) for dow in range(7)},
            'most_common_month': max(set(peak_months), key=peak_months.count),
            'most_common_dow': max(set(peak_days_of_week), key=peak_days_of_week.count)
        }
        
        print(f"📊 Correction pattern analysis complete:")
        print(f"   📈 Total corrections: {analysis['basic_stats']['total_corrections']}")
        print(f"   📉 Avg magnitude: {analysis['basic_stats']['magnitude_stats']['mean']:.1%}")
        print(f"   ⏱️  Avg duration: {analysis['basic_stats']['duration_stats']['mean']:.1f} days")
        print(f"   📅 Most common month: {analysis['seasonal_patterns']['most_common_month']}")
        
        return analysis
        
    def identify_major_events(self) -> List[Dict]:
        """Identify major correction events (>8%) for special analysis"""
        major_threshold = 0.08
        major_events = []
        
        for correction in self.correction_events:
            if correction['magnitude'] >= major_threshold:
                major_event = correction.copy()
                
                # Add severity classification
                if correction['magnitude'] >= 0.20:
                    major_event['severity'] = 'CRASH'
                elif correction['magnitude'] >= 0.15:
                    major_event['severity'] = 'SEVERE'
                elif correction['magnitude'] >= 0.12:
                    major_event['severity'] = 'MAJOR'
                else:
                    major_event['severity'] = 'SIGNIFICANT'
                
                major_events.append(major_event)
        
        # Sort by magnitude (largest first)
        major_events.sort(key=lambda x: x['magnitude'], reverse=True)
        
        print(f"🚨 Identified {len(major_events)} major correction events (≥{major_threshold*100:.0f}%)")
        
        return major_events

class CorrectionPlotter:
    """Visualize correction data"""
    
    def plot_corrections(self, price_data: pd.DataFrame, correction_events: List[Dict]):
        """Visualize identified corrections for validation"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import timedelta
            
            if price_data is None or price_data.empty:
                print("❌ No price data to plot")
                return
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot price data
            ax.plot(price_data['date'], price_data['underlying_price'], 
                   'b-', linewidth=1, alpha=0.7, label='SPY Price')
            
            # Highlight correction periods
            colors = ['red', 'orange', 'purple', 'brown', 'pink']
            for i, correction in enumerate(correction_events):
                color = colors[i % len(colors)]
                
                # Highlight the correction period
                peak_date = correction['peak_date']
                trough_date = correction['trough_date']
                
                # Find indices for the correction period
                peak_mask = price_data['date'] == peak_date
                trough_mask = price_data['date'] == trough_date
                
                if peak_mask.any() and trough_mask.any():
                    peak_idx = price_data[peak_mask].index[0]
                    trough_idx = price_data[trough_mask].index[0]
                    
                    # Plot correction period
                    correction_dates = price_data.loc[peak_idx:trough_idx, 'date']
                    correction_prices = price_data.loc[peak_idx:trough_idx, 'underlying_price']
                    
                    ax.plot(correction_dates, correction_prices, 
                           color=color, linewidth=3, alpha=0.8,
                           label=f"Correction {i+1}: {correction['magnitude']:.1%}")
                    
                    # Mark peak and trough
                    ax.scatter([peak_date], [correction['peak_price']], 
                             color=color, s=100, marker='^', zorder=5)
                    ax.scatter([trough_date], [correction['trough_price']], 
                             color=color, s=100, marker='v', zorder=5)
                    
                    # Add prediction windows (1-3 days before)
                    for days_before in range(1, 4):
                        pred_date = peak_date - timedelta(days=days_before)
                        pred_mask = price_data['date'] == pred_date
                        if pred_mask.any():
                            pred_idx = price_data[pred_mask].index[0]
                            pred_price = price_data.loc[pred_idx, 'underlying_price']
                            ax.scatter([pred_date], [pred_price], 
                                     color=color, s=50, marker='o', 
                                     alpha=0.6, zorder=4)
            
            # Formatting
            ax.set_title('SPY Corrections and Prediction Windows', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('SPY Price ($)', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            output_path = Path('analysis/outputs/correction_analysis.png')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved correction plot to {output_path}")
            
            plt.show()
            
        except ImportError:
            print("❌ Matplotlib not available for plotting")
        except Exception as e:
            print(f"❌ Error creating plot: {e}")