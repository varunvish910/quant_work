#!/usr/bin/env python3
"""
Create target labels for 4%+ correction prediction
Identifies correction events and labels days 1-3 before as prediction targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import yfinance as yf

class CorrectionTargetCreator:
    """
    Creates binary targets for predicting 4%+ corrections 1-3 days in advance
    """
    
    def __init__(self, correction_threshold: float = 0.04, lookback_days: int = 20):
        """
        Args:
            correction_threshold: Minimum drop % to qualify as correction (default 4%)
            lookback_days: Days to look back for peak before correction
        """
        self.correction_threshold = correction_threshold
        self.lookback_days = lookback_days
        self.correction_events = []
        self.price_data = None
        
    def load_price_data(self, start_date: str, end_date: str, ticker: str = "SPY") -> pd.DataFrame:
        """Load historical price data for target creation from options data"""
        try:
            # Load from options summary files which contain underlying_price
            data_files = []
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Find all summary files in the date range
            for year in range(start_dt.year, end_dt.year + 1):
                year_dir = Path(f"data")
                for file_path in year_dir.glob(f"SPY_summary_{year}*.csv"):
                    data_files.append(file_path)
            
            if not data_files:
                raise FileNotFoundError("No SPY summary files found in data directory")
            
            # Load and combine all data
            price_data = []
            for file_path in sorted(data_files):
                try:
                    df = pd.read_csv(file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Filter by date range
                    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                    
                    if len(df) > 0:
                        price_data.append(df[['date', 'underlying_price']])
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
                    continue
            
            if not price_data:
                raise ValueError("No price data found in the specified date range")
            
            # Combine all data
            combined_df = pd.concat(price_data, ignore_index=True)
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            combined_df = combined_df.drop_duplicates(subset=['date']).reset_index(drop=True)
            
            print(f"✅ Loaded {len(combined_df)} days of price data from {combined_df['date'].min()} to {combined_df['date'].max()}")
            return combined_df
            
        except Exception as e:
            print(f"❌ Error loading price data: {e}")
            # Fallback to yfinance
            try:
                import yfinance as yf
                print("🔄 Falling back to yfinance...")
                spy = yf.Ticker("SPY")
                hist = spy.history(start=start_date, end=end_date)
                if hist.empty:
                    raise ValueError("No data from yfinance")
                
                price_data = pd.DataFrame({
                    'date': hist.index,
                    'underlying_price': hist['Close']
                }).reset_index(drop=True)
                
                print(f"✅ Loaded {len(price_data)} days from yfinance")
                return price_data
            except Exception as e2:
                raise Exception(f"Failed to load price data from both sources: {e}, {e2}")
        
    def identify_corrections(self, price_data: pd.DataFrame) -> List[Dict]:
        """
        Identify all 4%+ correction events in the price data
        
        Returns:
            List of correction events with start_date, peak_date, trough_date, magnitude
        """
        if price_data is None or len(price_data) < self.lookback_days:
            return []
        
        # Store price data for later use
        self.price_data = price_data
        
        corrections = []
        prices = price_data['underlying_price'].values
        dates = price_data['date'].values
        
        # Find local peaks using rolling maximum
        rolling_max = pd.Series(prices).rolling(window=self.lookback_days, min_periods=1).max()
        
        # Identify peak points (where current price equals rolling max)
        peak_mask = prices == rolling_max.values
        
        # Find corrections by looking for significant drawdowns from peaks
        for i in range(len(prices)):
            if not peak_mask[i]:
                continue
                
            peak_price = prices[i]
            peak_date = dates[i]
            
            # Look forward for the next significant trough
            for j in range(i + 1, len(prices)):
                current_price = prices[j]
                drawdown = (peak_price - current_price) / peak_price
                
                # Check if this is a significant correction
                if drawdown >= self.correction_threshold:
                    # Find the actual trough (lowest point in this correction)
                    trough_idx = j
                    trough_price = current_price
                    
                    # Continue looking for the actual bottom
                    for k in range(j + 1, len(prices)):
                        if prices[k] < trough_price:
                            trough_price = prices[k]
                            trough_idx = k
                        # Stop if we hit a new peak (recovery)
                        elif prices[k] >= peak_price * 0.98:  # 98% recovery threshold
                            break
                    
                    # Calculate final drawdown
                    final_drawdown = (peak_price - trough_price) / peak_price
                    
                    if final_drawdown >= self.correction_threshold:
                        correction = {
                            'peak_date': peak_date,
                            'trough_date': dates[trough_idx],
                            'peak_price': peak_price,
                            'trough_price': trough_price,
                            'magnitude': final_drawdown,
                            'duration_days': (pd.to_datetime(dates[trough_idx]) - pd.to_datetime(peak_date)).days,
                            'peak_idx': i,
                            'trough_idx': trough_idx
                        }
                        corrections.append(correction)
                        
                        # Skip ahead to avoid overlapping corrections
                        i = trough_idx
                    break
        
        # Remove overlapping corrections (keep larger ones)
        filtered_corrections = []
        for correction in corrections:
            is_overlapping = False
            for existing in filtered_corrections:
                # Check if this correction overlaps with existing ones
                if (correction['peak_idx'] <= existing['trough_idx'] and 
                    correction['trough_idx'] >= existing['peak_idx']):
                    # Keep the larger correction
                    if correction['magnitude'] > existing['magnitude']:
                        filtered_corrections.remove(existing)
                        filtered_corrections.append(correction)
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_corrections.append(correction)
        
        self.correction_events = filtered_corrections
        print(f"🔍 Found {len(filtered_corrections)} correction events (≥{self.correction_threshold*100:.1f}%)")
        
        return filtered_corrections
        
    def create_prediction_targets(self, correction_events: List[Dict]) -> pd.DataFrame:
        """
        Create binary target labels: 1 for days 1-3 before corrections, 0 otherwise
        
        Args:
            correction_events: List of correction events from identify_corrections()
            
        Returns:
            DataFrame with date, target, days_to_correction, correction_magnitude
        """
        if not correction_events:
            return pd.DataFrame(columns=['date', 'target', 'days_to_correction', 'correction_magnitude'])
        
        # Create date range from price data
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_price_data() first.")
        
        # Initialize targets dataframe
        targets_df = self.price_data[['date']].copy()
        targets_df['target'] = 0
        targets_df['days_to_correction'] = np.nan
        targets_df['correction_magnitude'] = np.nan
        targets_df['correction_peak_date'] = pd.NaT
        
        # Sort corrections by magnitude (largest first) to handle overlaps
        sorted_corrections = sorted(correction_events, key=lambda x: x['magnitude'], reverse=True)
        
        for correction in sorted_corrections:
            peak_date = correction['peak_date']
            magnitude = correction['magnitude']
            
            # Find the peak date in our dataframe
            peak_mask = targets_df['date'] == peak_date
            if not peak_mask.any():
                continue
                
            peak_idx = targets_df[peak_mask].index[0]
            
            # Mark days 1-3 before the correction as targets
            for days_before in range(1, 4):
                target_idx = peak_idx - days_before
                if target_idx >= 0:
                    # Only set target if not already set (prioritize larger corrections)
                    if targets_df.loc[target_idx, 'target'] == 0:
                        targets_df.loc[target_idx, 'target'] = 1
                        targets_df.loc[target_idx, 'days_to_correction'] = days_before
                        targets_df.loc[target_idx, 'correction_magnitude'] = magnitude
                        targets_df.loc[target_idx, 'correction_peak_date'] = peak_date
        
        # Add some additional features
        targets_df['is_target'] = targets_df['target'] == 1
        targets_df['correction_category'] = pd.cut(
            targets_df['correction_magnitude'], 
            bins=[0, 0.05, 0.08, 0.12, 1.0], 
            labels=['Minor (4-5%)', 'Moderate (5-8%)', 'Major (8-12%)', 'Severe (12%+)'],
            include_lowest=True
        )
        
        print(f"🎯 Created {targets_df['target'].sum()} prediction targets")
        print(f"📊 Target distribution: {targets_df['target'].value_counts().to_dict()}")
        
        return targets_df
        
    def validate_targets(self, targets_df: pd.DataFrame) -> Dict:
        """
        Validate target distribution and timing
        
        Returns:
            Dictionary with validation metrics
        """
        if targets_df.empty:
            return {"error": "No targets to validate"}
        
        validation = {}
        
        # Basic statistics
        total_days = len(targets_df)
        target_days = targets_df['target'].sum()
        non_target_days = total_days - target_days
        
        validation['total_days'] = total_days
        validation['target_days'] = int(target_days)
        validation['non_target_days'] = int(non_target_days)
        validation['target_ratio'] = target_days / total_days if total_days > 0 else 0
        
        # Target distribution by days before correction
        if target_days > 0:
            days_dist = targets_df[targets_df['target'] == 1]['days_to_correction'].value_counts().sort_index()
            validation['days_to_correction_distribution'] = days_dist.to_dict()
            
            # Average days before correction
            validation['avg_days_before_correction'] = targets_df[targets_df['target'] == 1]['days_to_correction'].mean()
            
            # Correction magnitude distribution
            magnitude_dist = targets_df[targets_df['target'] == 1]['correction_magnitude'].describe()
            validation['correction_magnitude_stats'] = magnitude_dist.to_dict()
            
            # Category distribution
            category_dist = targets_df[targets_df['target'] == 1]['correction_category'].value_counts()
            validation['correction_category_distribution'] = category_dist.to_dict()
        
        # Check for data leakage (targets too close together)
        target_indices = targets_df[targets_df['target'] == 1].index
        if len(target_indices) > 1:
            gaps = target_indices[1:] - target_indices[:-1]
            validation['min_gap_between_targets'] = int(gaps.min())
            validation['avg_gap_between_targets'] = float(gaps.mean())
            
            # Flag potential leakage (gaps < 3 days)
            leakage_count = (gaps < 3).sum()
            validation['potential_leakage_count'] = int(leakage_count)
            validation['has_potential_leakage'] = leakage_count > 0
        
        # Temporal distribution
        if target_days > 0:
            target_dates = targets_df[targets_df['target'] == 1]['date']
            validation['first_target_date'] = str(target_dates.min())
            validation['last_target_date'] = str(target_dates.max())
            
            # Monthly distribution
            target_dates_series = pd.to_datetime(target_dates)
            monthly_dist = target_dates_series.dt.month.value_counts().sort_index()
            validation['monthly_target_distribution'] = monthly_dist.to_dict()
            
            # Day of week distribution
            dow_dist = target_dates_series.dt.dayofweek.value_counts().sort_index()
            validation['day_of_week_distribution'] = dow_dist.to_dict()
        
        # Quality checks
        validation['quality_checks'] = {
            'has_sufficient_targets': target_days >= 10,
            'balanced_ratio': 0.01 <= validation['target_ratio'] <= 0.3,
            'no_data_leakage': not validation.get('has_potential_leakage', False),
            'reasonable_timing': validation.get('avg_days_before_correction', 0) >= 1.0
        }
        
        print(f"✅ Validation complete:")
        print(f"   📊 Target ratio: {validation['target_ratio']:.3f} ({target_days}/{total_days})")
        print(f"   ⏰ Avg days before correction: {validation.get('avg_days_before_correction', 0):.1f}")
        print(f"   🔍 Quality checks passed: {sum(validation['quality_checks'].values())}/{len(validation['quality_checks'])}")
        
        return validation
        
    def export_targets(self, targets_df: pd.DataFrame, output_path: str):
        """Export targets to file for model training"""
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set date as index for better time series handling
            export_df = targets_df.set_index('date')
            
            # Export as parquet (preferred for ML)
            if output_path.suffix == '.parquet' or output_path.suffix == '':
                parquet_path = output_path.with_suffix('.parquet')
                export_df.to_parquet(parquet_path)
                print(f"💾 Exported targets to {parquet_path}")
                
                # Also export as CSV for easy inspection
                csv_path = output_path.with_suffix('.csv')
                export_df.to_csv(csv_path)
                print(f"📄 Also saved as CSV: {csv_path}")
                
            else:
                export_df.to_csv(output_path)
                print(f"💾 Exported targets to {output_path}")
            
            # Export metadata
            metadata_path = output_path.with_suffix('.json')
            metadata = {
                'export_date': datetime.now().isoformat(),
                'total_records': len(export_df),
                'target_count': int(export_df['target'].sum()),
                'date_range': {
                    'start': str(export_df.index.min()),
                    'end': str(export_df.index.max())
                },
                'columns': list(export_df.columns),
                'target_distribution': export_df['target'].value_counts().to_dict()
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"📋 Exported metadata to {metadata_path}")
            
        except Exception as e:
            print(f"❌ Error exporting targets: {e}")
            raise
        
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

class CorrectionAnalyzer:
    """
    Analyze characteristics of historical corrections for insights
    """
    
    def __init__(self, correction_events: List[Dict]):
        self.correction_events = correction_events
        
    def analyze_correction_patterns(self) -> Dict:
        """
        Analyze patterns in correction timing, magnitude, duration
        
        Returns:
            Dictionary with correction statistics and patterns
        """
        if not self.correction_events:
            return {"error": "No correction events to analyze"}
        
        analysis = {}
        
        # Basic statistics
        magnitudes = [c['magnitude'] for c in self.correction_events]
        durations = [c['duration_days'] for c in self.correction_events]
        peak_prices = [c['peak_price'] for c in self.correction_events]
        trough_prices = [c['trough_price'] for c in self.correction_events]
        
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
        
        # Duration patterns
        analysis['duration_patterns'] = {
            'quick_corrections_1_3_days': sum(1 for d in durations if 1 <= d <= 3),
            'medium_corrections_4_10_days': sum(1 for d in durations if 4 <= d <= 10),
            'long_corrections_11_plus_days': sum(1 for d in durations if d >= 11),
            'avg_duration_by_magnitude': {}
        }
        
        # Duration by magnitude category
        for category, min_mag, max_mag in [('minor', 0.04, 0.05), ('moderate', 0.05, 0.08), 
                                         ('major', 0.08, 0.12), ('severe', 0.12, 1.0)]:
            cat_corrections = [c for c in self.correction_events if min_mag <= c['magnitude'] < max_mag]
            if cat_corrections:
                cat_durations = [c['duration_days'] for c in cat_corrections]
                analysis['duration_patterns']['avg_duration_by_magnitude'][category] = {
                    'count': len(cat_corrections),
                    'avg_duration': np.mean(cat_durations),
                    'median_duration': np.median(cat_durations)
                }
        
        # Recovery patterns (if we have subsequent data)
        analysis['recovery_patterns'] = self._analyze_recovery_patterns()
        
        # Market context analysis
        analysis['market_context'] = self._analyze_market_context()
        
        # Clustering analysis
        analysis['clustering'] = self._analyze_correction_clustering()
        
        print(f"📊 Correction pattern analysis complete:")
        print(f"   📈 Total corrections: {analysis['basic_stats']['total_corrections']}")
        print(f"   📉 Avg magnitude: {analysis['basic_stats']['magnitude_stats']['mean']:.1%}")
        print(f"   ⏱️  Avg duration: {analysis['basic_stats']['duration_stats']['mean']:.1f} days")
        print(f"   📅 Most common month: {analysis['seasonal_patterns']['most_common_month']}")
        
        return analysis
        
    def identify_major_events(self) -> List[Dict]:
        """
        Identify major correction events (>8%) for special analysis
        
        Returns:
            List of major correction events with additional context
        """
        major_threshold = 0.08  # 8%
        major_events = []
        
        for correction in self.correction_events:
            if correction['magnitude'] >= major_threshold:
                # Add additional context for major events
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
                
                # Add market context
                major_event['market_context'] = self._get_market_context_for_event(correction)
                
                # Add historical ranking
                major_event['historical_rank'] = self._get_historical_ranking(correction)
                
                # Add recovery analysis
                major_event['recovery_analysis'] = self._analyze_event_recovery(correction)
                
                major_events.append(major_event)
        
        # Sort by magnitude (largest first)
        major_events.sort(key=lambda x: x['magnitude'], reverse=True)
        
        print(f"🚨 Identified {len(major_events)} major correction events (≥{major_threshold*100:.0f}%)")
        
        return major_events
    
    def _analyze_recovery_patterns(self) -> Dict:
        """Analyze how quickly markets recover from corrections"""
        recovery_analysis = {
            'avg_recovery_time': 0,
            'recovery_success_rate': 0,
            'recovery_patterns': {}
        }
        
        # This would require additional price data after corrections
        # For now, return basic structure
        return recovery_analysis
    
    def _analyze_market_context(self) -> Dict:
        """Analyze market context around corrections"""
        context = {
            'volatility_clusters': 0,
            'trend_context': {},
            'macro_events': []
        }
        
        # This would require additional market data
        # For now, return basic structure
        return context
    
    def _analyze_correction_clustering(self) -> Dict:
        """Analyze if corrections tend to cluster in time"""
        if len(self.correction_events) < 2:
            return {'clustering_detected': False}
        
        # Calculate time gaps between corrections
        peak_dates = sorted([pd.to_datetime(c['peak_date']) for c in self.correction_events])
        gaps = [(peak_dates[i+1] - peak_dates[i]).days for i in range(len(peak_dates)-1)]
        
        avg_gap = np.mean(gaps)
        clustering_threshold = 30  # days
        
        return {
            'clustering_detected': avg_gap < clustering_threshold,
            'avg_gap_days': avg_gap,
            'min_gap_days': min(gaps),
            'max_gap_days': max(gaps)
        }
    
    def _get_market_context_for_event(self, correction: Dict) -> Dict:
        """Get market context for a specific correction event"""
        return {
            'pre_correction_trend': 'unknown',  # Would need additional data
            'volatility_regime': 'unknown',
            'macro_environment': 'unknown'
        }
    
    def _get_historical_ranking(self, correction: Dict) -> int:
        """Get historical ranking of correction magnitude"""
        magnitudes = [c['magnitude'] for c in self.correction_events]
        magnitudes.sort(reverse=True)
        
        try:
            return magnitudes.index(correction['magnitude']) + 1
        except ValueError:
            return len(magnitudes) + 1
    
    def _analyze_event_recovery(self, correction: Dict) -> Dict:
        """Analyze recovery characteristics for a specific event"""
        return {
            'recovery_time': 'unknown',  # Would need additional data
            'recovery_pattern': 'unknown',
            'v_shaped': False,
            'u_shaped': False,
            'l_shaped': False
        }

# Example usage and testing
if __name__ == "__main__":
    
    # Initialize target creator
    creator = CorrectionTargetCreator(
        correction_threshold=0.04,  # 4% corrections
        lookback_days=20
    )
    
    # Load price data
    print("📊 Loading historical price data...")
    price_data = creator.load_price_data("2020-01-01", "2025-10-01")
    
    # Identify corrections
    print("🔍 Identifying 4%+ correction events...")
    corrections = creator.identify_corrections(price_data)
    print(f"Found {len(corrections)} correction events")
    
    # Create prediction targets
    print("🎯 Creating prediction targets...")
    targets = creator.create_prediction_targets(corrections)
    
    # Validate targets
    print("✅ Validating target distribution...")
    validation = creator.validate_targets(targets)
    
    # Export for model training
    print("💾 Exporting targets...")
    creator.export_targets(targets, "data/correction_targets.parquet")
    
    # Analyze patterns
    print("📊 Analyzing correction patterns...")
    analyzer = CorrectionAnalyzer(corrections)
    patterns = analyzer.analyze_correction_patterns()
    
    # Identify major events
    print("🚨 Identifying major correction events...")
    major_events = analyzer.identify_major_events()
    
    if major_events:
        print(f"Major events found:")
        for i, event in enumerate(major_events[:3]):  # Show top 3
            print(f"  {i+1}. {event['peak_date']}: {event['magnitude']:.1%} ({event['severity']})")
    
    print("🎯 Target creation complete!")