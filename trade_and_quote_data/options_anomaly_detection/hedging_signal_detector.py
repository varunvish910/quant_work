"""
SPY Options Hedging Signal Detector
==================================

This module implements hedging intelligence analysis to detect institutional
protective positioning that precedes market corrections. Unlike anomaly detection,
this system recognizes hedging as predictive market intelligence.

Key Features:
- Institutional vs Speculative hedging classification
- Temporal hedging buildup analysis (2-4 week windows)
- Support level concentration analysis
- Hedging momentum and acceleration detection
- Pre-correction warning signals

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HedgingSignalDetector:
    """
    Detects institutional hedging patterns that precede market corrections
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.hedging_history = {}
        self.support_levels_cache = {}
        
    def load_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load options data for a specific date"""
        try:
            year = date[:4]
            month = date[5:7]
            
            file_path = self.data_dir / year / month / f"SPY_options_snapshot_{date.replace('-', '')}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(date)
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {date}: {e}")
            return None
    
    def classify_hedging_type(self, puts_df: pd.DataFrame, spy_price: float) -> Dict:
        """
        Classify put activity as institutional vs speculative hedging
        """
        if puts_df.empty:
            return {'institutional_score': 0.0, 'speculative_score': 0.0}
        
        # Institutional hedging characteristics
        institutional_indicators = []
        
        # 1. Time to expiration (institutions use longer timeframes)
        dte_score = np.where(puts_df['dte'] >= 21, 1.0, 
                           np.where(puts_df['dte'] >= 14, 0.5, 0.1))
        institutional_indicators.append(dte_score)
        
        # 2. Moneyness (institutions buy deeper OTM for protection)
        moneyness_score = np.where(puts_df['moneyness'] <= 0.85, 1.0,
                                 np.where(puts_df['moneyness'] <= 0.95, 0.6, 0.2))
        institutional_indicators.append(moneyness_score)
        
        # 3. Block size (institutions trade larger blocks)
        avg_tx_size = puts_df['volume'] / (puts_df['transactions'] + 1e-6)
        block_size_score = np.where(avg_tx_size >= 50, 1.0,
                                  np.where(avg_tx_size >= 20, 0.7, 0.3))
        institutional_indicators.append(block_size_score)
        
        # 4. Volume/OI ratio (institutions hold, speculators flip)
        vol_oi_ratio = puts_df['volume'] / (puts_df['oi_proxy'] + 1e-6)
        holding_score = np.where(vol_oi_ratio <= 0.3, 1.0,
                               np.where(vol_oi_ratio <= 0.7, 0.6, 0.2))
        institutional_indicators.append(holding_score)
        
        # 5. Support level concentration
        support_levels = [spy_price * mult for mult in [0.95, 0.90, 0.85, 0.80]]
        support_score = np.zeros(len(puts_df))
        
        for level in support_levels:
            near_support = np.abs(puts_df['strike'] - level) <= 2.5
            support_score += near_support * 0.8
        
        support_score = np.clip(support_score, 0, 1)
        institutional_indicators.append(support_score)
        
        # Calculate weighted institutional score
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # Weights for each indicator
        institutional_score = np.average(institutional_indicators, weights=weights, axis=0)
        
        # Apply to OI values for weighted totals
        institutional_oi = (puts_df['oi_proxy'] * institutional_score).sum()
        speculative_oi = (puts_df['oi_proxy'] * (1 - institutional_score)).sum()
        
        total_oi = institutional_oi + speculative_oi
        
        return {
            'institutional_score': institutional_oi / (total_oi + 1e-6),
            'speculative_score': speculative_oi / (total_oi + 1e-6),
            'institutional_oi': institutional_oi,
            'speculative_oi': speculative_oi,
            'total_put_oi': total_oi,
            'institutional_contracts': np.sum(institutional_score > 0.7),
            'speculative_contracts': np.sum(institutional_score <= 0.7)
        }
    
    def analyze_temporal_hedging(self, date: str, lookback_days: int = 20) -> Dict:
        """
        Analyze hedging buildup over multiple weeks leading to the given date
        """
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=lookback_days)
        
        hedging_timeline = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if df is not None and len(df) > 0:
                    spy_price = df['underlying_price'].iloc[0]
                    puts = df[df['option_type'] == 'P']
                    
                    if not puts.empty:
                        hedging_class = self.classify_hedging_type(puts, spy_price)
                        
                        hedging_timeline.append({
                            'date': current_date,
                            'spy_price': spy_price,
                            'institutional_oi': hedging_class['institutional_oi'],
                            'speculative_oi': hedging_class['speculative_oi'],
                            'institutional_score': hedging_class['institutional_score'],
                            'total_put_oi': hedging_class['total_put_oi']
                        })
            
            current_date += timedelta(days=1)
        
        if len(hedging_timeline) < 5:
            return {'error': 'Insufficient data for temporal analysis'}
        
        # Convert to DataFrame for analysis
        timeline_df = pd.DataFrame(hedging_timeline)
        
        # Calculate temporal patterns
        analysis = self._analyze_hedging_momentum(timeline_df)
        
        return analysis
    
    def _analyze_hedging_momentum(self, timeline_df: pd.DataFrame) -> Dict:
        """
        Analyze momentum and acceleration in hedging activity
        """
        analysis = {}
        
        # 1. Hedging trend analysis
        inst_oi = timeline_df['institutional_oi'].values
        dates_numeric = np.arange(len(inst_oi))
        
        if len(inst_oi) >= 3:
            # Linear trend
            slope, intercept, r_value, _, _ = stats.linregress(dates_numeric, inst_oi)
            analysis['hedging_trend'] = slope
            analysis['hedging_r_squared'] = r_value ** 2
            
            # Acceleration (second derivative)
            if len(inst_oi) >= 5:
                first_diff = np.diff(inst_oi)
                second_diff = np.diff(first_diff)
                analysis['hedging_acceleration'] = np.mean(second_diff[-3:])  # Recent acceleration
            else:
                analysis['hedging_acceleration'] = 0.0
        else:
            analysis['hedging_trend'] = 0.0
            analysis['hedging_r_squared'] = 0.0
            analysis['hedging_acceleration'] = 0.0
        
        # 2. Recent vs historical comparison
        recent_period = timeline_df.tail(5)['institutional_oi'].mean()
        earlier_period = timeline_df.head(5)['institutional_oi'].mean()
        
        analysis['recent_vs_earlier_ratio'] = recent_period / (earlier_period + 1e-6)
        
        # 3. Institutional vs Speculative momentum
        inst_scores = timeline_df['institutional_score'].values
        if len(inst_scores) >= 3:
            inst_slope, _, inst_r, _, _ = stats.linregress(dates_numeric, inst_scores)
            analysis['institutional_momentum'] = inst_slope
            analysis['institutional_consistency'] = inst_r ** 2
        else:
            analysis['institutional_momentum'] = 0.0
            analysis['institutional_consistency'] = 0.0
        
        # 4. Volatility of hedging activity
        analysis['hedging_volatility'] = timeline_df['institutional_oi'].std()
        analysis['hedging_cv'] = analysis['hedging_volatility'] / (timeline_df['institutional_oi'].mean() + 1e-6)
        
        # 5. Current levels
        analysis['current_institutional_oi'] = timeline_df['institutional_oi'].iloc[-1]
        analysis['current_institutional_score'] = timeline_df['institutional_score'].iloc[-1]
        analysis['current_spy_price'] = timeline_df['spy_price'].iloc[-1]
        
        return analysis
    
    def generate_hedging_signals(self, temporal_analysis: Dict, 
                                current_day_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate trading signals based on hedging intelligence
        """
        if 'error' in temporal_analysis:
            return {'signal': 'insufficient_data', 'confidence': 0.0}
        
        signals = {}
        
        # 1. Hedging buildup signal
        hedging_trend = temporal_analysis.get('hedging_trend', 0)
        hedging_acceleration = temporal_analysis.get('hedging_acceleration', 0)
        recent_ratio = temporal_analysis.get('recent_vs_earlier_ratio', 1)
        
        # Strong hedging buildup indicates correction risk
        buildup_score = 0.0
        
        if hedging_trend > 0:  # Increasing hedging
            buildup_score += 0.3
        if hedging_acceleration > 0:  # Accelerating hedging
            buildup_score += 0.4
        if recent_ratio > 1.5:  # Recent surge vs earlier period
            buildup_score += 0.3
        
        # 2. Institutional conviction signal
        inst_momentum = temporal_analysis.get('institutional_momentum', 0)
        inst_consistency = temporal_analysis.get('institutional_consistency', 0)
        
        conviction_score = 0.0
        if inst_momentum > 0:  # Increasing institutional activity
            conviction_score += 0.4
        if inst_consistency > 0.7:  # Consistent pattern
            conviction_score += 0.6
        
        # 3. Combined signal strength
        combined_score = (buildup_score + conviction_score) / 2
        
        # 4. Signal interpretation
        if combined_score >= 0.8:
            signals['signal'] = 'strong_correction_warning'
            signals['confidence'] = combined_score
            signals['quality'] = 'high'
        elif combined_score >= 0.6:
            signals['signal'] = 'moderate_correction_risk'
            signals['confidence'] = combined_score
            signals['quality'] = 'medium'
        elif combined_score >= 0.4:
            signals['signal'] = 'elevated_hedging'
            signals['confidence'] = combined_score
            signals['quality'] = 'low'
        else:
            signals['signal'] = 'normal_hedging'
            signals['confidence'] = combined_score
            signals['quality'] = 'low'
        
        # 5. Additional signal metadata
        signals['hedging_trend'] = hedging_trend
        signals['hedging_acceleration'] = hedging_acceleration
        signals['institutional_momentum'] = inst_momentum
        signals['recent_vs_earlier_ratio'] = recent_ratio
        signals['buildup_score'] = buildup_score
        signals['conviction_score'] = conviction_score
        
        return signals
    
    def process_daily_hedging_signals(self, date: str, lookback_days: int = 20) -> Dict:
        """
        Process hedging signals for a single date with temporal context
        """
        logger.info(f"Processing hedging signals for {date}")
        
        # Load current day data
        current_df = self.load_daily_data(date)
        if current_df is None or len(current_df) == 0:
            return {'error': f'No data available for {date}'}
        
        # Analyze temporal hedging patterns
        temporal_analysis = self.analyze_temporal_hedging(date, lookback_days)
        
        # Generate signals
        signals = self.generate_hedging_signals(temporal_analysis, current_df)
        
        # Add current day metrics
        spy_price = current_df['underlying_price'].iloc[0]
        puts = current_df[current_df['option_type'] == 'P']
        
        if not puts.empty:
            current_hedging = self.classify_hedging_type(puts, spy_price)
            signals['current_day_metrics'] = current_hedging
        
        signals['date'] = date
        signals['spy_price'] = spy_price
        signals['temporal_analysis'] = temporal_analysis
        
        return signals
    
    def train_on_historical_hedging(self, start_date: str, end_date: str) -> Dict:
        """
        Analyze historical hedging patterns to calibrate signal thresholds
        """
        logger.info(f"Training hedging detector on {start_date} to {end_date}")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        training_data = []
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Process hedging signals for each day
                signals = self.process_daily_hedging_signals(date_str)
                
                if 'error' not in signals:
                    training_data.append(signals)
                    
                    if len(training_data) % 50 == 0:
                        logger.info(f"Processed {len(training_data)} days of hedging data")
            
            current_date += timedelta(days=1)
        
        if not training_data:
            return {'error': 'No training data found'}
        
        # Analyze historical patterns
        training_df = pd.DataFrame([
            {
                'date': d['date'],
                'signal': d['signal'],
                'confidence': d['confidence'],
                'hedging_trend': d.get('hedging_trend', 0),
                'hedging_acceleration': d.get('hedging_acceleration', 0),
                'institutional_momentum': d.get('institutional_momentum', 0)
            }
            for d in training_data
        ])
        
        # Calculate calibration metrics
        calibration = {
            'total_days': len(training_df),
            'signal_distribution': training_df['signal'].value_counts().to_dict(),
            'avg_confidence': training_df['confidence'].mean(),
            'high_confidence_days': len(training_df[training_df['confidence'] >= 0.8]),
            'hedging_trend_stats': training_df['hedging_trend'].describe().to_dict()
        }
        
        logger.info(f"Training complete: {calibration['total_days']} days processed")
        
        return {
            'calibration': calibration,
            'training_data': training_df
        }


def main():
    """
    Example usage of the hedging signal detector
    """
    # Initialize detector
    detector = HedgingSignalDetector()
    
    # Train on historical data
    training_results = detector.train_on_historical_hedging("2024-01-01", "2024-12-31")
    
    if 'error' not in training_results:
        print("Training completed successfully")
        print(f"Calibration: {training_results['calibration']}")
    
    # Process a specific date
    date = "2024-01-31"
    signals = detector.process_daily_hedging_signals(date)
    
    if 'error' not in signals:
        print(f"\nHedging signals for {date}:")
        print(f"Signal: {signals['signal']}")
        print(f"Confidence: {signals['confidence']:.3f}")
        print(f"Quality: {signals['quality']}")


if __name__ == "__main__":
    main()