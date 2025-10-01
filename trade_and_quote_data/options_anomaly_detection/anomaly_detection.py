"""
SPY Options Anomaly Detection - Core Detection Models
====================================================

This module implements various anomaly detection algorithms:
- Statistical methods (Z-score, percentile-based)
- Machine learning models (Isolation Forest, One-Class SVM)
- Ensemble methods for robust detection
- Signal generation and scoring

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy import stats
from hedging_signal_detector import HedgingSignalDetector
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsAnomalyDetector:
    """
    Multi-model anomaly detection system for SPY options
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []
        self.is_fitted = False
        
        # Add hedging signal detector
        self.hedging_detector = HedgingSignalDetector(str(data_dir))
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection
        """
        if df is None or len(df) == 0:
            return np.array([])
            
        # Select relevant features for anomaly detection
        feature_cols = [
            'oi_proxy', 'volume', 'transactions', 'moneyness', 'dte',
            'oi_percentile', 'volume_percentile', 'pc_oi_ratio',
            'oi_concentration', 'volume_concentration', 'turnover_rate',
            'tx_efficiency', 'oi_skew', 'activity_score',
            'otm_atm_call_ratio', 'institutional_momentum_score', 'call_skew_momentum'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) == 0:
            logger.warning("No relevant features found")
            return np.array([])
            
        # Extract features
        features = df[available_cols].fillna(0).values
        
        # Store feature columns for later use
        self.feature_columns = available_cols
        
        return features
    
    def fit_models(self, features: np.ndarray, contamination: float = 0.1):
        """
        Fit multiple anomaly detection models
        """
        if len(features) == 0:
            logger.error("No features to fit models")
            return
            
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # 1. Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.models['isolation_forest'].fit(features_scaled)
        
        # 2. One-Class SVM
        self.models['one_class_svm'] = OneClassSVM(
            nu=contamination,
            kernel='rbf',
            gamma='scale'
        )
        self.models['one_class_svm'].fit(features_scaled)
        
        # 3. DBSCAN for clustering-based detection
        self.models['dbscan'] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        self.models['dbscan'].fit(features_scaled)
        
        self.is_fitted = True
        logger.info("All models fitted successfully")
    
    def detect_anomalies(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using all fitted models
        """
        if not self.is_fitted:
            logger.error("Models not fitted yet")
            return {}
            
        if len(features) == 0:
            return {}
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        results = {}
        
        # Isolation Forest
        if_anomalies = self.models['isolation_forest'].predict(features_scaled)
        results['isolation_forest'] = (if_anomalies == -1).astype(int)
        
        # One-Class SVM
        svm_anomalies = self.models['one_class_svm'].predict(features_scaled)
        results['one_class_svm'] = (svm_anomalies == -1).astype(int)
        
        # DBSCAN
        dbscan_labels = self.models['dbscan'].fit_predict(features_scaled)
        results['dbscan'] = (dbscan_labels == -1).astype(int)
        
        # Statistical methods
        results['zscore'] = self._zscore_detection(features_scaled)
        results['iqr'] = self._iqr_detection(features_scaled)
        
        return results
    
    def _zscore_detection(self, features: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Z-score based anomaly detection
        """
        z_scores = np.abs(stats.zscore(features, axis=0))
        max_z_scores = np.max(z_scores, axis=1)
        return (max_z_scores > threshold).astype(int)
    
    def _iqr_detection(self, features: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """
        Interquartile Range based anomaly detection
        """
        Q1 = np.percentile(features, 25, axis=0)
        Q3 = np.percentile(features, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        anomalies = np.any((features < lower_bound) | (features > upper_bound), axis=1)
        return anomalies.astype(int)
    
    def ensemble_detection(self, features: np.ndarray, weights: Optional[Dict] = None) -> Dict:
        """
        Ensemble anomaly detection combining multiple methods
        """
        if not self.is_fitted:
            logger.error("Models not fitted yet")
            return {}
            
        # Get individual model results
        individual_results = self.detect_anomalies(features)
        
        if not individual_results:
            return {}
            
        # Default weights
        if weights is None:
            weights = {
                'isolation_forest': 0.3,
                'one_class_svm': 0.2,
                'dbscan': 0.2,
                'zscore': 0.15,
                'iqr': 0.15
            }
        
        # Calculate ensemble scores
        ensemble_scores = np.zeros(len(features))
        for model_name, anomalies in individual_results.items():
            if model_name in weights:
                ensemble_scores += weights[model_name] * anomalies
        
        # Create ensemble results
        ensemble_results = {
            'ensemble_score': ensemble_scores,
            'ensemble_anomaly': (ensemble_scores > 0.5).astype(int),
            'high_confidence': (ensemble_scores > 0.8).astype(int),
            'individual_results': individual_results
        }
        
        return ensemble_results
    
    def calculate_anomaly_metrics(self, df: pd.DataFrame, anomaly_results: Dict) -> Dict:
        """
        Calculate comprehensive anomaly metrics
        """
        if df is None or len(df) == 0:
            return {}
            
        metrics = {}
        
        # Basic counts
        total_contracts = len(df)
        metrics['total_contracts'] = total_contracts
        
        # Anomaly counts by method
        for method, anomalies in anomaly_results.get('individual_results', {}).items():
            anomaly_count = anomalies.sum()
            metrics[f'{method}_anomalies'] = anomaly_count
            metrics[f'{method}_anomaly_rate'] = anomaly_count / total_contracts if total_contracts > 0 else 0
        
        # Ensemble metrics
        if 'ensemble_anomaly' in anomaly_results:
            ensemble_anomalies = anomaly_results['ensemble_anomaly'].sum()
            metrics['ensemble_anomalies'] = ensemble_anomalies
            metrics['ensemble_anomaly_rate'] = ensemble_anomalies / total_contracts if total_contracts > 0 else 0
            
        if 'high_confidence' in anomaly_results:
            high_conf_anomalies = anomaly_results['high_confidence'].sum()
            metrics['high_confidence_anomalies'] = high_conf_anomalies
            metrics['high_confidence_rate'] = high_conf_anomalies / total_contracts if total_contracts > 0 else 0
        
        # Anomaly characteristics
        if 'ensemble_anomaly' in anomaly_results:
            anomaly_mask = anomaly_results['ensemble_anomaly'].astype(bool)
            if anomaly_mask.any():
                anomaly_df = df[anomaly_mask]
                
                # OI characteristics of anomalies
                metrics['anomaly_avg_oi'] = anomaly_df['oi_proxy'].mean() if 'oi_proxy' in anomaly_df.columns else 0
                metrics['anomaly_avg_volume'] = anomaly_df['volume'].mean() if 'volume' in anomaly_df.columns else 0
                metrics['anomaly_avg_moneyness'] = anomaly_df['moneyness'].mean() if 'moneyness' in anomaly_df.columns else 0
                
                # Put/Call distribution of anomalies
                if 'option_type' in anomaly_df.columns:
                    anomaly_calls = len(anomaly_df[anomaly_df['option_type'] == 'C'])
                    anomaly_puts = len(anomaly_df[anomaly_df['option_type'] == 'P'])
                    metrics['anomaly_pc_ratio'] = anomaly_puts / (anomaly_calls + 1e-6)
                else:
                    metrics['anomaly_pc_ratio'] = 1.0
        
        return metrics
    
    def generate_signals(self, df: pd.DataFrame, anomaly_results: Dict, 
                        date: str = None, threshold: float = 0.5) -> Dict:
        """
        Generate trading signals based on hedging intelligence and anomaly detection
        """
        if df is None or len(df) == 0:
            return {}
            
        signals = {}
        
        # 1. Get hedging signals (primary signal source)
        hedging_signals = {}
        if date:
            try:
                hedging_signals = self.hedging_detector.process_daily_hedging_signals(date)
            except Exception as e:
                logger.warning(f"Error getting hedging signals for {date}: {e}")
                hedging_signals = {}
        
        # 2. If we have hedging signals, use them as primary
        if 'signal' in hedging_signals and hedging_signals['signal'] != 'insufficient_data':
            # Map hedging signals to trading signals
            hedging_signal = hedging_signals['signal']
            
            if hedging_signal == 'strong_correction_warning':
                signals['direction'] = 'bearish'
                signals['strength'] = 0.9
                signals['confidence'] = hedging_signals.get('confidence', 0.8)
                signals['quality'] = 'high'
                signals['signal_source'] = 'hedging_intelligence'
                
            elif hedging_signal == 'moderate_correction_risk':
                signals['direction'] = 'bearish'
                signals['strength'] = 0.7
                signals['confidence'] = hedging_signals.get('confidence', 0.6)
                signals['quality'] = 'medium'
                signals['signal_source'] = 'hedging_intelligence'
                
            elif hedging_signal == 'elevated_hedging':
                signals['direction'] = 'bearish'
                signals['strength'] = 0.5
                signals['confidence'] = hedging_signals.get('confidence', 0.4)
                signals['quality'] = 'low'
                signals['signal_source'] = 'hedging_intelligence'
                
            else:  # normal_hedging
                signals['direction'] = 'neutral'
                signals['strength'] = 0.2
                signals['confidence'] = hedging_signals.get('confidence', 0.2)
                signals['quality'] = 'low'
                signals['signal_source'] = 'hedging_intelligence'
            
            # Add hedging-specific metadata
            signals['hedging_trend'] = hedging_signals.get('hedging_trend', 0)
            signals['hedging_acceleration'] = hedging_signals.get('hedging_acceleration', 0)
            signals['institutional_momentum'] = hedging_signals.get('institutional_momentum', 0)
            
        else:
            # 3. Fallback to anomaly-based signals if no hedging data
            if 'ensemble_anomaly' in anomaly_results:
                anomaly_mask = anomaly_results['ensemble_anomaly'].astype(bool)
                
                if anomaly_mask.any():
                    anomaly_df = df[anomaly_mask]
                    
                    # Signal strength based on anomaly characteristics
                    signal_strength = anomaly_results.get('ensemble_score', np.zeros(len(df)))
                    anomaly_scores = signal_strength[anomaly_mask]
                    
                    # Direction signals based on Put/Call ratio
                    if 'option_type' in anomaly_df.columns:
                        calls = anomaly_df[anomaly_df['option_type'] == 'C']
                        puts = anomaly_df[anomaly_df['option_type'] == 'P']
                        
                        if len(calls) > 0 and len(puts) > 0:
                            call_strength = calls['oi_proxy'].sum() if 'oi_proxy' in calls.columns else 0
                            put_strength = puts['oi_proxy'].sum() if 'oi_proxy' in puts.columns else 0
                            
                            # Bullish signal if calls dominate, bearish if puts dominate
                            signals['direction'] = 'bullish' if call_strength > put_strength else 'bearish'
                            signals['strength'] = abs(call_strength - put_strength) / (call_strength + put_strength + 1e-6)
                        else:
                            signals['direction'] = 'neutral'
                            signals['strength'] = 0.0
                    else:
                        signals['direction'] = 'neutral'
                        signals['strength'] = 0.0
                    
                    # Confidence level (reduced for anomaly-based signals)
                    signals['confidence'] = np.mean(anomaly_scores) * 0.7 if len(anomaly_scores) > 0 else 0.0
                    
                    # Signal quality (reduced for fallback)
                    signals['quality'] = 'medium' if signals['confidence'] > 0.6 else 'low'
                    signals['signal_source'] = 'anomaly_fallback'
                    
                else:
                    signals['direction'] = 'neutral'
                    signals['strength'] = 0.0
                    signals['confidence'] = 0.0
                    signals['quality'] = 'low'
                    signals['signal_source'] = 'anomaly_fallback'
            else:
                signals['direction'] = 'neutral'
                signals['strength'] = 0.0
                signals['confidence'] = 0.0
                signals['quality'] = 'low'
                signals['signal_source'] = 'no_data'
        
        return signals
    
    def process_daily_anomalies(self, date: str) -> Dict:
        """
        Process anomalies for a single date
        """
        from feature_engineering import OptionsFeatureEngine
        
        # Load and process features
        fe = OptionsFeatureEngine(str(self.data_dir))
        df, _ = fe.process_date(date)
        
        if df is None or len(df) == 0:
            return {}
        
        # Prepare features
        features = self.prepare_features(df)
        
        if len(features) == 0:
            return {}
        
        # Detect anomalies
        anomaly_results = self.ensemble_detection(features)
        
        # Calculate metrics
        metrics = self.calculate_anomaly_metrics(df, anomaly_results)
        
        # Generate signals (pass date for hedging analysis)
        signals = self.generate_signals(df, anomaly_results, date)
        
        return {
            'date': date,
            'metrics': metrics,
            'signals': signals,
            'anomaly_results': anomaly_results
        }
    
    def train_on_historical_data(self, start_date: str, end_date: str, 
                                contamination: float = 0.1):
        """
        Train models on historical data
        """
        logger.info(f"Training on historical data from {start_date} to {end_date}")
        
        # Load historical data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_features = []
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_date.weekday() < 5:
                df = self._load_daily_data(date_str)
                if df is not None and len(df) > 0:
                    features = self.prepare_features(df)
                    if len(features) > 0:
                        all_features.append(features)
            
            current_date += timedelta(days=1)
        
        if len(all_features) == 0:
            logger.error("No historical data found for training")
            return
        
        # Combine all features
        combined_features = np.vstack(all_features)
        logger.info(f"Training on {len(combined_features)} samples")
        
        # Fit models
        self.fit_models(combined_features, contamination)
        
        logger.info("Training completed successfully")
    
    def _load_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load daily data for training"""
        try:
            year = date[:4]
            month = date[5:7]
            
            file_path = self.data_dir / year / month / f"SPY_options_snapshot_{date}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(date)
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {date}: {e}")
            return None


def main():
    """
    Example usage of the anomaly detection system
    """
    # Initialize detector
    detector = OptionsAnomalyDetector()
    
    # Train on historical data
    detector.train_on_historical_data("2024-01-01", "2024-12-31")
    
    # Process a specific date
    date = "2025-01-31"
    results = detector.process_daily_anomalies(date)
    
    if results:
        print(f"Anomaly detection results for {date}:")
        print(f"Metrics: {results['metrics']}")
        print(f"Signals: {results['signals']}")


if __name__ == "__main__":
    main()
