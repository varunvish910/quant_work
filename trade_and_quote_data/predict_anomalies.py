#!/usr/bin/env python3
"""
SPY Options Anomaly Prediction System
====================================

This script uses the trained models to predict anomalies for new dates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import joblib
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyPredictor:
    """
    Predict anomalies using trained models
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data/options_chains/SPY"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.scaler = None
        self.models = {}
        self.feature_columns = []
        self.is_loaded = False
        
    def load_models(self):
        """Load trained models"""
        try:
            # Load scaler
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            
            # Load models
            model_files = {
                'random_forest': 'random_forest.pkl',
                'svm': 'svm.pkl',
                'isolation_forest': 'isolation_forest.pkl',
                'one_class_svm': 'one_class_svm.pkl'
            }
            
            for name, filename in model_files.items():
                self.models[name] = joblib.load(self.models_dir / filename)
            
            # Load feature columns
            with open(self.models_dir / "feature_columns.json", 'r') as f:
                self.feature_columns = json.load(f)
            
            self.is_loaded = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """Load options data for a specific date"""
        try:
            year = date[:4]
            month = date[5:7]
            date_formatted = date.replace('-', '')
            file_path = self.data_dir / year / month / f"SPY_options_snapshot_{date_formatted}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(date)
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {date}: {e}")
            return None
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for prediction (same as training)"""
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        
        # 1. OI Features
        features_df['oi_percentile'] = features_df['oi_proxy'].rank(pct=True)
        features_df['oi_zscore'] = stats.zscore(features_df['oi_proxy'])
        
        # Put/Call OI Ratios
        calls = features_df[features_df['option_type'] == 'C']
        puts = features_df[features_df['option_type'] == 'P']
        
        if len(calls) > 0 and len(puts) > 0:
            total_call_oi = calls['oi_proxy'].sum()
            total_put_oi = puts['oi_proxy'].sum()
            features_df['pc_oi_ratio'] = total_put_oi / (total_call_oi + 1e-6)
        else:
            features_df['pc_oi_ratio'] = 1.0
            
        # OI Concentration
        oi_total = features_df['oi_proxy'].sum()
        if oi_total > 0:
            oi_shares = features_df['oi_proxy'] / oi_total
            features_df['oi_concentration'] = (oi_shares ** 2).sum()
        else:
            features_df['oi_concentration'] = 0.0
            
        # OI Skewness (ATM vs OTM)
        atm_threshold = 0.95
        atm_options = features_df[features_df['moneyness'].between(1-atm_threshold, 1+atm_threshold)]
        otm_options = features_df[~features_df['moneyness'].between(1-atm_threshold, 1+atm_threshold)]
        
        if len(atm_options) > 0 and len(otm_options) > 0:
            atm_oi = atm_options['oi_proxy'].sum()
            otm_oi = otm_options['oi_proxy'].sum()
            features_df['oi_skew'] = otm_oi / (atm_oi + 1e-6)
        else:
            features_df['oi_skew'] = 1.0
        
        # 2. Volume Features
        features_df['volume_percentile'] = features_df['volume'].rank(pct=True)
        features_df['volume_zscore'] = stats.zscore(features_df['volume'])
        features_df['turnover_rate'] = features_df['volume'] / (features_df['oi_proxy'] + 1e-6)
        features_df['tx_efficiency'] = features_df['volume'] / (features_df['transactions'] + 1e-6)
        
        # Volume Concentration
        vol_total = features_df['volume'].sum()
        if vol_total > 0:
            vol_shares = features_df['volume'] / vol_total
            features_df['volume_concentration'] = (vol_shares ** 2).sum()
        else:
            features_df['volume_concentration'] = 0.0
        
        # 3. Price Features
        features_df['moneyness_squared'] = features_df['moneyness'] ** 2
        features_df['dte_percentile'] = features_df['dte'].rank(pct=True)
        
        # 4. Temporal Features
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['month'] = features_df['date'].dt.month
        features_df['quarter'] = features_df['date'].dt.quarter
        features_df['is_expiration_week'] = (features_df['dte'] <= 7).astype(int)
        
        # 5. Anomaly Features
        features_df['oi_anomaly_score'] = np.abs(features_df['oi_zscore'])
        features_df['volume_anomaly_score'] = np.abs(features_df['volume_zscore'])
        features_df['combined_anomaly_score'] = (
            0.4 * features_df['oi_anomaly_score'] +
            0.3 * features_df['volume_anomaly_score'] +
            0.3 * features_df['oi_concentration']
        )
        
        # 6. Institutional Momentum Features
        institutional = features_df[features_df['dte'] > 7].copy()
        if len(institutional) > 0:
            spy_price = institutional['underlying_price'].iloc[0]
            inst_calls = institutional[institutional['option_type'] == 'C']
            
            if len(inst_calls) > 0:
                atm_calls = inst_calls[
                    (inst_calls['strike'] >= spy_price - 5) & 
                    (inst_calls['strike'] <= spy_price + 5)
                ]
                otm_calls = inst_calls[inst_calls['strike'] > spy_price + 5]
                
                atm_oi = atm_calls['oi_proxy'].sum()
                otm_oi = otm_calls['oi_proxy'].sum()
                
                otm_atm_ratio = otm_oi / (atm_oi + 1e-6)
                momentum_score = np.clip((otm_atm_ratio - 2) / 6, 0, 1)
            else:
                otm_atm_ratio = 0.0
                momentum_score = 0.0
        else:
            otm_atm_ratio = 0.0
            momentum_score = 0.0
        
        # Apply to all rows
        features_df['otm_atm_call_ratio'] = otm_atm_ratio
        features_df['institutional_momentum_score'] = momentum_score
        
        return features_df
    
    def calculate_daily_features(self, df: pd.DataFrame) -> Dict:
        """Calculate daily aggregate features"""
        if df is None or len(df) == 0:
            return {}
            
        aggregates = {}
        
        # Basic aggregates
        aggregates['total_contracts'] = len(df)
        aggregates['total_volume'] = df['volume'].sum()
        aggregates['total_oi'] = df['oi_proxy'].sum()
        aggregates['avg_oi'] = df['oi_proxy'].mean()
        aggregates['median_oi'] = df['oi_proxy'].median()
        
        # Put/Call ratios
        calls = df[df['option_type'] == 'C']
        puts = df[df['option_type'] == 'P']
        
        if len(calls) > 0 and len(puts) > 0:
            aggregates['pc_ratio_volume'] = puts['volume'].sum() / calls['volume'].sum()
            aggregates['pc_ratio_oi'] = puts['oi_proxy'].sum() / calls['oi_proxy'].sum()
        else:
            aggregates['pc_ratio_volume'] = 1.0
            aggregates['pc_ratio_oi'] = 1.0
        
        # Concentration metrics
        aggregates['oi_concentration'] = df['oi_concentration'].iloc[0] if 'oi_concentration' in df.columns else 0.0
        aggregates['volume_concentration'] = df['volume_concentration'].iloc[0] if 'volume_concentration' in df.columns else 0.0
        
        # Anomaly metrics
        aggregates['avg_anomaly_score'] = df['combined_anomaly_score'].mean() if 'combined_anomaly_score' in df.columns else 0.0
        aggregates['max_anomaly_score'] = df['combined_anomaly_score'].max() if 'combined_anomaly_score' in df.columns else 0.0
        
        # Institutional features
        aggregates['otm_atm_call_ratio'] = df['otm_atm_call_ratio'].iloc[0] if 'otm_atm_call_ratio' in df.columns else 0.0
        aggregates['institutional_momentum_score'] = df['institutional_momentum_score'].iloc[0] if 'institutional_momentum_score' in df.columns else 0.0
        
        # Price level
        aggregates['underlying_price'] = df['underlying_price'].iloc[0] if 'underlying_price' in df.columns else 0.0
        
        return aggregates
    
    def predict_date(self, date: str) -> Dict:
        """Predict anomalies for a specific date"""
        if not self.is_loaded:
            self.load_models()
        
        # Load and process data
        df = self.load_daily_data(date)
        if df is None or len(df) == 0:
            return {'error': f'No data found for {date}'}
        
        # Calculate features
        df = self.calculate_features(df)
        daily_agg = self.calculate_daily_features(df)
        
        if not daily_agg:
            return {'error': f'Could not calculate features for {date}'}
        
        # Prepare features for prediction
        feature_df = pd.DataFrame([daily_agg])
        X = feature_df[self.feature_columns].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = {}
        
        for name, model in self.models.items():
            if name in ['random_forest', 'svm']:
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, 'predict_proba') else None
                predictions[name] = {
                    'prediction': int(pred),
                    'probability': float(prob) if prob is not None else None
                }
            else:
                pred = model.predict(X_scaled)[0]
                pred = int(pred == -1)  # Convert to binary
                predictions[name] = {
                    'prediction': pred,
                    'probability': None
                }
        
        # Enhanced ensemble prediction with thresholds and model agreement
        ensemble_probs = []
        weights = {
            'random_forest': 0.4,
            'svm': 0.3,
            'isolation_forest': 0.2,
            'one_class_svm': 0.1
        }
        
        # Count how many models predict anomaly
        anomaly_votes = 0
        total_models = len([p for p in predictions.values() if p['prediction'] is not None])
        
        for name, pred_dict in predictions.items():
            if pred_dict['probability'] is not None:
                ensemble_probs.append(weights[name] * pred_dict['probability'])
                if pred_dict['prediction'] == 1:
                    anomaly_votes += 1
            else:
                prob = float(pred_dict['prediction'])
                ensemble_probs.append(weights[name] * prob)
                if pred_dict['prediction'] == 1:
                    anomaly_votes += 1
        
        ensemble_prob = sum(ensemble_probs)
        
        # Enhanced ensemble logic: require both high confidence AND model agreement
        min_models_agree = 2  # At least 2 models must agree
        confidence_threshold = 0.6  # Higher threshold than 0.5
        
        ensemble_pred = int(
            ensemble_prob > confidence_threshold and 
            anomaly_votes >= min_models_agree
        )
        
        predictions['ensemble'] = {
            'prediction': ensemble_pred,
            'probability': ensemble_prob
        }
        
        # Add metadata
        result = {
            'date': date,
            'predictions': predictions,
            'features': daily_agg,
            'confidence': 'high' if ensemble_prob > 0.8 else 'medium' if ensemble_prob > 0.5 else 'low'
        }
        
        return result
    
    def predict_date_range(self, start_date: str, end_date: str) -> Dict:
        """Predict anomalies for a date range"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        results = {}
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_date.weekday() < 5:
                result = self.predict_date(date_str)
                if 'error' not in result:
                    results[date_str] = result
            
            current_date += timedelta(days=1)
        
        return results

def main():
    """Example usage"""
    predictor = AnomalyPredictor()
    
    # Predict for a specific date
    date = "2025-01-31"
    result = predictor.predict_date(date)
    
    if 'error' not in result:
        print(f"Prediction for {date}:")
        print(f"  Ensemble prediction: {result['predictions']['ensemble']['prediction']}")
        print(f"  Ensemble probability: {result['predictions']['ensemble']['probability']:.3f}")
        print(f"  Confidence: {result['confidence']}")
        
        print(f"\nIndividual model predictions:")
        for name, pred in result['predictions'].items():
            if name != 'ensemble':
                prob_str = f"{pred['probability']:.3f}" if pred['probability'] is not None else "N/A"
                print(f"  {name}: {pred['prediction']} ({prob_str})")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
