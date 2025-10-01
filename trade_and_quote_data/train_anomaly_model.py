#!/usr/bin/env python3
"""
Train SPY Options Anomaly Detection Model with Correction Targets
================================================================

This script trains the anomaly detection model using:
- Historical options data (2016-2024)
- Correction targets identified by target_creator.py
- Feature engineering pipeline
- Multiple ML models for ensemble detection

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
import joblib
import json
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectionAnomalyTrainer:
    """
    Train anomaly detection models using correction targets
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY", targets_file: str = "data/correction_targets.csv"):
        self.data_dir = Path(data_dir)
        self.targets_file = targets_file
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []
        self.is_fitted = False
        
    def load_targets(self) -> pd.DataFrame:
        """Load correction targets"""
        try:
            targets_df = pd.read_csv(self.targets_file)
            targets_df['date'] = pd.to_datetime(targets_df['date'])
            logger.info(f"Loaded {len(targets_df)} target records")
            return targets_df
        except Exception as e:
            logger.error(f"Error loading targets: {e}")
            return None
    
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """Load options data for a specific date"""
        try:
            year = date[:4]
            month = date[5:7]
            
            # Convert date to YYYYMMDD format
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
        """Calculate comprehensive features for anomaly detection"""
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
    
    def prepare_training_data(self, start_date: str, end_date: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and targets"""
        logger.info(f"Preparing training data from {start_date} to {end_date}")
        
        # Load targets
        targets_df = self.load_targets()
        if targets_df is None:
            raise ValueError("Could not load targets")
        
        # Filter targets for date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        targets_df = targets_df[(targets_df['date'] >= start) & (targets_df['date'] <= end)]
        
        logger.info(f"Found {len(targets_df)} target records in date range")
        
        # Collect daily features and targets
        daily_features = []
        daily_targets = []
        processed_dates = []
        
        # Get all available data files
        available_dates = []
        for year in range(start.year, end.year + 1):
            year_dir = self.data_dir / str(year)
            if year_dir.exists():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        for file_path in month_dir.glob("SPY_options_snapshot_*.parquet"):
                            date_str = file_path.stem.split('_')[-1]
                            try:
                                date_obj = pd.to_datetime(date_str)
                                if start <= date_obj <= end and date_obj.weekday() < 5:
                                    available_dates.append(date_obj)
                            except:
                                continue
        
        available_dates = sorted(set(available_dates))
        logger.info(f"Found {len(available_dates)} available trading days with data")
        
        for i, current_date in enumerate(available_dates):
            date_str = current_date.strftime('%Y-%m-%d')
            
            if i % 100 == 0:  # Log progress every 100 dates
                logger.info(f"Processing date {i+1}/{len(available_dates)}: {date_str}")
            
            # Load options data
            df = self.load_daily_data(date_str)
            if df is not None and len(df) > 0:
                try:
                    # Calculate features
                    df = self.calculate_features(df)
                    daily_agg = self.calculate_daily_features(df)
                    
                    if daily_agg:
                        # Get target for this date
                        target_row = targets_df[targets_df['date'] == current_date]
                        target = target_row['target'].iloc[0] if len(target_row) > 0 else 0
                        
                        daily_features.append(daily_agg)
                        daily_targets.append(target)
                        processed_dates.append(current_date)
                except Exception as e:
                    logger.error(f"Error processing {date_str}: {e}")
                    continue
        
        if len(daily_features) == 0:
            raise ValueError("No training data found")
        
        # Convert to arrays
        feature_df = pd.DataFrame(daily_features)
        feature_columns = [
            'total_contracts', 'total_volume', 'total_oi', 'avg_oi', 'median_oi',
            'pc_ratio_volume', 'pc_ratio_oi', 'oi_concentration', 'volume_concentration',
            'avg_anomaly_score', 'max_anomaly_score', 'otm_atm_call_ratio',
            'institutional_momentum_score', 'underlying_price'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_columns if col in feature_df.columns]
        X = feature_df[available_cols].fillna(0).values
        y = np.array(daily_targets)
        
        self.feature_columns = available_cols
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {np.bincount(y)}")
        logger.info(f"Processed dates: {len(processed_dates)} from {processed_dates[0]} to {processed_dates[-1]}")
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train multiple models for ensemble detection"""
        logger.info("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Random Forest Classifier
        logger.info("Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        
        # 2. SVM Classifier
        logger.info("Training SVM...")
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        self.models['svm'].fit(X_train_scaled, y_train)
        
        # 3. Isolation Forest (for anomaly detection)
        logger.info("Training Isolation Forest...")
        # Fixed contamination: using 5% instead of 2.24% for more realistic anomaly detection
        base_contamination = np.sum(y_train) / len(y_train)
        contamination = max(0.05, base_contamination)  # Use at least 5%
        logger.info(f"Using contamination rate: {contamination:.3f} (base: {base_contamination:.3f})")
        
        self.models['isolation_forest'] = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.models['isolation_forest'].fit(X_train_scaled)
        
        # 4. One-Class SVM
        logger.info("Training One-Class SVM...")
        self.models['one_class_svm'] = OneClassSVM(
            nu=contamination,
            kernel='rbf',
            gamma='scale'
        )
        self.models['one_class_svm'].fit(X_train_scaled)
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test)
        
        self.is_fitted = True
        logger.info("Model training completed")
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        logger.info("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            if name in ['random_forest', 'svm']:
                # Classification models
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = (y_pred == y_test).mean()
                precision = np.sum((y_pred == 1) & (y_test == 1)) / (np.sum(y_pred == 1) + 1e-6)
                recall = np.sum((y_pred == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
                }
                
                logger.info(f"{name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
                
            else:
                # Anomaly detection models
                y_pred = model.predict(X_test)
                y_pred = (y_pred == -1).astype(int)  # Convert to binary
                
                accuracy = (y_pred == y_test).mean()
                precision = np.sum((y_pred == 1) & (y_test == 1)) / (np.sum(y_pred == 1) + 1e-6)
                recall = np.sum((y_pred == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                logger.info(f"{name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Save evaluation results
        with open('analysis/outputs/model_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def predict_ensemble(self, X: np.ndarray) -> Dict:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Models not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        # Individual model predictions
        for name, model in self.models.items():
            if name in ['random_forest', 'svm']:
                pred = model.predict(X_scaled)
                prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                predictions[name] = {
                    'prediction': pred,
                    'probability': prob
                }
            else:
                pred = model.predict(X_scaled)
                pred = (pred == -1).astype(int)
                predictions[name] = {
                    'prediction': pred,
                    'probability': None
                }
        
        # Ensemble prediction (weighted average)
        ensemble_probs = []
        weights = {
            'random_forest': 0.4,
            'svm': 0.3,
            'isolation_forest': 0.2,
            'one_class_svm': 0.1
        }
        
        for name, pred_dict in predictions.items():
            if pred_dict['probability'] is not None:
                ensemble_probs.append(weights[name] * pred_dict['probability'])
            else:
                # Convert binary prediction to probability
                prob = pred_dict['prediction'].astype(float)
                ensemble_probs.append(weights[name] * prob)
        
        if ensemble_probs:
            ensemble_prob = np.sum(ensemble_probs, axis=0)
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
        else:
            ensemble_prob = np.zeros(X.shape[0])
            ensemble_pred = np.zeros(X.shape[0])
        
        predictions['ensemble'] = {
            'prediction': ensemble_pred,
            'probability': ensemble_prob
        }
        
        return predictions
    
    def save_models(self, output_dir: str = "models"):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, output_path / f"{name}.pkl")
        
        # Save feature columns
        with open(output_path / "feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'model_types': list(self.models.keys()),
            'is_fitted': self.is_fitted
        }
        
        with open(output_path / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {output_path}")
    
    def train_full_pipeline(self, start_date: str = "2016-01-01", end_date: str = "2024-12-31"):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        try:
            # Prepare training data
            X, y = self.prepare_training_data(start_date, end_date)
            
            # Train models
            self.train_models(X, y)
            
            # Save models
            self.save_models()
            
            logger.info("Training pipeline completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False

def main():
    """Main training function"""
    # Initialize trainer
    trainer = CorrectionAnomalyTrainer()
    
    # Run training pipeline
    success = trainer.train_full_pipeline("2016-01-01", "2024-12-31")
    
    if success:
        print("✅ Model training completed successfully!")
        print("📁 Models saved to: models/")
        print("📊 Evaluation results: analysis/outputs/model_evaluation.json")
    else:
        print("❌ Model training failed!")

if __name__ == "__main__":
    main()
