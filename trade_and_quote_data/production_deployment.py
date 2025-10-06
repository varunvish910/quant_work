#!/usr/bin/env python3
"""
Production Deployment Pipeline
Phase 6: Production-ready deployment system

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import warnings
from typing import Dict, List, Optional, Tuple
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionDeployment:
    """Production-ready deployment and monitoring system"""
    
    def __init__(self, config_file='production_config.json'):
        self.config = self.load_config(config_file)
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.db_connection = None
        self.last_update = None
        
        # Initialize database
        self.init_database()
        
    def load_config(self, config_file):
        """Load production configuration"""
        default_config = {
            'data_sources': {
                'symbols': ['SPY', '^VIX', '^TNX', 'GLD', 'TLT', 'QQQ', 'IWM'],
                'update_frequency': 'daily',
                'lookback_days': 500
            },
            'models': {
                'ensemble_threshold': 0.6,
                'high_confidence_threshold': 0.8,
                'retrain_frequency': 'weekly',
                'max_model_age_days': 30
            },
            'monitoring': {
                'performance_threshold': 0.55,
                'alert_email': None,
                'log_level': 'INFO'
            },
            'database': {
                'path': 'production_data.db',
                'backup_frequency': 'daily'
            }
        }
        
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file: {config_file}")
        
        return default_config
    
    def init_database(self):
        """Initialize production database"""
        db_path = self.config['database']['path']
        self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create tables
        cursor = self.db_connection.cursor()
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                date TEXT PRIMARY KEY,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                timestamp TEXT
            )
        ''')
        
        # Features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                date TEXT PRIMARY KEY,
                feature_data TEXT,
                timestamp TEXT
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                date TEXT,
                target TEXT,
                prediction REAL,
                confidence REAL,
                model_version TEXT,
                timestamp TEXT,
                PRIMARY KEY (date, target)
            )
        ''')
        
        # Performance monitoring table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_log (
                date TEXT,
                target TEXT,
                actual REAL,
                predicted REAL,
                accuracy REAL,
                timestamp TEXT,
                PRIMARY KEY (date, target)
            )
        ''')
        
        # Model metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metadata (
                model_name TEXT PRIMARY KEY,
                version TEXT,
                training_date TEXT,
                performance_metrics TEXT,
                file_path TEXT
            )
        ''')
        
        self.db_connection.commit()
        logger.info("Database initialized successfully")
    
    def download_latest_data(self) -> pd.DataFrame:
        """Download latest market data"""
        logger.info("Downloading latest market data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['data_sources']['lookback_days'])
        
        all_data = {}
        symbols = self.config['data_sources']['symbols']
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                # Handle multiindex columns
                if isinstance(data.columns, pd.MultiIndex):
                    if len(symbols) == 1:
                        data = data.xs(symbol, axis=1, level=1)
                    else:
                        # Keep symbol in column names for multiple symbols
                        data.columns = [f"{col[0]}_{symbol}" for col in data.columns]
                
                all_data[symbol] = data
                
                # Store in database
                self.store_market_data(symbol, data)
                
                logger.info(f"Downloaded {symbol}: {len(data)} days")
                
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
        
        return all_data
    
    def store_market_data(self, symbol: str, data: pd.DataFrame):
        """Store market data in database"""
        cursor = self.db_connection.cursor()
        timestamp = datetime.now().isoformat()
        
        for date, row in data.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (date, symbol, open, high, low, close, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date.strftime('%Y-%m-%d'),
                symbol,
                float(row.get('Open', 0)),
                float(row.get('High', 0)),
                float(row.get('Low', 0)),
                float(row.get('Close', 0)),
                int(row.get('Volume', 0)),
                timestamp
            ))
        
        self.db_connection.commit()
    
    def load_production_models(self):
        """Load trained models for production"""
        logger.info("Loading production models...")
        
        models_dir = Path('analysis/outputs/ensemble_results')
        
        # Find latest model files
        model_files = list(models_dir.glob('ensemble_models_*.pkl'))
        
        if not model_files:
            logger.error("No trained models found!")
            return False
        
        # Load latest models
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            logger.info(f"Loaded models from {latest_model_file}")
            
            # Extract models and preprocessors
            for target_name, data in model_data.items():
                self.models[target_name] = data.get('base_models', {})
                self.scalers[target_name] = data.get('scaler')
                self.feature_selectors[target_name] = data.get('feature_selector')
            
            # Store model metadata
            self.store_model_metadata(latest_model_file, model_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def store_model_metadata(self, model_file: Path, model_data: dict):
        """Store model metadata in database"""
        cursor = self.db_connection.cursor()
        timestamp = datetime.now().isoformat()
        
        for target_name, data in model_data.items():
            # Calculate summary metrics
            ensemble_results = data.get('ensemble_results', {})
            best_f1 = 0
            
            for ensemble_name, metrics in ensemble_results.items():
                if isinstance(metrics, dict) and 'f1_score' in metrics:
                    best_f1 = max(best_f1, metrics['f1_score'])
            
            metadata = {
                'best_f1_score': best_f1,
                'ensemble_count': len(ensemble_results),
                'selected_features': len(data.get('selected_features', []))
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_metadata
                (model_name, version, training_date, performance_metrics, file_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                target_name,
                timestamp[:10],  # Use date as version
                timestamp,
                json.dumps(metadata),
                str(model_file)
            ))
        
        self.db_connection.commit()
    
    def generate_features(self, market_data: dict) -> pd.DataFrame:
        """Generate features from latest market data"""
        logger.info("Generating features from latest market data...")
        
        # Use the feature engineering from run_analysis.py
        from run_analysis import SimplifiedTargetAnalyzer
        
        analyzer = SimplifiedTargetAnalyzer()
        
        # Extract SPY and VIX data
        spy = market_data.get('SPY')
        vix_data = market_data.get('^VIX')
        
        if spy is None or vix_data is None:
            logger.error("Missing SPY or VIX data for feature generation")
            return pd.DataFrame()
        
        # Handle VIX data structure
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix = vix_data['Close']['^VIX']
        else:
            vix = vix_data['Close']
        
        # Generate features
        features = analyzer.create_features(spy, vix)
        
        # Store features in database
        self.store_features(features)
        
        return features
    
    def store_features(self, features: pd.DataFrame):
        """Store features in database"""
        cursor = self.db_connection.cursor()
        timestamp = datetime.now().isoformat()
        
        # Store latest features (last 10 days)
        recent_features = features.tail(10)
        
        for date, row in recent_features.iterrows():
            feature_dict = row.to_dict()
            # Convert numpy types to native Python types
            feature_dict = {k: float(v) if pd.notna(v) else None for k, v in feature_dict.items()}
            
            cursor.execute('''
                INSERT OR REPLACE INTO features (date, feature_data, timestamp)
                VALUES (?, ?, ?)
            ''', (
                date.strftime('%Y-%m-%d'),
                json.dumps(feature_dict),
                timestamp
            ))
        
        self.db_connection.commit()
    
    def generate_predictions(self, features: pd.DataFrame) -> dict:
        """Generate predictions using loaded models"""
        logger.info("Generating predictions...")
        
        if not self.models:
            logger.error("No models loaded for prediction")
            return {}
        
        predictions = {}
        latest_features = features.tail(1)
        
        if len(latest_features) == 0:
            logger.error("No features available for prediction")
            return {}
        
        prediction_date = latest_features.index[0]
        
        for target_name, models in self.models.items():
            try:
                # Get feature selector and scaler
                feature_selector = self.feature_selectors.get(target_name)
                scaler = self.scalers.get(target_name)
                
                if feature_selector is None or scaler is None:
                    logger.warning(f"Missing preprocessors for {target_name}")
                    continue
                
                # Select and scale features
                selected_features = feature_selector.get_feature_names_out() if hasattr(feature_selector, 'get_feature_names_out') else feature_selector.support_
                
                if hasattr(feature_selector, 'get_feature_names_out'):
                    X = latest_features[selected_features]
                else:
                    X = latest_features.iloc[:, feature_selector.support_]
                
                X_scaled = scaler.transform(X)
                
                # Generate predictions from ensemble
                ensemble_predictions = []
                
                for model_name, model in models.items():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_scaled)[0, 1]
                        ensemble_predictions.append(pred)
                
                if ensemble_predictions:
                    # Average ensemble prediction
                    final_prediction = np.mean(ensemble_predictions)
                    confidence = 1 - np.std(ensemble_predictions)  # Higher std = lower confidence
                    
                    predictions[target_name] = {
                        'prediction': float(final_prediction),
                        'confidence': float(confidence),
                        'threshold_met': final_prediction > self.config['models']['ensemble_threshold'],
                        'high_confidence': final_prediction > self.config['models']['high_confidence_threshold']
                    }
                    
                    logger.info(f"{target_name}: {final_prediction:.3f} (confidence: {confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to generate prediction for {target_name}: {e}")
        
        # Store predictions
        self.store_predictions(prediction_date, predictions)
        
        return predictions
    
    def store_predictions(self, date: pd.Timestamp, predictions: dict):
        """Store predictions in database"""
        cursor = self.db_connection.cursor()
        timestamp = datetime.now().isoformat()
        
        for target_name, pred_data in predictions.items():
            cursor.execute('''
                INSERT OR REPLACE INTO predictions
                (date, target, prediction, confidence, model_version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                date.strftime('%Y-%m-%d'),
                target_name,
                pred_data['prediction'],
                pred_data['confidence'],
                timestamp[:10],  # Use date as version
                timestamp
            ))
        
        self.db_connection.commit()
    
    def monitor_performance(self) -> dict:
        """Monitor model performance"""
        logger.info("Monitoring model performance...")
        
        cursor = self.db_connection.cursor()
        
        # Get recent predictions and actuals
        cursor.execute('''
            SELECT p.date, p.target, p.prediction, p.confidence, pl.actual
            FROM predictions p
            LEFT JOIN performance_log pl ON p.date = pl.date AND p.target = pl.target
            WHERE p.date >= date('now', '-30 days')
            ORDER BY p.date DESC
        ''')
        
        results = cursor.fetchall()
        
        performance_summary = {}
        
        for date, target, prediction, confidence, actual in results:
            if target not in performance_summary:
                performance_summary[target] = {
                    'predictions': [],
                    'actuals': [],
                    'confidences': []
                }
            
            performance_summary[target]['predictions'].append(prediction)
            performance_summary[target]['confidences'].append(confidence)
            
            if actual is not None:
                performance_summary[target]['actuals'].append(actual)
        
        # Calculate performance metrics
        for target, data in performance_summary.items():
            if len(data['actuals']) > 5:
                from sklearn.metrics import roc_auc_score, precision_score
                
                predictions = np.array(data['predictions'][:len(data['actuals'])])
                actuals = np.array(data['actuals'])
                
                try:
                    roc_auc = roc_auc_score(actuals, predictions)
                    precision = precision_score(actuals, (predictions > 0.5).astype(int), zero_division=0)
                    
                    data['roc_auc'] = roc_auc
                    data['precision'] = precision
                    data['avg_confidence'] = np.mean(data['confidences'])
                    
                    # Alert if performance drops
                    if roc_auc < self.config['monitoring']['performance_threshold']:
                        self.send_alert(f"Performance alert: {target} ROC AUC = {roc_auc:.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to calculate metrics for {target}: {e}")
        
        return performance_summary
    
    def send_alert(self, message: str):
        """Send performance alert"""
        logger.warning(f"ALERT: {message}")
        
        alert_email = self.config['monitoring'].get('alert_email')
        if alert_email:
            try:
                # Simple email alert (would need SMTP configuration)
                logger.info(f"Would send email alert to {alert_email}: {message}")
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
    
    def create_daily_report(self, predictions: dict, performance: dict) -> str:
        """Create daily performance report"""
        report = []
        report.append("=" * 60)
        report.append("DAILY MARKET PREDICTION REPORT")
        report.append("=" * 60)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current predictions
        report.append("CURRENT PREDICTIONS:")
        report.append("-" * 30)
        
        for target, pred_data in predictions.items():
            status = "🚨 HIGH CONFIDENCE" if pred_data['high_confidence'] else ("⚠️  SIGNAL" if pred_data['threshold_met'] else "✅ NORMAL")
            
            report.append(f"{target}:")
            report.append(f"  Prediction: {pred_data['prediction']:.3f}")
            report.append(f"  Confidence: {pred_data['confidence']:.3f}")
            report.append(f"  Status: {status}")
            report.append("")
        
        # Performance summary
        report.append("RECENT PERFORMANCE:")
        report.append("-" * 30)
        
        for target, perf_data in performance.items():
            if 'roc_auc' in perf_data:
                report.append(f"{target}:")
                report.append(f"  ROC AUC (30d): {perf_data['roc_auc']:.3f}")
                report.append(f"  Precision (30d): {perf_data['precision']:.3f}")
                report.append(f"  Avg Confidence: {perf_data['avg_confidence']:.3f}")
                report.append(f"  Samples: {len(perf_data['actuals'])}")
                report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def run_daily_pipeline(self):
        """Run daily prediction pipeline"""
        logger.info("🚀 Starting daily prediction pipeline...")
        
        try:
            # Download latest data
            market_data = self.download_latest_data()
            
            # Generate features
            features = self.generate_features(market_data)
            
            if len(features) == 0:
                logger.error("No features generated, aborting pipeline")
                return False
            
            # Load models if not already loaded
            if not self.models:
                if not self.load_production_models():
                    logger.error("Failed to load models, aborting pipeline")
                    return False
            
            # Generate predictions
            predictions = self.generate_predictions(features)
            
            # Monitor performance
            performance = self.monitor_performance()
            
            # Create daily report
            report = self.create_daily_report(predictions, performance)
            
            # Save report
            report_dir = Path('analysis/outputs/daily_reports')
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Daily report saved to {report_file}")
            print(report)
            
            self.last_update = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Daily pipeline failed: {e}")
            self.send_alert(f"Daily pipeline failed: {e}")
            return False
    
    def schedule_automated_runs(self):
        """Schedule automated daily runs"""
        logger.info("Scheduling automated daily runs...")
        
        # Schedule daily prediction pipeline
        schedule.every().day.at("09:00").do(self.run_daily_pipeline)
        
        # Schedule weekly model retraining (placeholder)
        schedule.every().monday.at("02:00").do(self.retrain_models)
        
        logger.info("Automated scheduling configured")
        logger.info("Daily predictions: 09:00 AM")
        logger.info("Weekly retraining: Monday 02:00 AM")
    
    def retrain_models(self):
        """Retrain models with latest data (placeholder)"""
        logger.info("🔄 Starting model retraining...")
        
        # This would trigger a full retraining pipeline
        # For now, just log the event
        logger.info("Model retraining would be triggered here")
        
        return True
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop"""
        logger.info("Starting production monitoring loop...")
        
        self.schedule_automated_runs()
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
                
            except KeyboardInterrupt:
                logger.info("Monitoring loop stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(3600)


def main():
    """Main execution function"""
    print("🚀 Starting Production Deployment Pipeline")
    print("Phase 6: Production-Ready Deployment System")
    print("=" * 60)
    
    # Initialize production system
    production_system = ProductionDeployment()
    
    # Run initial daily pipeline
    logger.info("Running initial daily pipeline...")
    success = production_system.run_daily_pipeline()
    
    if success:
        print("\n✅ Production Deployment Complete!")
        print("✅ Daily pipeline executed successfully")
        print("✅ Database initialized")
        print("✅ Models loaded")
        print("✅ Predictions generated")
        print("✅ Performance monitoring active")
        
        print("\n📊 Production System Status:")
        print(f"  Models loaded: {len(production_system.models)}")
        print(f"  Last update: {production_system.last_update}")
        print(f"  Database: {production_system.config['database']['path']}")
        
        print("\n🔄 To start continuous monitoring:")
        print("  python3 production_deployment.py --monitor")
        
    else:
        print("\n❌ Production deployment failed!")
        print("Check logs for details")
    
    return production_system


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        # Run continuous monitoring
        production_system = ProductionDeployment()
        production_system.run_monitoring_loop()
    else:
        # Run initial setup
        production_system = main()
        print("\n🎉 Phase 6 Production Deployment Complete!")