#!/usr/bin/env python3
"""
Model Training Pipeline

Unified pipeline for training momentum-based pullback prediction models.
Handles data preparation, feature engineering, target creation, and model training.

USAGE:
======
from pipeline.model_trainer import ModelTrainer

# Basic training
trainer = ModelTrainer(ticker='SPY')
results = trainer.train_full_pipeline(
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Custom configuration
trainer = ModelTrainer(
    ticker='AAPL',
    model_type='ensemble',
    config_path='config/aapl_config.json'
)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from .data_loader import DataLoader
from ..features.feature_engine import FeatureEngine
from ..targets.target_factory import TargetFactory
from ..models.xgboost_predictor import XGBoostPullbackPredictor
from ..models.ensemble_predictor import EnsemblePullbackPredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified model training pipeline for pullback prediction.
    
    Orchestrates:
    - Data loading and validation
    - Feature engineering
    - Target creation
    - Model training and evaluation
    - Results analysis and saving
    """
    
    def __init__(self, ticker: str, model_type: str = 'xgboost',
                 config_path: Optional[str] = None,
                 output_dir: str = 'data/models'):
        """
        Initialize model trainer.
        
        Args:
            ticker: Stock/ETF ticker
            model_type: 'xgboost' or 'ensemble'
            config_path: Path to configuration file
            output_dir: Directory to save trained models
        """
        self.ticker = ticker.upper()
        self.model_type = model_type
        self.output_dir = output_dir
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_loader = DataLoader(
            ticker=self.ticker,
            data_source=self.config.get('data_source', 'yfinance')
        )
        
        self.feature_engine = FeatureEngine(
            include=self.config.get('feature_engines', ['momentum', 'volatility'])
        )
        
        self.target_factory = TargetFactory()
        
        # State variables
        self.data_raw = None
        self.data_features = None
        self.data_targets = None
        self.model = None
        self.training_results = {}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ModelTrainer initialized: {self.ticker}, {self.model_type}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"   📋 Loaded config from: {config_path}")
        else:
            # Default configuration
            config = {
                "data_source": "yfinance",
                "feature_engines": ["momentum", "volatility"],
                "target_config": {
                    "pullback_targets": {
                        "thresholds": [0.02, 0.05, 0.10],
                        "horizons": [5, 10, 15, 20]
                    },
                    "mean_reversion_targets": {
                        "sma_periods": [20, 50, 100, 200],
                        "horizons": [5, 10, 15, 20]
                    }
                },
                "model_params": {
                    "xgboost": {
                        "n_estimators": 1000,
                        "max_depth": 10,
                        "learning_rate": 0.03,
                        "n_features": 75
                    },
                    "ensemble": {
                        "model_weights": None,
                        "use_lstm": True
                    }
                },
                "training": {
                    "test_size": 0.2,
                    "validation_size": 0.15,
                    "random_state": 42,
                    "min_samples": 1000
                }
            }
        
        return config
    
    def train_full_pipeline(self, start_date: str, end_date: str = None,
                           target_name: str = 'pullback_5pct_10d',
                           save_model: bool = True) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            start_date: Training start date
            end_date: Training end date (optional)
            target_name: Target variable name
            save_model: Whether to save trained model
            
        Returns:
            Training results dictionary
        """
        print(f"\n🚀 Starting Full Training Pipeline for {self.ticker}")
        print(f"   • Model type: {self.model_type}")
        print(f"   • Period: {start_date} to {end_date or 'today'}")
        print(f"   • Target: {target_name}")
        
        # Step 1: Load data
        self.data_raw = self._load_and_validate_data(start_date, end_date)
        
        # Step 2: Create features
        self.data_features = self._create_features(self.data_raw)
        
        # Step 3: Create targets
        self.data_targets = self._create_targets(self.data_features)
        
        # Step 4: Prepare training data
        X_train, X_test, y_train, y_test = self._prepare_training_data(
            self.data_targets, target_name
        )
        
        # Step 5: Train model
        self.model = self._train_model(X_train, X_test, y_train, y_test)
        
        # Step 6: Evaluate model
        results = self._evaluate_model(X_test, y_test)
        
        # Step 7: Save model and results
        if save_model:
            model_path = self._save_model_and_results(target_name, results)
            results['model_path'] = model_path
        
        self.training_results = results
        
        print(f"\n✅ Training pipeline completed successfully!")
        print(f"   • Model performance: ROC-AUC = {results['metrics']['roc_auc']:.4f}")
        print(f"   • Training samples: {results['training_info']['train_samples']:,}")
        
        return results
    
    def _load_and_validate_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Load and validate data."""
        print(f"\n📥 Step 1: Loading data...")
        
        df = self.data_loader.load_and_prepare(
            start_date=start_date,
            end_date=end_date,
            add_returns=True,
            add_basic_features=True
        )
        
        # Validate data
        validation_results = self.data_loader.validate_data(df)
        
        if not validation_results['valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                print(f"   ⚠️  Warning: {warning}")
        
        # Check minimum samples requirement
        min_samples = self.config['training']['min_samples']
        if len(df) < min_samples:
            raise ValueError(f"Insufficient data: {len(df)} < {min_samples} required")
        
        print(f"   ✅ Data loaded and validated: {len(df):,} records")
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features using feature engine."""
        print(f"\n🔧 Step 2: Creating features...")
        
        df_features = self.feature_engine.create_all_features(
            df, 
            price_col='close',
            high_col='high' if 'high' in df.columns else None,
            low_col='low' if 'low' in df.columns else None
        )
        
        return df_features
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create targets using target factory."""
        print(f"\n🎯 Step 3: Creating targets...")
        
        df_targets = self.target_factory.create_all_targets(
            df,
            price_col='close',
            config=self.config.get('target_config')
        )
        
        # Validate targets
        validation = self.target_factory.validate_targets(df_targets)
        
        if validation['warnings']:
            for warning in validation['warnings'][:5]:  # Show first 5 warnings
                print(f"   ⚠️  {warning}")
        
        return df_targets
    
    def _prepare_training_data(self, df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for model training."""
        print(f"\n📊 Step 4: Preparing training data...")
        
        # Check if target exists
        if target_name not in df.columns:
            available_targets = [col for col in df.columns if 'pullback_' in col or 'mean_revert_' in col]
            raise ValueError(f"Target '{target_name}' not found. Available: {available_targets[:10]}")
        
        # Get feature columns
        feature_names = self.feature_engine.get_all_feature_names()
        available_features = [f for f in feature_names if f in df.columns]
        
        print(f"   • Available features: {len(available_features)}")
        print(f"   • Target: {target_name}")
        
        # Remove rows with missing targets
        df_clean = df.dropna(subset=[target_name]).copy()
        
        # Feature selection for model training
        exclude_cols = [
            'ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'
        ] + [col for col in df.columns if col.startswith(('pullback_', 'mean_revert_', 'any_movement_'))]
        
        feature_cols = [col for col in available_features if col not in exclude_cols]
        
        print(f"   • Training features: {len(feature_cols)}")
        print(f"   • Clean samples: {len(df_clean):,}")
        
        # Prepare X and y
        X = df_clean[feature_cols].copy()
        y = df_clean[target_name].copy()
        
        # Handle any remaining missing values in features
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Train-test split
        training_config = self.config['training']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=training_config['test_size'],
            random_state=training_config['random_state'],
            stratify=y
        )
        
        print(f"   • Train samples: {len(X_train):,} (positive: {y_train.sum():,}, {y_train.mean():.1%})")
        print(f"   • Test samples: {len(X_test):,} (positive: {y_test.sum():,}, {y_test.mean():.1%})")
        
        return X_train, X_test, y_train, y_test
    
    def _train_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series):
        """Train the specified model."""
        print(f"\n🤖 Step 5: Training {self.model_type} model...")
        
        if self.model_type == 'xgboost':
            model_params = self.config['model_params']['xgboost']
            model = XGBoostPullbackPredictor(**model_params)
            
            # Train with validation set
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config['training']['validation_size'],
                random_state=self.config['training']['random_state'],
                stratify=y_train
            )
            
            model.fit(
                X_train_split.values, y_train_split.values,
                X_val.values, y_val.values
            )
            
        elif self.model_type == 'ensemble':
            ensemble_params = self.config['model_params']['ensemble']
            model = EnsemblePullbackPredictor(**ensemble_params)
            
            # Train ensemble (it handles validation split internally)
            model.fit(X_train.values, y_train.values)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate trained model."""
        print(f"\n📈 Step 6: Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test.values)[:, 1]
        y_pred = self.model.predict(X_test.values)
        
        # Calculate metrics
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score, f1_score,
            accuracy_score, average_precision_score, log_loss
        )
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba)
        }
        
        # Feature importance (if available)
        feature_importance = {}
        try:
            if hasattr(self.model, 'get_feature_importance'):
                feature_importance = self.model.get_feature_importance()
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
        
        # Create results dictionary
        results = {
            'ticker': self.ticker,
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat(),
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_info': {
                'train_samples': len(X_test) * (1 / self.config['training']['test_size'] - 1),
                'test_samples': len(X_test),
                'features_used': X_test.shape[1],
                'positive_rate_train': y_test.mean(),  # Approximation
                'positive_rate_test': y_test.mean()
            },
            'config': self.config
        }
        
        # Print results
        print(f"   📊 Performance Metrics:")
        print(f"      • ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"      • Precision: {metrics['precision']:.4f}")
        print(f"      • Recall: {metrics['recall']:.4f}")
        print(f"      • F1-Score: {metrics['f1_score']:.4f}")
        
        # Top features
        if feature_importance:
            print(f"\n   🏆 Top 5 Features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                print(f"      {i+1}. {feature}: {importance:.4f}")
        
        return results
    
    def _save_model_and_results(self, target_name: str, results: Dict[str, Any]) -> str:
        """Save trained model and results."""
        print(f"\n💾 Step 7: Saving model and results...")
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.ticker}_{self.model_type}_{target_name}_{timestamp}"
        
        # Save model
        model_path = os.path.join(self.output_dir, f"{model_filename}.pkl")
        
        metadata = {
            'ticker': self.ticker,
            'target_name': target_name,
            'training_date': results['training_date'],
            'metrics': results['metrics']
        }
        
        if hasattr(self.model, 'save_model'):
            self.model.save_model(model_path, metadata=metadata)
        elif hasattr(self.model, 'save_ensemble'):
            self.model.save_ensemble(model_path, metadata=metadata)
        else:
            # Fallback to joblib
            import joblib
            joblib.dump({
                'model': self.model,
                'metadata': metadata
            }, model_path)
        
        # Save results as JSON
        results_path = os.path.join(self.output_dir, f"{model_filename}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   ✅ Model saved: {model_path}")
        print(f"   ✅ Results saved: {results_path}")
        
        return model_path
    
    def predict_latest(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Make predictions on latest data.
        
        Args:
            days_back: How many days of recent data to use
            
        Returns:
            Prediction results
        """
        if not self.model:
            raise ValueError("Model must be trained before making predictions")
        
        print(f"\n🔮 Making predictions for {self.ticker}...")
        
        # Get latest data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back*2)).strftime('%Y-%m-%d')
        
        df = self.data_loader.load_and_prepare(start_date, end_date)
        
        # Create features
        df_features = self.feature_engine.create_all_features(df)
        
        # Get latest features (last row)
        feature_names = self.feature_engine.get_all_feature_names()
        available_features = [f for f in feature_names if f in df_features.columns]
        
        exclude_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        feature_cols = [col for col in available_features if col not in exclude_cols]
        
        X_latest = df_features[feature_cols].iloc[-1:].fillna(0)
        
        # Make prediction
        pred_proba = self.model.predict_proba(X_latest.values)[0, 1]
        pred_binary = self.model.predict(X_latest.values)[0]
        
        results = {
            'ticker': self.ticker,
            'prediction_date': datetime.now().isoformat(),
            'latest_price': float(df_features['close'].iloc[-1]),
            'prediction_probability': float(pred_proba),
            'prediction_binary': int(pred_binary),
            'signal_strength': 'High' if pred_proba > 0.7 else 'Medium' if pred_proba > 0.4 else 'Low'
        }
        
        print(f"   🎯 Prediction: {pred_proba:.1%} probability")
        print(f"   📊 Signal strength: {results['signal_strength']}")
        
        return results
    
    def __str__(self) -> str:
        return f"ModelTrainer({self.ticker}, {self.model_type})"
    
    def __repr__(self) -> str:
        return f"ModelTrainer(ticker='{self.ticker}', model_type='{self.model_type}')"