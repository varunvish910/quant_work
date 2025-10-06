#!/usr/bin/env python3
"""
Ensemble Pullback Predictor

Combines XGBoost, Random Forest, and LSTM models for robust pullback prediction.
Uses weighted voting and model confidence scoring for enhanced performance.

USAGE:
======
from models.ensemble_predictor import EnsemblePullbackPredictor

# Basic usage
ensemble = EnsemblePullbackPredictor()
ensemble.fit(X_train, y_train)
predictions = ensemble.predict_proba(X_test)

# Custom weights
ensemble = EnsemblePullbackPredictor(
    model_weights={'xgboost': 0.5, 'random_forest': 0.3, 'lstm': 0.2}
)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import joblib
import os
from typing import Tuple, Dict, Any, Optional, List, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import LSTM dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️  TensorFlow not available. LSTM model will be disabled.")

from .xgboost_predictor import XGBoostPullbackPredictor

logger = logging.getLogger(__name__)


class EnsemblePullbackPredictor:
    """
    Ensemble model combining XGBoost, Random Forest, and LSTM for pullback prediction.
    
    Features:
    - Weighted voting ensemble
    - Individual model confidence scoring
    - Automatic weight optimization
    - Model performance monitoring
    - Fallback to available models
    """
    
    def __init__(self,
                 model_weights: Optional[Dict[str, float]] = None,
                 use_lstm: bool = True,
                 xgb_params: Optional[Dict] = None,
                 rf_params: Optional[Dict] = None,
                 lstm_params: Optional[Dict] = None,
                 random_state: int = 42):
        """
        Initialize ensemble predictor.
        
        Args:
            model_weights: Dictionary of model weights (auto-optimized if None)
            use_lstm: Whether to include LSTM model
            xgb_params: XGBoost parameters
            rf_params: Random Forest parameters
            lstm_params: LSTM parameters
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.use_lstm = use_lstm and LSTM_AVAILABLE
        
        # Default model weights
        if model_weights is None:
            if self.use_lstm:
                self.model_weights = {'xgboost': 0.5, 'random_forest': 0.3, 'lstm': 0.2}
            else:
                self.model_weights = {'xgboost': 0.6, 'random_forest': 0.4}
        else:
            self.model_weights = model_weights
        
        # Initialize models
        self._initialize_models(xgb_params, rf_params, lstm_params)
        
        # State variables
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {}
        self.model_performances = {}
        
        logger.info(f"EnsemblePullbackPredictor initialized: "
                   f"models={list(self.model_weights.keys())}")
    
    def _initialize_models(self, xgb_params: Optional[Dict] = None,
                          rf_params: Optional[Dict] = None,
                          lstm_params: Optional[Dict] = None):
        """Initialize individual models."""
        
        # XGBoost model
        xgb_defaults = {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_features': 75,
            'random_state': self.random_state
        }
        xgb_config = {**xgb_defaults, **(xgb_params or {})}
        self.xgboost_model = XGBoostPullbackPredictor(**xgb_config)
        
        # Random Forest model
        rf_defaults = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        rf_config = {**rf_defaults, **(rf_params or {})}
        self.random_forest = RandomForestClassifier(**rf_config)
        
        # LSTM model (if available)
        if self.use_lstm:
            lstm_defaults = {
                'sequence_length': 20,
                'lstm_units': 50,
                'dropout_rate': 0.3,
                'epochs': 100,
                'batch_size': 32
            }
            self.lstm_params = {**lstm_defaults, **(lstm_params or {})}
            self.lstm_model = None
            self.lstm_scaler = MinMaxScaler()
        
        self.models = {'xgboost': self.xgboost_model, 'random_forest': self.random_forest}
        if self.use_lstm:
            self.models['lstm'] = None  # Will be created during training
    
    def fit(self, X: np.ndarray, y: np.ndarray,
           X_val: Optional[np.ndarray] = None,
           y_val: Optional[np.ndarray] = None) -> 'EnsemblePullbackPredictor':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target vector
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Self for method chaining
        """
        print(f"\n🎯 Training Ensemble Pullback Predictor...")
        print(f"   • Models: {list(self.model_weights.keys())}")
        print(f"   • Training samples: {X.shape[0]:,}")
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split validation set if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
        
        # Train XGBoost
        print(f"\n   🚀 Training XGBoost model...")
        self.xgboost_model.fit(X_train, y_train, X_val, y_val)
        self.model_performances['xgboost'] = self._evaluate_model(
            self.xgboost_model, X_val, y_val, 'xgboost'
        )
        
        # Train Random Forest
        print(f"\n   🌲 Training Random Forest model...")
        self.random_forest.fit(X_train, y_train)
        self.model_performances['random_forest'] = self._evaluate_model(
            self.random_forest, X_val, y_val, 'random_forest'
        )
        
        # Train LSTM (if enabled)
        if self.use_lstm:
            print(f"\n   🧠 Training LSTM model...")
            try:
                self.lstm_model = self._train_lstm(X_train, y_train, X_val, y_val)
                self.models['lstm'] = self.lstm_model
                self.model_performances['lstm'] = self._evaluate_model(
                    self.lstm_model, X_val, y_val, 'lstm'
                )
            except Exception as e:
                logger.warning(f"LSTM training failed: {e}")
                print(f"   ❌ LSTM training failed: {e}")
                # Remove LSTM from ensemble
                self.model_weights.pop('lstm', None)
                self.use_lstm = False
        
        # Optimize weights based on validation performance
        self._optimize_weights(X_val, y_val)
        
        # Store training history
        self.training_history = {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'models_trained': list(self.model_weights.keys()),
            'final_weights': self.model_weights.copy(),
            'model_performances': self.model_performances.copy()
        }
        
        self.is_fitted = True
        
        print(f"\n   ✅ Ensemble training completed!")
        print(f"      • Final weights: {self.model_weights}")
        for model_name, perf in self.model_performances.items():
            print(f"      • {model_name}: ROC-AUC = {perf['roc_auc']:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = {}
        
        # Get predictions from each model
        if 'xgboost' in self.model_weights:
            predictions['xgboost'] = self.xgboost_model.predict_proba(X)[:, 1]
        
        if 'random_forest' in self.model_weights:
            predictions['random_forest'] = self.random_forest.predict_proba(X)[:, 1]
        
        if 'lstm' in self.model_weights and self.lstm_model:
            predictions['lstm'] = self._predict_lstm(X)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(X.shape[0])
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights[model_name]
            ensemble_pred += weight * pred
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Return as 2D array for consistency with sklearn
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray):
        """Train LSTM model."""
        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        # Prepare sequences for LSTM
        X_train_seq, y_train_seq = self._prepare_lstm_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self._prepare_lstm_sequences(X_val, y_val)
        
        # Scale features for LSTM
        X_train_scaled = self.lstm_scaler.fit_transform(
            X_train_seq.reshape(-1, X_train_seq.shape[-1])
        ).reshape(X_train_seq.shape)
        
        X_val_scaled = self.lstm_scaler.transform(
            X_val_seq.reshape(-1, X_val_seq.shape[-1])
        ).reshape(X_val_seq.shape)
        
        # Build LSTM model
        model = Sequential([
            LSTM(self.lstm_params['lstm_units'], 
                 return_sequences=True,
                 input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dropout(self.lstm_params['dropout_rate']),
            BatchNormalization(),
            
            LSTM(self.lstm_params['lstm_units'] // 2, 
                 return_sequences=False),
            Dropout(self.lstm_params['dropout_rate']),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(self.lstm_params['dropout_rate']),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_seq,
            validation_data=(X_val_scaled, y_val_seq),
            epochs=self.lstm_params['epochs'],
            batch_size=self.lstm_params['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        print(f"      • LSTM trained: {len(history.history['loss'])} epochs")
        
        return model
    
    def _prepare_lstm_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        seq_length = self.lstm_params['sequence_length']
        
        sequences = []
        targets = []
        
        for i in range(seq_length, len(X)):
            sequences.append(X[i-seq_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def _predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using LSTM model."""
        if not self.lstm_model:
            raise ValueError("LSTM model not trained")
        
        # Prepare sequences
        seq_length = self.lstm_params['sequence_length']
        
        if len(X) < seq_length:
            # Pad with first row if insufficient data
            X_padded = np.vstack([np.tile(X[0], (seq_length - len(X), 1)), X])
        else:
            X_padded = X
        
        sequences = []
        for i in range(seq_length, len(X_padded) + 1):
            sequences.append(X_padded[i-seq_length:i])
        
        if not sequences:
            return np.zeros(len(X))
        
        X_seq = np.array(sequences)
        
        # Scale sequences
        X_seq_scaled = self.lstm_scaler.transform(
            X_seq.reshape(-1, X_seq.shape[-1])
        ).reshape(X_seq.shape)
        
        # Predict
        predictions = self.lstm_model.predict(X_seq_scaled, verbose=0)
        
        return predictions.flatten()
    
    def _evaluate_model(self, model, X_val: np.ndarray, y_val: np.ndarray, 
                       model_name: str) -> Dict[str, float]:
        """Evaluate individual model performance."""
        try:
            if model_name == 'lstm':
                y_pred_proba = self._predict_lstm(X_val)
            elif model_name == 'xgboost':
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            else:  # random_forest
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            metrics = {
                'roc_auc': roc_auc_score(y_val, y_pred_proba),
                'avg_precision': average_precision_score(y_val, y_pred_proba)
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error evaluating {model_name}: {e}")
            return {'roc_auc': 0.0, 'avg_precision': 0.0}
    
    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Optimize ensemble weights based on validation performance."""
        # Simple performance-based weighting
        total_performance = sum(perf['roc_auc'] for perf in self.model_performances.values())
        
        if total_performance > 0:
            for model_name in self.model_weights:
                if model_name in self.model_performances:
                    performance = self.model_performances[model_name]['roc_auc']
                    self.model_weights[model_name] = performance / total_performance
        
        print(f"   ⚖️  Optimized weights: {self.model_weights}")
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from each model."""
        importance_dict = {}
        
        if 'xgboost' in self.model_weights:
            importance_dict['xgboost'] = self.xgboost_model.get_feature_importance()
        
        if 'random_forest' in self.model_weights:
            if self.feature_names:
                rf_importance = dict(zip(
                    self.feature_names, 
                    self.random_forest.feature_importances_
                ))
                importance_dict['random_forest'] = dict(
                    sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
                )
        
        # LSTM doesn't provide feature importance in traditional sense
        
        return importance_dict
    
    def save_ensemble(self, filepath: str, metadata: Optional[Dict] = None) -> str:
        """Save ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving")
        
        # Prepare ensemble data
        ensemble_data = {
            'model_weights': self.model_weights,
            'xgboost_model': self.xgboost_model,
            'random_forest': self.random_forest,
            'lstm_model': self.lstm_model if self.use_lstm else None,
            'lstm_scaler': self.lstm_scaler if self.use_lstm else None,
            'lstm_params': self.lstm_params if self.use_lstm else None,
            'use_lstm': self.use_lstm,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_performances': self.model_performances,
            'metadata': metadata or {}
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save ensemble
        joblib.dump(ensemble_data, filepath)
        
        print(f"   💾 Ensemble saved to: {filepath}")
        return filepath
    
    @classmethod
    def load_ensemble(cls, filepath: str) -> 'EnsemblePullbackPredictor':
        """Load ensemble model."""
        # Load ensemble data
        ensemble_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(
            model_weights=ensemble_data['model_weights'],
            use_lstm=ensemble_data['use_lstm']
        )
        
        # Restore state
        instance.xgboost_model = ensemble_data['xgboost_model']
        instance.random_forest = ensemble_data['random_forest']
        instance.lstm_model = ensemble_data.get('lstm_model')
        instance.lstm_scaler = ensemble_data.get('lstm_scaler')
        instance.lstm_params = ensemble_data.get('lstm_params', {})
        instance.feature_names = ensemble_data['feature_names']
        instance.training_history = ensemble_data['training_history']
        instance.model_performances = ensemble_data['model_performances']
        instance.is_fitted = True
        
        # Update models dict
        instance.models = {
            'xgboost': instance.xgboost_model,
            'random_forest': instance.random_forest
        }
        if instance.use_lstm and instance.lstm_model:
            instance.models['lstm'] = instance.lstm_model
        
        print(f"   📁 Ensemble loaded from: {filepath}")
        return instance
    
    def __str__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"EnsemblePullbackPredictor({status}, {list(self.model_weights.keys())})"
    
    def __repr__(self) -> str:
        return f"EnsemblePullbackPredictor(models={list(self.model_weights.keys())})"