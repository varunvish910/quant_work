#!/usr/bin/env python3
"""
XGBoost Pullback Predictor

XGBoost-based model for predicting pullbacks and mean reversion in momentum trading systems.
Optimized for high performance with robust preprocessing and feature selection.

USAGE:
======
from models.xgboost_predictor import XGBoostPullbackPredictor

# Basic usage
model = XGBoostPullbackPredictor()
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)

# Advanced usage with custom parameters
model = XGBoostPullbackPredictor(
    n_estimators=1500,
    max_depth=12,
    learning_rate=0.02,
    n_features=100
)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, log_loss
)
import xgboost as xgb
import joblib
import os
from typing import Tuple, Dict, Any, Optional, List, Union
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class XGBoostPullbackPredictor:
    """
    XGBoost-based pullback predictor for momentum trading systems.
    
    Features:
    - Optimized for pullback prediction with recall focus
    - Robust preprocessing with outlier handling
    - Feature selection using multiple methods
    - Cross-validation and hyperparameter optimization
    - Model persistence and versioning
    - Feature importance analysis
    """
    
    def __init__(self, 
                 n_estimators: int = 1000,
                 max_depth: int = 10,
                 learning_rate: float = 0.03,
                 n_features: int = 75,
                 feature_selection_method: str = 'f_classif',
                 scaling_method: str = 'robust',
                 random_state: int = 42,
                 class_weight: str = 'balanced',
                 early_stopping_rounds: int = 50):
        """
        Initialize XGBoost pullback predictor.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for boosting
            n_features: Number of features to select (None for all)
            feature_selection_method: 'f_classif' or 'mutual_info'
            scaling_method: 'robust', 'standard', or None
            random_state: Random state for reproducibility
            class_weight: 'balanced' or None for class weighting
            early_stopping_rounds: Early stopping rounds
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.feature_selection_method = feature_selection_method
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.class_weight = class_weight
        self.early_stopping_rounds = early_stopping_rounds
        
        # Initialize components
        self._initialize_model()
        self._initialize_preprocessors()
        
        # State variables
        self.feature_names = None
        self.selected_features = None
        self.is_fitted = False
        self.training_history = {}
        
        logger.info(f"XGBoostPullbackPredictor initialized: "
                   f"n_estimators={n_estimators}, max_depth={max_depth}, "
                   f"learning_rate={learning_rate}")
    
    def _initialize_model(self):
        """Initialize XGBoost model with optimized parameters."""
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = 3.0 if self.class_weight == 'balanced' else 1.0
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            
            # Optimized for pullback prediction
            objective='binary:logistic',
            eval_metric=['logloss', 'auc'],
            scale_pos_weight=scale_pos_weight,
            
            # Regularization
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=0.1,
            
            # Performance optimization
            tree_method='hist',
            max_bin=512,
            n_jobs=-1,
            
            # Early stopping
            early_stopping_rounds=self.early_stopping_rounds,
            
            # Reduce overfitting
            gamma=0.1
        )
    
    def _initialize_preprocessors(self):
        """Initialize preprocessing components."""
        # Scaler
        if self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        
        # Feature selector
        if self.n_features:
            if self.feature_selection_method == 'f_classif':
                self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            elif self.feature_selection_method == 'mutual_info':
                self.feature_selector = SelectKBest(mutual_info_classif, k=self.n_features)
            else:
                raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
        else:
            self.feature_selector = None
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str,
                    feature_cols: Optional[List[str]] = None,
                    exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for training/prediction.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (None for auto-detect)
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (X, y) arrays
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_list = [target_col] + (exclude_cols or [])
            
            # Common columns to exclude
            default_excludes = [
                'sip_timestamp', 'timestamp', 'date', 'datetime', 'Date',
                'symbol', 'ticker', 'index'
            ]
            exclude_list.extend([col for col in default_excludes if col in df.columns])
            
            feature_cols = [col for col in df.columns if col not in exclude_list]
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Remove rows with missing targets
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X.values, y.values
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill then backward fill
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining NaNs with median
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
        
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray,
           X_val: Optional[np.ndarray] = None,
           y_val: Optional[np.ndarray] = None,
           sample_weight: Optional[np.ndarray] = None) -> 'XGBoostPullbackPredictor':
        """
        Fit the XGBoost model with preprocessing.
        
        Args:
            X: Feature matrix
            y: Target vector
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (optional)
            
        Returns:
            Self for method chaining
        """
        print(f"\n🚀 Training XGBoost Pullback Predictor...")
        print(f"   • Training samples: {X.shape[0]:,}")
        print(f"   • Features: {X.shape[1]:,}")
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        # Select features
        X_selected = self._select_features(X_scaled, y, fit=True)
        
        # Prepare validation set if provided
        eval_set = []
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scale_features(X_val, fit=False)
            X_val_selected = self._select_features(X_val_scaled, None, fit=False)
            eval_set = [(X_selected, y), (X_val_selected, y_val)]
            print(f"   • Validation samples: {X_val.shape[0]:,}")
        else:
            eval_set = [(X_selected, y)]
        
        # Train model
        print(f"   • Training with {self.n_estimators} estimators...")
        
        self.model.fit(
            X_selected, y,
            eval_set=eval_set,
            sample_weight=sample_weight,
            verbose=False
        )
        
        # Store training history
        self.training_history = {
            'train_samples': X.shape[0],
            'features_original': X.shape[1],
            'features_selected': X_selected.shape[1],
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }
        
        self.is_fitted = True
        
        print(f"   ✅ Training completed!")
        print(f"      • Best iteration: {self.model.best_iteration}")
        print(f"      • Best score: {self.model.best_score:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Preprocess features
        X_scaled = self._scale_features(X, fit=False)
        X_selected = self._select_features(X_scaled, None, fit=False)
        
        return self.model.predict(X_selected)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Preprocess features
        X_scaled = self._scale_features(X, fit=False)
        X_selected = self._select_features(X_scaled, None, fit=False)
        
        return self.model.predict_proba(X_selected)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # Get importance scores
        if importance_type == 'gain':
            importance_scores = self.model.feature_importances_
        else:
            importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
            # Convert to array format
            feature_names = [f'f{i}' for i in range(len(self.selected_features))]
            importance_scores = np.array([importance_dict.get(name, 0) for name in feature_names])
        
        # Map to original feature names
        if self.selected_features is not None:
            selected_feature_names = [self.feature_names[i] for i in self.selected_features]
        else:
            selected_feature_names = self.feature_names
        
        # Create importance dictionary
        importance_dict = dict(zip(selected_feature_names, importance_scores))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y, y_proba),
            'avg_precision': average_precision_score(y, y_proba),
            'log_loss': log_loss(y, y_proba),
            'accuracy': (y_pred == y).mean(),
            'precision': ((y_pred == 1) & (y == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0,
            'recall': ((y_pred == 1) & (y == 1)).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 0,
            'f1_score': 2 * metrics.get('precision', 0) * metrics.get('recall', 0) / (metrics.get('precision', 0) + metrics.get('recall', 0)) if (metrics.get('precision', 0) + metrics.get('recall', 0)) > 0 else 0
        }
        
        # Fix F1 calculation
        precision = metrics['precision']
        recall = metrics['recall']
        if precision + recall > 0:
            metrics['f1_score'] = 2 * precision * recall / (precision + recall)
        else:
            metrics['f1_score'] = 0
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = 5,
                      scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        print(f"\n📊 Running {cv_folds}-fold cross-validation...")
        
        # Preprocess features
        X_scaled = self._scale_features(X, fit=True)
        X_selected = self._select_features(X_scaled, y, fit=True)
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        print(f"   📈 {scoring.upper()}: {results['mean']:.4f} ± {results['std']:.4f}")
        
        return results
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features using configured scaler."""
        if self.scaler is None:
            return X
        
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def _select_features(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                        fit: bool = False) -> np.ndarray:
        """Select features using configured selector."""
        if self.feature_selector is None:
            return X
        
        if fit:
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = self.feature_selector.get_support(indices=True)
            print(f"   • Selected {len(self.selected_features)} features from {X.shape[1]}")
            return X_selected
        else:
            return self.feature_selector.transform(X)
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None) -> str:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            metadata: Additional metadata to save
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'training_history': self.training_history,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_features': self.n_features,
                'feature_selection_method': self.feature_selection_method,
                'scaling_method': self.scaling_method,
                'class_weight': self.class_weight
            },
            'metadata': metadata or {}
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(model_data, filepath)
        
        print(f"   💾 Model saved to: {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str) -> 'XGBoostPullbackPredictor':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create instance
        hyperparams = model_data['hyperparameters']
        instance = cls(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            learning_rate=hyperparams['learning_rate'],
            n_features=hyperparams['n_features'],
            feature_selection_method=hyperparams['feature_selection_method'],
            scaling_method=hyperparams['scaling_method'],
            class_weight=hyperparams['class_weight']
        )
        
        # Restore state
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_selector = model_data['feature_selector']
        instance.feature_names = model_data['feature_names']
        instance.selected_features = model_data['selected_features']
        instance.training_history = model_data['training_history']
        instance.is_fitted = True
        
        print(f"   📁 Model loaded from: {filepath}")
        return instance
    
    def __str__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"XGBoostPullbackPredictor({status}, {self.n_estimators} estimators)"
    
    def __repr__(self) -> str:
        return (f"XGBoostPullbackPredictor(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, learning_rate={self.learning_rate})")