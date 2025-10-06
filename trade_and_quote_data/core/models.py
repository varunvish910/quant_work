"""
Model Definitions

All model classes for the early warning system.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from typing import Dict, List, Optional
import joblib
import json
from datetime import datetime

from utils.constants import (
    RF_PARAMS, XGB_PARAMS, ENSEMBLE_WEIGHTS,
    MODEL_REGISTRY_PATH, MODEL_METADATA_FILE, FEATURE_COLUMNS_FILE
)


class EarlyWarningModel:
    """Base class for early warning models"""
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize model
        
        Args:
            model_type: Type of model ('rf', 'xgboost', 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.metadata = {
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'training_config': {}
        }
    
    def build_model(self) -> any:
        """Build the model"""
        if self.model_type == 'rf':
            return self._build_random_forest()
        elif self.model_type == 'xgboost':
            return self._build_xgboost()
        elif self.model_type == 'ensemble':
            return self._build_ensemble()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_random_forest(self) -> RandomForestClassifier:
        """Build Random Forest model"""
        return RandomForestClassifier(**RF_PARAMS)
    
    def _build_xgboost(self) -> XGBClassifier:
        """Build XGBoost model"""
        return XGBClassifier(**XGB_PARAMS)
    
    def _build_ensemble(self) -> VotingClassifier:
        """Build ensemble of RF and XGBoost"""
        rf = self._build_random_forest()
        xgb = self._build_xgboost()
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft',
            weights=ENSEMBLE_WEIGHTS
        )
        
        return ensemble
    
    def fit(self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str]) -> 'EarlyWarningModel':
        """
        Train the model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_columns: List of feature column names
            
        Returns:
            Self for chaining
        """
        if self.model is None:
            self.model = self.build_model()
        
        self.feature_columns = feature_columns
        
        print(f"🎓 Training {self.model_type} model...")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Training samples: {len(X)}")
        print(f"   Positive rate: {y.mean():.2%}")
        
        # Train model
        self.model.fit(X[feature_columns], y)
        
        # Update metadata
        self.metadata['feature_count'] = len(feature_columns)
        self.metadata['training_samples'] = len(X)
        self.metadata['positive_rate'] = float(y.mean())
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        print(f"✅ Model trained successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X[self.feature_columns])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X[self.feature_columns])
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'ensemble':
            # Average importance across ensemble members
            rf_importance = self.model.named_estimators_['rf'].feature_importances_
            xgb_importance = self.model.named_estimators_['xgb'].feature_importances_
            importance = (rf_importance + xgb_importance) / 2
        else:
            importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, directory: str = MODEL_REGISTRY_PATH) -> None:
        """
        Save model to disk
        
        Args:
            directory: Directory to save model files
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        model_path = os.path.join(directory, f'{self.model_type}_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Save feature columns
        features_path = os.path.join(directory, FEATURE_COLUMNS_FILE)
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        print(f"💾 Features saved: {features_path}")
        
        # Save metadata
        metadata_path = os.path.join(directory, MODEL_METADATA_FILE)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Metadata saved: {metadata_path}")
    
    @classmethod
    def load(cls, directory: str = MODEL_REGISTRY_PATH, 
             model_type: str = 'ensemble') -> 'EarlyWarningModel':
        """
        Load model from disk
        
        Args:
            directory: Directory containing model files
            model_type: Type of model to load
            
        Returns:
            Loaded model instance
        """
        import os
        
        # Load model
        model_path = os.path.join(directory, f'{model_type}_model.pkl')
        model_obj = cls(model_type=model_type)
        model_obj.model = joblib.load(model_path)
        print(f"📂 Model loaded: {model_path}")
        
        # Load feature columns
        features_path = os.path.join(directory, FEATURE_COLUMNS_FILE)
        with open(features_path, 'r') as f:
            model_obj.feature_columns = json.load(f)
        print(f"📂 Features loaded: {len(model_obj.feature_columns)} features")
        
        # Load metadata
        metadata_path = os.path.join(directory, MODEL_METADATA_FILE)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_obj.metadata = json.load(f)
            print(f"📂 Metadata loaded")
        
        return model_obj


if __name__ == "__main__":
    # Test model creation
    print("Testing model classes...")
    
    # Create dummy data
    X = pd.DataFrame(np.random.randn(100, 10), 
                     columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 100))
    feature_columns = list(X.columns)
    
    # Test each model type
    for model_type in ['rf', 'xgboost', 'ensemble']:
        print(f"\n{'='*80}")
        print(f"Testing {model_type} model")
        print('='*80)
        
        model = EarlyWarningModel(model_type=model_type)
        model.fit(X, y, feature_columns)
        
        # Test prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Probabilities shape: {probabilities.shape}")
        
        # Test feature importance
        importance = model.get_feature_importance()
        print(f"\nTop 3 features:")
        print(importance.head(3))

