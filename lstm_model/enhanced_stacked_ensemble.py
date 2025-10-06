"""
Enhanced Stacked Ensemble with LSTM Integration
Extends the existing stacked ensemble to include LSTM predictions
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add the main project directory to path
sys.path.append('/Users/varun/code/quant_final_final/trade_and_quote_data')

try:
    # Try to import TensorFlow, but make it optional
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    
    # Set TensorFlow to use less resources
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
        
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, LSTM will be disabled")

class SimpleLSTMClassifier:
    """
    Simplified LSTM classifier for ensemble integration
    """
    
    def __init__(self, sequence_length=20, hidden_size=32, epochs=20, batch_size=32):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
        self.is_fitted = False
        
    def _create_sequences(self, X, y=None):
        """Create sequences for LSTM"""
        if len(X) < self.sequence_length:
            # If not enough data, return the data as is with padding
            if y is not None:
                return X.reshape(1, -1, X.shape[1]), np.array([y[-1]])
            return X.reshape(1, -1, X.shape[1])
        
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            if y is not None:
                targets.append(y[i])
        
        sequences = np.array(sequences)
        if y is not None:
            targets = np.array(targets)
            return sequences, targets
        return sequences
    
    def fit(self, X, y):
        """Fit the LSTM model"""
        if not TF_AVAILABLE:
            # Fallback to simple classifier
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            self.model.fit(X, y)
            self.classes_ = np.unique(y)
            self.is_fitted = True
            return self
        
        self.classes_ = np.unique(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Build simple LSTM model
        model = keras.Sequential([
            keras.Input(shape=(self.sequence_length, X.shape[1])),
            layers.LSTM(self.hidden_size, dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with reduced verbosity
        model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        self.model = model
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if not TF_AVAILABLE:
            # Use fallback model
            probs = self.model.predict_proba(X)
            return probs
        
        # Scale and create sequences
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        
        # Predict
        preds = self.model.predict(X_seq, verbose=0)
        
        # Convert to probability format
        probs = np.column_stack([1 - preds.flatten(), preds.flatten()])
        
        # Handle sequence padding if necessary
        if len(probs) < len(X):
            # Pad with mean probabilities for early samples
            mean_prob = np.mean(probs, axis=0)
            padding = np.tile(mean_prob, (len(X) - len(probs), 1))
            probs = np.vstack([padding, probs])
        
        return probs
    
    def predict(self, X):
        """Predict classes"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

class EnhancedStackedEnsemble:
    """
    Enhanced stacked ensemble that includes LSTM along with traditional models
    """
    
    def __init__(self, include_lstm=True):
        # Level 1: Base models
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.xgb = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
        self.lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=8, random_state=42, verbose=-1)
        
        # LSTM model (optional)
        self.include_lstm = include_lstm and TF_AVAILABLE
        if self.include_lstm:
            self.lstm = SimpleLSTMClassifier()
        
        # Level 2: Meta-learner
        self.meta = LogisticRegression(random_state=42, max_iter=1000)
        
        self.is_fitted = False
        self.model_names = ['RF', 'XGB', 'LGB']
        if self.include_lstm:
            self.model_names.append('LSTM')
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train enhanced stacked ensemble"""
        print(f"Training enhanced ensemble with models: {', '.join(self.model_names)}")
        
        # Train base models
        print("Training base models...")
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.lgb.fit(X, y)
        
        # Get predictions from traditional models
        base_predictions = []
        base_predictions.append(self.rf.predict_proba(X)[:, 1].reshape(-1, 1))
        base_predictions.append(self.xgb.predict_proba(X)[:, 1].reshape(-1, 1))
        base_predictions.append(self.lgb.predict_proba(X)[:, 1].reshape(-1, 1))
        
        # Train LSTM if available
        if self.include_lstm:
            print("Training LSTM model...")
            try:
                self.lstm.fit(X, y)
                lstm_pred = self.lstm.predict_proba(X)[:, 1].reshape(-1, 1)
                base_predictions.append(lstm_pred)
                print("✅ LSTM training completed")
            except Exception as e:
                print(f"❌ LSTM training failed: {e}")
                self.include_lstm = False
                self.model_names = self.model_names[:-1]  # Remove LSTM from list
        
        # Stack predictions
        meta_features = np.hstack(base_predictions)
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta.fit(meta_features, y)
        
        self.is_fitted = True
        print(f"✅ Enhanced ensemble trained with {len(self.model_names)} models")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get base model predictions
        base_predictions = []
        base_predictions.append(self.rf.predict_proba(X)[:, 1].reshape(-1, 1))
        base_predictions.append(self.xgb.predict_proba(X)[:, 1].reshape(-1, 1))
        base_predictions.append(self.lgb.predict_proba(X)[:, 1].reshape(-1, 1))
        
        # Add LSTM predictions if available
        if self.include_lstm:
            lstm_pred = self.lstm.predict_proba(X)[:, 1].reshape(-1, 1)
            base_predictions.append(lstm_pred)
        
        # Stack and predict with meta-learner
        meta_features = np.hstack(base_predictions)
        return self.meta.predict_proba(meta_features)
    
    def predict(self, X):
        """Predict classes"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
    
    def get_feature_importance(self):
        """Get feature importance from base models"""
        importance_dict = {}
        
        if hasattr(self.rf, 'feature_importances_'):
            importance_dict['RF'] = self.rf.feature_importances_
        if hasattr(self.xgb, 'feature_importances_'):
            importance_dict['XGB'] = self.xgb.feature_importances_
        if hasattr(self.lgb, 'feature_importances_'):
            importance_dict['LGB'] = self.lgb.feature_importances_
            
        return importance_dict
    
    def get_meta_weights(self):
        """Get meta-learner weights"""
        if hasattr(self.meta, 'coef_'):
            return dict(zip(self.model_names, self.meta.coef_[0]))
        return None

def test_enhanced_ensemble():
    """Test the enhanced ensemble with sample data"""
    print("Testing Enhanced Stacked Ensemble...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create some pattern for the target
    y = ((X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5) > 0).astype(int)
    
    print(f"Sample data: {X.shape}, Target distribution: {np.bincount(y)}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test enhanced ensemble
    ensemble = EnhancedStackedEnsemble(include_lstm=True)
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    probs = ensemble.predict_proba(X_test)
    preds = ensemble.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(preds == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Show meta-learner weights
    weights = ensemble.get_meta_weights()
    if weights:
        print("Meta-learner weights:")
        for model, weight in weights.items():
            print(f"  {model}: {weight:.4f}")
    
    print("✅ Enhanced ensemble test completed!")
    return ensemble

if __name__ == "__main__":
    test_enhanced_ensemble()