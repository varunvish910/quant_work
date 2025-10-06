"""
LSTM Integration Module for Existing Ensemble
This module extends your existing stacked ensemble to include LSTM capabilities
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Add the main project directory to path
main_project_path = '/Users/varun/code/quant_final_final/trade_and_quote_data'
sys.path.append(main_project_path)

class SimpleRecurrentClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple recurrent-like classifier using sklearn components
    Simulates LSTM behavior using feature engineering and ensemble methods
    """
    
    def __init__(self, lookback_window=20, random_state=42):
        self.lookback_window = lookback_window
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = None
        self.classes_ = None
        self.is_fitted = False
        
    def _create_lagged_features(self, X):
        """Create lagged features to simulate sequential dependencies"""
        if len(X) < self.lookback_window:
            # Not enough data, use what we have
            return X
        
        lagged_features = []
        
        for i in range(self.lookback_window, len(X)):
            # Get current features
            current = X[i].copy()
            
            # Add lagged features (last few timesteps)
            for lag in range(1, min(self.lookback_window + 1, i + 1)):
                lagged = X[i - lag]
                # Add statistical features of the lag
                current = np.concatenate([current, [
                    np.mean(lagged),  # Mean of lagged features
                    np.std(lagged),   # Std of lagged features
                    np.max(lagged),   # Max of lagged features
                    np.min(lagged),   # Min of lagged features
                ]])
            
            lagged_features.append(current)
        
        return np.array(lagged_features)
    
    def _create_sequence_features(self, X):
        """Create sequence-based features"""
        # Calculate rolling statistics
        df = pd.DataFrame(X)
        
        sequence_features = []
        
        for i in range(len(X)):
            # Get window of data
            start_idx = max(0, i - self.lookback_window + 1)
            window_data = df.iloc[start_idx:i+1]
            
            features = X[i].copy()
            
            if len(window_data) > 1:
                # Add sequence statistics
                features = np.concatenate([features, [
                    window_data.mean().mean(),    # Overall mean
                    window_data.std().mean(),     # Overall std
                    window_data.iloc[-1].mean() - window_data.iloc[0].mean(),  # Trend
                    len(window_data),             # Sequence length
                ]])
                
                # Add momentum features
                if len(window_data) >= 3:
                    recent_mean = window_data.iloc[-3:].mean().mean()
                    older_mean = window_data.iloc[:-3].mean().mean() if len(window_data) > 3 else recent_mean
                    momentum = recent_mean - older_mean
                    features = np.concatenate([features, [momentum]])
                else:
                    features = np.concatenate([features, [0]])
            else:
                # Pad with zeros for insufficient data
                features = np.concatenate([features, [0, 0, 0, 1, 0]])
            
            sequence_features.append(features)
        
        return np.array(sequence_features)
    
    def fit(self, X, y):
        """Fit the recurrent-like classifier"""
        self.classes_ = np.unique(y)
        
        # Create sequence features
        X_seq = self._create_sequence_features(X)
        
        # Align targets with sequence features
        if len(X_seq) < len(y):
            # Adjust targets to match sequence features
            y_seq = y[-len(X_seq):]
        else:
            y_seq = y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_seq)
        
        # Train multiple models for ensemble
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=self.random_state
        )
        self.models['rf_deep'] = RandomForestClassifier(
            n_estimators=50, max_depth=15, random_state=self.random_state + 1
        )
        
        # Train models
        for name, model in self.models.items():
            model.fit(X_scaled, y_seq)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Create sequence features
        X_seq = self._create_sequence_features(X)
        X_scaled = self.scaler.transform(X_seq)
        
        # Get predictions from all models
        all_probs = []
        for model in self.models.values():
            probs = model.predict_proba(X_scaled)
            all_probs.append(probs)
        
        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Handle length mismatch
        if len(avg_probs) < len(X):
            # Pad with the first prediction for missing samples
            padding = np.tile(avg_probs[0], (len(X) - len(avg_probs), 1))
            avg_probs = np.vstack([padding, avg_probs])
        
        return avg_probs
    
    def predict(self, X):
        """Predict classes"""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

def integrate_lstm_into_existing_ensemble():
    """
    Create a new stacked ensemble that includes the LSTM-like model
    This can be used as a drop-in replacement for the existing ensemble
    """
    
    try:
        # Try to import the existing stacked ensemble
        from core.stacked_ensemble import StackedEnsemble
        
        class LSTMEnhancedStackedEnsemble(StackedEnsemble):
            """Enhanced version of the existing stacked ensemble with LSTM"""
            
            def __init__(self, include_lstm=True):
                super().__init__()
                self.include_lstm = include_lstm
                
                if self.include_lstm:
                    self.lstm_model = SimpleRecurrentClassifier()
                    print("✅ LSTM-like model added to ensemble")
                
            def fit(self, X, y, X_val=None, y_val=None):
                """Train enhanced ensemble including LSTM"""
                print("Training enhanced stacked ensemble with LSTM-like model...")
                
                # Train base models (from parent class)
                super().fit(X, y, X_val, y_val)
                
                # Train LSTM-like model
                if self.include_lstm:
                    print("Training LSTM-like model...")
                    self.lstm_model.fit(X, y)
                    print("✅ LSTM-like model training completed")
                
                return self
            
            def predict_proba(self, X):
                """Enhanced prediction with LSTM"""
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                
                # Get base ensemble predictions
                base_probs = super().predict_proba(X)
                
                if not self.include_lstm:
                    return base_probs
                
                # Get LSTM predictions
                lstm_probs = self.lstm_model.predict_proba(X)
                
                # Weighted combination (you can adjust these weights)
                ensemble_weight = 0.7
                lstm_weight = 0.3
                
                combined_probs = (ensemble_weight * base_probs + 
                                lstm_weight * lstm_probs)
                
                return combined_probs
        
        return LSTMEnhancedStackedEnsemble
        
    except ImportError as e:
        print(f"Could not import existing StackedEnsemble: {e}")
        print("Creating standalone LSTM-enhanced ensemble...")
        
        class StandaloneLSTMEnsemble:
            """Standalone LSTM-enhanced ensemble"""
            
            def __init__(self):
                # Traditional models
                self.rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
                from xgboost import XGBClassifier
                self.xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
                import lightgbm as lgb
                self.lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=8, random_state=42, verbose=-1)
                
                # LSTM-like model
                self.lstm_model = SimpleRecurrentClassifier()
                
                # Meta-learner
                self.meta = LogisticRegression(random_state=42)
                
                self.is_fitted = False
            
            def fit(self, X, y, X_val=None, y_val=None):
                """Train ensemble"""
                print("Training standalone LSTM-enhanced ensemble...")
                
                # Train base models
                print("Training base models...")
                self.rf.fit(X, y)
                self.xgb.fit(X, y)
                self.lgb.fit(X, y)
                self.lstm_model.fit(X, y)
                
                # Get base predictions for meta-learner
                rf_pred = self.rf.predict_proba(X)[:, 1].reshape(-1, 1)
                xgb_pred = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)
                lgb_pred = self.lgb.predict_proba(X)[:, 1].reshape(-1, 1)
                lstm_pred = self.lstm_model.predict_proba(X)[:, 1].reshape(-1, 1)
                
                # Stack predictions
                meta_features = np.hstack([rf_pred, xgb_pred, lgb_pred, lstm_pred])
                
                # Train meta-learner
                print("Training meta-learner...")
                self.meta.fit(meta_features, y)
                
                self.is_fitted = True
                print("✅ LSTM-enhanced ensemble training completed")
                
                return self
            
            def predict_proba(self, X):
                """Predict probabilities"""
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                
                # Get base predictions
                rf_pred = self.rf.predict_proba(X)[:, 1].reshape(-1, 1)
                xgb_pred = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)
                lgb_pred = self.lgb.predict_proba(X)[:, 1].reshape(-1, 1)
                lstm_pred = self.lstm_model.predict_proba(X)[:, 1].reshape(-1, 1)
                
                # Stack and predict with meta-learner
                meta_features = np.hstack([rf_pred, xgb_pred, lgb_pred, lstm_pred])
                return self.meta.predict_proba(meta_features)
            
            def predict(self, X):
                """Predict classes"""
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        
        return StandaloneLSTMEnsemble

def test_lstm_integration():
    """Test the LSTM integration"""
    print("Testing LSTM integration with existing ensemble...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Generate features with some temporal patterns
    X = np.random.randn(n_samples, n_features)
    
    # Add some temporal dependencies to make the data more realistic
    for i in range(1, n_samples):
        X[i, :5] = 0.7 * X[i-1, :5] + 0.3 * np.random.randn(5)  # AR(1) process
    
    # Create target with temporal pattern
    y = np.zeros(n_samples)
    for i in range(20, n_samples):
        pattern = np.mean(X[i-20:i, 0]) + 0.5 * np.mean(X[i-10:i, 1])
        y[i] = (pattern > 0).astype(int)
    
    y = y.astype(int)
    
    print(f"Sample data: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Get enhanced ensemble class
    EnsembleClass = integrate_lstm_into_existing_ensemble()
    
    # Test the ensemble
    ensemble = EnsembleClass()
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    probs = ensemble.predict_proba(X_test)
    preds = ensemble.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(preds == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    print("✅ LSTM integration test completed successfully!")
    return ensemble

if __name__ == "__main__":
    test_lstm_integration()