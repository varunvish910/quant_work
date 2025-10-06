"""
LSTM-Enhanced Stacked Ensemble
Extends the existing stacked ensemble to include LSTM-like sequential modeling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from .stacked_ensemble import StackedEnsemble

class SequentialClassifier(BaseEstimator, ClassifierMixin):
    """
    Sequential classifier that captures temporal patterns
    Uses feature engineering to simulate LSTM-like behavior
    """
    
    def __init__(self, lookback_window=20, random_state=42):
        self.lookback_window = lookback_window
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.classes_ = None
        self.is_fitted = False
        
    def _create_sequence_features(self, X):
        """Create sequence-based features from tabular data"""
        df = pd.DataFrame(X)
        sequence_features = []
        
        for i in range(len(X)):
            # Get window of data
            start_idx = max(0, i - self.lookback_window + 1)
            window_data = df.iloc[start_idx:i+1]
            
            # Start with current features
            features = X[i].copy()
            
            if len(window_data) > 1:
                # Add rolling statistics
                features = np.concatenate([features, [
                    window_data.mean().mean(),    # Overall window mean
                    window_data.std().mean(),     # Overall window std
                    window_data.iloc[-1].mean() - window_data.iloc[0].mean(),  # Trend
                    len(window_data),             # Effective window length
                ]])
                
                # Add momentum features (short vs long term)
                if len(window_data) >= 5:
                    recent_mean = window_data.iloc[-3:].mean().mean()
                    older_mean = window_data.iloc[:-3].mean().mean()
                    momentum = recent_mean - older_mean
                    features = np.concatenate([features, [momentum]])
                    
                    # Add volatility features
                    volatility = window_data.std().std()
                    features = np.concatenate([features, [volatility]])
                else:
                    features = np.concatenate([features, [0, 0]])
                    
                # Add percentile features
                features = np.concatenate([features, [
                    (window_data.iloc[-1] > window_data.quantile(0.8)).mean(),  # Above 80th percentile
                    (window_data.iloc[-1] < window_data.quantile(0.2)).mean(),  # Below 20th percentile
                ]])
            else:
                # Pad with zeros for insufficient data
                features = np.concatenate([features, [0, 0, 0, 1, 0, 0, 0, 0]])
            
            sequence_features.append(features)
        
        return np.array(sequence_features)
    
    def fit(self, X, y):
        """Fit the sequential classifier"""
        self.classes_ = np.unique(y)
        
        # Create sequence features
        X_seq = self._create_sequence_features(X)
        
        # Align targets (sequence features might be shorter)
        y_seq = y[-len(X_seq):] if len(X_seq) < len(y) else y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_seq)
        
        # Train ensemble of models for robustness
        self.models['rf_seq'] = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=5,
            random_state=self.random_state
        )
        self.models['rf_deep'] = RandomForestClassifier(
            n_estimators=50, max_depth=20, min_samples_split=3,
            random_state=self.random_state + 1
        )
        
        # Train models
        for model in self.models.values():
            model.fit(X_scaled, y_seq)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using sequential features"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_seq = self._create_sequence_features(X)
        X_scaled = self.scaler.transform(X_seq)
        
        # Average predictions from all models
        all_probs = []
        for model in self.models.values():
            probs = model.predict_proba(X_scaled)
            all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        
        # Handle length mismatch by padding
        if len(avg_probs) < len(X):
            first_prob = avg_probs[0] if len(avg_probs) > 0 else np.array([0.5, 0.5])
            padding = np.tile(first_prob, (len(X) - len(avg_probs), 1))
            avg_probs = np.vstack([padding, avg_probs])
        
        return avg_probs
    
    def predict(self, X):
        """Predict classes"""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

class LSTMEnhancedStackedEnsemble(StackedEnsemble):
    """
    Enhanced version of StackedEnsemble that includes sequential modeling
    Drop-in replacement for the original StackedEnsemble
    """
    
    def __init__(self, include_sequential=True, sequential_weight=0.25):
        super().__init__()
        self.include_sequential = include_sequential
        self.sequential_weight = sequential_weight
        
        if self.include_sequential:
            self.sequential_model = SequentialClassifier()
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train enhanced ensemble with sequential model"""
        print("Training LSTM-enhanced stacked ensemble...")
        
        # Train base ensemble
        super().fit(X, y, X_val, y_val)
        
        # Train sequential model
        if self.include_sequential:
            print("Training sequential model...")
            self.sequential_model.fit(X, y)
            print("✅ Sequential model training completed")
        
        return self
    
    def predict_proba(self, X):
        """Enhanced prediction combining base ensemble with sequential model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get base ensemble predictions
        base_probs = super().predict_proba(X)
        
        if not self.include_sequential:
            return base_probs
        
        # Get sequential model predictions
        seq_probs = self.sequential_model.predict_proba(X)
        
        # Weighted combination
        base_weight = 1 - self.sequential_weight
        combined_probs = (base_weight * base_probs + 
                         self.sequential_weight * seq_probs)
        
        return combined_probs
    
    def get_model_weights(self):
        """Get the contribution weights of different models"""
        weights = {
            'base_ensemble': 1 - self.sequential_weight,
            'sequential_model': self.sequential_weight if self.include_sequential else 0
        }
        return weights

def compare_ensemble_performance():
    """
    Compare the performance of original vs LSTM-enhanced ensemble
    """
    print("Comparing ensemble performance...")
    
    # Generate sample time series data
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    # Create features with temporal dependencies
    X = np.random.randn(n_samples, n_features)
    
    # Add temporal patterns
    for i in range(1, n_samples):
        # AR(1) process for some features
        X[i, :5] = 0.6 * X[i-1, :5] + 0.4 * np.random.randn(5)
        # Moving average for others
        if i >= 10:
            X[i, 5:10] = 0.3 * np.mean(X[i-10:i, 5:10], axis=0) + 0.7 * np.random.randn(5)
    
    # Create target with temporal dependencies
    y = np.zeros(n_samples)
    for i in range(30, n_samples):
        # Target depends on recent patterns
        pattern_score = (
            0.3 * np.mean(X[i-20:i, 0]) +
            0.2 * np.mean(X[i-10:i, 1]) +
            0.1 * (X[i, 2] - X[i-5, 2]) +
            0.1 * np.std(X[i-15:i, 3]) +
            np.random.normal(0, 0.1)
        )
        y[i] = (pattern_score > 0).astype(int)
    
    y = y.astype(int)
    
    print(f"Dataset: {X.shape}, Target distribution: {np.bincount(y)}")
    
    # Split data chronologically
    split_idx = int(0.7 * len(X))
    val_split = int(0.85 * len(X))
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:val_split]
    y_val = y[split_idx:val_split]
    X_test = X[val_split:]
    y_test = y[val_split:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Test original ensemble
    print("\n=== Original Stacked Ensemble ===")
    original_ensemble = StackedEnsemble()
    original_ensemble.fit(X_train, y_train, X_val, y_val)
    
    original_probs = original_ensemble.predict_proba(X_test)
    original_preds = original_ensemble.predict(X_test)
    original_accuracy = np.mean(original_preds == y_test)
    
    print(f"Original ensemble accuracy: {original_accuracy:.4f}")
    
    # Test enhanced ensemble
    print("\n=== LSTM-Enhanced Stacked Ensemble ===")
    enhanced_ensemble = LSTMEnhancedStackedEnsemble()
    enhanced_ensemble.fit(X_train, y_train, X_val, y_val)
    
    enhanced_probs = enhanced_ensemble.predict_proba(X_test)
    enhanced_preds = enhanced_ensemble.predict(X_test)
    enhanced_accuracy = np.mean(enhanced_preds == y_test)
    
    print(f"Enhanced ensemble accuracy: {enhanced_accuracy:.4f}")
    
    # Show improvement
    improvement = enhanced_accuracy - original_accuracy
    print(f"\nAccuracy improvement: {improvement:.4f} ({improvement/original_accuracy*100:.2f}%)")
    
    # Show model weights
    weights = enhanced_ensemble.get_model_weights()
    print("Model contribution weights:")
    for model, weight in weights.items():
        print(f"  {model}: {weight:.3f}")
    
    return {
        'original_accuracy': original_accuracy,
        'enhanced_accuracy': enhanced_accuracy,
        'improvement': improvement
    }

if __name__ == "__main__":
    results = compare_ensemble_performance()
    print(f"\n✅ LSTM enhancement test completed!")
    print(f"Final improvement: {results['improvement']:.4f}")