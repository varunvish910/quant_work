"""
Stacked Ensemble Model

Combines RF, XGBoost, and LightGBM with a meta-learner.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

class StackedEnsemble:
    """Stacked ensemble with meta-learner"""
    
    def __init__(self):
        # Level 1: Base models
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        self.xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=8, random_state=42, verbose=-1)
        
        # Level 2: Meta-learner
        self.meta = LogisticRegression(random_state=42)
        
        self.is_fitted = False
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train stacked ensemble"""
        print("Training base models...")
        
        # Train base models
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.lgb.fit(X, y)
        
        # Get predictions from base models
        rf_pred = self.rf.predict_proba(X)[:, 1].reshape(-1, 1)
        xgb_pred = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)
        lgb_pred = self.lgb.predict_proba(X)[:, 1].reshape(-1, 1)
        
        # Stack predictions
        meta_features = np.hstack([rf_pred, xgb_pred, lgb_pred])
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta.fit(meta_features, y)
        
        self.is_fitted = True
        print("✅ Stacked ensemble trained")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get base model predictions
        rf_pred = self.rf.predict_proba(X)[:, 1].reshape(-1, 1)
        xgb_pred = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)
        lgb_pred = self.lgb.predict_proba(X)[:, 1].reshape(-1, 1)
        
        # Stack and predict with meta-learner
        meta_features = np.hstack([rf_pred, xgb_pred, lgb_pred])
        return self.meta.predict_proba(meta_features)
    
    def predict(self, X):
        """Predict classes"""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
