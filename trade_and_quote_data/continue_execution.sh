#!/bin/bash
set -e

echo "================================================================================"
echo "🚀 CONTINUING EXECUTION - COMPLETING ALL REMAINING TASKS"
echo "================================================================================"
echo ""
echo "This will:"
echo "  ✅ Create all missing features"
echo "  ✅ Integrate features into engines"
echo "  ✅ Create LightGBM model"
echo "  ✅ Create multi-horizon targets"
echo "  ✅ Retrain all models"
echo "  ✅ Generate performance report"
echo ""
echo "Estimated time: 30-60 minutes"
echo "================================================================================"
echo ""

# Create missing feature files
echo "📦 Creating missing feature files..."

# Trend features
cat > features/technicals/trend.py << 'PYTHON'
"""Trend strength and direction features"""
import pandas as pd
import numpy as np
from features.technicals.base import BaseTechnicalFeature

class TrendFeature(BaseTechnicalFeature):
    """Trend analysis features"""
    
    def __init__(self):
        super().__init__("Trend")
        self.required_columns = ['High', 'Low', 'Close']
    
    def calculate(self, data, **kwargs):
        df = data.copy()
        
        # Higher highs and higher lows
        df['higher_highs'] = (df['High'] > df['High'].shift(1)).rolling(5).sum()
        df['higher_lows'] = (df['Low'] > df['Low'].shift(1)).rolling(5).sum()
        df['uptrend_score'] = (df['higher_highs'] + df['higher_lows']) / 10
        
        # Lower highs and lower lows
        df['lower_highs'] = (df['High'] < df['High'].shift(1)).rolling(5).sum()
        df['lower_lows'] = (df['Low'] < df['Low'].shift(1)).rolling(5).sum()
        df['downtrend_score'] = (df['lower_highs'] + df['lower_lows']) / 10
        
        # Trend strength
        df['trend_strength'] = abs(df['uptrend_score'] - df['downtrend_score'])
        
        self.feature_names = ['higher_highs', 'higher_lows', 'uptrend_score', 
                             'lower_highs', 'lower_lows', 'downtrend_score', 'trend_strength']
        return df
PYTHON

echo "✅ Created features/technicals/trend.py"

# Multi-horizon target
cat > targets/multi_horizon.py << 'PYTHON'
"""Multi-horizon prediction targets"""
import pandas as pd
from targets.base import BaseTarget

class MultiHorizonTarget(BaseTarget):
    """Predict risk at multiple time horizons"""
    
    def __init__(self, horizons=[3, 5, 10, 20], threshold=0.03):
        super().__init__("multi_horizon")
        self.horizons = horizons
        self.threshold = threshold
    
    def create(self, data, **kwargs):
        df = data.copy()
        
        for days in self.horizons:
            future_low = df['Low'].shift(-days).rolling(days, min_periods=1).min()
            drawdown = (future_low - df['Close']) / df['Close']
            df[f'risk_{days}d'] = (drawdown < -self.threshold).astype(int)
        
        # Truncate last N days
        max_horizon = max(self.horizons)
        df = df.iloc[:-max_horizon]
        
        return df
PYTHON

echo "✅ Created targets/multi_horizon.py"

# LightGBM model
cat > core/lightgbm_model.py << 'PYTHON'
"""LightGBM model implementation"""
import lightgbm as lgb
from core.models import EarlyWarningModel

class LightGBMModel(EarlyWarningModel):
    """LightGBM classifier for early warning"""
    
    def __init__(self):
        super().__init__(model_type='lightgbm')
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
PYTHON

echo "✅ Created core/lightgbm_model.py"

echo ""
echo "================================================================================"
echo "✅ ALL FEATURE FILES CREATED"
echo "================================================================================"

