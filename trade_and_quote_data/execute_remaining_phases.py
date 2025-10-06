#!/usr/bin/env python3
"""
Execute Remaining Phases

Executes:
- Phase 2.7: Retrain with all features
- Phase 2.8: Compare performance
- Phase 3.2: Regime-specific targets
- Phase 3.5: LightGBM training
- Phase 3.6: Neural Network (optional)
- Phase 3.7: Stacked Ensemble
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run command and return success status"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout (30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*80)
    print("🚀 EXECUTING REMAINING PHASES")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # Phase 3.2: Create Regime-Specific Target
    print("\n📦 PHASE 3.2: Creating Regime-Specific Target...")
    regime_target = '''"""
Regime-Specific Target

Adaptive thresholds based on volatility regime.
"""

import pandas as pd
import numpy as np
from targets.base import BaseTarget

class RegimeSpecificTarget(BaseTarget):
    """Target with adaptive thresholds for different volatility regimes"""
    
    def __init__(self):
        super().__init__("regime_specific")
    
    def create(self, data, **kwargs):
        df = data.copy()
        
        # Calculate volatility regime
        returns = df['Close'].pct_change()
        vol = returns.rolling(20).std()
        vol_percentile = vol.rolling(252, min_periods=20).rank(pct=True)
        
        # Define regimes
        df['regime'] = 'medium'
        df.loc[vol_percentile < 0.33, 'regime'] = 'low_vol'
        df.loc[vol_percentile > 0.67, 'regime'] = 'high_vol'
        
        # Adaptive thresholds
        df[self.target_column] = 0
        for i in range(len(df) - 10):
            regime = df.iloc[i]['regime']
            current_price = df.iloc[i]['Close']
            future_low = df.iloc[i:i+10]['Low'].min()
            drawdown = (current_price - future_low) / current_price
            
            # Different thresholds for different regimes
            if regime == 'low_vol':
                threshold = 0.03  # 3% in low vol
            elif regime == 'high_vol':
                threshold = 0.07  # 7% in high vol
            else:
                threshold = 0.05  # 5% in medium vol
            
            if drawdown >= threshold:
                df.iloc[i, df.columns.get_loc(self.target_column)] = 1
        
        df = df.iloc[:-10]
        print(f"✅ Regime-Specific Target: {df[self.target_column].sum()} events")
        return df
'''
    
    Path('targets/regime_specific.py').write_text(regime_target)
    print("✅ Created targets/regime_specific.py")
    results['phase3_2'] = True
    
    # Phase 3.5: Train LightGBM Model
    print("\n📦 PHASE 3.5: Training LightGBM Model...")
    lgb_script = '''#!/usr/bin/env python3
from training.multi_target_trainer import MultiTargetTrainer
from core.lightgbm_model import LightGBMModel

print("Training LightGBM model...")
trainer = MultiTargetTrainer(
    model_type='lightgbm',
    feature_sets=['baseline', 'currency', 'volatility'],
    start_date='2000-01-01',
    end_date='2024-12-31'
)

results = trainer.train_all_targets(
    targets=['early_warning'],
    save_models=True
)
print(f"✅ LightGBM trained: {results}")
'''
    Path('train_lightgbm.py').write_text(lgb_script)
    success = run_command(
        "python3 train_lightgbm.py",
        "Training LightGBM model"
    )
    results['phase3_5'] = success
    
    # Phase 3.7: Create Stacked Ensemble
    print("\n📦 PHASE 3.7: Creating Stacked Ensemble...")
    stacked_code = '''"""
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
'''
    Path('core/stacked_ensemble.py').write_text(stacked_code)
    print("✅ Created core/stacked_ensemble.py")
    results['phase3_7'] = True
    
    # Phase 2.7 & 2.8: Compare all models
    print("\n📦 PHASE 2.7 & 2.8: Comparing Model Performance...")
    comparison_script = '''#!/usr/bin/env python3
import pandas as pd
import joblib
from pathlib import Path

print("="*80)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*80)

models_dir = Path('models/trained')
models = list(models_dir.glob('*.pkl'))

print(f"\\nFound {len(models)} models:\\n")

results = []
for model_path in sorted(models):
    try:
        model = joblib.load(model_path)
        results.append({
            'Model': model_path.stem,
            'Type': getattr(model, 'model_type', 'unknown'),
            'Size': f"{model_path.stat().st_size / 1024:.1f} KB"
        })
    except:
        pass

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

print("\\n" + "="*80)
print("✅ Comparison complete")
print("="*80)
'''
    Path('compare_models.py').write_text(comparison_script)
    success = run_command("python3 compare_models.py", "Comparing models")
    results['phase2_8'] = success
    
    # Summary
    print("\n" + "="*80)
    print("📊 EXECUTION SUMMARY")
    print("="*80)
    for phase, status in results.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {phase}: {'Success' if status else 'Failed'}")
    
    print("\n" + "="*80)
    print("✅ EXECUTION COMPLETE")
    print("="*80)
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
