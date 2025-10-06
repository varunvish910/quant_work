# Model Improvement Roadmap: SPY Pullback Prediction System

## Executive Summary

The current model has critical issues with 2024 predictions and shows signs of high false positive rates. This document provides a comprehensive phased approach to fix, improve, and scale the model system.

### Key Problems Identified:
1. **Poor 2024 Performance**: Model is not catching major 2024 drawdowns (April, August crash, December)
2. **Target Window Too Narrow**: Using 3-5 days when patterns show 7-8 days lead time
3. **High False Positive Rate**: 85%+ confidence signals have >50% false positive rate
4. **Feature Noise**: 139 features without proper selection/engineering
5. **Code Organization**: Scattered files need consolidation

### Alternative Approach: Start From Scratch
Given the complexity and issues in the current system, we recommend a clean-slate approach with systematic multi-target analysis to find the optimal prediction parameters.

---

## Phase 0: Clean Slate Multi-Target Analysis (Recommended First Step)

### 0.1 Multi-Target Grid Search
Instead of fixing the existing system, start fresh with a systematic analysis of all possible targets:

**Pullback Magnitudes**: 2%, 5%, 10%
**Time Horizons**: 5, 10, 15, 20 days
**Total Combinations**: 12 different prediction targets

```python
# Define target grid
targets = {
    'pullback_2pct_5d': {'magnitude': 0.02, 'days': 5},
    'pullback_2pct_10d': {'magnitude': 0.02, 'days': 10},
    'pullback_2pct_15d': {'magnitude': 0.02, 'days': 15},
    'pullback_2pct_20d': {'magnitude': 0.02, 'days': 20},
    'pullback_5pct_5d': {'magnitude': 0.05, 'days': 5},
    'pullback_5pct_10d': {'magnitude': 0.05, 'days': 10},
    'pullback_5pct_15d': {'magnitude': 0.05, 'days': 15},
    'pullback_5pct_20d': {'magnitude': 0.05, 'days': 20},
    'pullback_10pct_5d': {'magnitude': 0.10, 'days': 5},
    'pullback_10pct_10d': {'magnitude': 0.10, 'days': 10},
    'pullback_10pct_15d': {'magnitude': 0.10, 'days': 15},
    'pullback_10pct_20d': {'magnitude': 0.10, 'days': 20},
}
```

### 0.2 Feature Selection Strategy
Start with only the most predictive features based on market logic:

**Tier 1 Features (Core - 15 features)**:
- VIX level and momentum (3d, 5d, 10d)
- RSI and RSI extremes
- ADX (trend strength)
- Bollinger Band width and squeeze
- SPY volatility (10d, 20d)
- Volume patterns

**Tier 2 Features (Enhanced - 15 features)**:
- Options skew and put/call ratios
- Cross-asset correlations (TLT, GLD)
- USDJPY momentum (carry trade)
- VIX term structure
- Sector rotation signals

**Tier 3 Features (Experimental - 10 features)**:
- Seasonality factors
- Market microstructure
- Options flow metrics
- Economic indicators

### 0.3 Systematic Testing Framework
```python
class MultiTargetAnalyzer:
    def __init__(self):
        self.results = {}
        
    def analyze_target(self, magnitude, days, features):
        # 1. Create labels for this specific target
        # 2. Train model with 2016-2022 data
        # 3. Validate on 2023
        # 4. Test on 2024
        # 5. Record metrics
        
    def find_optimal_target(self):
        # Compare all targets based on:
        # - Precision at various thresholds
        # - Recall for major events
        # - False positive rate
        # - Profit factor in backtesting
```

### 0.4 Expected Outcomes
Different targets will excel at different tasks:
- **2% in 5-10 days**: Most signals, good for active trading
- **5% in 10-15 days**: Balanced risk/reward
- **10% in 15-20 days**: Rare but high-value crash detection

### 0.5 Implementation Steps
1. **Day 1**: Build multi-target labeling system
2. **Day 2**: Create feature tiers and selection pipeline  
3. **Day 3**: Train all 12 target models
4. **Day 4**: Analyze results and select best 3-4 targets
5. **Day 5**: Build ensemble of best targets
6. **Day 6**: Validate on 2024 events
7. **Day 7**: Create production pipeline

### 0.6 Advantages of Starting Fresh
- **Clean Architecture**: No legacy code debt
- **Systematic Approach**: Test all possibilities
- **Better Understanding**: Know exactly why each target works
- **Optimal Selection**: Data-driven target choice
- **Faster Development**: 1 week vs 3 weeks of fixes

### 0.7 Detailed Implementation Plan

#### Step 1: Create Multi-Target Analysis Script
```python
# analyze_optimal_targets.py
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score
import lightgbm as lgb
import yfinance as yf

class OptimalTargetFinder:
    def __init__(self):
        self.magnitudes = [0.02, 0.05, 0.10]  # 2%, 5%, 10%
        self.horizons = [5, 10, 15, 20]       # days
        self.results = []
        
    def create_target_labels(self, spy_data, magnitude, horizon):
        """Create binary labels for specific magnitude/horizon"""
        labels = pd.Series(0, index=spy_data.index)
        
        for i in range(len(spy_data) - horizon):
            future_prices = spy_data['Low'].iloc[i+1:i+horizon+1]
            current_price = spy_data['Close'].iloc[i]
            min_future = future_prices.min()
            
            if (min_future / current_price - 1) <= -magnitude:
                labels.iloc[i] = 1
                
        return labels
    
    def evaluate_target(self, magnitude, horizon, X_train, y_train, X_test, y_test):
        """Train and evaluate model for specific target"""
        # Train simple model
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics
        metrics = {
            'magnitude': magnitude,
            'horizon': horizon,
            'target_name': f'{int(magnitude*100)}pct_{horizon}d',
            'positive_rate_train': y_train.mean(),
            'positive_rate_test': y_test.mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'optimal_threshold': optimal_threshold,
            'precision_at_optimal': precisions[optimal_idx],
            'recall_at_optimal': recalls[optimal_idx],
            'f1_at_optimal': f1_scores[optimal_idx],
            'precision_at_80pct': precisions[thresholds >= 0.80][0] if any(thresholds >= 0.80) else 0,
            'feature_importance_top5': model.feature_importances_[:5]
        }
        
        return metrics, model
```

#### Step 2: Feature Engineering Pipeline
```python
class MinimalFeatureEngine:
    """Start with minimal, high-quality features"""
    
    def __init__(self):
        self.feature_groups = {
            'volatility': [
                'vix_level',
                'vix_momentum_3d',
                'vix_momentum_5d',
                'spy_volatility_10d',
                'spy_volatility_20d'
            ],
            'momentum': [
                'rsi_14',
                'rsi_extreme',
                'momentum_10d',
                'momentum_20d',
                'distance_from_high_20d'
            ],
            'trend': [
                'adx',
                'adx_strong',
                'bb_width',
                'bb_squeeze',
                'price_vs_sma200'
            ],
            'options': [
                'put_call_ratio',
                'iv_skew',
                'vix_term_structure',
                'options_flow_imbalance'
            ],
            'cross_asset': [
                'tlt_momentum_10d',
                'gld_momentum_10d',
                'usdjpy_momentum_5d',
                'dxy_level'
            ]
        }
```

#### Step 3: Walk-Forward Validation Framework
```python
class WalkForwardValidator:
    """
    Implement proper walk-forward validation to avoid overfitting
    WARNING: Do NOT optimize specifically for 2024 events!
    """
    
    def __init__(self, train_window_years=5, test_window_months=6):
        self.train_window = train_window_years * 252  # trading days
        self.test_window = test_window_months * 21   # trading days
        self.results = []
        
    def validate(self, data, features, target_creator):
        """
        Walk-forward validation from 2010 to 2024
        Each iteration:
        - Train on 5 years
        - Test on next 6 months
        - Move forward 6 months
        """
        
        start_date = pd.Timestamp('2010-01-01')
        end_date = pd.Timestamp('2024-12-31')
        
        current_date = start_date
        
        while current_date < end_date - pd.DateOffset(years=self.train_window_years):
            # Define windows
            train_start = current_date
            train_end = train_start + pd.DateOffset(years=self.train_window_years)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=6)
            
            # Split data
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]
            
            # Create targets
            train_labels = target_creator(train_data)
            test_labels = target_creator(test_data)
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
            
            model.fit(train_data[features], train_labels)
            
            # Evaluate
            predictions = model.predict_proba(test_data[features])[:, 1]
            
            # Store results
            self.results.append({
                'train_period': f"{train_start.date()} to {train_end.date()}",
                'test_period': f"{test_start.date()} to {test_end.date()}",
                'roc_auc': roc_auc_score(test_labels, predictions),
                'precision_50pct': precision_score(test_labels, predictions > 0.5),
                'recall_50pct': recall_score(test_labels, predictions > 0.5),
                'test_positive_rate': test_labels.mean()
            })
            
            # Move forward
            current_date += pd.DateOffset(months=6)
            
        return pd.DataFrame(self.results)

# IMPORTANT: Validate across multiple time periods, not just 2024!
```

#### Step 4: Generalization Testing (NOT 2024-Specific)
```python
def test_generalization(model, historical_crashes):
    """
    Test on multiple historical periods to ensure generalization
    WARNING: We want a model that works across different market regimes,
    not one that's overfit to 2024 patterns!
    """
    
    test_periods = {
        'dot_com_crash': ('2000-01-01', '2002-12-31'),
        'financial_crisis': ('2007-01-01', '2009-12-31'),
        'covid_crash': ('2020-01-01', '2020-12-31'),
        'rate_hike_cycle': ('2022-01-01', '2023-12-31'),
        'recent_period': ('2024-01-01', '2024-12-31')  # Just one of many test periods
    }
    
    results = {}
    for period_name, (start, end) in test_periods.items():
        # Test model on each period
        period_data = get_data(start, end)
        predictions = model.predict_proba(period_data)
        
        # Evaluate without targeting specific events
        results[period_name] = {
            'precision': calculate_precision(predictions),
            'recall': calculate_recall(predictions),
            'profit_factor': calculate_profit_factor(predictions)
        }
    
    return results
```

#### Step 4: Results Summary Dashboard
```python
def create_results_dashboard(all_results):
    """Create comprehensive comparison of all targets"""
    
    # Sort by key metrics
    df = pd.DataFrame(all_results)
    
    # Best for catching major events
    df['major_event_score'] = df['recall_at_optimal'] * df['precision_at_optimal']
    
    # Best for trading (balance of signals and accuracy)
    df['trading_score'] = df['f1_at_optimal'] * (1 - df['positive_rate_test'])
    
    # Best for risk management (high precision)
    df['risk_mgmt_score'] = df['precision_at_80pct']
    
    print("=" * 80)
    print("OPTIMAL TARGET ANALYSIS RESULTS")
    print("=" * 80)
    print()
    
    print("Top 3 for Catching Major Events:")
    print(df.nlargest(3, 'major_event_score')[['target_name', 'major_event_score', 'recall_at_optimal', 'precision_at_optimal']])
    print()
    
    print("Top 3 for Active Trading:")
    print(df.nlargest(3, 'trading_score')[['target_name', 'trading_score', 'f1_at_optimal', 'positive_rate_test']])
    print()
    
    print("Top 3 for Risk Management:")
    print(df.nlargest(3, 'risk_mgmt_score')[['target_name', 'risk_mgmt_score', 'precision_at_80pct']])
    
    return df
```

### 0.8 Expected Results and Recommendations

Based on market dynamics, we expect:

1. **Best Overall**: 5% pullback in 10-15 days
   - Balances frequency and significance
   - Catches most major events
   - Reasonable lead time for positioning

2. **Best for Active Trading**: 2% pullback in 5-10 days
   - More frequent signals
   - Good for options strategies
   - Requires tighter risk management

3. **Best for Crash Detection**: 10% pullback in 15-20 days
   - Rare but highly valuable
   - Focus on tail risk
   - Suitable for portfolio hedging

### 0.9 Migration Path from Current System

1. **Run Phase 0 in parallel** with existing system
2. **Compare results** on 2024 data
3. **If Phase 0 outperforms**, migrate completely
4. **If similar performance**, use ensemble approach
5. **Preserve valuable components** (data loaders, feature calculations)

---

## Phase 1: Critical Fixes (Days 1-3)

### 1.1 Fix Target Definition
**Problem**: Current 3-5 day window misses many events that have 7-8 day lead time

**Actions**:
```python
# Expand target window
class ImprovedEarlyWarningTarget(ForwardLookingTarget):
    def __init__(self):
        super().__init__(
            min_lead_days=3,
            max_lead_days=7,  # Expanded from 5
            drawdown_threshold=0.02
        )
```

**Implementation**:
- [ ] Update `targets/early_warning.py` with expanded window
- [ ] Add severity prediction (magnitude of drawdown)
- [ ] Create cluster detection logic (don't need every day in a cluster)

### 1.2 Fix 2024 Training Data Issues
**Problem**: Model trained only until 2022, missing recent market dynamics

**Actions**:
- [ ] Retrain with data through 2023 (keep 2024 for testing)
- [ ] Add walk-forward validation for 2024
- [ ] Implement proper train/validation/test splits:
  - Train: 2016-2021
  - Validation: 2022-2023
  - Test: 2024

### 1.3 Feature Selection & Engineering
**Problem**: 139 features causing overfitting and noise

**Actions**:
- [ ] Implement recursive feature elimination (RFE)
- [ ] Add feature importance analysis with SHAP
- [ ] Create feature groups:
  - Core (top 20-30 features)
  - Auxiliary (next 20-30)
  - Experimental (remainder)

**Key Features to Prioritize** (based on analysis):
1. VIX momentum (3d, 5d)
2. VIX level and regime
3. ADX (trend strength)
4. RSI extremes
5. Cross-asset flows (TLT, GLD)
6. Options skew metrics
7. Seasonality (Q3 sensitivity)

---

## Phase 2: Model Architecture Improvements (Days 4-7)

### 2.1 Ensemble Implementation
**Current**: Single LightGBM model
**Target**: Multi-model ensemble for robustness

```python
# Ensemble architecture
models = {
    'lightgbm': LightGBMModel(params_optimized),
    'xgboost': XGBoostModel(params_optimized),
    'random_forest': RandomForestModel(params_optimized),
    'neural_net': SimpleNN(layers=[128, 64, 32])  # For non-linear patterns
}

# Weighted ensemble based on validation performance
ensemble = WeightedEnsemble(models, weights='dynamic')
```

### 2.2 Multi-Horizon Models
**Problem**: Single model for all timeframes
**Solution**: Specialized models for different horizons

- **Short-term (1-3 days)**: High-frequency features, options flow
- **Medium-term (5-10 days)**: Trend exhaustion, momentum
- **Long-term (10-20 days)**: Macro features, seasonality

### 2.3 Regime-Aware Models
**Problem**: Same model for all market conditions
**Solution**: Separate models for different regimes

```python
regimes = {
    'bull': spy_return_20d > 0.05,
    'bear': spy_return_20d < -0.05,
    'sideways': abs(spy_return_20d) < 0.05
}
```

---

## Phase 3: Advanced Features (Days 8-10)

### 3.1 Microstructure Features
- [ ] Intraday volatility patterns
- [ ] Gap analysis (overnight vs intraday)
- [ ] Volume profile analysis
- [ ] Market depth indicators

### 3.2 Enhanced Options Features
- [ ] Skew acceleration (rate of change)
- [ ] Term structure steepness
- [ ] Put/call ratio momentum
- [ ] Options flow imbalance indicators

### 3.3 Cross-Asset Intelligence
- [ ] Yen carry trade indicators (critical for August 2024)
- [ ] Bond yield momentum and curve dynamics
- [ ] Currency volatility regime
- [ ] Commodity divergences

### 3.4 Alternative Data (Selective)
- [ ] VIX futures positioning (if available)
- [ ] ETF flows (SPY, sector rotation)
- [ ] ~~Google Trends~~ (Skip - too noisy)
- [ ] Economic surprise indices

---

## Phase 4: Backtesting & Validation (Days 11-13)

### 4.1 Comprehensive Backtesting Framework
```python
class BacktestEngine:
    def __init__(self):
        self.metrics = {
            'sharpe_ratio': None,
            'max_drawdown': None,
            'win_rate': None,
            'profit_factor': None,
            'calmar_ratio': None
        }
    
    def run_backtest(self, predictions, prices, strategy):
        # Implement realistic trading simulation
        # Include transaction costs, slippage
        # Test different position sizing methods
```

### 4.2 Walk-Forward Analysis
- [ ] 6-month rolling windows
- [ ] Quarterly retraining
- [ ] Hyperparameter stability testing
- [ ] Feature importance stability

### 4.3 Stress Testing
- [ ] Test on historical crashes (2008, 2020)
- [ ] Test on different market regimes
- [ ] Monte Carlo simulation for robustness

---

## Phase 5: Production Pipeline (Days 14-16)

### 5.1 Daily Retraining System
```python
class DailyPipeline:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engine = FeatureEngine()
        self.model_trainer = ModelTrainer()
        self.monitor = ModelMonitor()
    
    def daily_update(self):
        # 1. Download latest data
        # 2. Generate features
        # 3. Check for drift
        # 4. Retrain if needed
        # 5. Generate predictions
        # 6. Send alerts
```

### 5.2 Model Monitoring
- [ ] Performance degradation detection
- [ ] Feature drift monitoring
- [ ] Prediction distribution shifts
- [ ] Automated rollback on failure

### 5.3 Alert System
```python
class AlertSystem:
    def __init__(self):
        self.channels = ['email', 'sms', 'webhook']
        self.thresholds = {
            'high_confidence': 0.85,
            'critical': 0.95
        }
```

---

## Phase 6: Code Cleanup & Organization (Days 17-18)

### 6.1 File Consolidation Plan

**Keep These Core Files**:
```
├── core/
│   ├── data_loader.py
│   ├── features.py
│   └── models.py
├── targets/
│   ├── early_warning.py
│   ├── gradual_pullback.py
│   └── time_correction.py
├── engines/
│   ├── backtest.py
│   ├── train.py
│   └── predict.py
├── config/
│   └── unified_config.json
└── main.py  # Single entry point
```

**Consolidate These**:
- All train_*.py files → `engines/train.py` with arguments
- All analyze_*.py files → `engines/analyze.py` with modes
- All feature files → Organized feature modules

**Delete After Consolidation**:
- Temporary analysis scripts
- Duplicate training scripts
- One-off experiments

### 6.2 Documentation
- [ ] API documentation for each module
- [ ] Model card with assumptions and limitations
- [ ] Deployment guide
- [ ] Monitoring playbook

---

## Phase 7: Advanced Improvements (Days 19-21)

### 7.1 Probability Calibration
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probabilities for better reliability
calibrated_model = CalibratedClassifierCV(
    base_estimator=ensemble,
    method='isotonic',  # or 'sigmoid'
    cv=3
)
```

### 7.2 Cost-Sensitive Learning
```python
# Penalize false negatives more than false positives
class_weights = {
    0: 1.0,   # No pullback
    1: 3.0    # Pullback (higher weight)
}
```

### 7.3 Feature Interactions
```python
# Create interaction features for known relationships
interactions = [
    ('vix_momentum_3d', 'rsi'),
    ('bb_width', 'adx'),
    ('tlt_return_20d', 'vix_level'),
    ('usdjpy_momentum_5d', 'vix_spike')
]
```

---

## Implementation Priority

### Must Have (Critical for 2024 accuracy):
1. ✅ Expand target window to 3-7 days
2. ✅ Retrain with 2023 data
3. ✅ Feature selection (top 30-50)
4. ✅ Basic ensemble (LightGBM + XGBoost)
5. ✅ Walk-forward validation
6. ✅ Fix high false positive rate

### Should Have (Significant improvements):
1. ✅ Multi-horizon models
2. ✅ Enhanced options features
3. ✅ Backtesting framework
4. ✅ Daily pipeline
5. ✅ Code consolidation
6. ✅ SHAP analysis

### Nice to Have (Marginal gains):
1. ⚡ Neural network component
2. ⚡ International market testing
3. ⚡ Real-time streaming
4. ⚡ Mobile app
5. ⚡ Cloud deployment

---

## Success Metrics

### Primary Goals:
- **Catch 3/4 major 2024 clusters**: April, August, September, December
- **Reduce false positive rate**: <40% at 85% confidence
- **Maintain high precision**: >60% for actionable signals

### Stretch Goals:
- Catch all 4 major 2024 events
- Achieve 70%+ precision at 80% confidence
- Profitable backtesting results (Sharpe > 1.5)

---

## Next Steps

1. **Immediate**: Start with Phase 1.1 - Fix target window
2. **Day 1-3**: Complete Phase 1 (Critical Fixes)
3. **Day 4-7**: Implement basic ensemble
4. **Day 8-10**: Add key missing features
5. **Day 11-13**: Validate on 2024 data
6. **Day 14+**: Production pipeline

---

## Risk Mitigation

### Model Risks:
- **Overfitting**: Use proper validation, regularization
- **Regime changes**: Monitor for distribution shifts
- **Black swans**: Accept model limitations, use stop losses

### Implementation Risks:
- **Complexity**: Start simple, add incrementally
- **Data quality**: Implement data validation checks
- **Technical debt**: Regular refactoring sessions

---

## Conclusion

This roadmap provides a systematic approach to fixing the current model's issues and building a production-ready system. The phased approach allows for quick wins while building toward a comprehensive solution.

**Key insight**: The model's current problem isn't lack of features or complexity - it's poor target definition, training data recency, and overfitting. Fix these fundamentals first, then enhance.

Remember: **Real markets are messy. Perfect prediction is impossible. Focus on catching the big moves with high confidence.**
