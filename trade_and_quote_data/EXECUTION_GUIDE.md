# 🚀 Execution Guide: Phases 1-3

**Total Time: 2-3 hours**  
**Stop Before: Phase 3.5 (Options)**

---

## ⏱️ Time Breakdown

- **Phase 1:** 10-15 minutes (testing + data download)
- **Phase 2:** 30-45 minutes (feature creation + integration)
- **Phase 3:** 60-90 minutes (model training + evaluation)

---

## 📋 PHASE 1: VALIDATE & TEST (10-15 min)

### Step 1.1: Test Current System
```bash
cd /Users/varun/code/quant_final_final/trade_and_quote_data
python3 test_new_architecture.py
```
**Expected:** All 5 tests pass ✅

### Step 1.2: Download All Data
```bash
python3 download_all_data.py
```
**Downloads:**
- SPY (2000-2024)
- 10 Sector ETFs
- 4 Rotation indicators (MAGS, RSP, QQQ, QQQE)
- Currency pairs (USD/JPY, EUR/USD)
- Volatility indices (VIX, VIX9D, VVIX)

**Time:** 10-12 minutes

### Step 1.3: Verify Data
```bash
ls -lh data/ohlc/
ls -lh data/sectors/
ls -lh data/currency/
ls -lh data/volatility/
```
**Expected:** All directories have .parquet files

---

## 📋 PHASE 2: ADD FEATURES (30-45 min)

### Step 2.1: Create Volume Features
Create `features/technicals/volume.py`:

```python
"""Volume-based features"""
import pandas as pd
import numpy as np
from features.technicals.base import BaseTechnicalFeature

class VolumeFeature(BaseTechnicalFeature):
    """Volume analysis features"""
    
    def __init__(self):
        super().__init__("Volume")
        self.required_columns = ['Close', 'Volume']
    
    def calculate(self, data, **kwargs):
        df = data.copy()
        
        # Volume moving average
        df['volume_ma20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma20']
        
        # On-Balance Volume
        price_change = df['Close'].diff()
        df['obv'] = (df['Volume'] * np.sign(price_change)).cumsum()
        
        # Volume-price divergence
        price_trend = df['Close'].rolling(20).apply(lambda x: 1 if x[-1] > x[0] else -1)
        volume_trend = df['Volume'].rolling(20).apply(lambda x: 1 if x[-1] > x[0] else -1)
        df['volume_price_divergence'] = (price_trend != volume_trend).astype(int)
        
        self.feature_names = ['volume_ma20', 'volume_ratio', 'obv', 'volume_price_divergence']
        return df
```

### Step 2.2: Create Market Breadth Features
Create `features/market/breadth.py`:

```python
"""Market breadth indicators"""
import pandas as pd
from features.market.base import BaseMarketFeature

class MarketBreadthFeature(BaseMarketFeature):
    """Calculate market breadth from sector data"""
    
    def __init__(self):
        super().__init__("MarketBreadth")
    
    def calculate(self, data, sector_data=None, **kwargs):
        df = data.copy()
        
        if sector_data and len(sector_data) > 0:
            # Count sectors above their 50-day MA
            sectors_above_ma = []
            for date in df.index:
                count = 0
                for symbol, sector_df in sector_data.items():
                    if date in sector_df.index:
                        ma50 = sector_df['Close'].rolling(50).mean()
                        if date in ma50.index and sector_df.loc[date, 'Close'] > ma50.loc[date]:
                            count += 1
                sectors_above_ma.append(count / len(sector_data))
            
            df['sector_breadth'] = sectors_above_ma
            self.feature_names = ['sector_breadth']
        
        return df
```

### Step 2.3: Update Technical Engine
Edit `engines/technical_engine.py`:

```python
from features.technicals.volume import VolumeFeature

class TechnicalFeatureEngine(BaseFeatureEngine):
    def _initialize_features(self, config):
        # Existing features
        self.add_feature(MomentumFeature())
        self.add_feature(RSIFeature())
        self.add_feature(VolatilityFeature())
        # ... other existing features ...
        
        # NEW: Add volume features
        self.add_feature(VolumeFeature())
```

### Step 2.4: Update Market Engine
Edit `engines/market_engine.py`:

```python
from features.market.breadth import MarketBreadthFeature
from features.market.rotation_indicators import RotationIndicatorFeature

class MarketFeatureEngine(BaseFeatureEngine):
    def _initialize_features(self, config):
        self.add_feature(SectorRotationFeature())
        
        # NEW: Add breadth and rotation features
        self.add_feature(MarketBreadthFeature())
        self.add_feature(RotationIndicatorFeature())
```

### Step 2.5: Test New Features
```bash
python3 -c "
from features.technicals.volume import VolumeFeature
from features.market.breadth import MarketBreadthFeature
from features.market.rotation_indicators import RotationIndicatorFeature
print('✅ All feature imports successful')
"
```

---

## 📋 PHASE 3: IMPROVE MODELS (60-90 min)

### Step 3.1: Install LightGBM
```bash
pip install lightgbm
```

### Step 3.2: Create Multi-Horizon Target
Create `targets/multi_horizon.py`:

```python
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
            future_low = df['Low'].shift(-days).rolling(days).min()
            drawdown = (future_low - df['Close']) / df['Close']
            df[f'risk_{days}d'] = (drawdown < -self.threshold).astype(int)
        
        return df
```

### Step 3.3: Add LightGBM Model
Edit `core/models.py`, add:

```python
import lightgbm as lgb

class LightGBMModel(EarlyWarningModel):
    """LightGBM classifier"""
    
    def __init__(self):
        super().__init__(model_type='lightgbm')
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            random_state=42
        )
```

### Step 3.4: Train Multi-Target Models
```bash
python3 train_all_targets.py
```
**Time:** 30-60 minutes  
**Output:** Separate models for each target in `models/trained/`

### Step 3.5: Compare Models
```bash
python3 -c "
import pandas as pd
import joblib
from pathlib import Path

models_dir = Path('models/trained')
models = list(models_dir.glob('*_ensemble_*.pkl'))

print('📊 Trained Models:')
for model_path in models:
    print(f'  ✅ {model_path.name}')
"
```

### Step 3.6: Evaluate Best Model
```bash
python3 daily_usage_example.py
```
**Check:** Risk probability and feature importance

---

## ✅ VERIFICATION CHECKLIST

After completing all phases:

- [ ] All tests pass (`test_new_architecture.py`)
- [ ] Data downloaded (14 ETFs + SPY + currency + volatility)
- [ ] New features created (volume, breadth, rotation)
- [ ] Engines updated with new features
- [ ] LightGBM installed
- [ ] Multi-target models trained
- [ ] Models saved in `models/trained/`
- [ ] Predictions working (`daily_usage_example.py`)

---

## 📊 Expected Results

### Feature Count
- **Before:** 65 features
- **After Phase 2:** 80+ features
- **New features:** ~15-20

### Model Performance
- **Current ROC AUC:** 64%
- **Target ROC AUC:** 70%+
- **Expected improvement:** 3-6%

### Models Trained
- Early Warning Model
- Mean Reversion Model
- (Optional) Multi-Horizon Models

---

## 🚨 Common Issues

### Issue: Data download fails
**Solution:** Check internet connection, try again

### Issue: yfinance rate limiting
**Solution:** Add delays between downloads (already in script)

### Issue: Model training OOM
**Solution:** Reduce data range or use smaller model

### Issue: Import errors
**Solution:** Check all files are created in correct locations

---

## 🎯 Quick Commands

```bash
# Full execution (automated)
python3 execute_phases_1_to_3.py

# Or step-by-step:
python3 test_new_architecture.py          # Phase 1.1
python3 download_all_data.py              # Phase 1.2
# Create feature files manually            # Phase 2
python3 train_all_targets.py              # Phase 3
python3 daily_usage_example.py            # Verify
```

---

## 📝 Notes

- **Phase 3.5 (Options) is SKIPPED** as requested
- Total execution time: 2-3 hours
- Can be paused and resumed at any phase
- All data is cached after download
- Models are saved incrementally

---

**Ready to execute? Start with Phase 1!**

```bash
python3 test_new_architecture.py
```
