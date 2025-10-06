# Codebase Cleanup and Refactoring Plan

## Overview
The current codebase has grown organically with many experimental scripts, duplicate functionality, and scattered analysis files. This plan outlines a systematic approach to clean up and reorganize the code before implementing the Model Improvement Roadmap.

## Current Issues
1. **28+ individual training scripts** with overlapping functionality
2. **20+ analysis scripts** that were one-time experiments
3. **Duplicate feature engineering** across multiple files
4. **No clear entry points** for common tasks
5. **Mixed production and experimental code**
6. **Scattered model files** without clear versioning

## Phase 1: Backup and Inventory (30 minutes)

### 1.1 Create Complete Backup
```bash
# Create timestamped backup of entire codebase
tar -czf backup/pre_cleanup_$(date +%Y%m%d_%H%M%S).tar.gz \
  --exclude='data' \
  --exclude='backup' \
  --exclude='output' \
  --exclude='__pycache__' \
  .
```

### 1.2 Document Current Scripts
Create inventory of what each script does:
- Training scripts: purpose, target, features used
- Analysis scripts: one-time vs reusable
- Utility scripts: which are still needed

## Phase 2: Consolidate Training Scripts (2 hours)

### 2.1 Create Unified Training System
**Replace all train_*.py files with:**
```
training/
├── train.py              # Main CLI entry point
├── configs/
│   ├── targets.yaml      # Target definitions
│   ├── features.yaml     # Feature groups
│   └── models.yaml       # Model configurations
└── experiments/
    └── archive/          # Move old training scripts here
```

**New unified interface:**
```python
# train.py - Single entry point for all training
python train.py \
  --target "pullback_2pct_5d" \
  --features "tier1" \
  --model "lightgbm" \
  --validation "walk-forward"
```

### 2.2 Scripts to Consolidate
**Keep and refactor:**
- `train_all_three_targets.py` → Integrate into unified system
- `train_improved_model.py` → Extract improvements

**Archive (move to experiments/archive/):**
- `train_pullback_*.py` (6 files)
- `train_early_warning_*.py` (2 files)
- `train_lightgbm.py`
- `train_with_all_features.py`
- `retrain_*.py` (4 files)

## Phase 3: Consolidate Analysis Scripts (1.5 hours)

### 3.1 Create Analysis Framework
**Replace all analyze_*.py files with:**
```
analysis/
├── analyze.py           # Main analysis CLI
├── reports/
│   ├── performance.py   # Model performance analysis
│   ├── features.py      # Feature importance
│   └── predictions.py   # Prediction analysis
└── archive/            # Old analysis scripts
```

**New interface:**
```python
# analyze.py - Single entry point for analysis
python analyze.py --type performance --model latest --period 2024
python analyze.py --type features --top 20
python analyze.py --type predictions --threshold 0.8
```

### 3.2 Scripts to Handle
**Extract useful functions then archive:**
- `analyze_2024_drawdowns.py`
- `analyze_april_2024.py`
- `analyze_false_positives.py`
- `analyze_high_confidence_signals.py`
- `analyze_feature_importance.py`
- `check_critical_clusters.py`

**Delete after extracting insights:**
- `analyze_false_positive_sources.py`
- `test_*.py` files (unless they're actual unit tests)

## Phase 4: Reorganize Core Components (1 hour)

### 4.1 Clean Feature System
**Current:** Features scattered across multiple modules
**Target:** Centralized feature registry

```
features/
├── __init__.py
├── registry.py          # Central feature registry
├── core/               # Essential features only
│   ├── volatility.py   # VIX, realized vol
│   ├── momentum.py     # RSI, momentum
│   └── trend.py        # ADX, BB
├── advanced/           # Options, cross-asset
└── experimental/       # New/testing features
```

### 4.2 Model Organization
```
models/
├── definitions/        # Model architectures
│   ├── lightgbm.py
│   ├── xgboost.py
│   └── ensemble.py
├── trained/           # Saved models with metadata
│   └── archive/       # Old models
└── configs/           # Model hyperparameters
```

## Phase 5: Create Clear Entry Points (1 hour)

### 5.1 Main CLI Application
```python
# cli.py - Primary interface
commands:
  data:
    download    # Download all required data
    update      # Update existing data
    validate    # Check data integrity
    
  train:
    single      # Train single model
    ensemble    # Train ensemble
    backtest    # Walk-forward validation
    
  predict:
    batch       # Batch predictions
    realtime    # Real-time predictions
    
  analyze:
    performance # Model performance
    features    # Feature analysis
    predictions # Prediction analysis
```

### 5.2 Simplified Main.py
```python
# main.py - Quick start for common tasks
def train_optimal_model():
    """Train the recommended model configuration"""
    
def generate_predictions():
    """Generate predictions for tomorrow"""
    
def update_and_retrain():
    """Daily update workflow"""
```

## Phase 6: Clean Up Utilities (30 minutes)

### 6.1 Remove Redundant Files
**Delete:**
- `execute_*.py` files (old execution scripts)
- `implement_*.py` files (old implementation scripts)
- `generate_*.py` files (move logic to predictions module)
- `compare_models.py` (integrate into analysis)
- `visualize_*.py` (integrate into analysis)

### 6.2 Consolidate Documentation
**Keep:**
- `README.md` (update with new structure)
- `MODEL_IMPROVEMENT_ROADMAP.md`

**Archive:**
- Other .md files to `docs/archive/`

## Phase 7: Data and Config Cleanup (30 minutes)

### 7.1 Data Organization
```
data/
├── raw/              # Original downloaded data
├── processed/        # Feature-engineered data
├── predictions/      # Model predictions
└── cache/           # Temporary files
```

### 7.2 Configuration
```
config/
├── data_sources.yaml    # Data download configs
├── features.yaml        # Feature definitions
├── models.yaml         # Model configurations
└── trading.yaml        # Trading parameters
```

## Implementation Order

### Day 1: Preparation (2 hours)
1. Create full backup
2. Set up new directory structure
3. Create base classes for unified system

### Day 2: Core Consolidation (4 hours)
1. Consolidate training scripts
2. Consolidate analysis scripts
3. Test unified interfaces

### Day 3: Cleanup and Testing (3 hours)
1. Archive old files
2. Update imports and dependencies
3. Run integration tests
4. Update documentation

## Success Criteria
- [ ] Single entry point for training
- [ ] Single entry point for analysis
- [ ] No duplicate code
- [ ] Clear separation of production vs experimental
- [ ] All functionality preserved
- [ ] Clean directory structure
- [ ] Updated documentation

## File Count Targets
- Training scripts: 28 → 3
- Analysis scripts: 20 → 3
- Total .py files: ~80 → ~40
- Clear improvement in maintainability

## Post-Cleanup Structure
```
trade_and_quote_data/
├── cli.py              # Main CLI interface
├── main.py             # Quick start functions
├── config/             # All configurations
├── core/               # Core data and model classes
├── features/           # Feature engineering
├── models/             # Model definitions
├── training/           # Training framework
├── analysis/           # Analysis framework
├── data/              # Data storage
├── tests/             # Unit tests
├── docs/              # Documentation
└── archive/           # Old code for reference
```

## Next Steps
After cleanup is complete:
1. Run full test suite
2. Verify all functionality works
3. Begin Phase 0 of Model Improvement Roadmap
4. Use clean codebase for faster development

---

**Ready to execute this cleanup plan? The clean codebase will make the Model Improvement Roadmap much easier to implement.**
