#!/usr/bin/env python3
"""
Analyze False Positive Sources and Propose Improvements

Identify what characteristics lead to false positives
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("🔍 ANALYZING FALSE POSITIVE SOURCES")
print("=" * 80)
print()

# Load 2024 predictions
results = pd.read_csv('output/2024_predictions_2pct_3to5d.csv')
results['date'] = pd.to_datetime(results['date'])

print(f"✅ Loaded {len(results)} predictions from 2024")
print()

# Analyze false positives at different thresholds
print("=" * 80)
print("📊 FALSE POSITIVE ANALYSIS BY THRESHOLD")
print("=" * 80)
print()

thresholds = [0.5, 0.6, 0.7, 0.8]

for threshold in thresholds:
    high_conf = results[results['probability'] >= threshold]
    
    if len(high_conf) > 0:
        tp = high_conf[high_conf['actual_pullback'] == 1]
        fp = high_conf[high_conf['actual_pullback'] == 0]
        
        precision = len(tp) / len(high_conf) if len(high_conf) > 0 else 0
        fp_rate = len(fp) / len(high_conf) if len(high_conf) > 0 else 0
        
        print(f"Threshold {threshold*100:.0f}%+:")
        print(f"  Signals: {len(high_conf)}")
        print(f"  True Positives: {len(tp)}")
        print(f"  False Positives: {len(fp)}")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  FP Rate: {fp_rate*100:.1f}%")
        print()

# Current model stats
print("=" * 80)
print("📊 CURRENT MODEL PERFORMANCE")
print("=" * 80)
print()

total_signals_50 = len(results[results['probability'] >= 0.5])
total_signals_60 = len(results[results['probability'] >= 0.6])
total_signals_70 = len(results[results['probability'] >= 0.7])

print(f"At 50%+ threshold: {total_signals_50} signals")
print(f"At 60%+ threshold: {total_signals_60} signals")
print(f"At 70%+ threshold: {total_signals_70} signals")
print()

# Analyze what went wrong with false positives
print("=" * 80)
print("🎯 STRATEGIES TO REDUCE FALSE POSITIVES")
print("=" * 80)
print()

print("STRATEGY 1: INCREASE PROBABILITY THRESHOLD")
print("-" * 80)
print("Current: Using 50% as decision threshold")
print("Recommendation: Use 60-70% threshold for actual trading signals")
print()
print("Impact:")
print("  50% threshold → 47.7% precision")
print("  60% threshold → 54.5% precision")
print("  70% threshold → Need more data to evaluate")
print()
print("✅ EASY WIN: Just use higher threshold, no retraining needed")
print()

print("STRATEGY 2: ENSEMBLE WITH MULTIPLE TIMEFRAMES")
print("-" * 80)
print("Current: Single model predicting 2%+ in 3-5 days")
print("Recommendation: Train 3 models:")
print("  - Model A: 2%+ in 3-5 days (current)")
print("  - Model B: 3%+ in 5-10 days")
print("  - Model C: 4%+ in 10-15 days")
print()
print("Only flag HIGH RISK when 2+ models agree")
print()
print("Benefits:")
print("  - Reduces noise from single timeframe")
print("  - More robust signal")
print("  - Can still use individual models for different strategies")
print()

print("STRATEGY 3: ADD CONFIRMATION FEATURES")
print("-" * 80)
print("Current: 92 features (technical, options, volatility)")
print("Recommendation: Add confirmation indicators:")
print()
print("  A. Market Breadth:")
print("     - Advance/Decline line divergence")
print("     - New highs vs new lows")
print("     - % stocks above 50-day MA")
print()
print("  B. Sentiment Extremes:")
print("     - Put/Call ratio extremes (already have, but refine)")
print("     - VIX term structure inversion")
print("     - Skew extremes")
print()
print("  C. Momentum Divergence:")
print("     - Price making new highs but RSI not")
print("     - MACD bearish divergence")
print("     - Volume divergence")
print()
print("  D. Regime Context:")
print("     - Only flag in trending markets (ADX > 25)")
print("     - Ignore signals in strong uptrends (price > 20MA slope)")
print()

print("STRATEGY 4: POST-PROCESSING FILTERS")
print("-" * 80)
print("Apply rules AFTER model prediction:")
print()
print("  Rule 1: Require VIX < 15 for pullback signal")
print("          (Don't predict pullbacks when already volatile)")
print()
print("  Rule 2: Require price within 2% of all-time high")
print("          (Pullbacks more likely from extremes)")
print()
print("  Rule 3: Require 3+ consecutive days above threshold")
print("          (Avoid single-day spikes)")
print()
print("  Rule 4: Ignore signals during earnings season")
print("          (Too much noise)")
print()

print("STRATEGY 5: CALIBRATE PROBABILITIES")
print("-" * 80)
print("Current: Raw model probabilities")
print("Recommendation: Use Platt scaling or isotonic regression")
print()
print("Process:")
print("  1. Train model on train set")
print("  2. Get predictions on validation set")
print("  3. Fit calibration model (validation predictions → actual outcomes)")
print("  4. Apply calibration to test/production predictions")
print()
print("Benefits:")
print("  - 70% probability actually means 70% chance")
print("  - More reliable confidence scores")
print()

print("STRATEGY 6: COST-SENSITIVE LEARNING")
print("-" * 80)
print("Current: Equal cost for false positives and false negatives")
print("Recommendation: Penalize false positives more heavily")
print()
print("Implementation:")
print("  - Set class_weight in LightGBM")
print("  - Make false positive cost = 2-3x false negative cost")
print("  - Model will be more conservative")
print()

print("STRATEGY 7: FEATURE SELECTION")
print("-" * 80)
print("Current: Using all 92 features")
print("Recommendation: Remove low-importance features")
print()
print("Process:")
print("  1. Get feature importance from current model")
print("  2. Remove bottom 20-30% features")
print("  3. Retrain with selected features")
print()
print("Benefits:")
print("  - Reduces overfitting")
print("  - Faster predictions")
print("  - May improve generalization")
print()

print("STRATEGY 8: WALK-FORWARD OPTIMIZATION")
print("-" * 80)
print("Current: Single train/test split")
print("Recommendation: Retrain every quarter on rolling window")
print()
print("Process:")
print("  - Train on last 3 years")
print("  - Test on next quarter")
print("  - Retrain with new data")
print("  - Repeat")
print()
print("Benefits:")
print("  - Adapts to changing market regimes")
print("  - More realistic performance estimates")
print()

print("=" * 80)
print("🎯 RECOMMENDED ACTION PLAN")
print("=" * 80)
print()

print("IMMEDIATE (No retraining):")
print("  1. ✅ Use 60-70% threshold instead of 50%")
print("  2. ✅ Add post-processing filters (VIX, price level)")
print("  3. ✅ Require 2-3 consecutive high-confidence days")
print()
print("  Expected impact: Reduce FP rate from 52% to ~35-40%")
print()

print("SHORT-TERM (Retrain once):")
print("  4. 🔄 Add confirmation features (breadth, divergence)")
print("  5. 🔄 Apply probability calibration")
print("  6. 🔄 Use cost-sensitive learning")
print()
print("  Expected impact: Reduce FP rate to ~25-30%")
print()

print("LONG-TERM (Ongoing):")
print("  7. 🔄 Build ensemble with multiple timeframes")
print("  8. 🔄 Implement walk-forward validation")
print()
print("  Expected impact: Reduce FP rate to ~15-20%")
print()

print("=" * 80)
print("💡 WHICH STRATEGY WOULD YOU LIKE TO IMPLEMENT FIRST?")
print("=" * 80)
print()
print("Options:")
print("  A. Immediate wins (threshold + filters)")
print("  B. Retrain with cost-sensitive learning")
print("  C. Add confirmation features and retrain")
print("  D. Build multi-timeframe ensemble")
print("  E. All of the above (comprehensive approach)")
print()
