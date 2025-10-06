#!/usr/bin/env python3
"""
Unified Model Training Script

Consolidates all training functionality into a single, parameterized script.
Supports multiple model types, optimization strategies, and feature sets.

Usage:
    python scripts/train_model.py --model gradual_decline
    python scripts/train_model.py --model early_warning --optimize precision
    python scripts/train_model.py --model crash_risk --features sector_rotation
    python scripts/train_model.py --model multi_scenario --target pullback_4pct
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all model types
from gradual_decline_analyzer import GradualDeclineDetector
from gradual_decline_analyzer.targets import GradualDeclineTargetCreator
from gradual_decline_analyzer.models import GradualDeclineEnsemble

# Import core modules
from core.data_loader import DataLoader
from core.features import FeatureEngine
from training.trainer import ModelTrainer

# Import crash risk models
from crash_risk_buildup.models import EnsemblePredictor, XGBoostPredictor
from crash_risk_buildup.targets import TargetFactory

# Training modes configuration
TRAINING_MODES = {
    'gradual_decline': {
        'description': 'Detect gradual declines over 7-40 days',
        'model_class': GradualDeclineEnsemble,
        'default_target': 'gradual_decline',
        'window_days': (7, 40),
        'threshold': 0.05
    },
    'early_warning': {
        'description': 'Early detection of 4%+ pullbacks (5-10 days)',
        'model_class': 'EarlyWarningModel',
        'default_target': 'pullback_4pct',
        'window_days': (5, 10),
        'threshold': 0.04
    },
    'crash_risk': {
        'description': 'Detect crash risk buildup (3-7 days)',
        'model_class': EnsemblePredictor,
        'default_target': 'pullback_5pct',
        'window_days': (3, 7),
        'threshold': 0.05
    },
    'multi_scenario': {
        'description': 'Test multiple scenarios and thresholds',
        'model_class': 'MultiScenarioModel',
        'default_target': 'multi',
        'window_days': (5, 30),
        'threshold': [0.04, 0.05, 0.10]
    }
}

OPTIMIZATION_STRATEGIES = {
    'precision': 'Optimize for high precision (fewer false positives)',
    'recall': 'Optimize for high recall (catch more events)',
    'f1': 'Balance precision and recall',
    'rule_based': 'Pure rules-based approach',
    'hybrid': 'Combine rules and ML',
    'targeted': 'Target specific dates/events'
}

FEATURE_SETS = {
    'standard': 'Default feature set',
    'streamlined': 'Reduced features (no slow macro)',
    'sector_rotation': 'Enhanced sector rotation features',
    'minimal': 'Minimal feature set for speed'
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified model training script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py --model gradual_decline
  python scripts/train_model.py --model early_warning --optimize precision
  python scripts/train_model.py --model crash_risk --features sector_rotation
  python scripts/train_model.py --model multi_scenario --target pullback_4pct
        """
    )
    
    parser.add_argument('--model', 
                       choices=list(TRAINING_MODES.keys()),
                       required=True,
                       help='Model type to train')
    
    parser.add_argument('--optimize',
                       choices=list(OPTIMIZATION_STRATEGIES.keys()),
                       default='f1',
                       help='Optimization strategy')
    
    parser.add_argument('--features',
                       choices=list(FEATURE_SETS.keys()),
                       default='standard',
                       help='Feature set to use')
    
    parser.add_argument('--target',
                       help='Target type (overrides model default)')
    
    parser.add_argument('--start-date',
                       type=str,
                       default='2000-01-01',
                       help='Start date for training data')
    
    parser.add_argument('--end-date',
                       type=str,
                       default='2024-12-31',
                       help='End date for training data')
    
    parser.add_argument('--output-dir',
                       type=Path,
                       default=Path('models/trained'),
                       help='Output directory for trained models')
    
    parser.add_argument('--config',
                       type=str,
                       help='Path to configuration file')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file"""
    if not config_path or not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)

def train_gradual_decline(args, config):
    """Train gradual decline detection model"""
    print("🚀 Training Gradual Decline Detection Model")
    print("=" * 60)
    
    # Load data
    print(f"📥 Loading SPY data from {args.start_date} to {args.end_date}")
    loader = DataLoader(start_date=args.start_date, end_date=args.end_date)
    spy = loader.load_spy_data()
    print(f"✅ Loaded {len(spy)} records")
    
    # Calculate features
    print("🔧 Calculating features...")
    detector = GradualDeclineDetector()
    features_df, feature_cols = detector.calculate_features(spy)
    print(f"✅ Calculated {len(feature_cols)} features")
    
    # Create targets
    print("🎯 Creating targets...")
    target_creator = GradualDeclineTargetCreator(features_df)
    windows = config.get('windows', [7, 14, 20, 30])
    threshold = config.get('threshold', 0.05)
    targets_df = target_creator.create_multi_timeframe_targets(
        windows=windows, 
        threshold=threshold
    )
    
    # Merge features and targets
    combined_df = features_df.copy()
    for col in targets_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = targets_df[col]
    
    # Split data
    train_end = config.get('train_end', '2022-12-31')
    val_end = config.get('val_end', '2023-12-31')
    
    train_data = combined_df[combined_df.index <= train_end]
    val_data = combined_df[(combined_df.index > train_end) & (combined_df.index <= val_end)]
    test_data = combined_df[combined_df.index > val_end]
    
    print(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Train model
    print("🤖 Training ensemble model...")
    model = GradualDeclineEnsemble()
    
    # Get target columns
    target_cols = [c for c in targets_df.columns if "decline_" in c]
    
    # Train on each target
    results = {}
    for target_col in target_cols:
        print(f"   Training on {target_col}...")
        
        # Prepare data
        train_X = train_data[feature_cols]
        train_y = train_data[target_col].fillna(0)
        val_X = val_data[feature_cols]
        val_y = val_data[target_col].fillna(0)
        
        # Train
        model.fit(train_X, train_y, val_X, val_y)
        
        # Evaluate
        test_X = test_data[feature_cols]
        test_y = test_data[target_col].fillna(0)
        score = model.score(test_X, test_y)
        results[target_col] = score
        
        print(f"   {target_col} score: {score:.3f}")
    
    # Save model
    output_dir = args.output_dir / 'gradual_decline_ensemble'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(output_dir)
    print(f"✅ Model saved to {output_dir}")
    
    return model, results

def train_early_warning(args, config):
    """Train early warning model"""
    print("🚀 Training Early Warning Model")
    print("=" * 60)
    
    # Use existing trainer for early warning
    trainer = ModelTrainer(
        model_type='ensemble',
        feature_sets=[args.features],
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Train model
    model = trainer.train(
        target_type=args.target or 'early_warning',
        save_model=True
    )
    
    return model, trainer.get_feature_importance()

def train_crash_risk(args, config):
    """Train crash risk detection model"""
    print("🚀 Training Crash Risk Detection Model")
    print("=" * 60)
    
    # Load data
    loader = DataLoader(start_date=args.start_date, end_date=args.end_date)
    data = loader.load_all_data()
    
    # Use crash risk trainer
    from crash_risk_buildup.pipeline.model_trainer import ModelTrainer as CrashRiskTrainer
    
    trainer = CrashRiskTrainer(
        ticker='SPY',
        model_type='ensemble',
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Train model
    results = trainer.train_full_pipeline()
    
    return trainer.model, results

def train_multi_scenario(args, config):
    """Train multi-scenario model"""
    print("🚀 Training Multi-Scenario Model")
    print("=" * 60)
    
    # Load data
    loader = DataLoader(start_date=args.start_date, end_date=args.end_date)
    spy = loader.load_spy_data()
    
    # Test different scenarios
    scenarios = [
        {'threshold': 0.05, 'window': 5},
        {'threshold': 0.05, 'window': 10},
        {'threshold': 0.05, 'window': 20},
        {'threshold': 0.10, 'window': 5},
        {'threshold': 0.10, 'window': 10},
        {'threshold': 0.10, 'window': 20},
    ]
    
    results = {}
    for i, scenario in enumerate(scenarios):
        print(f"   Testing scenario {i+1}: {scenario['threshold']*100}% threshold, {scenario['window']} day window")
        
        # Create targets for this scenario
        # (Implementation would go here)
        
        # Train model for this scenario
        # (Implementation would go here)
        
        results[f"scenario_{i+1}"] = scenario
    
    return None, results

def main():
    """Main training function"""
    args = parse_args()
    config = load_config(args.config)
    
    print("🎯 UNIFIED MODEL TRAINING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Optimization: {args.optimize}")
    print(f"Features: {args.features}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print("=" * 80)
    
    # Route to appropriate training function
    if args.model == 'gradual_decline':
        model, results = train_gradual_decline(args, config)
    elif args.model == 'early_warning':
        model, results = train_early_warning(args, config)
    elif args.model == 'crash_risk':
        model, results = train_crash_risk(args, config)
    elif args.model == 'multi_scenario':
        model, results = train_multi_scenario(args, config)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    print("\n✅ Training complete!")
    print(f"Results: {results}")
    
    return model, results

if __name__ == "__main__":
    model, results = main()
