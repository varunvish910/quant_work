#!/usr/bin/env python3
"""
Unified Training System - Single entry point for all model training

Usage:
    python train.py --target pullback_2pct_5d --features tier1 --model lightgbm
    python train.py --target early_warning_4pct_30d --features enhanced --model ensemble
    python train.py --config config_name.yaml
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import get_model
from core.features import get_feature_set
from core.targets import get_target_definition
from core.data_loader import load_training_data
from scripts.train_model import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Unified Model Training System')
    parser.add_argument('--target', required=True, 
                       choices=['pullback_2pct_5d', 'pullback_4pct_30d', 'early_warning_2pct_3to5d', 
                               'early_warning_4pct_30d', 'mean_reversion', 'crash_detection'],
                       help='Target variable to predict')
    parser.add_argument('--features', default='tier1',
                       choices=['tier1', 'tier2', 'enhanced', 'all', 'reduced_volatility'],
                       help='Feature set to use')
    parser.add_argument('--model', default='lightgbm',
                       choices=['lightgbm', 'xgboost', 'ensemble', 'stacked'],
                       help='Model type to train')
    parser.add_argument('--validation', default='walk-forward',
                       choices=['walk-forward', 'time-series', 'stratified'],
                       help='Validation strategy')
    parser.add_argument('--config', help='YAML config file (overrides other args)')
    parser.add_argument('--output-dir', default='models/trained',
                       help='Output directory for trained models')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(
        target=args.target,
        feature_set=args.features,
        model_type=args.model,
        validation_strategy=args.validation,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Train model
    print(f"Training {args.model} for {args.target} using {args.features} features...")
    results = trainer.train()
    
    # Save results
    trainer.save_model(results)
    trainer.generate_report(results)
    
    print(f"✅ Training completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()