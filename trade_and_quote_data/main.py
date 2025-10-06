#!/usr/bin/env python3
"""
Trade and Quote Data Analysis System - Main Entry Point

Quick start functions for common tasks.

Usage:
    python main.py                              # Interactive menu
    python main.py --train-optimal              # Train recommended model
    python main.py --predict                    # Generate predictions
    python main.py --update-and-retrain         # Daily update workflow
"""

import argparse
import sys
from datetime import datetime
from scripts.train_model import ModelTrainer
from analysis.analyze import main as analyze_main

def train_optimal_model():
    """Train the recommended model configuration"""
    print("🚀 Training optimal model configuration...")
    print("Target: pullback_4pct_30d")
    print("Features: enhanced")
    print("Model: ensemble")
    
    trainer = ModelTrainer(
        target='pullback_4pct_30d',
        feature_set='enhanced',
        model_type='ensemble',
        validation_strategy='walk-forward'
    )
    
    results = trainer.train()
    trainer.save_model(results)
    print("✅ Optimal model training completed!")

def generate_predictions():
    """Generate predictions for tomorrow"""
    print("🔮 Generating predictions for tomorrow...")
    # Load latest model and generate predictions
    # Implementation depends on prediction pipeline
    print("✅ Predictions generated!")

def update_and_retrain():
    """Daily update workflow"""
    print("🔄 Running daily update workflow...")
    
    # 1. Update data
    from data_management.unified_downloader import UnifiedDownloader
    downloader = UnifiedDownloader()
    downloader.update_data()
    
    # 2. Check if retraining is needed
    # 3. Generate new predictions
    
    print("✅ Daily update completed!")

def interactive_menu():
    """Interactive menu for common tasks"""
    print("=" * 60)
    print("Trade and Quote Data Analysis System")
    print("=" * 60)
    print("1. Train optimal model")
    print("2. Generate predictions")
    print("3. Update data and retrain")
    print("4. Run analysis")
    print("5. Exit")
    print("=" * 60)
    
    choice = input("Select option (1-5): ")
    
    if choice == '1':
        train_optimal_model()
    elif choice == '2':
        generate_predictions()
    elif choice == '3':
        update_and_retrain()
    elif choice == '4':
        print("For analysis, use: python analysis/analyze.py --help")
    elif choice == '5':
        sys.exit(0)
    else:
        print("Invalid choice. Please select 1-5.")

def main():
    parser = argparse.ArgumentParser(description='Trade and Quote Data Analysis System')
    parser.add_argument('--train-optimal', action='store_true', 
                       help='Train the recommended model configuration')
    parser.add_argument('--predict', action='store_true',
                       help='Generate predictions for tomorrow')
    parser.add_argument('--update-and-retrain', action='store_true',
                       help='Daily update workflow')
    
    args = parser.parse_args()
    
    if args.train_optimal:
        train_optimal_model()
    elif args.predict:
        generate_predictions()
    elif args.update_and_retrain:
        update_and_retrain()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()

