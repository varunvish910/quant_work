#!/usr/bin/env python3
"""
Momentum-Based Pullback Prediction System - Main Entry Point

Unified interface for training models, making predictions, and managing the system.
Supports any ticker with configurable targets and model types.

USAGE:
======
# Train SPY model with defaults
python main.py train --ticker SPY --start 2020-01-01 --end 2024-01-01

# Train ensemble model for AAPL
python main.py train --ticker AAPL --model ensemble --config config/aapl_config.json

# Make predictions
python main.py predict --ticker SPY --model-path data/models/SPY_xgboost_latest.pkl

# Update model with new data
python main.py update --ticker SPY --days 30

# Generate trading signals
python main.py signals --ticker SPY --date today

COMMANDS:
=========
- train: Train new model
- predict: Make predictions on latest data
- update: Update existing model with new data
- signals: Generate trading signals
- backtest: Run backtesting analysis
"""

import argparse
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.model_trainer import ModelTrainer
from pipeline.data_loader import DataLoader


def setup_logging():
    """Setup logging configuration."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('momentum_prediction.log'),
            logging.StreamHandler()
        ]
    )


def train_model(args) -> Dict[str, Any]:
    """Train new model."""
    print(f"🚀 Training {args.model} model for {args.ticker}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        ticker=args.ticker,
        model_type=args.model,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Run training
    results = trainer.train_full_pipeline(
        start_date=args.start,
        end_date=args.end,
        target_name=args.target,
        save_model=True
    )
    
    print(f"\n✅ Training completed!")
    print(f"   Model: {results.get('model_path', 'N/A')}")
    print(f"   ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    
    return results


def make_predictions(args) -> Dict[str, Any]:
    """Make predictions on latest data."""
    print(f"🔮 Making predictions for {args.ticker}")
    
    if args.model_path:
        # Load specific model
        if 'ensemble' in args.model_path:
            from models.ensemble_predictor import EnsemblePullbackPredictor
            model = EnsemblePullbackPredictor.load_ensemble(args.model_path)
        else:
            from models.xgboost_predictor import XGBoostPullbackPredictor
            model = XGBoostPullbackPredictor.load_model(args.model_path)
        
        # Create temporary trainer for predictions
        trainer = ModelTrainer(ticker=args.ticker)
        trainer.model = model
        
        results = trainer.predict_latest(days_back=args.days_back)
    else:
        # Train new model and predict
        trainer = ModelTrainer(ticker=args.ticker, model_type=args.model)
        
        # Quick training on recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        trainer.train_full_pipeline(start_date, end_date, save_model=False)
        results = trainer.predict_latest(days_back=args.days_back)
    
    print(f"\n🎯 Prediction Results:")
    print(f"   Ticker: {results['ticker']}")
    print(f"   Probability: {results['prediction_probability']:.1%}")
    print(f"   Signal: {'BUY' if results['prediction_binary'] else 'HOLD'}")
    print(f"   Strength: {results['signal_strength']}")
    
    return results


def update_model(args) -> Dict[str, Any]:
    """Update existing model with new data."""
    print(f"🔄 Updating model for {args.ticker}")
    
    # For now, retrain the model (incremental training to be implemented)
    trainer = ModelTrainer(
        ticker=args.ticker,
        model_type=args.model,
        output_dir=args.output_dir
    )
    
    # Use recent data for update
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days*2)).strftime('%Y-%m-%d')
    
    results = trainer.train_full_pipeline(
        start_date=start_date,
        end_date=end_date,
        save_model=True
    )
    
    print(f"✅ Model updated successfully!")
    return results


def generate_signals(args) -> Dict[str, Any]:
    """Generate trading signals."""
    print(f"📊 Generating trading signals for {args.ticker}")
    
    # Use prediction function with signal interpretation
    results = make_predictions(args)
    
    # Enhanced signal generation
    probability = results['prediction_probability']
    
    if probability >= 0.7:
        signal = "STRONG SELL"
        action = "Consider reducing position or hedging"
    elif probability >= 0.5:
        signal = "SELL"
        action = "Monitor for pullback opportunity"
    elif probability >= 0.3:
        signal = "NEUTRAL"
        action = "Hold current position"
    else:
        signal = "BUY"
        action = "Consider accumulating on weakness"
    
    signal_results = {
        **results,
        'trading_signal': signal,
        'recommended_action': action,
        'confidence': 'High' if abs(probability - 0.5) > 0.2 else 'Medium'
    }
    
    print(f"\n📈 Trading Signal:")
    print(f"   Signal: {signal}")
    print(f"   Action: {action}")
    print(f"   Confidence: {signal_results['confidence']}")
    
    return signal_results


def run_backtest(args) -> Dict[str, Any]:
    """Run backtesting analysis."""
    print(f"📊 Running backtest for {args.ticker}")
    print("⚠️  Backtesting functionality coming soon!")
    
    # Placeholder for backtesting
    return {
        'status': 'not_implemented',
        'message': 'Backtesting functionality will be added in future release'
    }


def create_default_config(ticker: str) -> Dict[str, Any]:
    """Create default configuration for ticker."""
    
    # SPY-optimized config
    if ticker == 'SPY':
        return {
            "data_source": "yfinance",
            "feature_engines": ["momentum", "volatility"],
            "target_config": {
                "pullback_targets": {
                    "thresholds": [0.02, 0.03, 0.05, 0.07, 0.10],
                    "horizons": [5, 10, 15, 20, 30]
                },
                "mean_reversion_targets": {
                    "sma_periods": [9, 20, 50, 100, 200],
                    "horizons": [5, 10, 15, 20],
                    "reversion_threshold": 0.005
                }
            },
            "model_params": {
                "xgboost": {
                    "n_estimators": 1500,
                    "max_depth": 12,
                    "learning_rate": 0.02,
                    "n_features": 100
                }
            }
        }
    
    # Tech stock config (AAPL, MSFT, etc.)
    elif ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']:
        return {
            "data_source": "yfinance", 
            "feature_engines": ["momentum", "volatility"],
            "target_config": {
                "pullback_targets": {
                    "thresholds": [0.03, 0.05, 0.10, 0.15],
                    "horizons": [3, 5, 10, 15]
                },
                "mean_reversion_targets": {
                    "sma_periods": [10, 20, 50],
                    "horizons": [3, 5, 10],
                    "reversion_threshold": 0.02
                }
            },
            "model_params": {
                "xgboost": {
                    "n_estimators": 1000,
                    "max_depth": 10,
                    "learning_rate": 0.03,
                    "n_features": 75
                }
            }
        }
    
    # Default config for other tickers
    else:
        return {
            "data_source": "yfinance",
            "feature_engines": ["momentum", "volatility"],
            "target_config": {
                "pullback_targets": {
                    "thresholds": [0.02, 0.05, 0.10],
                    "horizons": [5, 10, 15, 20]
                },
                "mean_reversion_targets": {
                    "sma_periods": [20, 50, 100, 200],
                    "horizons": [5, 10, 15, 20]
                }
            },
            "model_params": {
                "xgboost": {
                    "n_estimators": 1000,
                    "max_depth": 10,
                    "learning_rate": 0.03,
                    "n_features": 75
                }
            }
        }


def setup_config_directory():
    """Setup configuration directory with default configs."""
    config_dir = 'config'
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(f'{config_dir}/ticker_configs', exist_ok=True)
    
    # Create default SPY config
    spy_config_path = f'{config_dir}/spy_config.json'
    if not os.path.exists(spy_config_path):
        with open(spy_config_path, 'w') as f:
            json.dump(create_default_config('SPY'), f, indent=2)
    
    return config_dir


def main():
    """Main entry point."""
    setup_logging()
    setup_config_directory()
    
    # Main argument parser
    parser = argparse.ArgumentParser(
        description='Momentum-Based Pullback Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train new model')
    train_parser.add_argument('--ticker', required=True, help='Ticker symbol (e.g., SPY)')
    train_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    train_parser.add_argument('--end', help='End date (YYYY-MM-DD, default: today)')
    train_parser.add_argument('--model', choices=['xgboost', 'ensemble'], default='xgboost',
                             help='Model type (default: xgboost)')
    train_parser.add_argument('--target', default='pullback_5pct_10d',
                             help='Target variable name (default: pullback_5pct_10d)')
    train_parser.add_argument('--config', help='Path to configuration file')
    train_parser.add_argument('--output-dir', default='data/models',
                             help='Output directory for models (default: data/models)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--ticker', required=True, help='Ticker symbol')
    predict_parser.add_argument('--model', choices=['xgboost', 'ensemble'], default='xgboost',
                               help='Model type (if no model-path provided)')
    predict_parser.add_argument('--model-path', help='Path to trained model file')
    predict_parser.add_argument('--days-back', type=int, default=30,
                               help='Days of data to use for prediction (default: 30)')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update existing model')
    update_parser.add_argument('--ticker', required=True, help='Ticker symbol')
    update_parser.add_argument('--model', choices=['xgboost', 'ensemble'], default='xgboost',
                              help='Model type (default: xgboost)')
    update_parser.add_argument('--days', type=int, default=90,
                              help='Days of new data to include (default: 90)')
    update_parser.add_argument('--output-dir', default='data/models',
                              help='Output directory for models')
    
    # Signals command
    signals_parser = subparsers.add_parser('signals', help='Generate trading signals')
    signals_parser.add_argument('--ticker', required=True, help='Ticker symbol')
    signals_parser.add_argument('--model', choices=['xgboost', 'ensemble'], default='xgboost',
                               help='Model type')
    signals_parser.add_argument('--model-path', help='Path to trained model file')
    signals_parser.add_argument('--date', default='today', help='Date for signals (default: today)')
    signals_parser.add_argument('--days-back', type=int, default=30,
                               help='Days of data to use (default: 30)')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--ticker', required=True, help='Ticker symbol')
    backtest_parser.add_argument('--start', required=True, help='Backtest start date')
    backtest_parser.add_argument('--end', help='Backtest end date (default: today)')
    backtest_parser.add_argument('--model', choices=['xgboost', 'ensemble'], default='xgboost',
                                help='Model type')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        print(f"🎯 Momentum Pullback Prediction System")
        print(f"   Command: {args.command}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Route to appropriate function
        if args.command == 'train':
            results = train_model(args)
        elif args.command == 'predict':
            results = make_predictions(args)
        elif args.command == 'update':
            results = update_model(args)
        elif args.command == 'signals':
            results = generate_signals(args)
        elif args.command == 'backtest':
            results = run_backtest(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return
        
        print(f"\n🎉 Command '{args.command}' completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error executing command '{args.command}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()