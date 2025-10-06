#!/usr/bin/env python3
"""
Command-Line Interface for SPX Early Warning System

Unified entry point for all operations: training, prediction, and analysis.
"""

import click
from datetime import datetime
import sys

from training.trainer import ModelTrainer
from training.validator import ModelValidator
from core.models import EarlyWarningModel
from core.data_loader import DataLoader
from core.features import FeatureEngine
from utils.constants import (
    TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE
)


@click.group()
def cli():
    """SPX Early Warning System - Unified CLI"""
    pass


@cli.command()
@click.option('--model', '-m', type=click.Choice(['rf', 'xgboost', 'ensemble']), 
              default='ensemble', help='Model type to train')
@click.option('--features', '-f', multiple=True, 
              default=['baseline'], help='Feature sets to use (baseline, currency, volatility, all)')
@click.option('--target', '-t', type=click.Choice(['early_warning', 'mean_reversion', 'pullback']),
              default='early_warning', help='Target type')
@click.option('--start-date', default=TRAIN_START_DATE, help='Start date for data')
@click.option('--end-date', default=TEST_END_DATE, help='End date for data')
@click.option('--no-save', is_flag=True, help='Do not save trained model')
def train(model, features, target, start_date, end_date, no_save):
    """Train a new model"""
    
    click.echo("=" * 80)
    click.echo("🚀 SPX EARLY WARNING SYSTEM - TRAINING")
    click.echo("=" * 80)
    click.echo(f"Model: {model}")
    click.echo(f"Features: {', '.join(features)}")
    click.echo(f"Target: {target}")
    click.echo(f"Date range: {start_date} to {end_date}")
    click.echo("=" * 80)
    
    try:
        # Convert features tuple to list
        feature_list = list(features)
        
        # Create trainer
        trainer = ModelTrainer(
            model_type=model,
            feature_sets=feature_list,
            start_date=start_date,
            end_date=end_date
        )
        
        # Train model
        trained_model = trainer.train(
            target_type=target,
            save_model=not no_save
        )
        
        # Show feature importance
        click.echo("\n" + "=" * 80)
        click.echo("📊 TOP 20 MOST IMPORTANT FEATURES")
        click.echo("=" * 80)
        importance = trainer.get_feature_importance(top_n=20)
        click.echo(importance.to_string(index=False))
        
        # Validate 2024 events
        click.echo("\n" + "=" * 80)
        click.echo("🚨 VALIDATING 2024 CRITICAL EVENTS")
        click.echo("=" * 80)
        
        validator = ModelValidator(
            model=trained_model,
            features_data=trainer.features_data,
            feature_columns=trainer.feature_engine.get_feature_columns()
        )
        
        event_results = validator.validate_2024_events()
        
        # Evaluate test set
        test_metrics = trainer.evaluate_test_set(target_type=target)
        
        click.echo("\n✅ Training complete!")
        
    except Exception as e:
        click.echo(f"\n❌ Training failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', type=click.Choice(['rf', 'xgboost', 'ensemble']),
              default='ensemble', help='Model type to load')
@click.option('--date', '-d', help='Date for prediction (YYYY-MM-DD). Default: today')
@click.option('--days', '-n', type=int, default=5, help='Number of days to predict')
def predict(model, date, days):
    """Make predictions with trained model"""
    
    click.echo("=" * 80)
    click.echo("🔮 SPX EARLY WARNING SYSTEM - PREDICTION")
    click.echo("=" * 80)
    
    try:
        # Load model
        click.echo(f"Loading {model} model...")
        trained_model = EarlyWarningModel.load(model_type=model)
        
        # Determine prediction date
        if date:
            pred_date = datetime.strptime(date, '%Y-%m-%d')
        else:
            pred_date = datetime.now()
        
        click.echo(f"Prediction date: {pred_date.date()}")
        click.echo(f"Prediction window: {days} days")
        
        # Load recent data
        start_date = (pred_date - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = pred_date.strftime('%Y-%m-%d')
        
        loader = DataLoader(start_date=start_date, end_date=end_date)
        data = loader.load_all_data()
        
        # Calculate features
        feature_engine = FeatureEngine(feature_sets=['all'])  # Use all available
        features = feature_engine.calculate_features(
            spy_data=data['spy'],
            sector_data=data.get('sectors'),
            currency_data=data.get('currency'),
            volatility_data=data.get('volatility')
        )
        
        # Get latest data
        latest_features = features.tail(days)
        
        # Make predictions
        predictions = trained_model.predict(latest_features)
        probabilities = trained_model.predict_proba(latest_features)[:, 1]
        
        # Display results
        click.echo("\n" + "=" * 80)
        click.echo("PREDICTIONS")
        click.echo("=" * 80)
        
        for i, (idx, pred, prob) in enumerate(zip(latest_features.index, predictions, probabilities)):
            status = "🚨 WARNING" if pred == 1 else "✅ CLEAR"
            click.echo(f"{idx.date()}: {status} (probability: {prob:.2%})")
        
        # Summary
        warning_days = predictions.sum()
        if warning_days > 0:
            click.echo(f"\n⚠️  WARNING: {warning_days}/{days} days show early warning signal")
        else:
            click.echo(f"\n✅ All clear for next {days} days")
        
    except Exception as e:
        click.echo(f"\n❌ Prediction failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--model', '-m', type=click.Choice(['rf', 'xgboost', 'ensemble']),
              default='ensemble', help='Model type to load')
def analyze(start_date, end_date, model):
    """Analyze model performance over a date range"""
    
    click.echo("=" * 80)
    click.echo("📊 SPX EARLY WARNING SYSTEM - ANALYSIS")
    click.echo("=" * 80)
    
    try:
        # Load model
        click.echo(f"Loading {model} model...")
        trained_model = EarlyWarningModel.load(model_type=model)
        
        # Load data
        loader = DataLoader(start_date=TRAIN_START_DATE, end_date=end_date)
        data = loader.load_all_data()
        
        # Calculate features
        feature_engine = FeatureEngine(feature_sets=['all'])
        features = feature_engine.calculate_features(
            spy_data=data['spy'],
            sector_data=data.get('sectors'),
            currency_data=data.get('currency'),
            volatility_data=data.get('volatility')
        )
        
        # Create targets
        from core.targets import TargetCreator
        creator = TargetCreator(data['spy'])
        features_with_targets = creator.create_early_warning_target()
        features = features.join(features_with_targets['early_warning_target'], how='inner')
        
        # Create validator
        validator = ModelValidator(
            model=trained_model,
            features_data=features,
            feature_columns=feature_engine.get_feature_columns()
        )
        
        # Run backtest
        results = validator.backtest(start_date=start_date, end_date=end_date)
        
        click.echo("\n✅ Analysis complete!")
        
    except Exception as e:
        click.echo(f"\n❌ Analysis failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
def info():
    """Show system information and available models"""
    
    click.echo("=" * 80)
    click.echo("SPX EARLY WARNING SYSTEM - INFORMATION")
    click.echo("=" * 80)
    
    click.echo("\n📊 Available Feature Sets:")
    click.echo("  - baseline: Core 8 features (volatility, MA distance, momentum, sector rotation)")
    click.echo("  - currency: USD/JPY, DXY, EUR/USD features for carry trade detection")
    click.echo("  - volatility: VIX, VVIX, term structure features")
    click.echo("  - all: All feature sets combined")
    
    click.echo("\n🎯 Available Targets:")
    click.echo("  - early_warning: Signal 3-5 days before 5%+ drawdowns")
    click.echo("  - mean_reversion: Predict bounces after pullbacks")
    click.echo("  - pullback: Identify healthy pullback patterns")
    
    click.echo("\n🤖 Available Models:")
    click.echo("  - rf: Random Forest")
    click.echo("  - xgboost: XGBoost")
    click.echo("  - ensemble: Voting ensemble of RF + XGBoost (recommended)")
    
    click.echo("\n📅 Data Split:")
    click.echo(f"  - Train: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    click.echo(f"  - Validation: {TRAIN_END_DATE} to {VAL_END_DATE}")
    click.echo(f"  - Test: {VAL_END_DATE} to {TEST_END_DATE}")
    
    click.echo("\n" + "=" * 80)


if __name__ == '__main__':
    # Add pandas import for predict command
    import pandas as pd
    cli()

