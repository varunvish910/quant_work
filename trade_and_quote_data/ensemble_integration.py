#!/usr/bin/env python3
"""
Ensemble Integration & Optimization
Phase 4: Advanced multi-model ensemble system

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
import logging
from datetime import datetime
from pathlib import Path
import json
import pickle
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleIntegration:
    """Advanced ensemble integration and optimization system"""
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        self.optimization_results = {}
        self.feature_selectors = {}
        self.scalers = {}
        
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        logger.info("Creating base models for ensemble...")
        
        base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'logistic': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        
        return base_models
    
    def optimize_hyperparameters(self, model, X_train, y_train, param_grid, model_name):
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            # Create model with trial parameters
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                    'random_state': 42,
                    'class_weight': 'balanced'
                }
                model_trial = RandomForestClassifier(**params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
                model_trial = XGBClassifier(**params)
                
            elif model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
                model_trial = GradientBoostingClassifier(**params)
            
            else:
                return 0.5  # Skip optimization for other models
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model_trial.fit(X_tr, y_tr)
                y_pred = model_trial.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        logger.info(f"Best {model_name} ROC AUC: {study.best_value:.3f}")
        return study.best_params
    
    def feature_selection_recursive(self, X_train, y_train, base_model, target_features=30):
        """Recursive feature elimination for ensemble"""
        from sklearn.feature_selection import RFE
        
        logger.info(f"Performing recursive feature elimination to {target_features} features...")
        
        # Use Random Forest for feature selection
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rfe = RFE(estimator=rf_selector, n_features_to_select=target_features, step=5)
        rfe.fit(X_train, y_train)
        
        selected_features = X_train.columns[rfe.support_]
        logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features, rfe
    
    def create_voting_ensemble(self, base_models, X_train, y_train):
        """Create voting classifier ensemble"""
        logger.info("Creating voting ensemble...")
        
        # Prepare models for voting
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Hard voting
        voting_hard = VotingClassifier(estimators=estimators, voting='hard')
        voting_hard.fit(X_train, y_train)
        
        # Soft voting
        voting_soft = VotingClassifier(estimators=estimators, voting='soft')
        voting_soft.fit(X_train, y_train)
        
        return {
            'voting_hard': voting_hard,
            'voting_soft': voting_soft
        }
    
    def create_stacking_ensemble(self, base_models, X_train, y_train):
        """Create stacking classifier ensemble"""
        logger.info("Creating stacking ensemble...")
        
        # Prepare base estimators
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta-learners
        meta_learners = {
            'logistic': LogisticRegression(random_state=42, class_weight='balanced'),
            'rf_meta': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
            'xgb_meta': XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
        }
        
        stacking_models = {}
        for meta_name, meta_model in meta_learners.items():
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_model,
                cv=TimeSeriesSplit(n_splits=3),
                stack_method='predict_proba'
            )
            stacking_clf.fit(X_train, y_train)
            stacking_models[f'stacking_{meta_name}'] = stacking_clf
        
        return stacking_models
    
    def create_weighted_ensemble(self, base_models, X_train, y_train, X_val, y_val):
        """Create weighted ensemble based on validation performance"""
        logger.info("Creating weighted ensemble...")
        
        # Get predictions from each model
        predictions = {}
        weights = {}
        
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            
            # Calculate weight based on F1 score
            y_pred_binary = (y_pred > 0.5).astype(int)
            f1 = f1_score(y_val, y_pred_binary)
            
            predictions[name] = y_pred
            weights[name] = f1
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0/len(weights) for k in weights.keys()}
        
        logger.info("Model weights:")
        for name, weight in weights.items():
            logger.info(f"  {name}: {weight:.3f}")
        
        return weights, predictions
    
    def optimize_ensemble_weights(self, predictions_dict, y_true):
        """Optimize ensemble weights using Optuna"""
        logger.info("Optimizing ensemble weights...")
        
        model_names = list(predictions_dict.keys())
        
        def objective(trial):
            # Suggest weights for each model
            weights = []
            for i, name in enumerate(model_names):
                if i < len(model_names) - 1:
                    weight = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
                    weights.append(weight)
                else:
                    # Last weight is constrained to sum to 1
                    weights.append(max(0.0, 1.0 - sum(weights)))
            
            # Normalize weights
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
            
            # Calculate weighted prediction
            weighted_pred = np.zeros(len(y_true))
            for i, name in enumerate(model_names):
                weighted_pred += weights[i] * predictions_dict[name]
            
            # Evaluate with F1 score
            y_pred_binary = (weighted_pred > 0.5).astype(int)
            return f1_score(y_true, y_pred_binary)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200, show_progress_bar=False)
        
        # Extract optimal weights
        optimal_weights = {}
        weights = []
        for i, name in enumerate(model_names):
            if i < len(model_names) - 1:
                weight = study.best_params[f'weight_{name}']
                weights.append(weight)
            else:
                weights.append(max(0.0, 1.0 - sum(weights)))
        
        # Normalize
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        
        for i, name in enumerate(model_names):
            optimal_weights[name] = weights[i]
        
        logger.info(f"Optimal ensemble F1: {study.best_value:.3f}")
        logger.info("Optimal weights:")
        for name, weight in optimal_weights.items():
            logger.info(f"  {name}: {weight:.3f}")
        
        return optimal_weights
    
    def evaluate_ensemble(self, ensemble_pred, y_true, ensemble_name):
        """Evaluate ensemble performance"""
        
        # Binary predictions
        y_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_true, ensemble_pred),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
            'accuracy': np.mean(y_true == y_pred_binary)
        }
        
        # High confidence metrics
        high_conf_mask = ensemble_pred > 0.8
        if np.sum(high_conf_mask) > 0:
            metrics['precision_80'] = precision_score(y_true[high_conf_mask], 
                                                    y_pred_binary[high_conf_mask], 
                                                    zero_division=0)
            metrics['n_signals_80'] = np.sum(high_conf_mask)
        else:
            metrics['precision_80'] = 0.0
            metrics['n_signals_80'] = 0
        
        # Ultra high confidence metrics
        ultra_conf_mask = ensemble_pred > 0.9
        if np.sum(ultra_conf_mask) > 0:
            metrics['precision_90'] = precision_score(y_true[ultra_conf_mask], 
                                                    y_pred_binary[ultra_conf_mask], 
                                                    zero_division=0)
            metrics['n_signals_90'] = np.sum(ultra_conf_mask)
        else:
            metrics['precision_90'] = 0.0
            metrics['n_signals_90'] = 0
        
        logger.info(f"{ensemble_name} Performance:")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Precision@80%: {metrics['precision_80']:.3f} ({metrics['n_signals_80']} signals)")
        logger.info(f"  Precision@90%: {metrics['precision_90']:.3f} ({metrics['n_signals_90']} signals)")
        
        return metrics
    
    def run_ensemble_analysis(self, features, targets):
        """Run complete ensemble analysis"""
        logger.info("Starting comprehensive ensemble analysis...")
        
        results = {}
        
        for target_name in targets.columns:
            logger.info(f"\n=== Analyzing target: {target_name} ===")
            
            target = targets[target_name]
            
            # Skip if insufficient positive samples
            if target.sum() < 50:
                logger.warning(f"Skipping {target_name}: insufficient positive samples ({target.sum()})")
                continue
            
            # Align data
            common_index = features.index.intersection(target.index)
            X = features.loc[common_index]
            y = target.loc[common_index]
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Time-based split
            split_idx = int(len(X) * 0.7)
            val_split_idx = int(len(X) * 0.85)
            
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:val_split_idx]
            y_val = y.iloc[split_idx:val_split_idx]
            X_test = X.iloc[val_split_idx:]
            y_test = y.iloc[val_split_idx:]
            
            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Feature selection
            selected_features, feature_selector = self.feature_selection_recursive(
                X_train, y_train, None, target_features=min(30, len(X_train.columns))
            )
            
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_selected),
                index=X_train_selected.index,
                columns=X_train_selected.columns
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val_selected),
                index=X_val_selected.index,
                columns=X_val_selected.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_selected),
                index=X_test_selected.index,
                columns=X_test_selected.columns
            )
            
            # Create and optimize base models
            base_models = self.create_base_models()
            
            # Optimize key models
            optimized_models = {}
            for name, model in base_models.items():
                if name in ['random_forest', 'xgboost', 'gradient_boosting']:
                    best_params = self.optimize_hyperparameters(
                        model, X_train_scaled, y_train, {}, name
                    )
                    
                    if name == 'random_forest':
                        optimized_models[name] = RandomForestClassifier(**best_params)
                    elif name == 'xgboost':
                        optimized_models[name] = XGBClassifier(**best_params)
                    elif name == 'gradient_boosting':
                        optimized_models[name] = GradientBoostingClassifier(**best_params)
                else:
                    optimized_models[name] = model
            
            # Train optimized models
            for name, model in optimized_models.items():
                model.fit(X_train_scaled, y_train)
            
            # Create different ensemble types
            ensemble_results = {}
            
            # 1. Voting ensembles
            voting_ensembles = self.create_voting_ensemble(optimized_models, X_train_scaled, y_train)
            for name, ensemble in voting_ensembles.items():
                pred = ensemble.predict_proba(X_test_scaled)[:, 1]
                ensemble_results[name] = self.evaluate_ensemble(pred, y_test, name)
                ensemble_results[name]['predictions'] = pred
            
            # 2. Stacking ensembles
            stacking_ensembles = self.create_stacking_ensemble(optimized_models, X_train_scaled, y_train)
            for name, ensemble in stacking_ensembles.items():
                pred = ensemble.predict_proba(X_test_scaled)[:, 1]
                ensemble_results[name] = self.evaluate_ensemble(pred, y_test, name)
                ensemble_results[name]['predictions'] = pred
            
            # 3. Weighted ensemble with optimized weights
            weights, val_predictions = self.create_weighted_ensemble(
                optimized_models, X_train_scaled, y_train, X_val_scaled, y_val
            )
            
            # Get test predictions for weighted ensemble
            test_predictions = {}
            for name, model in optimized_models.items():
                test_predictions[name] = model.predict_proba(X_test_scaled)[:, 1]
            
            optimal_weights = self.optimize_ensemble_weights(val_predictions, y_val)
            
            # Calculate optimized weighted prediction
            weighted_pred = np.zeros(len(y_test))
            for name, weight in optimal_weights.items():
                weighted_pred += weight * test_predictions[name]
            
            ensemble_results['weighted_optimized'] = self.evaluate_ensemble(
                weighted_pred, y_test, 'weighted_optimized'
            )
            ensemble_results['weighted_optimized']['predictions'] = weighted_pred
            ensemble_results['weighted_optimized']['weights'] = optimal_weights
            
            # Store results
            results[target_name] = {
                'ensemble_results': ensemble_results,
                'selected_features': list(selected_features),
                'feature_selector': feature_selector,
                'scaler': scaler,
                'base_models': optimized_models
            }
        
        return results
    
    def save_ensemble_results(self, results):
        """Save ensemble analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('analysis/outputs/ensemble_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save performance metrics
        performance_summary = {}
        for target, data in results.items():
            performance_summary[target] = {}
            for ensemble_name, metrics in data['ensemble_results'].items():
                # Remove predictions for JSON serialization
                clean_metrics = {k: v for k, v in metrics.items() if k != 'predictions'}
                performance_summary[target][ensemble_name] = clean_metrics
        
        results_file = output_dir / f'ensemble_performance_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        # Save models and scalers
        models_file = output_dir / f'ensemble_models_{timestamp}.pkl'
        with open(models_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Ensemble results saved to {results_file}")
        logger.info(f"Models saved to {models_file}")
        
        return output_dir


def main():
    """Main execution function"""
    print("🎯 Starting Ensemble Integration & Optimization")
    print("Phase 4: Advanced Multi-Model Ensemble System")
    print("=" * 60)
    
    # Load data from previous phases
    from run_analysis import SimplifiedTargetAnalyzer
    
    # Get enhanced features and targets
    analyzer = SimplifiedTargetAnalyzer()
    spy, vix = analyzer.download_data()
    features = analyzer.create_features(spy, vix)
    
    # Create target matrix (focus on best performing targets)
    targets = pd.DataFrame(index=features.index)
    
    # Best targets from Phase 1
    best_targets = [
        (0.02, 20),  # 2pct_20d - champion
        (0.02, 15),  # 2pct_15d - strong performer
        (0.02, 10),  # 2pct_10d - good performer
    ]
    
    for magnitude, horizon in best_targets:
        target = analyzer.create_target(spy, magnitude, horizon)
        targets[f'{int(magnitude*100)}pct_{horizon}d'] = target
    
    # Add VIX spike targets
    vix_targets = analyzer.create_vix_spike_targets(vix)
    targets['vix_spike_10d'] = vix_targets['vix_spike_10d']  # Best VIX performer
    
    # Initialize ensemble system
    ensemble_system = EnsembleIntegration()
    
    # Run ensemble analysis
    results = ensemble_system.run_ensemble_analysis(features, targets)
    
    # Save results
    output_dir = ensemble_system.save_ensemble_results(results)
    
    # Print comprehensive summary
    print(f"\n✅ Ensemble Analysis Complete!")
    print(f"Targets analyzed: {len(results)}")
    print(f"Results saved to: {output_dir}")
    
    # Performance summary
    print("\n📊 Ensemble Performance Summary:")
    for target, data in results.items():
        print(f"\n{target}:")
        
        best_ensemble = None
        best_f1 = 0
        
        for ensemble_name, metrics in data['ensemble_results'].items():
            f1 = metrics['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_ensemble = ensemble_name
            
            print(f"  {ensemble_name}: F1={f1:.3f}, ROC AUC={metrics['roc_auc']:.3f}, "
                  f"Precision@80%={metrics['precision_80']:.3f}")
        
        print(f"  🏆 Best: {best_ensemble} (F1={best_f1:.3f})")
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n🎉 Phase 4 Ensemble Integration Complete!")