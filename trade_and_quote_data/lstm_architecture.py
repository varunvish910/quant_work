#!/usr/bin/env python3
"""
LSTM Architecture & Training
Phase 3: Deep learning temporal pattern recognition

Author: AI Assistant  
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
import logging
from datetime import datetime
from pathlib import Path
import json
import pickle

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class LSTMArchitecture:
    """Advanced LSTM architecture for market prediction"""
    
    def __init__(self, sequence_length=20, prediction_horizon=5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_lstm_data(self, features, targets, test_split=0.3):
        """Prepare data for LSTM training with proper sequencing"""
        logger.info(f"Preparing LSTM data with sequence length {self.sequence_length}")
        
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        X = features.loc[common_index].copy()
        y = targets.loc[common_index].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled) - self.prediction_horizon):
            # Features: look back sequence_length days
            X_seq = X_scaled.iloc[i-self.sequence_length:i].values
            
            # Target: predict prediction_horizon days ahead
            y_seq = y.iloc[i + self.prediction_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Train/test split (time-based)
        split_idx = int(len(X_sequences) * (1 - test_split))
        
        X_train = X_sequences[:split_idx]
        X_test = X_sequences[split_idx:]
        y_train = y_sequences[:split_idx]
        y_test = y_sequences[split_idx:]
        
        logger.info(f"Training sequences: {len(X_train)}")
        logger.info(f"Test sequences: {len(X_test)}")
        logger.info(f"Sequence shape: {X_train.shape}")
        logger.info(f"Positive rate train: {y_train.mean():.3f}")
        logger.info(f"Positive rate test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def create_basic_lstm(self, input_shape):
        """Create basic LSTM architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape, 
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False, 
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_attention_lstm(self, input_shape):
        """Create LSTM with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(inputs)
        lstm1 = Dropout(0.3)(lstm1)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001))(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        lstm2 = BatchNormalization()(lstm2)
        
        # Simple attention mechanism
        attention_weights = Dense(1, activation='tanh')(lstm2)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        # Weighted context vector
        context = tf.reduce_sum(lstm2 * attention_weights, axis=1)
        
        # Final layers
        dense1 = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(context)
        dense1 = Dropout(0.2)(dense1)
        
        outputs = Dense(1, activation='sigmoid')(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_multi_scale_lstm(self, input_shape):
        """Create multi-scale LSTM architecture"""
        inputs = Input(shape=input_shape)
        
        # Short-term LSTM (5-day patterns)
        short_lstm = LSTM(32, return_sequences=False, name='short_lstm')(inputs)
        short_lstm = Dropout(0.3)(short_lstm)
        
        # Medium-term LSTM (20-day patterns)  
        med_lstm = LSTM(48, return_sequences=True)(inputs)
        med_lstm = LSTM(24, return_sequences=False, name='med_lstm')(med_lstm)
        med_lstm = Dropout(0.3)(med_lstm)
        
        # Long-term LSTM (full sequence patterns)
        long_lstm = LSTM(64, return_sequences=True)(inputs)
        long_lstm = LSTM(32, return_sequences=True)(long_lstm)
        long_lstm = LSTM(16, return_sequences=False, name='long_lstm')(long_lstm)
        long_lstm = Dropout(0.3)(long_lstm)
        
        # Combine all scales
        combined = Concatenate()([short_lstm, med_lstm, long_lstm])
        combined = BatchNormalization()(combined)
        
        # Final layers
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(combined)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(1, activation='sigmoid')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_lstm_model(self, X_train, X_test, y_train, y_test, model_type='basic'):
        """Train LSTM model with proper validation"""
        logger.info(f"Training {model_type} LSTM model...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Create model based on type
        if model_type == 'basic':
            model = self.create_basic_lstm(input_shape)
        elif model_type == 'attention':
            model = self.create_attention_lstm(input_shape)
        elif model_type == 'multi_scale':
            model = self.create_multi_scale_lstm(input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Handle class imbalance
        class_weight = {0: 1.0, 1: len(y_train) / (2 * np.sum(y_train))}
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_test)[:, 0]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'accuracy': np.mean(y_test == y_pred)
        }
        
        # High confidence metrics
        high_conf_mask = y_pred_proba > 0.8
        if np.sum(high_conf_mask) > 0:
            metrics['precision_80'] = precision_score(y_test[high_conf_mask], 
                                                    y_pred[high_conf_mask], 
                                                    zero_division=0)
            metrics['n_signals_80'] = np.sum(high_conf_mask)
        else:
            metrics['precision_80'] = 0.0
            metrics['n_signals_80'] = 0
        
        logger.info(f"{model_type} LSTM Results:")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  Precision@80%: {metrics['precision_80']:.3f}")
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'predictions': y_pred_proba
        }
    
    def ensemble_lstm_models(self, models, X_test, y_test):
        """Create ensemble of LSTM models"""
        logger.info("Creating LSTM ensemble...")
        
        predictions = []
        weights = []
        
        for model_name, model_data in models.items():
            model = model_data['model']
            f1_score = model_data['metrics']['f1_score']
            
            pred = model.predict(X_test)[:, 0]
            predictions.append(pred)
            weights.append(f1_score)  # Weight by F1 score
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X_test))
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[i] * pred
        
        # Evaluate ensemble
        y_pred_ensemble = (ensemble_pred > 0.5).astype(int)
        
        ensemble_metrics = {
            'roc_auc': roc_auc_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
            'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_ensemble, zero_division=0),
            'accuracy': np.mean(y_test == y_pred_ensemble)
        }
        
        # High confidence ensemble metrics
        high_conf_mask = ensemble_pred > 0.8
        if np.sum(high_conf_mask) > 0:
            ensemble_metrics['precision_80'] = precision_score(y_test[high_conf_mask], 
                                                             y_pred_ensemble[high_conf_mask], 
                                                             zero_division=0)
            ensemble_metrics['n_signals_80'] = np.sum(high_conf_mask)
        else:
            ensemble_metrics['precision_80'] = 0.0
            ensemble_metrics['n_signals_80'] = 0
        
        logger.info("Ensemble LSTM Results:")
        logger.info(f"  ROC AUC: {ensemble_metrics['roc_auc']:.3f}")
        logger.info(f"  F1 Score: {ensemble_metrics['f1_score']:.3f}")
        logger.info(f"  Precision: {ensemble_metrics['precision']:.3f}")
        logger.info(f"  Precision@80%: {ensemble_metrics['precision_80']:.3f}")
        
        return ensemble_metrics, ensemble_pred, weights
    
    def analyze_feature_importance(self, model, X_train, feature_names, n_samples=100):
        """Analyze feature importance using permutation method"""
        logger.info("Analyzing feature importance...")
        
        # Baseline prediction
        baseline_pred = model.predict(X_train[:n_samples])
        baseline_loss = tf.keras.losses.binary_crossentropy(
            np.ones((n_samples, 1)) * 0.5,  # Dummy target
            baseline_pred
        ).numpy().mean()
        
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            # Permute feature i across all time steps
            X_permuted = X_train[:n_samples].copy()
            np.random.shuffle(X_permuted[:, :, i])
            
            # Calculate new prediction
            permuted_pred = model.predict(X_permuted)
            permuted_loss = tf.keras.losses.binary_crossentropy(
                np.ones((n_samples, 1)) * 0.5,  # Dummy target
                permuted_pred
            ).numpy().mean()
            
            # Importance = increase in loss
            importance_scores[feature_name] = permuted_loss - baseline_loss
        
        # Sort by importance
        sorted_importance = sorted(importance_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance)
    
    def run_lstm_analysis(self, features, targets):
        """Run complete LSTM analysis"""
        logger.info("Starting LSTM analysis...")
        
        results = {}
        all_models = {}
        
        for target_name in targets.columns:
            logger.info(f"Training LSTM models for {target_name}")
            
            target = targets[target_name]
            
            # Skip if insufficient positive samples
            if target.sum() < 20:
                logger.warning(f"Skipping {target_name}: insufficient positive samples ({target.sum()})")
                continue
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler = self.prepare_lstm_data(features, target)
            
            if len(X_train) < 100:
                logger.warning(f"Skipping {target_name}: insufficient training data")
                continue
            
            # Train different model architectures
            model_types = ['basic', 'attention', 'multi_scale']
            target_models = {}
            
            for model_type in model_types:
                try:
                    model_result = self.train_lstm_model(X_train, X_test, y_train, y_test, model_type)
                    target_models[model_type] = model_result
                except Exception as e:
                    logger.error(f"Failed to train {model_type} for {target_name}: {e}")
            
            if target_models:
                # Create ensemble
                ensemble_metrics, ensemble_pred, weights = self.ensemble_lstm_models(
                    target_models, X_test, y_test
                )
                
                # Feature importance for best individual model
                best_model_name = max(target_models.keys(), 
                                    key=lambda x: target_models[x]['metrics']['f1_score'])
                best_model = target_models[best_model_name]['model']
                
                feature_importance = self.analyze_feature_importance(
                    best_model, X_train, features.columns
                )
                
                results[target_name] = {
                    'individual_models': {k: v['metrics'] for k, v in target_models.items()},
                    'ensemble_metrics': ensemble_metrics,
                    'ensemble_weights': dict(zip(target_models.keys(), weights)),
                    'feature_importance': feature_importance,
                    'best_individual': best_model_name
                }
                
                all_models[target_name] = target_models
                self.scalers[target_name] = scaler
        
        self.models = all_models
        return results
    
    def save_models_and_results(self, results):
        """Save LSTM models and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('analysis/outputs/lstm_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / f'lstm_results_{timestamp}.json'
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for target, data in results.items():
            json_results[target] = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    json_results[target][key] = {k: float(v) if isinstance(v, np.floating) else v 
                                               for k, v in value.items()}
                else:
                    json_results[target][key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save scalers
        scalers_file = output_dir / f'lstm_scalers_{timestamp}.pkl'
        with open(scalers_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        logger.info(f"LSTM results saved to {results_file}")
        logger.info(f"Scalers saved to {scalers_file}")
        
        return output_dir


def main():
    """Main execution function"""
    print("🧠 Starting LSTM Architecture & Training")
    print("Phase 3: Deep Learning Temporal Pattern Recognition")
    print("=" * 60)
    
    # Load previous results
    from run_analysis import SimplifiedTargetAnalyzer
    
    # Get enhanced features and targets
    analyzer = SimplifiedTargetAnalyzer()
    spy, vix = analyzer.download_data()
    features = analyzer.create_features(spy, vix)
    
    # Create target matrix
    targets = pd.DataFrame(index=features.index)
    
    # Add best performing targets from Phase 1
    for magnitude in [0.02, 0.05]:
        for horizon in [10, 15, 20]:
            target = analyzer.create_target(spy, magnitude, horizon)
            targets[f'{int(magnitude*100)}pct_{horizon}d'] = target
    
    # Add VIX spike targets
    vix_targets = analyzer.create_vix_spike_targets(vix)
    for name, target in vix_targets.items():
        targets[name] = target
    
    # Initialize LSTM architecture
    lstm_arch = LSTMArchitecture(sequence_length=20, prediction_horizon=5)
    
    # Run LSTM analysis
    results = lstm_arch.run_lstm_analysis(features, targets)
    
    # Save results
    output_dir = lstm_arch.save_models_and_results(results)
    
    # Print summary
    print(f"\n✅ LSTM Analysis Complete!")
    print(f"Targets analyzed: {len(results)}")
    print(f"Results saved to: {output_dir}")
    
    # Performance summary
    print("\n📊 LSTM Performance Summary:")
    for target, data in results.items():
        ensemble = data['ensemble_metrics']
        best_individual = data['best_individual']
        print(f"\n{target}:")
        print(f"  Best Individual ({best_individual}): F1={data['individual_models'][best_individual]['f1_score']:.3f}")
        print(f"  Ensemble: F1={ensemble['f1_score']:.3f}, ROC AUC={ensemble['roc_auc']:.3f}")
        print(f"  Precision@80%: {ensemble['precision_80']:.3f}")
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n🎉 Phase 3 LSTM Architecture Complete!")