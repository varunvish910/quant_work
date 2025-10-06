"""
Modern LSTM Stock Prediction Model
Adapted from the original stocks_rnn implementation to work with current TensorFlow/Keras
Compatible with TensorFlow 2.x and modern Python

Original Copyright 2016 Tencia Lee (Apache License 2.0)
Modernization updates for TensorFlow 2.x compatibility
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import os
import sys
from collections import deque
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModernConfig:
    """Configuration for the modern LSTM model"""
    def __init__(self, size='small'):
        if size.lower() == 'small':
            self.hidden_size = 200
            self.num_layers = 2
            self.dropout_rate = 0.0
            self.learning_rate = 0.001
            self.max_grad_norm = 5.0
        elif size.lower() == 'medium':
            self.hidden_size = 650
            self.num_layers = 2
            self.dropout_rate = 0.5
            self.learning_rate = 0.001
            self.max_grad_norm = 5.0
        elif size.lower() == 'large':
            self.hidden_size = 1500
            self.num_layers = 2
            self.dropout_rate = 0.65
            self.learning_rate = 0.001
            self.max_grad_norm = 10.0
        else:
            raise ValueError(f"Unknown config size: {size}")
        
        # Common parameters
        self.num_steps = 10  # sequence length
        self.batch_size = 32
        self.epochs = 50
        self.patience = 10
        self.validation_split = 0.2

class ModernStockLSTM:
    """Modern LSTM implementation using TensorFlow 2.x/Keras"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self, input_shape):
        """Build the LSTM model using Keras functional API"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # LSTM layers
        for i in range(self.config.num_layers):
            return_sequences = (i < self.config.num_layers - 1)
            model.add(layers.LSTM(
                self.config.hidden_size,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.max_grad_norm
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, data, sequence_length):
        """Prepare sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, data, validation_data=None):
        """Train the LSTM model"""
        # Normalize the data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X, y = self.prepare_sequences(data_scaled, self.config.num_steps)
        
        # Reshape for LSTM input
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model if not already built
        if self.model is None:
            self.build_model((self.config.num_steps, 1))
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.config.patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X, y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Normalize the data using the fitted scaler
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X, _ = self.prepare_sequences(data_scaled, self.config.num_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Make predictions
        predictions_scaled = self.model.predict(X)
        
        # Inverse transform to original scale
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def evaluate(self, data):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Normalize the data
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X, y = self.prepare_sequences(data_scaled, self.config.num_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Evaluate
        loss, mae = self.model.evaluate(X, y, verbose=0)
        
        # Also get predictions for more detailed analysis
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        actual = self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        
        return {
            'loss': loss,
            'mae': mae,
            'predictions': predictions,
            'actual': actual,
            'mse': np.mean((predictions - actual) ** 2),
            'rmse': np.sqrt(np.mean((predictions - actual) ** 2))
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save(filepath)
        
        # Also save the scaler
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Also load the scaler
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

def generate_sample_data(n_samples=1000, seed=42):
    """Generate sample stock return data for testing"""
    np.random.seed(seed)
    
    # Generate synthetic stock returns with some patterns
    t = np.linspace(0, 4*np.pi, n_samples)
    trend = 0.0001 * t
    seasonal = 0.01 * np.sin(t) + 0.005 * np.sin(2*t)
    noise = np.random.normal(0, 0.02, n_samples)
    
    returns = trend + seasonal + noise
    
    # Add some momentum and mean reversion
    for i in range(1, len(returns)):
        momentum = 0.1 * returns[i-1]
        mean_reversion = -0.05 * np.mean(returns[max(0, i-10):i])
        returns[i] += momentum + mean_reversion
    
    return returns

def train_and_evaluate_lstm(config_size='small', use_sample_data=True):
    """Train and evaluate the LSTM model"""
    
    print(f"Training LSTM model with {config_size} configuration...")
    
    # Create configuration
    config = ModernConfig(config_size)
    
    # Create model
    lstm_model = ModernStockLSTM(config)
    
    if use_sample_data:
        # Generate sample data
        print("Using sample synthetic data...")
        data = generate_sample_data(2000)
        
        # Split data
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
    else:
        # TODO: Load real stock data
        raise NotImplementedError("Real data loading not implemented yet")
    
    print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train the model
    print("Training model...")
    history = lstm_model.train(train_data)
    
    # Evaluate on test data
    print("Evaluating model...")
    test_results = lstm_model.evaluate(test_data)
    
    print(f"Test Results:")
    print(f"  MSE: {test_results['mse']:.6f}")
    print(f"  RMSE: {test_results['rmse']:.6f}")
    print(f"  MAE: {test_results['mae']:.6f}")
    
    return lstm_model, test_results, history

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM stock prediction model')
    parser.add_argument('--config_size', type=str, default='small', 
                       choices=['small', 'medium', 'large'],
                       help='Model configuration size')
    parser.add_argument('--use_sample_data', action='store_true', default=True,
                       help='Use synthetic sample data')
    
    args = parser.parse_args()
    
    # Train and evaluate
    model, results, history = train_and_evaluate_lstm(
        config_size=args.config_size,
        use_sample_data=args.use_sample_data
    )
    
    print("Training completed!")