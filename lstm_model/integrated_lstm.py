"""
LSTM Model Integration for Existing Ensemble System
Adapts LSTM to work with the current data format and ensemble structure
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    LSTM Classifier that conforms to sklearn interface for ensemble integration
    """
    
    def __init__(self, 
                 sequence_length=30,
                 hidden_size=128, 
                 num_layers=2,
                 dropout_rate=0.3,
                 learning_rate=0.001,
                 epochs=50,
                 batch_size=32,
                 patience=10,
                 random_state=42):
        """
        Initialize LSTM classifier
        
        Args:
            sequence_length: Number of time steps to look back
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            random_state: Random seed
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.classes_ = None
        self.n_features_in_ = None
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _create_sequences(self, X, y=None):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        # Ensure we have enough data for sequences
        if len(X) < self.sequence_length:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        for i in range(self.sequence_length, len(X)):
            sequence = X[i-self.sequence_length:i]
            sequences.append(sequence)
            
            if y is not None:
                targets.append(y[i])
        
        sequences = np.array(sequences)
        if y is not None:
            targets = np.array(targets)
            return sequences, targets
        
        return sequences
    
    def _build_model(self, input_shape):
        """Build the LSTM model"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)
            model.add(layers.LSTM(
                self.hidden_size,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
        
        # Dense layers for classification
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        n_classes = len(self.classes_)
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Compile model
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y):
        """
        Fit the LSTM model
        
        Args:
            X: Feature matrix (samples, features)
            y: Target vector
        """
        # Store classes and feature info
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Build model
        input_shape = (self.sequence_length, self.n_features_in_)
        self.model = self._build_model(input_shape)
        
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.patience,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                monitor='val_loss'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.history_ = history
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Handle binary vs multiclass
        if len(self.classes_) == 2:
            # Binary classification
            probs = np.column_stack([1 - predictions.flatten(), predictions.flatten()])
        else:
            # Multiclass
            probs = predictions
        
        # Pad probabilities for sequences that we couldn't predict
        if len(probs) < len(X):
            # For the first sequence_length samples, use the mean probability
            mean_probs = np.mean(probs, axis=0)
            padding = np.tile(mean_probs, (self.sequence_length, 1))
            probs = np.vstack([padding, probs])
        
        return probs
    
    def predict(self, X):
        """
        Predict classes
        
        Args:
            X: Feature matrix
            
        Returns:
            Class predictions
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Save Keras model
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Save scaler and other attributes
        model_data = {
            'scaler': self.scaler,
            'classes_': self.classes_,
            'n_features_in_': self.n_features_in_,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        
        print(f"LSTM model saved to {filepath} and {model_path}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        # Load model data
        model_data = joblib.load(filepath)
        
        self.scaler = model_data['scaler']
        self.classes_ = model_data['classes_']
        self.n_features_in_ = model_data['n_features_in_']
        self.sequence_length = model_data['sequence_length']
        self.hidden_size = model_data['hidden_size']
        self.num_layers = model_data['num_layers']
        self.dropout_rate = model_data['dropout_rate']
        self.random_state = model_data['random_state']
        
        # Load Keras model
        model_path = filepath.replace('.pkl', '_model.h5')
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        
        return self

def create_price_features(df):
    """
    Create price-based features for LSTM from OHLCV data
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with enhanced features
    """
    features = df.copy()
    
    # Price returns
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility features
    features['volatility_5d'] = features['returns'].rolling(5).std()
    features['volatility_20d'] = features['returns'].rolling(20).std()
    
    # Price ratios
    features['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
    features['oc_ratio'] = (df['Close'] - df['Open']) / df['Open']
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        features[f'ma_{period}'] = df['Close'].rolling(period).mean()
        features[f'close_ma_{period}_ratio'] = df['Close'] / features[f'ma_{period}']
    
    # Volume features
    features['volume_ma_20'] = df['Volume'].rolling(20).mean()
    features['volume_ratio'] = df['Volume'] / features['volume_ma_20']
    
    # Technical indicators
    features['rsi'] = calculate_rsi(df['Close'])
    features['bb_upper'], features['bb_lower'] = calculate_bollinger_bands(df['Close'])
    features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    # Drop original OHLCV columns and NaN values
    features = features.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    features = features.dropna()
    
    return features

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, period=20, std_mult=2):
    """Calculate Bollinger Bands"""
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = ma + (std * std_mult)
    lower = ma - (std * std_mult)
    return upper, lower

def test_lstm_integration():
    """Test LSTM integration with sample data"""
    
    print("Testing LSTM integration...")
    
    # Load sample data
    try:
        # Try to load actual data
        import sys
        sys.path.append('/Users/varun/code/quant_final_final/trade_and_quote_data')
        df = pd.read_parquet('/Users/varun/code/quant_final_final/trade_and_quote_data/data/sectors/XLK.parquet')
        print(f"Loaded real data: {df.shape}")
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        # Generate synthetic data for testing
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        price = 100
        prices = []
        volumes = []
        
        for _ in range(1000):
            price *= (1 + np.random.normal(0, 0.02))
            prices.append(price)
            volumes.append(np.random.randint(1000000, 5000000))
        
        df = pd.DataFrame({
            'Close': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Open': [prices[i-1] if i > 0 else prices[i] for i in range(len(prices))],
            'Volume': volumes
        }, index=dates)
        
        print(f"Generated synthetic data: {df.shape}")
    
    # Create features
    features = create_price_features(df)
    print(f"Features created: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Create synthetic targets (pullback prediction)
    # Simulate pullback detection based on future returns
    future_returns = df['Close'].pct_change(5).shift(-5)  # 5-day forward return
    targets = (future_returns < -0.02).astype(int)  # 2% pullback
    
    # Align targets with features
    targets = targets.reindex(features.index).fillna(0).astype(int)
    
    print(f"Target distribution: {np.bincount(targets)}")
    
    # Split data
    split_idx = int(0.8 * len(features))
    X_train = features.iloc[:split_idx].values
    y_train = targets.iloc[:split_idx].values
    X_test = features.iloc[split_idx:].values
    y_test = targets.iloc[split_idx:].values
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Initialize and train LSTM
    lstm = LSTMClassifier(
        sequence_length=20,
        hidden_size=64,
        num_layers=1,
        epochs=10,  # Reduced for testing
        patience=5
    )
    
    print("Training LSTM...")
    lstm.fit(X_train, y_train)
    
    # Test predictions
    print("Making predictions...")
    probs = lstm.predict_proba(X_test)
    preds = lstm.predict(X_test)
    
    print(f"Prediction probabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Prediction distribution: {np.bincount(preds)}")
    
    # Calculate accuracy
    accuracy = np.mean(preds == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    print("✅ LSTM integration test completed successfully!")
    
    return lstm

if __name__ == "__main__":
    test_lstm_integration()