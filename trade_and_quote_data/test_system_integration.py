#!/usr/bin/env python3
"""
Test System Integration
Simple test to validate the core components work together

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_download():
    """Test basic data download functionality"""
    logger.info("Testing data download...")
    
    try:
        # Download SPY data
        spy = yf.download('SPY', start='2023-01-01', end='2024-01-01', progress=False)
        logger.info(f"SPY data downloaded: {len(spy)} days")
        
        # Download VIX data
        vix = yf.download('^VIX', start='2023-01-01', end='2024-01-01', progress=False)
        logger.info(f"VIX data downloaded: {len(vix)} days")
        
        return True, spy, vix
        
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        return False, None, None

def test_target_creation():
    """Test target label creation"""
    logger.info("Testing target creation...")
    
    success, spy, vix = test_data_download()
    if not success:
        return False
    
    try:
        # Create simple pullback target
        labels = pd.Series(0, index=spy.index, name='pullback_5pct_10d')
        
        for i in range(len(spy) - 10):
            current_price = spy['Close'].iloc[i]
            future_lows = spy['Low'].iloc[i+1:i+11]
            
            if len(future_lows) > 0:
                min_future = future_lows.min()
                if pd.notna(min_future) and pd.notna(current_price) and current_price > 0:
                    drawdown = (min_future / current_price) - 1
                    
                    if drawdown <= -0.05:  # 5% pullback
                        labels.iloc[i] = 1
        
        positive_rate = labels.mean()
        logger.info(f"Target created: {labels.sum()} positive labels ({positive_rate:.3f} rate)")
        
        return True, labels
        
    except Exception as e:
        logger.error(f"Target creation failed: {e}")
        return False, None

def test_feature_creation():
    """Test basic feature creation"""
    logger.info("Testing feature creation...")
    
    success, spy, vix = test_data_download()
    if not success:
        return False
    
    try:
        features = pd.DataFrame(index=spy.index)
        
        # Basic features
        features['returns'] = spy['Close'].pct_change()
        features['sma_20'] = spy['Close'].rolling(20).mean()
        features['rsi'] = calculate_rsi(spy['Close'])
        features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
        
        # VIX features
        if 'Close' in vix.columns:
            vix_close = vix['Close']
            features['vix_level'] = vix_close.reindex(spy.index, method='ffill')
            features['vix_momentum'] = features['vix_level'].pct_change(5)
        
        # Drop NaN values
        features = features.dropna()
        
        logger.info(f"Features created: {len(features.columns)} features, {len(features)} observations")
        
        return True, features
        
    except Exception as e:
        logger.error(f"Feature creation failed: {e}")
        return False, None

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def test_model_training():
    """Test basic model training"""
    logger.info("Testing model training...")
    
    # Get features and targets
    feature_success, features = test_feature_creation()
    target_success, targets = test_target_creation()
    
    if not (feature_success and target_success):
        return False
    
    try:
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        X = features.loc[common_index]
        y = targets.loc[common_index]
        
        if len(X) < 100:
            logger.warning(f"Insufficient data: {len(X)} samples")
            return False
        
        # Split data
        split_point = int(len(X) * 0.7)
        X_train = X.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_test = y.iloc[split_point:]
        
        logger.info(f"Training set: {len(X_train)} samples, {y_train.sum()} positive")
        logger.info(f"Test set: {len(X_test)} samples, {y_test.sum()} positive")
        
        if y_train.sum() < 5:
            logger.warning("Too few positive samples for training")
            return False
        
        # Simple model using scikit-learn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"Model performance: ROC AUC={roc_auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False

def run_integration_test():
    """Run complete integration test"""
    logger.info("Starting system integration test...")
    
    tests = [
        ("Data Download", test_data_download),
        ("Target Creation", test_target_creation),
        ("Feature Creation", test_feature_creation),
        ("Model Training", test_model_training)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            if test_name == "Data Download":
                success, _, _ = test_func()
            elif test_name in ["Target Creation", "Feature Creation"]:
                success, _ = test_func()
            else:
                success = test_func()
            
            results[test_name] = success
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name}: {status}")
            
        except Exception as e:
            logger.error(f"{test_name} test error: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("=" * 50)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 50)
    
    all_passed = True
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    logger.info("=" * 50)
    overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    logger.info(f"Overall: {overall_status}")
    
    return all_passed

if __name__ == "__main__":
    success = run_integration_test()
    
    if success:
        print("\n🎉 System integration test completed successfully!")
        print("Ready to run the full optimal target finder analysis.")
    else:
        print("\n❌ System integration test failed!")
        print("Please fix the issues before running the full analysis.")