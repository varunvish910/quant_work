#!/usr/bin/env python3
"""
Setup script for Momentum-Based Pullback Prediction System
"""

import os
import sys
import subprocess
import json

def create_directories():
    """Create necessary directories."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/models',
        'config/ticker_configs',
        'logs',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_default_configs():
    """Create default configuration files."""
    
    # SPY configuration
    spy_config = {
        "ticker": "SPY",
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
            },
            "ensemble": {
                "model_weights": None,
                "use_lstm": True
            }
        },
        "training": {
            "test_size": 0.2,
            "validation_size": 0.15,
            "random_state": 42,
            "min_samples": 1000
        }
    }
    
    with open('config/spy_config.json', 'w') as f:
        json.dump(spy_config, f, indent=2)
    
    print("✅ Created SPY configuration")
    
    # QQQ configuration (tech-focused)
    qqq_config = spy_config.copy()
    qqq_config["ticker"] = "QQQ"
    qqq_config["target_config"]["pullback_targets"]["thresholds"] = [0.03, 0.05, 0.08, 0.12]
    
    with open('config/ticker_configs/qqq_config.json', 'w') as f:
        json.dump(qqq_config, f, indent=2)
    
    print("✅ Created QQQ configuration")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def verify_installation():
    """Verify that key dependencies are working."""
    print("🔍 Verifying installation...")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import yfinance
        print("✅ Core dependencies verified")
        
        # Optional dependencies
        try:
            import tensorflow
            print("✅ TensorFlow available (LSTM models enabled)")
        except ImportError:
            print("⚠️  TensorFlow not available (LSTM models disabled)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_example_script():
    """Create an example usage script."""
    example_script = '''#!/usr/bin/env python3
"""
Example usage of the Momentum Pullback Prediction System
"""

def run_spy_example():
    """Run complete example for SPY."""
    print("🚀 Running SPY Momentum Prediction Example")
    
    # Train model
    import subprocess
    import sys
    
    print("\\n📚 Step 1: Training SPY model...")
    cmd = [
        sys.executable, "main.py", "train",
        "--ticker", "SPY",
        "--start", "2022-01-01", 
        "--end", "2024-01-01",
        "--model", "xgboost"
    ]
    subprocess.run(cmd)
    
    print("\\n🔮 Step 2: Making predictions...")
    cmd = [
        sys.executable, "main.py", "predict",
        "--ticker", "SPY"
    ]
    subprocess.run(cmd)
    
    print("\\n📊 Step 3: Generating trading signals...")
    cmd = [
        sys.executable, "main.py", "signals", 
        "--ticker", "SPY"
    ]
    subprocess.run(cmd)

if __name__ == '__main__':
    run_spy_example()
'''
    
    with open('example.py', 'w') as f:
        f.write(example_script)
    
    print("✅ Created example.py")

def main():
    """Main setup function."""
    print("🎯 Setting up Momentum-Based Pullback Prediction System")
    print("=" * 60)
    
    # Create directories
    create_directories()
    print()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed due to dependency issues")
        return False
    print()
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup failed due to verification issues")
        return False
    print()
    
    # Create configurations
    create_default_configs()
    print()
    
    # Create example
    create_example_script()
    print()
    
    print("✅ Setup completed successfully!")
    print("\n🎉 Next steps:")
    print("1. Run the example: python example.py")
    print("2. Train your first model: python main.py train --ticker SPY --start 2022-01-01")
    print("3. Make predictions: python main.py predict --ticker SPY")
    print("4. Read the README.md for detailed usage instructions")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)