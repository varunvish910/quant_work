#!/usr/bin/env python3
"""
Test Predictions - Check for anomalies across different years
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from predict_anomalies import AnomalyPredictor
import json
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_year_predictions(year: int):
    """Test predictions for a specific year"""
    print("=" * 80)
    print(f"🔮 Testing {year} Predictions")
    print("=" * 80)
    
    try:
        # Initialize predictor
        predictor = AnomalyPredictor()
        
        # Load model
        model_path = Path("models/correction_predictor.joblib")
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return
        
        predictor.load_model(str(model_path))
        
        # Load data for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        print(f"📊 Loading data for {year}...")
        predictions = predictor.predict_corrections(start_date, end_date)
        
        if predictions is None or predictions.empty:
            print(f"❌ No predictions generated for {year}")
            return
        
        # Analyze predictions
        print(f"\n📈 Prediction Summary for {year}:")
        print(f"   Total days: {len(predictions)}")
        print(f"   Flagged days: {predictions['prediction'].sum()}")
        print(f"   Flag rate: {predictions['prediction'].mean():.2%}")
        
        # Show flagged dates
        flagged = predictions[predictions['prediction'] == 1]
        if not flagged.empty:
            print(f"\n🚨 Flagged dates in {year}:")
            for _, row in flagged.iterrows():
                confidence = row.get('confidence', 'N/A')
                print(f"   {row['date']}: confidence={confidence}")
        else:
            print(f"\n✅ No correction signals found in {year}")
        
        # Save results
        output_path = Path(f"analysis/outputs/{year}_predictions.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error testing {year} predictions: {e}")
        raise

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Test anomaly predictions for specific years")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025], 
                        help="Years to test (default: 2024 2025)")
    parser.add_argument("--year", type=int, help="Single year to test")
    
    args = parser.parse_args()
    
    # Determine which years to test
    if args.year:
        years = [args.year]
    else:
        years = args.years
    
    # Test each year
    for year in years:
        try:
            test_year_predictions(year)
            print("\n")
        except Exception as e:
            print(f"❌ Failed to test {year}: {e}")
            continue
    
    print("🎯 Testing complete!")

if __name__ == "__main__":
    main()