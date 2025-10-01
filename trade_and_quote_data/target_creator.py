#!/usr/bin/env python3
"""
Create target labels for 4%+ correction prediction
Identifies correction events and labels days 1-3 before as prediction targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import yfinance as yf

from correction_detector import CorrectionDetector
from price_data_loader import PriceDataLoader  
from target_generator import TargetGenerator

class CorrectionTargetCreator:
    """
    Creates binary targets for predicting 4%+ corrections 1-3 days in advance
    """
    
    def __init__(self, correction_threshold: float = 0.04, lookback_days: int = 20):
        """
        Args:
            correction_threshold: Minimum drop % to qualify as correction (default 4%)
            lookback_days: Days to look back for peak before correction
        """
        self.correction_threshold = correction_threshold
        self.lookback_days = lookback_days
        self.correction_events = []
        self.price_data = None
        
        # Initialize components
        self.detector = CorrectionDetector(correction_threshold, lookback_days)
        self.loader = PriceDataLoader()
        self.generator = TargetGenerator()
        
    def load_price_data(self, start_date: str, end_date: str, ticker: str = "SPY") -> pd.DataFrame:
        """Load historical price data for target creation"""
        self.price_data = self.loader.load_price_data(start_date, end_date, ticker)
        return self.price_data
        
    def identify_corrections(self, price_data: pd.DataFrame) -> List[Dict]:
        """Identify all correction events in the price data"""
        self.correction_events = self.detector.identify_corrections(price_data)
        return self.correction_events
        
    def create_prediction_targets(self, correction_events: List[Dict]) -> pd.DataFrame:
        """Create binary target labels for correction prediction"""
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_price_data() first.")
        return self.generator.create_prediction_targets(self.price_data, correction_events)
        
    def validate_targets(self, targets_df: pd.DataFrame) -> Dict:
        """Validate target distribution and timing"""
        return self.generator.validate_targets(targets_df)
        
    def export_targets(self, targets_df: pd.DataFrame, output_path: str):
        """Export targets to file for model training"""
        self.generator.export_targets(targets_df, output_path)
        
    def plot_corrections(self, price_data: pd.DataFrame, correction_events: List[Dict]):
        """Visualize identified corrections for validation"""
        from correction_analyzer import CorrectionPlotter
        plotter = CorrectionPlotter()
        plotter.plot_corrections(price_data, correction_events)

# Example usage and testing
if __name__ == "__main__":
    
    # Initialize target creator
    creator = CorrectionTargetCreator(
        correction_threshold=0.04,  # 4% corrections
        lookback_days=20
    )
    
    # Load price data
    print("📊 Loading historical price data...")
    price_data = creator.load_price_data("2016-01-01", "2023-12-31")
    
    # Identify corrections
    print("🔍 Identifying 4%+ correction events...")
    corrections = creator.identify_corrections(price_data)
    print(f"Found {len(corrections)} correction events")
    
    # Create prediction targets
    print("🎯 Creating prediction targets...")
    targets = creator.create_prediction_targets(corrections)
    
    # Validate targets
    print("✅ Validating target distribution...")
    validation = creator.validate_targets(targets)
    
    # Export for model training
    print("💾 Exporting targets...")
    creator.export_targets(targets, "data/correction_targets.parquet")
    
    # Analyze patterns
    print("📊 Analyzing correction patterns...")
    from correction_analyzer import CorrectionAnalyzer
    analyzer = CorrectionAnalyzer(corrections)
    patterns = analyzer.analyze_correction_patterns()
    
    # Identify major events
    print("🚨 Identifying major correction events...")
    major_events = analyzer.identify_major_events()
    
    if major_events:
        print(f"Major events found:")
        for i, event in enumerate(major_events[:3]):  # Show top 3
            print(f"  {i+1}. {event['peak_date']}: {event['magnitude']:.1%} ({event['severity']})")
    
    print("🎯 Target creation complete!")