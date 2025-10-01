"""
SPY Options Anomaly Detection - Main Orchestration Script
========================================================

This is the main entry point for the SPY options anomaly detection system.
It orchestrates the entire pipeline from data loading to signal generation.

Usage:
    python main.py --start-date 2024-01-01 --end-date 2024-12-31 --mode analysis
    python main.py --start-date 2025-01-01 --end-date 2025-01-31 --mode detection
    python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31

Author: AI Assistant
Date: 2025-10-01
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import our modules
from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector
from analysis_engine import OptionsAnalysisEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AnomalyDetectionPipeline:
    """
    Main pipeline for SPY options anomaly detection
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = data_dir
        self.feature_engine = OptionsFeatureEngine(data_dir)
        self.anomaly_detector = OptionsAnomalyDetector(data_dir)
        self.analysis_engine = OptionsAnalysisEngine(data_dir)
        
    def run_analysis(self, start_date: str, end_date: str) -> Dict:
        """
        Run comprehensive historical analysis
        """
        logger.info(f"Running analysis from {start_date} to {end_date}")
        
        # Analyze historical patterns
        patterns = self.analysis_engine.analyze_historical_patterns(start_date, end_date)
        
        if not patterns:
            logger.error("No patterns found in historical data")
            return {}
        
        # Generate report
        report_file = f"analysis_report_{start_date}_{end_date}.html"
        self.analysis_engine.generate_report(patterns, report_file)
        
        logger.info(f"Analysis completed. Report saved to {report_file}")
        return patterns
    
    def run_detection(self, start_date: str, end_date: str) -> Dict:
        """
        Run anomaly detection on specified date range
        """
        logger.info(f"Running anomaly detection from {start_date} to {end_date}")
        
        # Train models on historical data (use previous year for training)
        train_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        train_end = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Training models on {train_start} to {train_end}")
        self.anomaly_detector.train_on_historical_data(train_start, train_end)
        
        # Run detection on target period
        detection_results = {}
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_date.weekday() < 5:
                logger.info(f"Processing {date_str}")
                results = self.anomaly_detector.process_daily_anomalies(date_str)
                
                if results:
                    detection_results[date_str] = results
                    
                    # Log key metrics
                    metrics = results.get('metrics', {})
                    signals = results.get('signals', {})
                    
                    logger.info(f"{date_str}: {metrics.get('ensemble_anomaly_rate', 0):.2%} anomaly rate, "
                              f"{signals.get('direction', 'neutral')} signal, "
                              f"{signals.get('quality', 'low')} quality")
            
            current_date += timedelta(days=1)
        
        # Save results
        self._save_detection_results(detection_results, start_date, end_date)
        
        logger.info(f"Detection completed. Processed {len(detection_results)} days")
        return detection_results
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """
        Run backtesting on historical data
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Train models on first half of data
        mid_date = (datetime.strptime(start_date, '%Y-%m-%d') + 
                   (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')) / 2)
        mid_date_str = mid_date.strftime('%Y-%m-%d')
        
        logger.info(f"Training on {start_date} to {mid_date_str}")
        self.anomaly_detector.train_on_historical_data(start_date, mid_date_str)
        
        # Backtest on second half
        backtest_results = self.analysis_engine.backtest_anomaly_detection(
            mid_date_str, end_date, self.anomaly_detector
        )
        
        if backtest_results:
            logger.info("Backtest completed successfully")
            logger.info(f"Average anomaly rate: {backtest_results.get('avg_anomaly_rate', 0):.2%}")
            logger.info(f"High quality signals: {backtest_results.get('high_quality_signals', 0)}")
        
        return backtest_results
    
    def run_feature_engineering(self, start_date: str, end_date: str) -> Dict:
        """
        Run feature engineering on historical data
        """
        logger.info(f"Running feature engineering from {start_date} to {end_date}")
        
        # Process date range
        daily_aggregates = self.feature_engine.process_date_range(start_date, end_date)
        
        if daily_aggregates:
            # Save aggregates
            self.feature_engine.save_aggregates(daily_aggregates, 
                                              f"daily_aggregates_{start_date}_{end_date}.csv")
            
            logger.info(f"Feature engineering completed. Processed {len(daily_aggregates)} days")
            
            # Log sample metrics
            sample_date = list(daily_aggregates.keys())[0]
            sample_metrics = daily_aggregates[sample_date]
            logger.info(f"Sample metrics for {sample_date}: {sample_metrics}")
        
        return daily_aggregates
    
    def _save_detection_results(self, results: Dict, start_date: str, end_date: str):
        """Save detection results to file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for date, data in results.items():
            serializable_results[date] = {
                'metrics': data.get('metrics', {}),
                'signals': data.get('signals', {}),
                'anomaly_results': {
                    k: v.tolist() if hasattr(v, 'tolist') else v 
                    for k, v in data.get('anomaly_results', {}).items()
                }
            }
        
        # Save to JSON
        output_file = f"detection_results_{start_date}_{end_date}.json"
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
    
    def run_full_pipeline(self, start_date: str, end_date: str) -> Dict:
        """
        Run the complete pipeline: feature engineering -> detection -> analysis
        """
        logger.info("Running full anomaly detection pipeline")
        
        # 1. Feature engineering
        logger.info("Step 1: Feature Engineering")
        features = self.run_feature_engineering(start_date, end_date)
        
        # 2. Anomaly detection
        logger.info("Step 2: Anomaly Detection")
        detection = self.run_detection(start_date, end_date)
        
        # 3. Analysis
        logger.info("Step 3: Analysis")
        analysis = self.run_analysis(start_date, end_date)
        
        return {
            'features': features,
            'detection': detection,
            'analysis': analysis
        }


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description='SPY Options Anomaly Detection System')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--mode', required=True, 
                       choices=['analysis', 'detection', 'backtest', 'features', 'full'],
                       help='Mode to run')
    parser.add_argument('--data-dir', default='data/options_chains/SPY',
                       help='Data directory path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(args.data_dir)
    
    try:
        if args.mode == 'analysis':
            results = pipeline.run_analysis(args.start_date, args.end_date)
        elif args.mode == 'detection':
            results = pipeline.run_detection(args.start_date, args.end_date)
        elif args.mode == 'backtest':
            results = pipeline.run_backtest(args.start_date, args.end_date)
        elif args.mode == 'features':
            results = pipeline.run_feature_engineering(args.start_date, args.end_date)
        elif args.mode == 'full':
            results = pipeline.run_full_pipeline(args.start_date, args.end_date)
        
        logger.info(f"Pipeline completed successfully in {args.mode} mode")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
