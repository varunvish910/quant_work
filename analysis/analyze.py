#!/usr/bin/env python3
"""
Unified Analysis System - Single entry point for all analysis tasks

Usage:
    python analyze.py --type performance --model latest --period 2024
    python analyze.py --type features --top 20
    python analyze.py --type predictions --threshold 0.8
    python analyze.py --type backtest --start 2023-01-01 --end 2024-12-31
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.reports.performance import PerformanceAnalyzer
from analysis.reports.features import FeatureAnalyzer
from analysis.reports.predictions import PredictionAnalyzer
from analysis.reports.backtest import BacktestAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Unified Analysis System')
    parser.add_argument('--type', required=True,
                       choices=['performance', 'features', 'predictions', 'backtest'],
                       help='Type of analysis to perform')
    
    # Performance analysis args
    parser.add_argument('--model', default='latest',
                       help='Model to analyze (latest, specific path, or model name)')
    parser.add_argument('--period', default='2024',
                       help='Time period for analysis')
    
    # Feature analysis args
    parser.add_argument('--top', type=int, default=20,
                       help='Number of top features to analyze')
    
    # Prediction analysis args
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold for analysis')
    
    # Backtest args
    parser.add_argument('--start', default='2023-01-01',
                       help='Backtest start date')
    parser.add_argument('--end', default='2024-12-31',
                       help='Backtest end date')
    
    # General args
    parser.add_argument('--output', default='output',
                       help='Output directory for reports')
    parser.add_argument('--format', choices=['text', 'json', 'html'], default='text',
                       help='Output format')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize analyzer based on type
    if args.type == 'performance':
        analyzer = PerformanceAnalyzer(
            model=args.model,
            period=args.period,
            output_dir=args.output,
            verbose=args.verbose
        )
    elif args.type == 'features':
        analyzer = FeatureAnalyzer(
            top_n=args.top,
            output_dir=args.output,
            verbose=args.verbose
        )
    elif args.type == 'predictions':
        analyzer = PredictionAnalyzer(
            threshold=args.threshold,
            output_dir=args.output,
            verbose=args.verbose
        )
    elif args.type == 'backtest':
        analyzer = BacktestAnalyzer(
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output,
            verbose=args.verbose
        )
    
    # Run analysis
    print(f"Running {args.type} analysis...")
    results = analyzer.analyze()
    
    # Generate report
    analyzer.generate_report(results, format=args.format)
    
    print(f"✅ Analysis completed. Report saved to {args.output}")

if __name__ == "__main__":
    main()