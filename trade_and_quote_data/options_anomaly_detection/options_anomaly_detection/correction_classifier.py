#!/usr/bin/env python3
"""
Simple classifier to test if institutional flow features predict 4%+ corrections
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from target_creator import CorrectionTargetCreator
from feature_extractor import HistoricalFeatureExtractor

class CorrectionPredictor:
    """
    Test if institutional flow features can predict corrections 1-3 days in advance
    """
    
    def __init__(self):
        self.target_creator = CorrectionTargetCreator(correction_threshold=0.04)
        self.feature_extractor = HistoricalFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_training_dataset(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features and targets for the date range"""
        print(f"📊 Creating training dataset from {start_date} to {end_date}")
        
        # Step 1: Create correction targets using price data
        print("🎯 Creating correction targets...")
        price_data = self.target_creator.load_price_data(start_date, end_date)
        corrections = self.target_creator.identify_corrections(price_data)
        targets_df = self.target_creator.create_prediction_targets(corrections)
        
        print(f"   Found {len(corrections)} corrections")
        print(f"   Created {targets_df['target'].sum()} prediction targets")
        
        # Step 2: Extract features for the same period
        print("🔧 Extracting institutional flow features...")
        features_df = self.feature_extractor.extract_features_batch(start_date, end_date)
        
        if features_df.empty:
            print("❌ No features extracted - using dummy data for demonstration")
            return self._create_dummy_dataset()
        
        # Step 3: Merge features with targets
        print("🔗 Merging features with targets...")
        
        # Align dates (convert feature dates to match target format)
        features_df['date_key'] = features_df['date'].dt.strftime('%Y-%m-%d')
        targets_df['date_key'] = targets_df['date'].dt.strftime('%Y-%m-%d')
        
        # Merge on date
        merged_df = pd.merge(features_df, targets_df[['date_key', 'target']], 
                           on='date_key', how='inner')
        
        print(f"✅ Merged dataset: {len(merged_df)} records")
        print(f"   Positive targets: {merged_df['target'].sum()}")
        print(f"   Target ratio: {merged_df['target'].mean():.3f}")
        
        return merged_df, targets_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector"""
        
        # Select feature columns (exclude metadata)
        feature_cols = [col for col in df.columns if col.startswith(('downward_', 'bigmove_', 'risk_'))]
        feature_cols += ['downward_composite', 'bigmove_composite']
        
        # Remove any non-numeric features
        numeric_features = []
        for col in feature_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
        
        if not numeric_features:
            raise ValueError("No numeric features found")
        
        X = df[numeric_features].fillna(0).values
        y = df['target'].values
        
        print(f"📊 Feature matrix: {X.shape}")
        print(f"   Features: {numeric_features}")
        
        return X, y, numeric_features
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple models and return results"""
        print("🤖 Training prediction models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y if y.sum() > 1 else None
        )
        
        results = {}
        
        # 1. Logistic Regression (interpretable)
        print("   Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, class_weight='balanced')
        lr_model.fit(X_train, y_train)
        
        lr_scores = cross_val_score(lr_model, X_scaled, y, cv=3, scoring='precision')
        lr_pred = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test)[:, 1]
        
        results['logistic'] = {
            'model': lr_model,
            'cv_precision': lr_scores.mean(),
            'test_predictions': lr_pred,
            'test_probabilities': lr_proba,
            'test_targets': y_test,
            'feature_importance': lr_model.coef_[0]
        }
        
        # 2. Random Forest (ensemble)
        print("   Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        
        rf_scores = cross_val_score(rf_model, X_scaled, y, cv=3, scoring='precision')
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'model': rf_model,
            'cv_precision': rf_scores.mean(),
            'test_predictions': rf_pred,
            'test_probabilities': rf_proba,
            'test_targets': y_test,
            'feature_importance': rf_model.feature_importances_
        }
        
        # Store best model
        if results['logistic']['cv_precision'] > results['random_forest']['cv_precision']:
            self.model = lr_model
            print(f"   Best model: Logistic Regression (CV Precision: {lr_scores.mean():.3f})")
        else:
            self.model = rf_model
            print(f"   Best model: Random Forest (CV Precision: {rf_scores.mean():.3f})")
        
        return results
    
    def evaluate_models(self, results: Dict, feature_names: List[str]) -> Dict:
        """Evaluate model performance and feature importance"""
        print("\n📊 MODEL EVALUATION RESULTS")
        print("=" * 50)
        
        evaluation = {}
        
        for model_name, result in results.items():
            print(f"\n🤖 {model_name.upper()} RESULTS:")
            print("-" * 30)
            
            y_true = result['test_targets']
            y_pred = result['test_predictions']
            y_proba = result['test_probabilities']
            
            # Classification metrics
            print(f"Cross-Val Precision: {result['cv_precision']:.3f}")
            
            if len(np.unique(y_true)) > 1:  # Only if we have both classes
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred))
                
                print("Confusion Matrix:")
                print(confusion_matrix(y_true, y_pred))
            
            # Feature importance
            importance = result['feature_importance']
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(importance)  # Use absolute values
            }).sort_values('importance', ascending=False)
            
            print(f"\n📈 Top 5 Most Important Features:")
            for _, row in feature_imp_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
            
            evaluation[model_name] = {
                'cv_precision': result['cv_precision'],
                'feature_importance': feature_imp_df,
                'predictions': {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                }
            }
        
        return evaluation
    
    def analyze_signal_combinations(self, df: pd.DataFrame) -> Dict:
        """Analyze which signal combinations work best"""
        print("\n🔍 SIGNAL COMBINATION ANALYSIS")
        print("=" * 50)
        
        # Define signal categories
        signal_tests = {
            'downward_only': ['downward_composite'],
            'bigmove_only': ['bigmove_composite'], 
            'combined_signals': ['downward_composite', 'bigmove_composite'],
            'individual_downward': [col for col in df.columns if col.startswith('downward_') and col != 'downward_composite'],
            'individual_bigmove': [col for col in df.columns if col.startswith('bigmove_') and col != 'bigmove_composite']
        }
        
        results = {}
        
        for test_name, features in signal_tests.items():
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                continue
                
            print(f"\n📊 Testing {test_name}:")
            print(f"   Features: {available_features}")
            
            # Simple correlation analysis
            correlations = []
            for feature in available_features:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    corr = df[feature].corr(df['target'])
                    correlations.append((feature, corr))
            
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"   Correlations with correction targets:")
            for feature, corr in correlations:
                print(f"     {feature}: {corr:.3f}")
            
            results[test_name] = correlations
        
        return results
    
    def _create_dummy_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create dummy dataset for testing when no real data available"""
        print("🎭 Creating dummy dataset for testing...")
        
        np.random.seed(42)
        n_samples = 100
        
        # Create dummy features
        dummy_data = {
            'date': pd.date_range(start='2024-01-01', periods=n_samples, freq='B'),
            'spy_price': 500 + np.random.randn(n_samples) * 20,
            'downward_distribution_score': np.random.exponential(1, n_samples),
            'downward_put_accumulation': np.random.exponential(1.5, n_samples),
            'downward_call_exit_signal': np.random.exponential(1, n_samples),
            'bigmove_tension_index': np.random.exponential(2, n_samples),
            'bigmove_asymmetry_score': np.random.normal(0, 2, n_samples),
            'risk_otm_atm_ratio': np.random.exponential(3, n_samples),
            'downward_composite': np.random.poisson(2, n_samples),
            'bigmove_composite': np.random.poisson(1.5, n_samples)
        }
        
        df = pd.DataFrame(dummy_data)
        
        # Create some realistic targets (corrections more likely when signals high)
        correction_prob = (
            0.1 + 
            0.1 * (df['downward_composite'] / 6) + 
            0.1 * (df['bigmove_composite'] / 5)
        )
        df['target'] = np.random.binomial(1, correction_prob)
        
        # Create dummy targets dataframe
        targets_df = df[['date', 'target']].copy()
        
        print(f"   Dummy data: {len(df)} samples, {df['target'].sum()} positive targets")
        
        return df, targets_df
    
    def run_full_analysis(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """Run complete correction prediction analysis"""
        print("🎯 CORRECTION PREDICTION ANALYSIS")
        print("=" * 60)
        print(f"Goal: Predict 4%+ corrections 1-3 days in advance")
        print(f"Period: {start_date} to {end_date}")
        
        try:
            # Step 1: Create dataset
            merged_df, targets_df = self.create_training_dataset(start_date, end_date)
            
            # Step 2: Prepare features
            X, y, feature_names = self.prepare_features(merged_df)
            self.feature_names = feature_names
            
            # Step 3: Train models
            model_results = self.train_models(X, y)
            
            # Step 4: Evaluate performance
            evaluation = self.evaluate_models(model_results, feature_names)
            
            # Step 5: Analyze signal combinations
            signal_analysis = self.analyze_signal_combinations(merged_df)
            
            # Step 6: Summary
            print(f"\n🎯 FINAL SUMMARY")
            print("=" * 50)
            print(f"Dataset: {len(merged_df)} days, {y.sum()} correction targets")
            print(f"Best Model CV Precision: {max([r['cv_precision'] for r in model_results.values()]):.3f}")
            
            best_features = evaluation['logistic']['feature_importance'].head(3)
            print(f"Top predictive features:")
            for _, row in best_features.iterrows():
                print(f"  - {row['feature']}: {row['importance']:.3f}")
            
            return {
                'dataset': merged_df,
                'model_results': model_results,
                'evaluation': evaluation,
                'signal_analysis': signal_analysis
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None

def main():
    """Run the correction prediction analysis"""
    predictor = CorrectionPredictor()
    
    # Run analysis (will use dummy data if real data not available)
    results = predictor.run_full_analysis("2024-01-01", "2024-12-31")
    
    if results:
        print("\n✅ Analysis complete!")
        print("💡 Next steps:")
        print("   1. Gather more historical options data")
        print("   2. Test on known correction events (July 2024, Feb 2025)")
        print("   3. Optimize feature combinations")
        print("   4. Build real-time prediction pipeline")
    else:
        print("\n❌ Analysis failed - check data availability")

if __name__ == "__main__":
    main()