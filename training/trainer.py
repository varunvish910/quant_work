"""
Unified Model Training Pipeline

Consolidates training logic from all phase files into a single, configurable interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from core.data_loader import DataLoader
from core.features import FeatureEngine
from core.targets import TargetCreator
from core.models import EarlyWarningModel
from utils.constants import (
    TRAIN_START_DATE, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE,
    DRAWDOWN_THRESHOLD, EARLY_WARNING_DAYS, LOOKFORWARD_DAYS
)


class ModelTrainer:
    """Unified model training pipeline"""
    
    def __init__(self, 
                 model_type: str = 'ensemble',
                 feature_sets: Optional[List[str]] = None,
                 start_date: str = TRAIN_START_DATE,
                 end_date: str = TEST_END_DATE):
        """
        Initialize ModelTrainer
        
        Args:
            model_type: Type of model to train ('rf', 'xgboost', 'ensemble')
            feature_sets: List of feature sets to use ['baseline', 'currency', 'volatility', 'all']
            start_date: Start date for data loading
            end_date: End date for data loading
        """
        self.model_type = model_type
        self.feature_sets = feature_sets or ['baseline']
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize components
        self.data_loader = DataLoader(start_date=start_date, end_date=end_date)
        self.feature_engine = FeatureEngine(feature_sets=self.feature_sets)
        self.model = None
        
        # Data storage
        self.raw_data = None
        self.features_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def load_data(self) -> Dict[str, any]:
        """
        Load all required data
        
        Returns:
            Dictionary of loaded data
        """
        print("=" * 80)
        print("📥 LOADING DATA")
        print("=" * 80)
        
        # Determine what data to load based on feature sets
        include_currency = 'currency' in self.feature_sets or 'all' in self.feature_sets
        include_volatility = 'volatility' in self.feature_sets or 'all' in self.feature_sets
        
        self.raw_data = self.data_loader.load_all_data(
            include_sectors=True,  # Always include sectors for baseline features
            include_currency=include_currency,
            include_volatility=include_volatility
        )
        
        return self.raw_data
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Calculate all features
        
        Returns:
            DataFrame with features
        """
        print("\n" + "=" * 80)
        print("🔧 FEATURE ENGINEERING")
        print("=" * 80)
        
        self.features_data = self.feature_engine.calculate_features(
            spy_data=self.raw_data['spy'],
            sector_data=self.raw_data.get('sectors'),
            currency_data=self.raw_data.get('currency'),
            volatility_data=self.raw_data.get('volatility')
        )
        
        return self.features_data
    
    def create_targets(self, 
                      target_type: str = 'early_warning',
                      target_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create targets for training
        
        Args:
            target_type: Type of target ('early_warning', 'mean_reversion', 'pullback', 'all')
            target_params: Parameters for target creation
            
        Returns:
            DataFrame with features and targets
        """
        print("\n" + "=" * 80)
        print("🎯 TARGET CREATION")
        print("=" * 80)
        
        target_params = target_params or {}
        
        # Create target creator from original SPY data (not features)
        creator = TargetCreator(self.raw_data['spy'])
        
        if target_type == 'early_warning':
            data_with_target = creator.create_early_warning_target(**target_params)
            target_column = 'early_warning_target'
        elif target_type == 'mean_reversion':
            data_with_target = creator.create_mean_reversion_target(**target_params)
            target_column = 'mean_reversion_target'
        elif target_type == 'pullback':
            data_with_target = creator.create_pullback_target(**target_params)
            target_column = 'pullback_target'
        elif target_type == 'all':
            data_with_target = creator.create_all_targets()
            target_column = 'early_warning_target'  # Default to early warning
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Merge targets with features on index
        self.features_data = self.features_data.join(
            data_with_target[[col for col in data_with_target.columns if 'target' in col]],
            how='inner'
        )
        
        print(f"✅ Targets merged with features: {len(self.features_data)} records")
        
        return self.features_data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        print("\n" + "=" * 80)
        print("✂️  DATA SPLITTING")
        print("=" * 80)
        
        self.train_data, self.val_data, self.test_data = self.data_loader.train_test_split(
            self.features_data,
            train_end=TRAIN_END_DATE,
            val_end=VAL_END_DATE
        )
        
        return self.train_data, self.val_data, self.test_data
    
    def train(self, 
             target_type: str = 'early_warning',
             target_params: Optional[Dict] = None,
             save_model: bool = True) -> EarlyWarningModel:
        """
        Complete training pipeline
        
        Args:
            target_type: Type of target to train on
            target_params: Parameters for target creation
            save_model: Whether to save the trained model
            
        Returns:
            Trained model
        """
        print("\n" + "=" * 80)
        print("🚀 STARTING TRAINING PIPELINE")
        print("=" * 80)
        print(f"Model type: {self.model_type}")
        print(f"Feature sets: {self.feature_sets}")
        print(f"Target type: {target_type}")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Engineer features
        self.engineer_features()
        
        # Step 3: Create targets
        self.create_targets(target_type=target_type, target_params=target_params)
        
        # Step 4: Split data
        self.split_data()
        
        # Step 5: Train model
        print("\n" + "=" * 80)
        print("🎓 MODEL TRAINING")
        print("=" * 80)
        
        # Get feature columns and target
        feature_columns = self.feature_engine.get_feature_columns()
        target_column = f'{target_type}_target' if target_type != 'all' else 'early_warning_target'
        
        # Prepare training data
        X_train = self.train_data[feature_columns]
        y_train = self.train_data[target_column]
        
        # Create and train model
        self.model = EarlyWarningModel(model_type=self.model_type)
        self.model.fit(X_train, y_train, feature_columns)
        
        # Step 6: Evaluate on validation set
        print("\n" + "=" * 80)
        print("📊 VALIDATION PERFORMANCE")
        print("=" * 80)
        
        X_val = self.val_data[feature_columns]
        y_val = self.val_data[target_column]
        
        val_predictions = self.model.predict(X_val)
        val_probabilities = self.model.predict_proba(X_val)[:, 1]
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        print("\nValidation Set Performance:")
        print(classification_report(y_val, val_predictions, 
                                   target_names=['No Warning', 'Early Warning']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, val_predictions))
        
        if len(np.unique(y_val)) > 1:
            roc_auc = roc_auc_score(y_val, val_probabilities)
            print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        # Step 7: Save model
        if save_model:
            print("\n" + "=" * 80)
            print("💾 SAVING MODEL")
            print("=" * 80)
            self.model.save()
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)
        
        return self.model
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with top features and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = self.model.get_feature_importance()
        return importance_df.head(top_n)
    
    def evaluate_test_set(self, target_type: str = 'early_warning') -> Dict:
        """
        Evaluate model on test set (2024 data)
        
        Args:
            target_type: Type of target used for training
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        print("\n" + "=" * 80)
        print("📊 TEST SET EVALUATION (2024)")
        print("=" * 80)
        
        feature_columns = self.feature_engine.get_feature_columns()
        target_column = f'{target_type}_target'
        
        X_test = self.test_data[feature_columns]
        y_test = self.test_data[target_column]
        
        test_predictions = self.model.predict(X_test)
        test_probabilities = self.model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
        
        print("\nTest Set Performance:")
        print(classification_report(y_test, test_predictions,
                                   target_names=['No Warning', 'Early Warning']))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, test_predictions)
        print(cm)
        
        metrics = {
            'precision': precision_score(y_test, test_predictions),
            'recall': recall_score(y_test, test_predictions),
            'confusion_matrix': cm.tolist()
        }
        
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, test_probabilities)
            print(f"\nROC AUC Score: {roc_auc:.4f}")
            metrics['roc_auc'] = roc_auc
        
        return metrics


if __name__ == "__main__":
    # Test training pipeline
    print("Testing ModelTrainer...")
    
    # Test with baseline features only
    trainer = ModelTrainer(
        model_type='ensemble',
        feature_sets=['baseline'],
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    model = trainer.train(
        target_type='early_warning',
        save_model=False  # Don't save during testing
    )
    
    # Show feature importance
    print("\n" + "=" * 80)
    print("TOP FEATURES")
    print("=" * 80)
    print(trainer.get_feature_importance(top_n=10))
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_test_set()

