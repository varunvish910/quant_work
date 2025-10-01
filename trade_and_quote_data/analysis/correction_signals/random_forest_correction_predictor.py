"""
Random Forest Correction Predictor
=================================

Implements Random Forest model for predicting 5-10% corrections
based on hedging patterns. Uses the identified features from
correction prediction analysis.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class RandomForestCorrectionPredictor:
    """
    Random Forest model for predicting 5-10% corrections
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.training_data = None
        
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """Load options data for a specific date"""
        try:
            year = date[:4]
            month = date[5:7]
            date_formatted = date.replace('-', '')
            file_path = self.data_dir / year / month / f"SPY_options_snapshot_{date_formatted}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                else:
                    df['date'] = pd.to_datetime(date)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()
    
    def extract_features(self, df: pd.DataFrame, spy_price: float) -> dict:
        """Extract features for Random Forest model"""
        if df.empty:
            return {}
        
        puts = df[df['option_type'] == 'P'].copy()
        calls = df[df['option_type'] == 'C'].copy()
        
        if puts.empty:
            return {}
        
        features = {}
        
        # 1. Institutional Dominance (90.5% frequency in corrections)
        if 'dte' in df.columns:
            institutional_puts = puts[puts['dte'] > 7]
            total_put_oi = puts['oi_proxy'].sum()
            institutional_pct = institutional_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
            features['institutional_dominance'] = 1 if institutional_pct > 0.8 else 0
        else:
            features['institutional_dominance'] = 0
        
        # 2. High Defensive Positioning (60% frequency in corrections)
        deep_otm_puts = puts[puts['strike'] < spy_price * 0.85]
        deep_otm_pct = deep_otm_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
        features['high_defensive'] = 1 if deep_otm_pct > 0.1 else 0
        
        # 3. Hedging Intensity Trend
        total_call_oi = calls['oi_proxy'].sum() if not calls.empty else 0
        pc_ratio = total_put_oi / (total_call_oi + 1e-6)
        features['hedging_intensity'] = min(1.0, (pc_ratio - 1) * 2)
        
        # 4. Volume/OI Ratio (accumulation vs speculation)
        total_put_vol = puts['volume'].sum()
        vol_oi_ratio = total_put_vol / (total_put_oi + 1e-6)
        features['vol_oi_ratio'] = vol_oi_ratio
        
        # 5. Strike Concentration
        strike_oi = puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
        top_5_oi = strike_oi.head(5).sum()
        concentration = top_5_oi / (total_put_oi + 1e-6)
        features['strike_concentration'] = concentration
        
        # 6. Long-term Positioning
        if 'dte' in df.columns:
            long_term_puts = puts[puts['dte'] > 60]
            long_term_pct = long_term_puts['oi_proxy'].sum() / (total_put_oi + 1e-6)
            features['long_term_positioning'] = long_term_pct
        else:
            features['long_term_positioning'] = 0
        
        # 7. ATM vs OTM Ratio
        atm_puts = puts[abs(puts['strike'] - spy_price) <= 10]
        otm_puts = puts[puts['strike'] < spy_price * 0.95]
        
        atm_oi = atm_puts['oi_proxy'].sum()
        otm_oi = otm_puts['oi_proxy'].sum()
        features['atm_otm_ratio'] = atm_oi / (otm_oi + 1e-6)
        
        # 8. Put/Call Ratio
        features['pc_ratio'] = pc_ratio
        
        return features
    
    def build_training_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build training dataset with features and labels"""
        
        print(f"📊 BUILDING RANDOM FOREST TRAINING DATASET")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 50)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Build price data for correction detection
        price_data = []
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime('%Y-%m-%d')
                df = self.load_daily_data(date_str)
                
                if not df.empty and 'underlying_price' in df.columns:
                    spy_price = df['underlying_price'].iloc[0]
                    price_data.append({
                        'date': current_date,
                        'spy_price': spy_price
                    })
            
            current_date += timedelta(days=1)
        
        if not price_data:
            return pd.DataFrame()
        
        df_prices = pd.DataFrame(price_data)
        df_prices = df_prices.sort_values('date').reset_index(drop=True)
        
        # Create labels (corrections in next 10 days)
        labels = []
        for i in range(len(df_prices)):
            current_price = df_prices.iloc[i]['spy_price']
            current_date = df_prices.iloc[i]['date']
            
            # Look ahead 10 days for correction
            future_prices = df_prices[df_prices['date'] > current_date].head(10)
            
            if not future_prices.empty:
                min_future_price = future_prices['spy_price'].min()
                decline_pct = (current_price - min_future_price) / current_price
                
                # Label: 1 if 5-10% correction, 0 otherwise
                label = 1 if 0.05 <= decline_pct <= 0.10 else 0
            else:
                label = 0
            
            labels.append(label)
        
        df_prices['correction_label'] = labels
        
        # Extract features for each day
        feature_data = []
        for i, row in df_prices.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            df = self.load_daily_data(date_str)
            
            if not df.empty and 'underlying_price' in df.columns:
                spy_price = df['underlying_price'].iloc[0]
                features = self.extract_features(df, spy_price)
                
                if features:
                    features['date'] = row['date']
                    features['spy_price'] = spy_price
                    features['correction_label'] = row['correction_label']
                    feature_data.append(features)
        
        if feature_data:
            df_features = pd.DataFrame(feature_data)
            print(f"✅ Built training dataset with {len(df_features)} samples")
            print(f"   • Corrections: {df_features['correction_label'].sum()}")
            print(f"   • No corrections: {len(df_features) - df_features['correction_label'].sum()}")
            return df_features
        else:
            print("❌ No feature data found")
            return pd.DataFrame()
    
    def train_model(self, df: pd.DataFrame) -> dict:
        """Train Random Forest model"""
        
        if df.empty:
            return {}
        
        print(f"\n🤖 TRAINING RANDOM FOREST MODEL")
        print("=" * 40)
        
        # Prepare features and target
        feature_columns = [
            'institutional_dominance', 'high_defensive', 'hedging_intensity',
            'vol_oi_ratio', 'strike_concentration', 'long_term_positioning',
            'atm_otm_ratio', 'pc_ratio'
        ]
        
        X = df[feature_columns].fillna(0)
        y = df['correction_label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='roc_auc')
        
        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc_score': auc_score,
            'feature_importance': self.feature_importance_,
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        print(f"✅ Model trained successfully")
        print(f"   • CV AUC Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print(f"   • Overall AUC: {results['auc_score']:.3f}")
        print(f"\n📊 Feature Importance:")
        for _, row in self.feature_importance_.head(5).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        return results
    
    def predict_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict corrections for given data"""
        
        if self.model is None:
            print("❌ Model not trained yet")
            return pd.DataFrame()
        
        feature_columns = [
            'institutional_dominance', 'high_defensive', 'hedging_intensity',
            'vol_oi_ratio', 'strike_concentration', 'long_term_positioning',
            'atm_otm_ratio', 'pc_ratio'
        ]
        
        X = df[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['correction_prediction'] = predictions
        df_result['correction_probability'] = probabilities
        
        return df_result
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance_
            }, filepath)
            print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance_ = model_data['feature_importance']
            print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def create_prediction_chart(self, df: pd.DataFrame, save_path: str = None):
        """Create prediction visualization"""
        
        if df.empty:
            print("No data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Random Forest Correction Predictions', fontsize=16, fontweight='bold')
        
        # Plot 1: SPY Price with Predictions
        ax1.plot(df['date'], df['spy_price'], label='SPY Price', linewidth=2, color='blue', alpha=0.8)
        
        # Highlight correction predictions
        correction_dates = df[df['correction_prediction'] == 1]['date']
        for date in correction_dates:
            ax1.axvline(x=date, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax1.set_title('SPY Price with Correction Predictions')
        ax1.set_ylabel('SPY Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correction Probabilities
        ax2.plot(df['date'], df['correction_probability'], label='Correction Probability', 
                linewidth=2, color='red', alpha=0.8)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        
        ax2.set_title('Correction Probability Over Time')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Feature Importance
        if self.feature_importance_ is not None:
            self.feature_importance_.plot(kind='barh', x='feature', y='importance', ax=ax3)
            ax3.set_title('Feature Importance')
            ax3.set_xlabel('Importance')
        
        # Plot 4: Prediction Distribution
        ax4.hist(df['correction_probability'], bins=20, alpha=0.7, color='red')
        ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        ax4.set_title('Prediction Probability Distribution')
        ax4.set_xlabel('Probability')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()


def main():
    """Main Random Forest correction prediction"""
    
    # Initialize predictor
    predictor = RandomForestCorrectionPredictor()
    
    print("🤖 RANDOM FOREST CORRECTION PREDICTOR")
    print("Training model on hedging patterns")
    print("=" * 50)
    
    # Build training dataset
    df_training = predictor.build_training_dataset('2020-01-01', '2024-12-31')
    
    if df_training.empty:
        print("❌ No training data available")
        return
    
    # Train model
    results = predictor.train_model(df_training)
    
    if not results:
        print("❌ Model training failed")
        return
    
    # Save model
    predictor.save_model('random_forest_correction_model.joblib')
    
    # Test on recent data
    print(f"\n🔍 TESTING ON RECENT DATA")
    print("=" * 30)
    
    df_recent = predictor.build_training_dataset('2025-01-01', '2025-09-30')
    if not df_recent.empty:
        df_predictions = predictor.predict_corrections(df_recent)
        
        # Show recent predictions
        recent_predictions = df_predictions[df_predictions['correction_prediction'] == 1]
        if not recent_predictions.empty:
            print(f"🎯 Recent correction predictions:")
            for _, row in recent_predictions.iterrows():
                print(f"   • {row['date'].strftime('%Y-%m-%d')}: {row['correction_probability']:.2f}")
        else:
            print("   • No corrections predicted in recent data")
        
        # Create visualization
        predictor.create_prediction_chart(df_predictions, 'random_forest_predictions.png')
    
    # Save results
    df_training.to_csv('random_forest_training_data.csv', index=False)
    with open('random_forest_results.txt', 'w') as f:
        f.write(f"Random Forest Correction Predictor Results\n")
        f.write(f"==========================================\n\n")
        f.write(f"CV AUC Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}\n")
        f.write(f"Overall AUC: {results['auc_score']:.3f}\n\n")
        f.write(f"Feature Importance:\n")
        for _, row in results['feature_importance'].iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.3f}\n")
        f.write(f"\nClassification Report:\n{results['classification_report']}")
    
    print(f"\n💾 Results saved")


if __name__ == "__main__":
    main()
