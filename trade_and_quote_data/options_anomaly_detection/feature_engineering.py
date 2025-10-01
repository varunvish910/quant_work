"""
SPY Options Anomaly Detection - Feature Engineering Pipeline
============================================================

This module builds comprehensive features for anomaly detection using:
- Open Interest (OI) proxy data
- Volume and transaction patterns
- Price-OI correlations
- Temporal patterns and seasonality
- Market microstructure indicators

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsFeatureEngine:
    """
    Feature engineering pipeline for SPY options anomaly detection
    """
    
    def __init__(self, data_dir: str = "data/options_chains/SPY"):
        self.data_dir = Path(data_dir)
        self.features_cache = {}
        self.hedging_history = {}  # Cache for temporal hedging analysis
        
    def load_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load options data for a specific date"""
        try:
            year = date[:4]
            month = date[5:7]
            day = date[8:10]
            
            file_path = self.data_dir / year / month / f"SPY_options_snapshot_{date}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(date)
                return df
            else:
                logger.warning(f"No data found for {date}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {date}: {e}")
            return None
    
    def calculate_oi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate core OI-based features
        """
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        
        # 1. OI Distribution Features
        features_df['oi_percentile'] = features_df['oi_proxy'].rank(pct=True)
        features_df['oi_zscore'] = stats.zscore(features_df['oi_proxy'])
        
        # 2. Put/Call OI Ratios
        calls = features_df[features_df['option_type'] == 'C']
        puts = features_df[features_df['option_type'] == 'P']
        
        if len(calls) > 0 and len(puts) > 0:
            total_call_oi = calls['oi_proxy'].sum()
            total_put_oi = puts['oi_proxy'].sum()
            features_df['pc_oi_ratio'] = total_put_oi / (total_call_oi + 1e-6)
        else:
            features_df['pc_oi_ratio'] = 1.0
            
        # 3. OI Concentration (Herfindahl Index)
        oi_total = features_df['oi_proxy'].sum()
        if oi_total > 0:
            oi_shares = features_df['oi_proxy'] / oi_total
            features_df['oi_concentration'] = (oi_shares ** 2).sum()
        else:
            features_df['oi_concentration'] = 0.0
            
        # 4. OI Skewness (ATM vs OTM)
        atm_threshold = 0.95  # Within 5% of ATM
        atm_options = features_df[features_df['moneyness'].between(1-atm_threshold, 1+atm_threshold)]
        otm_options = features_df[~features_df['moneyness'].between(1-atm_threshold, 1+atm_threshold)]
        
        if len(atm_options) > 0 and len(otm_options) > 0:
            atm_oi = atm_options['oi_proxy'].sum()
            otm_oi = otm_options['oi_proxy'].sum()
            features_df['oi_skew'] = otm_oi / (atm_oi + 1e-6)
        else:
            features_df['oi_skew'] = 1.0
            
        return features_df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features
        """
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        
        # 1. Volume Distribution
        features_df['volume_percentile'] = features_df['volume'].rank(pct=True)
        features_df['volume_zscore'] = stats.zscore(features_df['volume'])
        
        # 2. Volume-OI Interaction
        features_df['volume_oi_ratio'] = features_df['volume'] / (features_df['oi_proxy'] + 1e-6)
        features_df['turnover_rate'] = features_df['volume'] / (features_df['oi_proxy'] + 1e-6)
        
        # 3. Transaction Efficiency
        features_df['tx_efficiency'] = features_df['volume'] / (features_df['transactions'] + 1e-6)
        features_df['avg_tx_size'] = features_df['volume'] / (features_df['transactions'] + 1e-6)
        
        # 4. Volume Concentration
        vol_total = features_df['volume'].sum()
        if vol_total > 0:
            vol_shares = features_df['volume'] / vol_total
            features_df['volume_concentration'] = (vol_shares ** 2).sum()
        else:
            features_df['volume_concentration'] = 0.0
            
        return features_df
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features
        """
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        
        # 1. Moneyness Features
        features_df['moneyness_squared'] = features_df['moneyness'] ** 2
        features_df['log_moneyness'] = np.log(features_df['moneyness'] + 1e-6)
        
        # 2. Strike Distribution
        features_df['strike_percentile'] = features_df['strike'].rank(pct=True)
        
        # 3. DTE Features
        features_df['dte_percentile'] = features_df['dte'].rank(pct=True)
        features_df['dte_squared'] = features_df['dte'] ** 2
        
        # 4. Price-Volume Relationship
        features_df['price_volume_correlation'] = features_df['close'].corr(features_df['volume'])
        
        return features_df
    
    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal and seasonal features
        """
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        
        # 1. Day of Week
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
        features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
        
        # 2. Month and Quarter
        features_df['month'] = features_df['date'].dt.month
        features_df['quarter'] = features_df['date'].dt.quarter
        
        # 3. Expiration Cycle
        features_df['days_to_expiry'] = features_df['dte']
        features_df['is_expiration_week'] = (features_df['dte'] <= 7).astype(int)
        features_df['is_expiration_day'] = (features_df['dte'] <= 1).astype(int)
        
        # 4. VIX-like Volatility Proxy
        # Use OI concentration as a volatility proxy
        features_df['volatility_proxy'] = features_df['oi_concentration']
        
        return features_df
    
    def calculate_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features specifically for anomaly detection
        """
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        
        # 1. OI Anomaly Scores
        features_df['oi_anomaly_score'] = np.abs(features_df['oi_zscore'])
        features_df['oi_extreme'] = (features_df['oi_anomaly_score'] > 2).astype(int)
        
        # 2. Volume Anomaly Scores
        features_df['volume_anomaly_score'] = np.abs(features_df['volume_zscore'])
        features_df['volume_extreme'] = (features_df['volume_anomaly_score'] > 2).astype(int)
        
        # 3. Combined Anomaly Score
        features_df['combined_anomaly_score'] = (
            0.4 * features_df['oi_anomaly_score'] +
            0.3 * features_df['volume_anomaly_score'] +
            0.3 * features_df['oi_concentration']
        )
        
        # 4. Unusual Activity Detection
        features_df['unusual_activity'] = (
            (features_df['oi_extreme'] == 1) |
            (features_df['volume_extreme'] == 1) |
            (features_df['turnover_rate'] > features_df['turnover_rate'].quantile(0.95))
        ).astype(int)
        
        return features_df
    
    def calculate_institutional_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate institutional momentum features including OTM/ATM ratios
        """
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        
        # Filter for institutional timeframe (>7 DTE)
        institutional = features_df[features_df['dte'] > 7].copy()
        
        if len(institutional) == 0:
            # If no institutional data, set default values
            features_df['otm_atm_call_ratio'] = 0.0
            features_df['institutional_momentum_score'] = 0.0
            features_df['call_skew_momentum'] = 0.0
            return features_df
        
        # Get current SPY price
        spy_price = institutional['underlying_price'].iloc[0]
        
        # Filter calls only for momentum analysis
        inst_calls = institutional[institutional['option_type'] == 'C']
        
        if len(inst_calls) > 0:
            # 1. OTM/ATM Call Ratio (key institutional momentum indicator)
            atm_calls = inst_calls[
                (inst_calls['strike'] >= spy_price - 5) & 
                (inst_calls['strike'] <= spy_price + 5)
            ]
            otm_calls = inst_calls[inst_calls['strike'] > spy_price + 5]
            
            atm_oi = atm_calls['oi_proxy'].sum()
            otm_oi = otm_calls['oi_proxy'].sum()
            
            otm_atm_ratio = otm_oi / (atm_oi + 1e-6)
            
            # 2. Call Skew Momentum (Near OTM vs Far OTM)
            near_otm = inst_calls[
                (inst_calls['strike'] > spy_price + 5) & 
                (inst_calls['strike'] <= spy_price + 25)
            ]
            far_otm = inst_calls[inst_calls['strike'] > spy_price + 25]
            
            near_otm_oi = near_otm['oi_proxy'].sum()
            far_otm_oi = far_otm['oi_proxy'].sum()
            
            call_skew = near_otm_oi / (far_otm_oi + 1e-6)
            
            # 3. Institutional Momentum Score
            # Higher ratio = More upside positioning = Higher momentum
            # Normalize around typical range (2-8)
            momentum_score = np.clip((otm_atm_ratio - 2) / 6, 0, 1)
            
        else:
            otm_atm_ratio = 0.0
            call_skew = 0.0
            momentum_score = 0.0
        
        # Apply to all rows in the dataset
        features_df['otm_atm_call_ratio'] = otm_atm_ratio
        features_df['call_skew_momentum'] = call_skew
        features_df['institutional_momentum_score'] = momentum_score
        
        # 4. Momentum Regime Classification
        if otm_atm_ratio > 6:
            momentum_regime = 'high_momentum'
        elif otm_atm_ratio > 4:
            momentum_regime = 'moderate_momentum'
        else:
            momentum_regime = 'low_momentum'
            
        features_df['momentum_regime'] = momentum_regime
        
        return features_df
    
    def calculate_hedging_features(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        Calculate hedging-specific features for anomaly detection
        """
        if df is None or len(df) == 0:
            return df
            
        features_df = df.copy()
        spy_price = features_df['underlying_price'].iloc[0] if 'underlying_price' in features_df.columns else 600
        
        # Separate puts and calls
        puts = features_df[features_df['option_type'] == 'P'].copy()
        calls = features_df[features_df['option_type'] == 'C'].copy()
        
        if puts.empty:
            # Add default hedging features if no puts
            default_hedging_features = [
                'institutional_hedging_score', 'speculative_hedging_score',
                'deep_otm_put_activity', 'support_level_concentration',
                'hedging_volume_ratio', 'temporal_hedging_momentum'
            ]
            for feature in default_hedging_features:
                features_df[feature] = 0.0
            return features_df
        
        # 1. Institutional vs Speculative Classification
        institutional_scores = self._classify_institutional_hedging(puts, spy_price)
        
        # Apply scores to all contracts (puts get their specific scores, calls get 0)
        features_df['institutional_hedging_score'] = 0.0
        features_df['speculative_hedging_score'] = 0.0
        
        put_mask = features_df['option_type'] == 'P'
        features_df.loc[put_mask, 'institutional_hedging_score'] = institutional_scores
        features_df.loc[put_mask, 'speculative_hedging_score'] = 1.0 - institutional_scores
        
        # 2. Deep OTM Put Activity
        deep_otm_puts = puts[puts['moneyness'] <= 0.85]
        deep_otm_activity = deep_otm_puts['oi_proxy'].sum() / (puts['oi_proxy'].sum() + 1e-6)
        features_df['deep_otm_put_activity'] = deep_otm_activity
        
        # 3. Support Level Concentration
        support_concentration = self._calculate_support_concentration(puts, spy_price)
        features_df['support_level_concentration'] = support_concentration
        
        # 4. Hedging Volume Ratio
        hedging_volume = puts[puts['dte'] >= 14]['volume'].sum()  # Longer-term puts
        total_volume = features_df['volume'].sum()
        hedging_volume_ratio = hedging_volume / (total_volume + 1e-6)
        features_df['hedging_volume_ratio'] = hedging_volume_ratio
        
        # 5. Temporal Hedging Momentum (if we have history)
        temporal_momentum = self._calculate_temporal_hedging_momentum(date, puts, spy_price)
        features_df['temporal_hedging_momentum'] = temporal_momentum
        
        return features_df
    
    def _classify_institutional_hedging(self, puts_df: pd.DataFrame, spy_price: float) -> np.ndarray:
        """
        Classify put contracts as institutional vs speculative
        """
        if puts_df.empty:
            return np.array([])
        
        # Institutional indicators
        institutional_scores = []
        
        # 1. Time to expiration (institutions prefer longer timeframes)
        dte_score = np.where(puts_df['dte'] >= 21, 1.0, 
                           np.where(puts_df['dte'] >= 14, 0.6, 0.2))
        institutional_scores.append(dte_score)
        
        # 2. Moneyness (institutions buy deeper OTM for protection)
        moneyness_score = np.where(puts_df['moneyness'] <= 0.85, 1.0,
                                 np.where(puts_df['moneyness'] <= 0.95, 0.6, 0.2))
        institutional_scores.append(moneyness_score)
        
        # 3. Volume/OI ratio (institutions hold, speculators trade)
        vol_oi_ratio = puts_df['volume'] / (puts_df['oi_proxy'] + 1e-6)
        holding_score = np.where(vol_oi_ratio <= 0.3, 1.0,
                               np.where(vol_oi_ratio <= 0.7, 0.6, 0.2))
        institutional_scores.append(holding_score)
        
        # 4. Block size indicator
        avg_tx_size = puts_df['volume'] / (puts_df['transactions'] + 1e-6)
        block_score = np.where(avg_tx_size >= 50, 1.0,
                             np.where(avg_tx_size >= 20, 0.6, 0.3))
        institutional_scores.append(block_score)
        
        # Weighted average
        weights = [0.3, 0.3, 0.25, 0.15]
        institutional_score = np.average(institutional_scores, weights=weights, axis=0)
        
        return institutional_score
    
    def _calculate_support_concentration(self, puts_df: pd.DataFrame, spy_price: float) -> float:
        """
        Calculate concentration of put activity at key support levels
        """
        if puts_df.empty:
            return 0.0
        
        support_levels = [spy_price * mult for mult in [0.95, 0.90, 0.85, 0.80]]
        total_support_oi = 0.0
        
        for level in support_levels:
            near_support = np.abs(puts_df['strike'] - level) <= 2.5
            support_oi = puts_df[near_support]['oi_proxy'].sum()
            total_support_oi += support_oi
        
        total_put_oi = puts_df['oi_proxy'].sum()
        return total_support_oi / (total_put_oi + 1e-6)
    
    def _calculate_temporal_hedging_momentum(self, date: str, puts_df: pd.DataFrame, spy_price: float) -> float:
        """
        Calculate hedging momentum over time (simplified version)
        """
        try:
            # Get institutional put activity for current date
            institutional_scores = self._classify_institutional_hedging(puts_df, spy_price)
            current_institutional_oi = (puts_df['oi_proxy'] * institutional_scores).sum()
            
            # Store in history
            date_key = pd.to_datetime(date).strftime('%Y-%m-%d')
            self.hedging_history[date_key] = current_institutional_oi
            
            # Calculate momentum if we have history
            if len(self.hedging_history) >= 5:
                recent_dates = sorted(self.hedging_history.keys())[-5:]
                recent_values = [self.hedging_history[d] for d in recent_dates]
                
                # Simple momentum calculation
                if len(recent_values) >= 2:
                    momentum = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
                    # Normalize by current level
                    return momentum / (recent_values[-1] + 1e-6)
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def calculate_daily_aggregates(self, df: pd.DataFrame) -> Dict:
        """
        Calculate daily aggregate features for the entire options chain
        """
        if df is None or len(df) == 0:
            return {}
            
        aggregates = {}
        
        # 1. Basic Aggregates
        aggregates['total_contracts'] = len(df)
        aggregates['total_volume'] = df['volume'].sum()
        aggregates['total_oi'] = df['oi_proxy'].sum()
        aggregates['avg_oi'] = df['oi_proxy'].mean()
        aggregates['median_oi'] = df['oi_proxy'].median()
        
        # 2. Put/Call Ratios
        calls = df[df['option_type'] == 'C']
        puts = df[df['option_type'] == 'P']
        
        if len(calls) > 0 and len(puts) > 0:
            aggregates['pc_ratio_volume'] = puts['volume'].sum() / calls['volume'].sum()
            aggregates['pc_ratio_oi'] = puts['oi_proxy'].sum() / calls['oi_proxy'].sum()
            aggregates['pc_ratio_contracts'] = len(puts) / len(calls)
        else:
            aggregates['pc_ratio_volume'] = 1.0
            aggregates['pc_ratio_oi'] = 1.0
            aggregates['pc_ratio_contracts'] = 1.0
            
        # 3. Concentration Metrics
        aggregates['oi_concentration'] = df['oi_concentration'].iloc[0] if 'oi_concentration' in df.columns else 0.0
        aggregates['volume_concentration'] = df['volume_concentration'].iloc[0] if 'volume_concentration' in df.columns else 0.0
        
        # 4. Anomaly Metrics
        aggregates['anomaly_contracts'] = df['unusual_activity'].sum() if 'unusual_activity' in df.columns else 0
        aggregates['anomaly_rate'] = aggregates['anomaly_contracts'] / aggregates['total_contracts']
        
        # 5. Price Level
        aggregates['underlying_price'] = df['underlying_price'].iloc[0] if 'underlying_price' in df.columns else 0.0
        
        # 6. Institutional Momentum Features
        if 'otm_atm_call_ratio' in df.columns:
            aggregates['otm_atm_call_ratio'] = df['otm_atm_call_ratio'].iloc[0]
            aggregates['institutional_momentum_score'] = df['institutional_momentum_score'].iloc[0]
            aggregates['call_skew_momentum'] = df['call_skew_momentum'].iloc[0]
            aggregates['momentum_regime'] = df['momentum_regime'].iloc[0]
        else:
            aggregates['otm_atm_call_ratio'] = 0.0
            aggregates['institutional_momentum_score'] = 0.0
            aggregates['call_skew_momentum'] = 0.0
            aggregates['momentum_regime'] = 'unknown'
        
        return aggregates
    
    def process_date(self, date: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Process a single date and return features + aggregates
        """
        logger.info(f"Processing features for {date}")
        
        # Load data
        df = self.load_daily_data(date)
        if df is None:
            return None, None
            
        # Calculate all features
        df = self.calculate_oi_features(df)
        df = self.calculate_volume_features(df)
        df = self.calculate_price_features(df)
        df = self.calculate_temporal_features(df)
        df = self.calculate_anomaly_features(df)
        df = self.calculate_institutional_momentum_features(df)
        
        # Add hedging-aware features
        df = self.calculate_hedging_features(df, date)
        
        # Calculate daily aggregates
        aggregates = self.calculate_daily_aggregates(df)
        
        return df, aggregates
    
    def process_date_range(self, start_date: str, end_date: str) -> Dict[str, Dict]:
        """
        Process a range of dates and return daily aggregates
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        daily_aggregates = {}
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_date.weekday() < 5:
                _, aggregates = self.process_date(date_str)
                if aggregates:
                    daily_aggregates[date_str] = aggregates
                    
            current_date += timedelta(days=1)
            
        return daily_aggregates
    
    def save_features(self, df: pd.DataFrame, date: str, output_dir: str = "data/features"):
        """
        Save processed features to parquet file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        year = date[:4]
        month = date[5:7]
        
        file_path = output_path / year / month / f"SPY_features_{date}.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved features to {file_path}")
        
    def save_aggregates(self, aggregates: Dict[str, Dict], output_file: str = "data/daily_aggregates.csv"):
        """
        Save daily aggregates to CSV
        """
        df = pd.DataFrame.from_dict(aggregates, orient='index')
        df.index.name = 'date'
        df.to_csv(output_file)
        logger.info(f"Saved daily aggregates to {output_file}")


def main():
    """
    Example usage of the feature engineering pipeline
    """
    # Initialize feature engine
    fe = OptionsFeatureEngine()
    
    # Process a single date
    date = "2025-01-31"
    df, aggregates = fe.process_date(date)
    
    if df is not None:
        print(f"Processed {len(df)} contracts for {date}")
        print(f"Features: {list(df.columns)}")
        print(f"Aggregates: {aggregates}")
        
        # Save features
        fe.save_features(df, date)
    
    # Process a date range
    start_date = "2025-01-01"
    end_date = "2025-01-31"
    
    daily_aggregates = fe.process_date_range(start_date, end_date)
    print(f"Processed {len(daily_aggregates)} trading days")
    
    # Save aggregates
    fe.save_aggregates(daily_aggregates)


if __name__ == "__main__":
    main()
