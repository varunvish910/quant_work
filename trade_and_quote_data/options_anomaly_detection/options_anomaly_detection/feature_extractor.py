#!/usr/bin/env python3
"""
Extract features from historical options data using existing specialized predictors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Import existing predictors (copy from temp files)
sys.path.append('/tmp')

class HistoricalFeatureExtractor:
    """
    Extract prediction features from historical options data
    Uses existing specialized predictors to create feature vectors
    """
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.features_cache = {}
        
    def load_options_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load options data for a specific date"""
        try:
            # Format: data/options_chains/SPY/YYYY/MM/SPY_options_snapshot_YYYYMMDD.parquet
            year = date[:4]
            month = date[4:6]
            
            possible_files = [
                self.data_dir / f"options_chains/SPY/{year}/{month}/SPY_options_snapshot_{date}.parquet",
                self.data_dir / f"SPY_options_snapshot_{date}.parquet",
                self.data_dir / f"SPY_{date}.parquet"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    print(f"✅ Loaded {len(df)} options contracts for {date}")
                    return df
            
            print(f"❌ No options data found for {date}")
            return None
            
        except Exception as e:
            print(f"❌ Error loading data for {date}: {e}")
            return None
    
    def extract_downward_signals(self, options_df: pd.DataFrame, spy_price: float) -> Dict:
        """Extract bearish/downward move signals"""
        if options_df.empty:
            return self._empty_downward_signals()
        
        # Filter institutional options (>7 DTE)
        inst_calls = options_df[(options_df['dte'] > 7) & (options_df['option_type'] == 'C')]
        inst_puts = options_df[(options_df['dte'] > 7) & (options_df['option_type'] == 'P')]
        
        signals = {}
        
        # 1. Distribution Score (institutional selling)
        if len(inst_calls) > 0:
            call_volume = inst_calls['volume'].sum()
            call_oi = inst_calls['oi_proxy'].sum()
            distribution_ratio = call_volume / (call_oi + 1e-6)
            signals['distribution_score'] = min(distribution_ratio * 2, 5.0)
        else:
            signals['distribution_score'] = 0
        
        # 2. Put Accumulation Score
        if len(inst_puts) > 0:
            defensive_puts = inst_puts[
                (inst_puts['strike'] >= spy_price * 0.95) & 
                (inst_puts['strike'] <= spy_price * 1.05)
            ]
            if len(defensive_puts) > 0:
                put_oi_concentration = defensive_puts['oi_proxy'].sum() / inst_puts['oi_proxy'].sum()
                signals['put_accumulation'] = put_oi_concentration * 3
            else:
                signals['put_accumulation'] = 0
        else:
            signals['put_accumulation'] = 0
        
        # 3. Call Exit Signal
        if len(inst_calls) > 0:
            profit_taking_calls = inst_calls[
                (inst_calls['strike'] >= spy_price * 0.98) & 
                (inst_calls['strike'] <= spy_price * 1.02)
            ]
            if len(profit_taking_calls) > 0:
                exit_volume = profit_taking_calls['volume'].sum()
                exit_ratio = exit_volume / (inst_calls['volume'].sum() + 1e-6)
                signals['call_exit_signal'] = exit_ratio * 3
            else:
                signals['call_exit_signal'] = 0
        else:
            signals['call_exit_signal'] = 0
        
        # 4. Support Weakness
        support_level = spy_price * 0.975
        support_options = options_df[
            (options_df['dte'] > 7) & 
            (options_df['strike'] >= support_level - 5) & 
            (options_df['strike'] <= support_level + 5)
        ]
        
        if len(support_options) > 0:
            support_puts = support_options[support_options['option_type'] == 'P']
            support_calls = support_options[support_options['option_type'] == 'C']
            
            if len(support_puts) > 0 and len(support_calls) > 0:
                support_pc_ratio = support_puts['oi_proxy'].sum() / support_calls['oi_proxy'].sum()
                signals['support_weakness'] = min(support_pc_ratio, 3.0)
            else:
                signals['support_weakness'] = 1.0
        else:
            signals['support_weakness'] = 1.0
        
        # 5. Skew Inversion
        total_put_vol = inst_puts['volume'].sum() if len(inst_puts) > 0 else 0
        total_call_vol = inst_calls['volume'].sum() if len(inst_calls) > 0 else 0
        
        if total_call_vol > 0:
            current_pc_vol_ratio = total_put_vol / total_call_vol
            signals['skew_inversion'] = min(current_pc_vol_ratio / 0.8, 2.0)
        else:
            signals['skew_inversion'] = 1.0
        
        # 6. Volume Profile Shift
        if len(inst_puts) > 0:
            downside_puts = inst_puts[inst_puts['strike'] < spy_price * 0.98]
            upside_calls = inst_calls[inst_calls['strike'] > spy_price * 1.02] if len(inst_calls) > 0 else pd.DataFrame()
            
            downside_volume = downside_puts['volume'].sum()
            upside_volume = upside_calls['volume'].sum() if len(upside_calls) > 0 else 1
            
            signals['volume_profile_shift'] = min(downside_volume / (upside_volume + 1e-6), 2.0)
        else:
            signals['volume_profile_shift'] = 1.0
        
        return signals
    
    def extract_big_move_signals(self, options_df: pd.DataFrame, spy_price: float) -> Dict:
        """Extract big move prediction signals"""
        if options_df.empty:
            return self._empty_big_move_signals()
        
        inst_calls = options_df[(options_df['dte'] > 7) & (options_df['option_type'] == 'C')]
        inst_puts = options_df[(options_df['dte'] > 7) & (options_df['option_type'] == 'P')]
        
        signals = {}
        
        # 1. Tension Index (OI concentration vs price action)
        atm_calls = inst_calls[(inst_calls['strike'] >= spy_price - 10) & (inst_calls['strike'] <= spy_price + 10)]
        far_calls = inst_calls[inst_calls['strike'] > spy_price + 20]
        
        if len(atm_calls) > 0 and len(far_calls) > 0:
            atm_concentration = atm_calls['oi_proxy'].sum()
            far_concentration = far_calls['oi_proxy'].sum() 
            signals['tension_index'] = far_concentration / (atm_concentration + 1e-6)
        else:
            signals['tension_index'] = 0
        
        # 2. Put/Call Asymmetry Score
        total_call_oi = inst_calls['oi_proxy'].sum()
        total_put_oi = inst_puts['oi_proxy'].sum()
        
        if total_call_oi > 0 and total_put_oi > 0:
            pc_ratio = total_put_oi / total_call_oi
            signals['asymmetry_score'] = (pc_ratio - 1.0) * 3
        else:
            signals['asymmetry_score'] = 0
        
        # 3. Coiling Pattern (low volume, high OI)
        total_volume = inst_calls['volume'].sum() + inst_puts['volume'].sum()
        total_oi = inst_calls['oi_proxy'].sum() + inst_puts['oi_proxy'].sum()
        
        if total_oi > 0:
            volume_oi_ratio = total_volume / total_oi
            signals['coiling_pattern'] = int(volume_oi_ratio < 0.3)  # Convert to int
        else:
            signals['coiling_pattern'] = 0
        
        # 4. Volume/OI Disconnect
        signals['volume_oi_disconnect'] = total_volume / (total_oi + 1e-6)
        
        # 5. Strike Distribution Anomaly
        call_strikes = inst_calls['strike'].values
        if len(call_strikes) > 0:
            strike_std = np.std(call_strikes)
            expected_std = spy_price * 0.15
            signals['strike_anomaly'] = strike_std / expected_std
        else:
            signals['strike_anomaly'] = 1.0
        
        # 6. Put/Call Momentum
        signals['put_call_momentum'] = signals['asymmetry_score'] * 0.5
        
        return signals
    
    def extract_risk_signals(self, options_df: pd.DataFrame, spy_price: float) -> Dict:
        """Extract risk assessment signals"""
        if options_df.empty:
            return self._empty_risk_signals()
        
        inst_calls = options_df[(options_df['dte'] > 7) & (options_df['option_type'] == 'C')]
        
        if len(inst_calls) == 0:
            return self._empty_risk_signals()
        
        # Calculate key risk metrics
        atm_calls = inst_calls[
            (inst_calls['strike'] >= spy_price - 5) & 
            (inst_calls['strike'] <= spy_price + 5)
        ]
        otm_calls = inst_calls[inst_calls['strike'] > spy_price + 5]
        
        atm_oi = atm_calls['oi_proxy'].sum()
        otm_oi = otm_calls['oi_proxy'].sum()
        
        # Risk concentration metrics
        far_otm = inst_calls[inst_calls['strike'] > spy_price + 25]
        extreme_otm = inst_calls[inst_calls['strike'] > spy_price + 50]
        
        signals = {
            'otm_atm_ratio': otm_oi / (atm_oi + 1e-6),
            'total_call_oi': inst_calls['oi_proxy'].sum(),
            'far_otm_concentration': far_otm['oi_proxy'].sum() / inst_calls['oi_proxy'].sum(),
            'extreme_concentration': extreme_otm['oi_proxy'].sum() / inst_calls['oi_proxy'].sum(),
            'volume_turnover': inst_calls['volume'].sum() / (inst_calls['oi_proxy'].sum() + 1e-6)
        }
        
        return signals
    
    def extract_all_features(self, date: str) -> Optional[Dict]:
        """Extract all features for a given date"""
        options_df = self.load_options_data(date)
        if options_df is None:
            return None
        
        try:
            spy_price = options_df['underlying_price'].iloc[0]
            
            features = {
                'date': date,
                'spy_price': spy_price,
                'total_contracts': len(options_df)
            }
            
            # Extract all signal types
            downward_signals = self.extract_downward_signals(options_df, spy_price)
            big_move_signals = self.extract_big_move_signals(options_df, spy_price)
            risk_signals = self.extract_risk_signals(options_df, spy_price)
            
            # Combine all features
            features.update({f"downward_{k}": v for k, v in downward_signals.items()})
            features.update({f"bigmove_{k}": v for k, v in big_move_signals.items()})
            features.update({f"risk_{k}": v for k, v in risk_signals.items()})
            
            # Composite scores
            features['downward_composite'] = sum([
                downward_signals.get('distribution_score', 0) > 2.0,
                downward_signals.get('put_accumulation', 0) > 1.5,
                downward_signals.get('call_exit_signal', 0) > 1.8,
                downward_signals.get('support_weakness', 0) > 1.5,
                downward_signals.get('skew_inversion', 0) > 1.2,
                downward_signals.get('volume_profile_shift', 0) > 1.3
            ])
            
            features['bigmove_composite'] = sum([
                big_move_signals.get('tension_index', 0) > 3.0,
                abs(big_move_signals.get('asymmetry_score', 0)) > 2.0,
                big_move_signals.get('coiling_pattern', 0) > 0,
                big_move_signals.get('volume_oi_disconnect', 0) > 2.5,
                big_move_signals.get('strike_anomaly', 0) > 1.5
            ])
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features for {date}: {e}")
            return None
    
    def extract_features_batch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract features for a date range"""
        print(f"🔄 Extracting features from {start_date} to {end_date}")
        
        # Generate date range (business days only)
        date_range = pd.bdate_range(start=start_date, end=end_date)
        
        features_list = []
        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            
            features = self.extract_all_features(date_str)
            if features:
                features_list.append(features)
                if len(features_list) % 10 == 0:
                    print(f"   Processed {len(features_list)} days...")
        
        if features_list:
            df = pd.DataFrame(features_list)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)
            
            print(f"✅ Extracted features for {len(df)} days")
            return df
        else:
            print("❌ No features extracted")
            return pd.DataFrame()
    
    def _empty_downward_signals(self) -> Dict:
        """Return empty downward signals"""
        return {
            'distribution_score': 0,
            'put_accumulation': 0,
            'call_exit_signal': 0,
            'support_weakness': 1.0,
            'skew_inversion': 1.0,
            'volume_profile_shift': 1.0
        }
    
    def _empty_big_move_signals(self) -> Dict:
        """Return empty big move signals"""
        return {
            'tension_index': 0,
            'asymmetry_score': 0,
            'coiling_pattern': 0,
            'volume_oi_disconnect': 0,
            'strike_anomaly': 1.0,
            'put_call_momentum': 0
        }
    
    def _empty_risk_signals(self) -> Dict:
        """Return empty risk signals"""
        return {
            'otm_atm_ratio': 1.0,
            'total_call_oi': 0,
            'far_otm_concentration': 0,
            'extreme_concentration': 0,
            'volume_turnover': 0
        }

# Test function
def test_feature_extraction():
    """Test feature extraction with sample data"""
    print("🧪 Testing feature extractor...")
    
    extractor = HistoricalFeatureExtractor()
    
    # Test with a recent date (if data exists)
    test_date = "20240930"  # September 30, 2024
    features = extractor.extract_all_features(test_date)
    
    if features:
        print(f"✅ Successfully extracted {len(features)} features")
        print("📊 Sample features:")
        for key, value in list(features.items())[:10]:
            print(f"   {key}: {value}")
        return True
    else:
        print("❌ No features extracted - check data availability")
        return False

if __name__ == "__main__":
    success = test_feature_extraction()
    print(f"\n{'✅ Test passed' if success else '❌ Test failed'}")