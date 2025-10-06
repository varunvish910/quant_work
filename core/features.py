"""
Unified Feature Engineering Module

Consolidates all feature calculation logic from:
- phase1_baseline_early_warning.py (baseline features)
- phase2_currency_features.py (currency features)
- phase3_volatility_features.py (volatility features)
- rally_analyzer/features/ (rally features)

CRITICAL: All features use ONLY real market data, never synthetic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.constants import (
    SECTOR_ETFS, VOLATILITY_WINDOW, SHORT_MA_WINDOW,
    MEDIUM_MA_WINDOW, LONG_MA_WINDOW, MOMENTUM_SHORT,
    MOMENTUM_MEDIUM, MOMENTUM_LONG
)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class FeatureEngine:
    """Unified feature calculation engine"""
    
    def __init__(self, feature_sets: Optional[List[str]] = None):
        """
        Initialize FeatureEngine
        
        Args:
            feature_sets: List of feature sets to calculate
                         Options: ['baseline', 'currency', 'volatility', 'options', 'all']
                         Default: ['baseline']
        """
        if feature_sets is None:
            feature_sets = ['baseline']
        
        if 'all' in feature_sets:
            self.feature_sets = ['baseline', 'currency', 'volatility', 'options']
        else:
            self.feature_sets = feature_sets
        
        self.feature_columns = []
    
    def calculate_features(self, 
                          spy_data: pd.DataFrame,
                          sector_data: Optional[Dict[str, pd.DataFrame]] = None,
                          currency_data: Optional[Dict[str, pd.DataFrame]] = None,
                          volatility_data: Optional[Dict[str, pd.DataFrame]] = None,
                          options_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all requested feature sets
        
        Args:
            spy_data: SPY OHLCV DataFrame
            sector_data: Dictionary of sector symbol -> DataFrame
            currency_data: Dictionary of currency name -> DataFrame
            volatility_data: Dictionary of volatility index name -> DataFrame
            options_data: Options chain data (if available)
            
        Returns:
            DataFrame with all calculated features
        """
        print("=" * 80)
        print("🔧 CALCULATING FEATURES")
        print("=" * 80)
        print(f"Feature sets: {', '.join(self.feature_sets)}")
        
        # Start with SPY data
        features_df = spy_data[['Close', 'High', 'Low', 'Volume']].copy()
        self.feature_columns = []
        
        # Calculate each feature set
        if 'baseline' in self.feature_sets:
            features_df = self._add_baseline_features(features_df, sector_data)
        
        if 'currency' in self.feature_sets:
            if currency_data is not None:
                features_df = self._add_currency_features(features_df, currency_data)
            else:
                print("⚠️  Currency data not provided, skipping currency features")
        
        if 'volatility' in self.feature_sets:
            if volatility_data is not None:
                features_df = self._add_volatility_features(features_df, volatility_data)
            else:
                print("⚠️  Volatility data not provided, skipping volatility features")
        
        if 'options' in self.feature_sets:
            if options_data is not None:
                features_df = self._add_options_features(features_df, options_data)
            else:
                print("⚠️  Options data not provided, skipping options features")
        
        # Clean data
        features_df = self._clean_features(features_df)
        
        print("=" * 80)
        print(f"✅ FEATURE CALCULATION COMPLETE: {len(self.feature_columns)} features")
        print("=" * 80)
        
        return features_df
    
    def _add_baseline_features(self, df: pd.DataFrame, 
                               sector_data: Optional[Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Add baseline 8 features from Phase 1
        
        Features:
        1. Volatility features (2): volatility_20d, atr_14
        2. Moving average distances (2): price_vs_sma200, price_vs_sma50
        3. Momentum (1): return_50d
        4. Sector rotation (3): xlu_vs_xlk, xlv_vs_xlk, defensive_rotation
        """
        print("📊 Calculating baseline features...")
        
        # 1. Volatility features
        df['volatility_20d'] = df['Close'].pct_change().rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)
        df['atr_14'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # 2. Moving average distance features
        df['sma_200'] = df['Close'].rolling(LONG_MA_WINDOW, min_periods=50).mean()
        df['sma_50'] = df['Close'].rolling(MEDIUM_MA_WINDOW, min_periods=20).mean()
        df['price_vs_sma200'] = ((df['Close'] - df['sma_200']) / df['sma_200']) * 100
        df['price_vs_sma50'] = ((df['Close'] - df['sma_50']) / df['sma_50']) * 100
        
        # 3. Momentum features
        df['return_50d'] = df['Close'].pct_change(MEDIUM_MA_WINDOW) * 100
        
        baseline_features = [
            'volatility_20d', 'atr_14', 'price_vs_sma200', 'price_vs_sma50', 'return_50d'
        ]
        
        # 4. Sector rotation features (if sector data available)
        if sector_data is not None and 'XLU' in sector_data and 'XLK' in sector_data and 'XLV' in sector_data:
            xlu_return = sector_data['XLU']['Close'].pct_change(20) * 100
            xlk_return = sector_data['XLK']['Close'].pct_change(20) * 100
            xlv_return = sector_data['XLV']['Close'].pct_change(20) * 100
            
            df['xlu_vs_xlk'] = xlu_return - xlk_return
            df['xlv_vs_xlk'] = xlv_return - xlk_return
            df['defensive_rotation'] = ((xlu_return > xlk_return) & (xlv_return > xlk_return)).astype(int)
            
            baseline_features.extend(['xlu_vs_xlk', 'xlv_vs_xlk', 'defensive_rotation'])
        else:
            print("   ⚠️  Sector data incomplete, skipping sector rotation features")
        
        # 5. Rotation indicator features (MAGS, RSP, QQQ, QQQE)
        if sector_data is not None:
            rotation_features_added = []
            
            # MAGS vs SPY (Concentration risk)
            if 'MAGS' in sector_data:
                mags_return = sector_data['MAGS']['Close'].pct_change(20) * 100
                spy_return = df['Close'].pct_change(20) * 100
                df['mags_vs_spy'] = mags_return - spy_return
                rotation_features_added.append('mags_vs_spy')
            
            # RSP vs SPY (Breadth indicator)
            if 'RSP' in sector_data:
                rsp_return = sector_data['RSP']['Close'].pct_change(20) * 100
                spy_return = df['Close'].pct_change(20) * 100
                df['rsp_vs_spy'] = rsp_return - spy_return
                # Negative RSP vs SPY = narrow leadership = risk
                df['narrow_leadership'] = (df['rsp_vs_spy'] < -2).astype(int)
                rotation_features_added.extend(['rsp_vs_spy', 'narrow_leadership'])
            
            # QQQ vs SPY (Tech concentration)
            if 'QQQ' in sector_data:
                qqq_return = sector_data['QQQ']['Close'].pct_change(20) * 100
                spy_return = df['Close'].pct_change(20) * 100
                df['qqq_vs_spy'] = qqq_return - spy_return
                rotation_features_added.append('qqq_vs_spy')
            
            # QQQE vs QQQ (Tech breadth)
            if 'QQQE' in sector_data and 'QQQ' in sector_data:
                qqqe_return = sector_data['QQQE']['Close'].pct_change(20) * 100
                qqq_return = sector_data['QQQ']['Close'].pct_change(20) * 100
                df['qqqe_vs_qqq'] = qqqe_return - qqq_return
                rotation_features_added.append('qqqe_vs_qqq')
            
            if rotation_features_added:
                baseline_features.extend(rotation_features_added)
                print(f"   ✅ Rotation indicators: {len(rotation_features_added)} features")
        
        # 6. Regime detection features (CRITICAL FOR PULLBACK PREDICTION)
        from features.technicals.regime_detection import RegimeDetectionFeature, MomentumExhaustionFeature
        
        regime_detector = RegimeDetectionFeature()
        df = regime_detector.calculate(df)
        baseline_features.extend(regime_detector.feature_names)
        print(f"   ✅ Regime detection: {len(regime_detector.feature_names)} features")
        
        momentum_detector = MomentumExhaustionFeature()
        df = momentum_detector.calculate(df)
        baseline_features.extend(momentum_detector.feature_names)
        print(f"   ✅ Momentum exhaustion: {len(momentum_detector.feature_names)} features")
        
        self.feature_columns.extend(baseline_features)
        print(f"   ✅ Baseline features: {len(baseline_features)}")
        
        return df
    
    def _add_currency_features(self, df: pd.DataFrame, 
                               currency_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add currency features from Phase 2
        
        Critical for detecting carry trade unwinds (July 2024 event)
        """
        print("💱 Calculating currency features...")
        
        currency_features = []
        
        # 1. USD/JPY Features - CRITICAL for carry trade detection
        if 'USDJPY' in currency_data:
            usdjpy = currency_data['USDJPY']['Close']
            
            df['usdjpy_level'] = usdjpy
            df['usdjpy_momentum_3d'] = usdjpy.pct_change(3) * 100
            df['usdjpy_momentum_5d'] = usdjpy.pct_change(5) * 100
            df['usdjpy_momentum_10d'] = usdjpy.pct_change(10) * 100
            df['usdjpy_acceleration_3d'] = df['usdjpy_momentum_3d'].diff()
            df['usdjpy_volatility'] = usdjpy.pct_change().rolling(20).std() * 100
            
            # Carry trade unwind risk signal
            df['yen_carry_unwind_risk'] = (
                (df['usdjpy_momentum_5d'] < -1.0) &  # Yen strengthening >1%
                (df['usdjpy_acceleration_3d'] < 0)   # Accelerating yen strength
            ).astype(int)
            
            currency_features.extend([
                'usdjpy_level', 'usdjpy_momentum_3d', 'usdjpy_momentum_5d',
                'usdjpy_momentum_10d', 'usdjpy_acceleration_3d', 'usdjpy_volatility',
                'yen_carry_unwind_risk'
            ])
        
        # 2. Dollar Index (DXY) Features
        if 'DXY' in currency_data:
            dxy = currency_data['DXY']['Close']
            
            df['dxy_level'] = dxy
            df['dxy_momentum_5d'] = dxy.pct_change(5) * 100
            df['dxy_momentum_10d'] = dxy.pct_change(10) * 100
            df['dxy_rsi'] = calculate_rsi(dxy, 14)
            
            currency_features.extend(['dxy_level', 'dxy_momentum_5d', 'dxy_momentum_10d', 'dxy_rsi'])
        
        # 3. EUR/USD Features
        if 'EURUSD' in currency_data:
            eurusd = currency_data['EURUSD']['Close']
            df['eurusd_momentum_5d'] = eurusd.pct_change(5) * 100
            currency_features.append('eurusd_momentum_5d')
        
        self.feature_columns.extend(currency_features)
        print(f"   ✅ Currency features: {len(currency_features)}")
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, 
                                 volatility_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add volatility features from Phase 3
        
        Critical for distinguishing normal pullbacks from crisis events
        """
        print("📊 Calculating volatility features...")
        
        volatility_features = []
        
        # 1. VIX Features
        if 'VIX' in volatility_data:
            vix = volatility_data['VIX']['Close']
            
            # Basic VIX metrics
            df['vix_level'] = vix
            df['vix_percentile_252d'] = vix.rolling(252, min_periods=20).rank(pct=True)
            
            # VIX momentum
            df['vix_momentum_3d'] = vix.pct_change(3) * 100
            df['vix_momentum_5d'] = vix.pct_change(5) * 100
            df['vix_momentum_10d'] = vix.pct_change(10) * 100
            
            # VIX regime
            df['vix_regime'] = pd.cut(vix, bins=[0, 15, 25, 35, 100], 
                                     labels=[1, 2, 3, 4], include_lowest=True).astype(float)
            
            # VIX spike detection
            df['vix_spike'] = (vix.pct_change() > 15).astype(int)
            
            # VIX mean reversion
            vix_ma20 = vix.rolling(20).mean()
            df['vix_vs_ma20'] = ((vix - vix_ma20) / vix_ma20) * 100
            df['vix_extreme_high'] = (df['vix_vs_ma20'] > 50).astype(int)
            
            volatility_features.extend([
                'vix_level', 'vix_percentile_252d', 'vix_momentum_3d', 'vix_momentum_5d',
                'vix_momentum_10d', 'vix_regime', 'vix_spike', 'vix_vs_ma20', 'vix_extreme_high'
            ])
        
        # 2. VIX Term Structure
        if 'VIX9D' in volatility_data and 'VIX' in volatility_data:
            vix = volatility_data['VIX']['Close']
            vix9d = volatility_data['VIX9D']['Close']
            
            df['vix_term_structure'] = vix - vix9d
            df['vix_backwardation'] = (df['vix_term_structure'] < -2).astype(int)
            df['vix_term_momentum'] = df['vix_term_structure'].diff()
            
            volatility_features.extend(['vix_term_structure', 'vix_backwardation', 'vix_term_momentum'])
        
        # 3. VVIX (Volatility of Volatility)
        if 'VVIX' in volatility_data:
            vvix = volatility_data['VVIX']['Close']
            
            df['vvix_level'] = vvix
            df['vvix_percentile_252d'] = vvix.rolling(252, min_periods=20).rank(pct=True)
            df['vvix_momentum_5d'] = vvix.pct_change(5) * 100
            
            volatility_features.extend(['vvix_level', 'vvix_percentile_252d', 'vvix_momentum_5d'])
        
        # 4. Realized vs Implied Volatility
        if 'VIX' in volatility_data:
            vix = volatility_data['VIX']['Close']
            realized_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            df['realized_vol_20d'] = realized_vol
            df['vix_vs_realized'] = vix - realized_vol
            df['vol_risk_premium'] = df['vix_vs_realized']
            
            volatility_features.extend(['realized_vol_20d', 'vix_vs_realized', 'vol_risk_premium'])
        
        # 5. Volatility regime detection
        if 'vix_level' in df.columns and 'volatility_20d' in df.columns:
            df['vol_regime_transition'] = (
                (df['vix_level'] > 20) & 
                (df['vix_momentum_5d'] > 10)
            ).astype(int)
            
            volatility_features.append('vol_regime_transition')
        
        self.feature_columns.extend(volatility_features)
        print(f"   ✅ Volatility features: {len(volatility_features)}")
        
        return df
    
    def _add_options_features(self, df: pd.DataFrame, 
                             options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add options-specific features (placeholder for future expansion)
        
        This would include:
        - Put/Call ratio
        - Unusual options activity
        - Hedging intensity
        - IV skew
        """
        print("📈 Calculating options features...")
        print("   ⚠️  Options features not yet implemented")
        
        # Placeholder for future options features
        options_features = []
        
        self.feature_columns.extend(options_features)
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean calculated features
        
        Strategy:
        - Drop rows missing critical features (Close, volatility)
        - Forward/backward fill other features with limited NaN values
        """
        print("🧹 Cleaning features...")
        
        # Critical features that must exist
        critical_features = ['Close', 'volatility_20d', 'atr_14']
        available_critical = [f for f in critical_features if f in df.columns]
        
        initial_len = len(df)
        df = df.dropna(subset=available_critical)
        dropped = initial_len - len(df)
        
        if dropped > 0:
            print(f"   Dropped {dropped} rows with missing critical features")
        
        # Forward/backward fill other features
        for feature in self.feature_columns:
            if feature in df.columns:
                df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill')
        
        # Final check for remaining NaN values
        nan_counts = df[self.feature_columns].isna().sum()
        if nan_counts.sum() > 0:
            print(f"   ⚠️  Remaining NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            # Fill remaining NaN with 0
            df[self.feature_columns] = df[self.feature_columns].fillna(0)
        
        print(f"   ✅ Clean data: {len(df)} records with {len(self.feature_columns)} features")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of calculated feature columns"""
        return self.feature_columns.copy()


if __name__ == "__main__":
    # Test feature engineering
    from core.data_loader import DataLoader
    
    print("Testing FeatureEngine...")
    loader = DataLoader(start_date='2020-01-01', end_date='2024-12-31')
    data = loader.load_all_data()
    
    # Test each feature set
    for feature_set in [['baseline'], ['baseline', 'currency'], ['baseline', 'currency', 'volatility'], ['all']]:
        print("\n" + "=" * 80)
        print(f"Testing feature set: {feature_set}")
        print("=" * 80)
        
        engine = FeatureEngine(feature_sets=feature_set)
        features = engine.calculate_features(
            spy_data=data['spy'],
            sector_data=data.get('sectors'),
            currency_data=data.get('currency'),
            volatility_data=data.get('volatility')
        )
        
        print(f"\nResult: {len(features)} records, {len(engine.get_feature_columns())} features")
        print(f"Features: {engine.get_feature_columns()}")

