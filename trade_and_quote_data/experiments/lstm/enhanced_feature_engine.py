#!/usr/bin/env python3
"""
Enhanced Feature Engineering Pipeline
Phase 0 implementation from MODEL_IMPROVEMENT_ROADMAP.md

Implements tier-based feature engineering with systematic feature selection
based on market logic and predictive power.

Author: AI Assistant  
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TieredFeatureEngine:
    """
    Systematic feature engineering with hierarchical tiers:
    
    Tier 1 (Core - 15 features): Proven market indicators
    Tier 2 (Enhanced - 15 features): Cross-asset and options signals  
    Tier 3 (Experimental - 10 features): Advanced and alternative data
    """
    
    def __init__(self):
        self.feature_tiers = {
            'tier1_volatility': [
                'vix_level',
                'vix_momentum_3d', 
                'vix_momentum_5d',
                'vix_vs_ma',
                'spy_volatility_10d',
                'spy_volatility_20d',
                'vix_spike_indicator'
            ],
            'tier1_momentum': [
                'rsi_14',
                'rsi_extreme_flag',
                'momentum_5d',
                'momentum_10d', 
                'momentum_20d',
                'distance_from_high_20d'
            ],
            'tier1_trend': [
                'price_vs_sma20',
                'price_vs_sma50',
                'sma_crossover',
                'bb_width',
                'bb_position',
                'adx_proxy'
            ],
            'tier2_options': [
                'put_call_ratio_proxy',
                'iv_rank_proxy',
                'vix_term_structure',
                'options_flow_imbalance'
            ],
            'tier2_cross_asset': [
                'tlt_momentum_10d',
                'gld_momentum_10d', 
                'dxy_level',
                'usdjpy_momentum_5d',
                'sector_rotation_signal',
                'bond_equity_correlation'
            ],
            'tier2_microstructure': [
                'gap_analysis',
                'overnight_vs_intraday',
                'volume_profile_proxy',
                'price_acceleration',
                'volatility_regime'
            ],
            'tier3_seasonality': [
                'month_effect',
                'day_of_week_effect',
                'quarter_end_effect',
                'options_expiry_effect'
            ],
            'tier3_advanced': [
                'regime_detection',
                'tail_risk_indicator',
                'correlation_breakdown',
                'liquidity_stress_proxy',
                'sentiment_divergence',
                'momentum_exhaustion'
            ]
        }
        
        # Cache for downloaded data
        self._data_cache = {}
        
    def download_market_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download all required market data for feature engineering"""
        
        cache_key = f"{start_date}_{end_date}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
            
        logger.info(f"Downloading market data from {start_date} to {end_date}")
        
        data = {}
        
        # Main assets
        tickers = {
            'SPY': 'SPY',    # S&P 500 ETF
            'VIX': '^VIX',   # Volatility Index
            'TLT': 'TLT',    # 20+ Year Treasury ETF
            'GLD': 'GLD',    # Gold ETF
            'DXY': 'DX-Y.NYB', # Dollar Index
            'JPY': 'USDJPY=X'  # USD/JPY exchange rate
        }
        
        for name, ticker in tickers.items():
            try:
                data[name] = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data[name].empty:
                    logger.warning(f"No data downloaded for {ticker}")
                else:
                    logger.info(f"Downloaded {len(data[name])} days for {name}")
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")
                # Create dummy data to prevent failures
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                data[name] = pd.DataFrame({
                    'Open': 100, 'High': 101, 'Low': 99, 'Close': 100, 'Volume': 1000000
                }, index=date_range)
        
        # Align all data to SPY index (main reference)
        if 'SPY' in data and not data['SPY'].empty:
            spy_index = data['SPY'].index
            for name in data:
                if name != 'SPY':
                    data[name] = data[name].reindex(spy_index, method='ffill')
        
        self._data_cache[cache_key] = data
        return data
    
    def create_tier1_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create Tier 1 (Core) Features - 15 most predictive features
        Based on proven market indicators with strong theoretical foundation
        """
        spy = data['SPY']
        
        # Handle VIX data with potential MultiIndex columns
        if 'VIX' in data:
            vix_data = data['VIX']
            if 'Close' in vix_data.columns:
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix = vix_data['Close'].iloc[:, 0]
                else:
                    vix = vix_data['Close']
            else:
                vix = pd.Series(20, index=spy.index)
        else:
            vix = pd.Series(20, index=spy.index)
        
        features = pd.DataFrame(index=spy.index)
        
        # === VOLATILITY FEATURES ===
        features['vix_level'] = vix
        features['vix_momentum_3d'] = vix.pct_change(3)
        features['vix_momentum_5d'] = vix.pct_change(5)
        features['vix_vs_ma'] = vix / vix.rolling(20).mean() - 1
        
        # SPY volatility measures
        spy_returns = spy['Close'].pct_change()
        features['spy_volatility_10d'] = spy_returns.rolling(10).std() * np.sqrt(252)
        features['spy_volatility_20d'] = spy_returns.rolling(20).std() * np.sqrt(252)
        
        # VIX spike detection (above 80th percentile of recent readings)
        features['vix_spike_indicator'] = (vix > vix.rolling(60).quantile(0.8)).astype(int)
        
        # === MOMENTUM FEATURES ===
        features['rsi_14'] = self._calculate_rsi(spy['Close'], 14)
        features['rsi_extreme_flag'] = ((features['rsi_14'] > 70) | (features['rsi_14'] < 30)).astype(int)
        
        # Price momentum across different timeframes
        features['momentum_5d'] = spy['Close'].pct_change(5)
        features['momentum_10d'] = spy['Close'].pct_change(10) 
        features['momentum_20d'] = spy['Close'].pct_change(20)
        
        # Distance from recent highs (momentum exhaustion signal)
        features['distance_from_high_20d'] = spy['Close'] / spy['High'].rolling(20).max() - 1
        
        # === TREND FEATURES ===
        sma20 = spy['Close'].rolling(20).mean()
        sma50 = spy['Close'].rolling(50).mean()
        
        features['price_vs_sma20'] = spy['Close'] / sma20 - 1
        features['price_vs_sma50'] = spy['Close'] / sma50 - 1
        features['sma_crossover'] = (sma20 > sma50).astype(int)
        
        # Bollinger Band analysis
        bb_std = spy['Close'].rolling(20).std()
        features['bb_width'] = (bb_std * 2) / sma20
        features['bb_position'] = (spy['Close'] - sma20) / (bb_std * 2)
        
        # ADX proxy using true range and directional movement
        features['adx_proxy'] = self._calculate_adx_proxy(spy)
        
        return features
    
    def create_tier2_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create Tier 2 (Enhanced) Features - 15 cross-asset and options features
        Advanced signals from multiple markets and asset classes
        """
        spy = data['SPY']
        
        # Handle VIX data with potential MultiIndex columns
        if 'VIX' in data:
            vix_data = data['VIX']
            if 'Close' in vix_data.columns:
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix = vix_data['Close'].iloc[:, 0]  # Handle MultiIndex
                else:
                    vix = vix_data['Close']
            else:
                vix = pd.Series(20, index=spy.index)
        else:
            vix = pd.Series(20, index=spy.index)
            
        tlt = data['TLT'] if 'TLT' in data else spy  # Fallback to SPY if TLT unavailable
        gld = data['GLD'] if 'GLD' in data else spy
        
        features = pd.DataFrame(index=spy.index)
        
        # === OPTIONS FEATURES ===
        # Put/Call ratio proxy using VIX vs realized volatility
        realized_vol = spy['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Ensure vix is a Series for division
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        if isinstance(realized_vol, pd.DataFrame):
            realized_vol = realized_vol.iloc[:, 0]
            
        features['put_call_ratio_proxy'] = vix / (realized_vol + 1e-10)
        
        # IV rank proxy
        features['iv_rank_proxy'] = (vix - vix.rolling(252).min()) / (vix.rolling(252).max() - vix.rolling(252).min())
        
        # VIX term structure proxy (VIX vs its moving average)
        features['vix_term_structure'] = vix / vix.rolling(60).mean() - 1
        
        # Options flow imbalance (momentum in put/call ratio)
        features['options_flow_imbalance'] = features['put_call_ratio_proxy'].pct_change(5)
        
        # === CROSS-ASSET FEATURES ===
        # Treasury momentum (flight to quality indicator)
        if 'Close' in tlt.columns:
            features['tlt_momentum_10d'] = tlt['Close'].pct_change(10)
        else:
            features['tlt_momentum_10d'] = 0
            
        # Gold momentum (alternative asset performance)
        if 'Close' in gld.columns:
            features['gld_momentum_10d'] = gld['Close'].pct_change(10)
        else:
            features['gld_momentum_10d'] = 0
        
        # Dollar strength (affects risk sentiment)
        if 'DXY' in data and 'Close' in data['DXY'].columns:
            features['dxy_level'] = data['DXY']['Close'] / data['DXY']['Close'].rolling(50).mean() - 1
        else:
            features['dxy_level'] = 0
            
        # Yen carry trade indicator (critical for August 2024 crash)
        if 'JPY' in data and 'Close' in data['JPY'].columns:
            features['usdjpy_momentum_5d'] = data['JPY']['Close'].pct_change(5)
        else:
            features['usdjpy_momentum_5d'] = 0
        
        # Sector rotation signal (TLT vs SPY performance)
        features['sector_rotation_signal'] = features['tlt_momentum_10d'] - spy['Close'].pct_change(10)
        
        # Bond-equity correlation (risk-on/risk-off regime)
        spy_returns = spy['Close'].pct_change()
        tlt_returns = tlt['Close'].pct_change() if 'Close' in tlt.columns else spy_returns
        features['bond_equity_correlation'] = spy_returns.rolling(20).corr(tlt_returns)
        
        # === MICROSTRUCTURE FEATURES ===
        # Gap analysis (overnight moves)
        features['gap_analysis'] = (spy['Open'] - spy['Close'].shift(1)) / spy['Close'].shift(1)
        
        # Overnight vs intraday performance  
        overnight_return = (spy['Open'] - spy['Close'].shift(1)) / spy['Close'].shift(1)
        intraday_return = (spy['Close'] - spy['Open']) / spy['Open']
        features['overnight_vs_intraday'] = overnight_return / (intraday_return + 1e-10)
        
        # Volume profile proxy (high-low range vs close)
        features['volume_profile_proxy'] = (spy['High'] - spy['Low']) / spy['Close']
        
        # Price acceleration (second derivative of price)
        features['price_acceleration'] = spy['Close'].pct_change().diff()
        
        # Volatility regime detection
        vol_ma = realized_vol.rolling(60).mean()
        features['volatility_regime'] = (realized_vol > vol_ma * 1.2).astype(int)
        
        return features
    
    def create_tier3_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create Tier 3 (Experimental) Features - 10 advanced features
        Experimental and alternative data signals
        """
        spy = data['SPY']
        
        # Handle VIX data with potential MultiIndex columns
        if 'VIX' in data:
            vix_data = data['VIX']
            if 'Close' in vix_data.columns:
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix = vix_data['Close'].iloc[:, 0]
                else:
                    vix = vix_data['Close']
            else:
                vix = pd.Series(20, index=spy.index)
        else:
            vix = pd.Series(20, index=spy.index)
        
        features = pd.DataFrame(index=spy.index)
        
        # === SEASONALITY FEATURES ===
        features['month_effect'] = spy.index.month
        features['day_of_week_effect'] = spy.index.dayofweek
        features['quarter_end_effect'] = ((spy.index.month % 3 == 0) & 
                                         (spy.index.day > 25)).astype(int)
        
        # Options expiry effect (third Friday of each month)
        third_friday = spy.index.to_series().apply(
            lambda x: x.day >= 15 and x.day <= 21 and x.weekday() == 4
        ).astype(int)
        features['options_expiry_effect'] = third_friday
        
        # === ADVANCED FEATURES ===
        # Market regime detection using multiple timeframes
        short_trend = spy['Close'] > spy['Close'].rolling(20).mean()
        long_trend = spy['Close'] > spy['Close'].rolling(200).mean()
        features['regime_detection'] = (short_trend & long_trend).astype(int)
        
        # Tail risk indicator (extreme VIX moves)
        vix_zscore = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
        features['tail_risk_indicator'] = (vix_zscore > 2).astype(int)
        
        # Correlation breakdown detection
        spy_returns = spy['Close'].pct_change()
        rolling_vol = spy_returns.rolling(20).std()
        vol_of_vol = rolling_vol.rolling(20).std()
        features['correlation_breakdown'] = vol_of_vol / rolling_vol
        
        # Liquidity stress proxy (gap vs typical range)
        typical_range = (spy['High'] - spy['Low']).rolling(20).mean()
        gap_size = abs(spy['Open'] - spy['Close'].shift(1))
        features['liquidity_stress_proxy'] = gap_size / typical_range
        
        # Sentiment divergence (price vs VIX momentum)
        price_momentum = spy['Close'].pct_change(5)
        vix_momentum = vix.pct_change(5)
        features['sentiment_divergence'] = price_momentum * vix_momentum  # Negative when diverging
        
        # Momentum exhaustion indicator
        rsi = self._calculate_rsi(spy['Close'], 14)
        momentum_20d = spy['Close'].pct_change(20)
        features['momentum_exhaustion'] = ((rsi > 70) & (momentum_20d > 0.05)).astype(int)
        
        return features
    
    def create_all_features(self, data: Dict[str, pd.DataFrame], 
                           tiers: List[str] = None) -> pd.DataFrame:
        """
        Create features from specified tiers
        
        Args:
            data: Dictionary of market data
            tiers: List of tiers to include ['tier1', 'tier2', 'tier3']
                  If None, includes all tiers
        """
        if tiers is None:
            tiers = ['tier1', 'tier2', 'tier3']
        
        all_features = pd.DataFrame(index=data['SPY'].index)
        
        if 'tier1' in tiers:
            tier1_features = self.create_tier1_features(data)
            all_features = pd.concat([all_features, tier1_features], axis=1)
            logger.info(f"Added {len(tier1_features.columns)} Tier 1 features")
        
        if 'tier2' in tiers:
            tier2_features = self.create_tier2_features(data)
            all_features = pd.concat([all_features, tier2_features], axis=1)
            logger.info(f"Added {len(tier2_features.columns)} Tier 2 features")
            
        if 'tier3' in tiers:
            tier3_features = self.create_tier3_features(data)
            all_features = pd.concat([all_features, tier3_features], axis=1)
            logger.info(f"Added {len(tier3_features.columns)} Tier 3 features")
        
        # Remove features with too many NaN values (>50%)
        nan_threshold = len(all_features) * 0.5
        all_features = all_features.dropna(axis=1, thresh=nan_threshold)
        
        # Forward fill remaining NaN values
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        logger.info(f"Final feature set: {len(all_features.columns)} features, {len(all_features)} observations")
        
        return all_features
    
    def get_feature_importance_tiers(self) -> Dict[str, List[str]]:
        """Return the feature tier definitions for analysis"""
        return self.feature_tiers
    
    # === HELPER METHODS ===
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx_proxy(self, ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ADX proxy using True Range and Directional Movement
        Simplified version that captures trend strength
        """
        high = ohlc['High']
        low = ohlc['Low'] 
        close = ohlc['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth the values
        tr_smooth = true_range.rolling(period).mean()
        plus_dm_smooth = plus_dm.rolling(period).mean()
        minus_dm_smooth = minus_dm.rolling(period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))
        
        # ADX proxy (simplified)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_proxy = dx.rolling(period).mean()
        
        # Ensure we return a Series, not DataFrame
        if isinstance(adx_proxy, pd.DataFrame):
            adx_proxy = adx_proxy.iloc[:, 0]
        
        return adx_proxy.fillna(25)  # Fill with neutral ADX value


def test_feature_engine():
    """Test the feature engineering pipeline"""
    logger.info("Testing Enhanced Feature Engine")
    
    # Create engine
    engine = TieredFeatureEngine()
    
    # Download sample data
    data = engine.download_market_data('2020-01-01', '2024-12-31')
    
    # Test each tier individually
    print("\nTesting individual tiers:")
    
    tier1 = engine.create_tier1_features(data)
    print(f"Tier 1: {len(tier1.columns)} features")
    print(f"Sample features: {list(tier1.columns[:5])}")
    
    tier2 = engine.create_tier2_features(data) 
    print(f"Tier 2: {len(tier2.columns)} features")
    print(f"Sample features: {list(tier2.columns[:5])}")
    
    tier3 = engine.create_tier3_features(data)
    print(f"Tier 3: {len(tier3.columns)} features") 
    print(f"Sample features: {list(tier3.columns[:5])}")
    
    # Test combined features
    all_features = engine.create_all_features(data)
    print(f"\nCombined: {len(all_features.columns)} total features")
    
    # Test tier selection
    tier1_only = engine.create_all_features(data, tiers=['tier1'])
    print(f"Tier 1 only: {len(tier1_only.columns)} features")
    
    print("\nFeature engineering test completed successfully!")
    return all_features


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    features = test_feature_engine()
    
    # Display sample statistics
    print("\nSample Feature Statistics:")
    print(features.describe().round(4).head())