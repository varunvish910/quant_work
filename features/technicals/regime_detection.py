"""
Regime Detection Features

Identifies market regime (trending vs range-bound) to improve pullback prediction.
Critical for filtering false signals in choppy markets.
"""

import pandas as pd
import numpy as np
from features.technicals.base import BaseTechnicalFeature


class RegimeDetectionFeature(BaseTechnicalFeature):
    """
    Comprehensive regime detection using ADX and Bollinger Band width.
    
    Key insight: Pullbacks happen in TRENDING markets, not range-bound markets.
    ADX < 20 = range-bound (no pullbacks, just oscillation)
    ADX > 25 = trending (pullbacks possible)
    """
    
    def __init__(self, adx_period: int = 14, bb_period: int = 20):
        super().__init__("RegimeDetection")
        self.adx_period = adx_period
        self.bb_period = bb_period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate regime detection features"""
        df = data.copy()
        
        # 1. ADX Calculation (Average Directional Index)
        df = self._calculate_adx(df)
        
        # 2. Bollinger Band Width
        df = self._calculate_bb_width(df)
        
        # 3. Regime Classification
        df = self._classify_regime(df)
        
        # Track feature names
        self.feature_names = [
            'adx', 'plus_di', 'minus_di',
            'adx_regime_score',
            'is_trending', 'is_range_bound',
            'bb_width', 'bb_width_percentile',
            'bb_squeeze', 'bb_expansion',
            'regime_confidence'
        ]
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index)
        
        ADX measures trend strength (not direction):
        - ADX < 20: Range-bound/choppy
        - ADX 20-25: Weak trend
        - ADX 25-40: Strong trend
        - ADX > 40: Very strong trend
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        # Only keep positive movements
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When both are positive, only keep the larger one
        plus_dm[(plus_dm > 0) & (minus_dm > 0) & (plus_dm < minus_dm)] = 0
        minus_dm[(plus_dm > 0) & (minus_dm > 0) & (minus_dm < plus_dm)] = 0
        
        # Smooth with Wilder's smoothing (exponential moving average)
        alpha = 1 / self.adx_period
        
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
        
        # Calculate DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def _calculate_bb_width(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Band width as volatility regime indicator
        
        Narrow bands = low volatility = range-bound
        Wide bands = high volatility = trending
        """
        close = df['Close']
        
        # Calculate Bollinger Bands
        bb_middle = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        
        # Band width as percentage of middle band
        bb_width = (bb_upper - bb_lower) / bb_middle * 100
        
        # Historical percentile of band width (252 trading days = 1 year)
        bb_width_percentile = bb_width.rolling(window=252, min_periods=20).rank(pct=True)
        
        df['bb_width'] = bb_width
        df['bb_width_percentile'] = bb_width_percentile
        
        # Squeeze detection (low volatility before breakout)
        df['bb_squeeze'] = (bb_width_percentile < 0.1).astype(int)
        
        # Expansion detection (high volatility)
        df['bb_expansion'] = (bb_width_percentile > 0.9).astype(int)
        
        return df
    
    def _classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime using ADX and BB width
        
        Composite scoring system for robust regime detection
        """
        # ADX-based regime score
        def adx_score(adx_val):
            if adx_val < 20:
                return 0  # Range-bound
            elif adx_val < 25:
                return 1  # Weak trend
            elif adx_val < 40:
                return 2  # Strong trend
            else:
                return 3  # Very strong trend
        
        df['adx_regime_score'] = df['adx'].apply(adx_score)
        
        # Binary regime indicators
        df['is_trending'] = (df['adx'] > 25).astype(int)
        df['is_range_bound'] = (df['adx'] < 20).astype(int)
        
        # Regime confidence (0-1 scale)
        # High confidence when ADX is extreme or BB width is extreme
        adx_confidence = df['adx'].clip(0, 50) / 50  # Normalize to 0-1
        bb_confidence = df['bb_width_percentile'].fillna(0.5)
        
        # Average confidence from both indicators
        df['regime_confidence'] = (adx_confidence + bb_confidence) / 2
        
        return df


class MomentumExhaustionFeature(BaseTechnicalFeature):
    """
    Momentum exhaustion indicators for detecting tops.
    
    RSI overbought + MACD divergence = classic top signals
    """
    
    def __init__(self, rsi_period: int = 14):
        super().__init__("MomentumExhaustion")
        self.rsi_period = rsi_period
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate momentum exhaustion features"""
        df = data.copy()
        
        # 1. RSI (Relative Strength Index)
        df = self._calculate_rsi(df)
        
        # 2. MACD (Moving Average Convergence Divergence)
        df = self._calculate_macd(df)
        
        # 3. Price position relative to highs
        df = self._calculate_price_position(df)
        
        self.feature_names = [
            'rsi', 'rsi_overbought', 'rsi_extreme',
            'macd', 'macd_signal', 'macd_histogram',
            'distance_from_20d_high', 'at_new_highs',
            'extended_move'
        ]
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index)
        
        RSI > 70 = Overbought (pullback risk)
        RSI > 80 = Extremely overbought (high pullback risk)
        """
        close = df['Close']
        delta = close.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss using Wilder's smoothing
        avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df['rsi'] = rsi
        df['rsi_overbought'] = (rsi > 70).astype(int)
        df['rsi_extreme'] = (rsi > 80).astype(int)
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        MACD crossing below signal = momentum weakening
        """
        close = df['Close']
        
        # Calculate MACD line (12-day EMA - 26-day EMA)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        
        # Calculate signal line (9-day EMA of MACD)
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        
        # Calculate histogram
        macd_histogram = macd - macd_signal
        
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        return df
    
    def _calculate_price_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price position relative to recent highs
        
        At new highs = potential pullback zone
        """
        close = df['Close']
        high = df['High']
        
        # 20-day high
        high_20d = high.rolling(window=20).max()
        
        # Distance from 20-day high (as percentage)
        distance_from_high = (high_20d - close) / high_20d * 100
        
        df['distance_from_20d_high'] = distance_from_high
        df['at_new_highs'] = (distance_from_high < 0.5).astype(int)
        
        # Extended move (price > 2 std devs above 20-day MA)
        ma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        df['extended_move'] = (close > ma_20 + 2 * std_20).astype(int)
        
        return df


if __name__ == "__main__":
    # Test the features
    print("Testing Regime Detection Features...")
    
    import yfinance as yf
    
    # Download test data
    spy = yf.download('SPY', start='2020-01-01', end='2024-10-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # Test regime detection
    regime_feature = RegimeDetectionFeature()
    result = regime_feature.calculate(spy)
    
    print(f"\n✅ Regime features calculated")
    print(f"Features: {regime_feature.feature_names}")
    print(f"\nSample data (last 5 rows):")
    print(result[['adx', 'is_trending', 'bb_width_percentile', 'regime_confidence']].tail())
    
    # Test momentum exhaustion
    momentum_feature = MomentumExhaustionFeature()
    result = momentum_feature.calculate(result)
    
    print(f"\n✅ Momentum features calculated")
    print(f"Features: {momentum_feature.feature_names}")
    print(f"\nSample data (last 5 rows):")
    print(result[['rsi', 'rsi_overbought', 'macd', 'distance_from_20d_high']].tail())
