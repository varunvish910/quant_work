"""
Cross-Asset Correlation Features

Features based on relationships between:
- SPY and bonds (TLT)
- SPY and gold (GLD)
- SPY and volatility
- Risk-on/risk-off indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from features.base import BaseFeature


class CrossAssetFeature(BaseFeature):
    """
    Cross-asset correlation and risk regime features.
    """
    
    def __init__(self):
        super().__init__("CrossAsset")
        self.feature_names = []
        self.tlt_data = None
        self.gld_data = None
    
    def _download_asset(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download asset data from yfinance"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data
        except Exception as e:
            print(f"   ⚠️  Failed to download {symbol}: {e}")
            return None
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate cross-asset correlation features.
        
        Args:
            data: DataFrame with SPY OHLC data
            
        Returns:
            DataFrame with cross-asset features added
        """
        df = data.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        
        print(f"   📊 Downloading cross-asset data...")
        
        # Download TLT (20+ Year Treasury Bond ETF)
        print(f"      Loading TLT (bonds)...")
        tlt = self._download_asset('TLT', start_date, end_date)
        
        # Download GLD (Gold ETF)
        print(f"      Loading GLD (gold)...")
        gld = self._download_asset('GLD', start_date, end_date)
        
        # 1. SPY-TLT Correlation (Flight to Safety)
        if tlt is not None and len(tlt) > 0:
            # Align TLT with SPY
            tlt_aligned = tlt['Close'].reindex(df.index, method='ffill')
            
            # Calculate returns
            spy_returns = df['Close'].pct_change()
            tlt_returns = tlt_aligned.pct_change()
            
            # Rolling correlation (20-day)
            df['spy_tlt_corr_20d'] = spy_returns.rolling(20).corr(tlt_returns)
            
            # Correlation regime
            df['spy_tlt_negative_corr'] = (df['spy_tlt_corr_20d'] < -0.3).astype(int)  # Flight to safety
            df['spy_tlt_positive_corr'] = (df['spy_tlt_corr_20d'] > 0.3).astype(int)  # Risk-on
            
            # TLT momentum (bonds rallying = risk-off)
            df['tlt_return_5d'] = tlt_aligned.pct_change(5) * 100
            df['tlt_return_20d'] = tlt_aligned.pct_change(20) * 100
            
            # Bond strength (TLT up + SPY down = flight to safety)
            df['flight_to_safety_score'] = (df['tlt_return_5d'] > 0).astype(int) * (spy_returns.rolling(5).sum() < 0).astype(int)
            
            print(f"      ✅ Added TLT features")
        else:
            print(f"      ⚠️  Skipping TLT features")
        
        # 2. SPY-GLD Correlation (Risk-Off Behavior)
        if gld is not None and len(gld) > 0:
            # Align GLD with SPY
            gld_aligned = gld['Close'].reindex(df.index, method='ffill')
            
            # Calculate returns
            gld_returns = gld_aligned.pct_change()
            
            # Rolling correlation (20-day)
            df['spy_gld_corr_20d'] = spy_returns.rolling(20).corr(gld_returns)
            
            # Gold momentum
            df['gld_return_5d'] = gld_aligned.pct_change(5) * 100
            df['gld_return_20d'] = gld_aligned.pct_change(20) * 100
            
            # Gold strength (GLD up + SPY down = risk-off)
            df['gold_strength_score'] = (df['gld_return_5d'] > 0).astype(int) * (spy_returns.rolling(5).sum() < 0).astype(int)
            
            print(f"      ✅ Added GLD features")
        else:
            print(f"      ⚠️  Skipping GLD features")
        
        # 3. Risk Regime Composite
        # Combine multiple signals for overall risk regime
        risk_off_signals = []
        
        if 'spy_tlt_negative_corr' in df.columns:
            risk_off_signals.append(df['spy_tlt_negative_corr'])
        if 'flight_to_safety_score' in df.columns:
            risk_off_signals.append(df['flight_to_safety_score'])
        if 'gold_strength_score' in df.columns:
            risk_off_signals.append(df['gold_strength_score'])
        
        if len(risk_off_signals) > 0:
            df['risk_off_composite'] = sum(risk_off_signals) / len(risk_off_signals)
            df['is_risk_off_regime'] = (df['risk_off_composite'] >= 0.5).astype(int)
        
        # 4. Cross-Asset Momentum Divergence
        # When bonds/gold rally but stocks don't = warning sign
        if 'tlt_return_5d' in df.columns and 'gld_return_5d' in df.columns:
            spy_return_5d = df['Close'].pct_change(5) * 100
            
            # Both TLT and GLD up, but SPY flat/down
            df['defensive_assets_outperform'] = (
                (df['tlt_return_5d'] > 1) & 
                (df['gld_return_5d'] > 1) & 
                (spy_return_5d < 1)
            ).astype(int)
        
        # Store feature names (only those that were actually created)
        self.feature_names = [col for col in df.columns if col not in data.columns]
        
        print(f"   ✅ Created {len(self.feature_names)} cross-asset features")
        
        return df


if __name__ == "__main__":
    # Test cross-asset features
    import yfinance as yf
    
    print("Testing CrossAssetFeature...")
    spy = yf.download('SPY', start='2024-01-01', end='2024-12-31', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    feature = CrossAssetFeature()
    result = feature.calculate(spy)
    
    print(f"\n✅ Created {len(feature.feature_names)} cross-asset features:")
    for feat in feature.feature_names:
        print(f"   - {feat}")
    
    if len(feature.feature_names) > 0:
        print(f"\nSample data:")
        print(result[feature.feature_names].tail(10))
