"""
Market Rotation Indicators

Analyzes rotation between different market segments using specialized ETFs:
- MAGS vs SPY: Magnificent 7 concentration risk
- RSP vs SPY: Equal-weight vs cap-weight (breadth)
- QQQ vs SPY: Tech leadership
- QQQE vs QQQ: Tech breadth
"""

import pandas as pd
import numpy as np
from features.market.base import BaseMarketFeature


class RotationIndicatorFeature(BaseMarketFeature):
    """
    Market rotation indicators using specialized ETFs
    
    Key Signals:
    - MAGS outperforming SPY = Concentration risk (few stocks driving market)
    - RSP outperforming SPY = Broad market strength (healthy)
    - QQQ outperforming SPY = Tech leadership (risk-on)
    - QQQE outperforming QQQ = Tech breadth improving (healthy tech)
    """
    
    def __init__(self, lookback_periods: list = None):
        super().__init__("RotationIndicators")
        self.lookback_periods = lookback_periods or [5, 10, 20]
        self.required_etfs = ['MAGS', 'RSP', 'QQQ', 'QQQE']
    
    def calculate(self, data: pd.DataFrame, sector_data: dict = None, **kwargs) -> pd.DataFrame:
        """Calculate rotation indicators"""
        if sector_data is None:
            return data
        
        df = data.copy()
        
        # Check if we have the required ETFs
        available_etfs = [etf for etf in self.required_etfs if etf in sector_data]
        if len(available_etfs) < 2:
            print(f"   ⚠️  Rotation indicators: Only {len(available_etfs)}/4 ETFs available")
            return df
        
        # Calculate relative performance for each lookback period
        for period in self.lookback_periods:
            spy_return = df['Close'].pct_change(period) * 100
            
            # MAGS vs SPY (Concentration Risk)
            if 'MAGS' in sector_data:
                mags_return = sector_data['MAGS']['Close'].pct_change(period) * 100
                df[f'mags_vs_spy_{period}d'] = mags_return - spy_return
                self.feature_names.append(f'mags_vs_spy_{period}d')
            
            # RSP vs SPY (Breadth Indicator)
            if 'RSP' in sector_data:
                rsp_return = sector_data['RSP']['Close'].pct_change(period) * 100
                df[f'rsp_vs_spy_{period}d'] = rsp_return - spy_return
                self.feature_names.append(f'rsp_vs_spy_{period}d')
            
            # QQQ vs SPY (Tech Leadership)
            if 'QQQ' in sector_data:
                qqq_return = sector_data['QQQ']['Close'].pct_change(period) * 100
                df[f'qqq_vs_spy_{period}d'] = qqq_return - spy_return
                self.feature_names.append(f'qqq_vs_spy_{period}d')
            
            # QQQE vs QQQ (Tech Breadth)
            if 'QQQE' in sector_data and 'QQQ' in sector_data:
                qqqe_return = sector_data['QQQE']['Close'].pct_change(period) * 100
                qqq_return = sector_data['QQQ']['Close'].pct_change(period) * 100
                df[f'qqqe_vs_qqq_{period}d'] = qqqe_return - qqq_return
                self.feature_names.append(f'qqqe_vs_qqq_{period}d')
        
        # Binary risk signals
        if 'MAGS' in sector_data and 'RSP' in sector_data:
            # Concentration risk: MAGS outperforming AND RSP underperforming
            mags_outperform = df.get(f'mags_vs_spy_{self.lookback_periods[0]}d', 0) > 2.0
            rsp_underperform = df.get(f'rsp_vs_spy_{self.lookback_periods[0]}d', 0) < -1.0
            df['concentration_risk'] = (mags_outperform & rsp_underperform).astype(int)
            self.feature_names.append('concentration_risk')
        
        if 'RSP' in sector_data:
            # Broad market strength: RSP outperforming SPY
            df['broad_market_strength'] = (df.get(f'rsp_vs_spy_{self.lookback_periods[0]}d', 0) > 1.0).astype(int)
            self.feature_names.append('broad_market_strength')
        
        if 'QQQ' in sector_data:
            # Tech leadership: QQQ significantly outperforming
            df['tech_leadership'] = (df.get(f'qqq_vs_spy_{self.lookback_periods[0]}d', 0) > 2.0).astype(int)
            self.feature_names.append('tech_leadership')
        
        if 'QQQE' in sector_data and 'QQQ' in sector_data:
            # Healthy tech: Equal-weight tech outperforming cap-weight
            df['healthy_tech_breadth'] = (df.get(f'qqqe_vs_qqq_{self.lookback_periods[0]}d', 0) > 0.5).astype(int)
            self.feature_names.append('healthy_tech_breadth')
        
        print(f"   ✅ Rotation indicators: {len(self.feature_names)} features from {len(available_etfs)} ETFs")
        
        return df
    
    def validate_sector_data(self, sector_data: dict) -> bool:
        """Validate we have rotation indicator ETFs"""
        if not sector_data:
            return False
        
        available = [etf for etf in self.required_etfs if etf in sector_data]
        if len(available) < 2:
            print(f"   ⚠️  Warning: Only {len(available)}/4 rotation ETFs available")
            print(f"   Available: {available}")
            print(f"   Missing: {set(self.required_etfs) - set(available)}")
        
        return len(available) >= 2  # Need at least 2 to be useful


class ConcentrationRiskFeature(BaseMarketFeature):
    """
    Concentration risk analyzer
    
    High concentration risk when:
    - MAGS significantly outperforms SPY (few stocks driving market)
    - RSP underperforms SPY (narrow market)
    - This combination often precedes corrections
    """
    
    def __init__(self):
        super().__init__("ConcentrationRisk")
        self.required_etfs = ['MAGS', 'RSP']
    
    def calculate(self, data: pd.DataFrame, sector_data: dict = None, **kwargs) -> pd.DataFrame:
        """Calculate concentration risk metrics"""
        if sector_data is None or not all(etf in sector_data for etf in self.required_etfs):
            return data
        
        df = data.copy()
        
        # Calculate returns
        spy_return_20d = df['Close'].pct_change(20) * 100
        mags_return_20d = sector_data['MAGS']['Close'].pct_change(20) * 100
        rsp_return_20d = sector_data['RSP']['Close'].pct_change(20) * 100
        
        # Concentration score
        # High when MAGS >> SPY and RSP << SPY
        mags_spread = mags_return_20d - spy_return_20d
        rsp_spread = rsp_return_20d - spy_return_20d
        
        df['concentration_score'] = mags_spread - rsp_spread
        
        # Extreme concentration (risk signal)
        df['extreme_concentration'] = (
            (mags_spread > 3.0) & (rsp_spread < -2.0)
        ).astype(int)
        
        # Concentration trend (getting worse)
        df['concentration_increasing'] = (
            df['concentration_score'].diff(5) > 2.0
        ).astype(int)
        
        self.feature_names = [
            'concentration_score',
            'extreme_concentration',
            'concentration_increasing'
        ]
        
        return df


if __name__ == "__main__":
    # Test the feature
    print("Testing RotationIndicatorFeature...")
    
    import yfinance as yf
    
    # Download test data
    spy = yf.download('SPY', start='2024-01-01', end='2024-10-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    sector_data = {}
    for symbol in ['MAGS', 'RSP', 'QQQ', 'QQQE']:
        data = yf.download(symbol, start='2024-01-01', end='2024-10-01', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) > 0:
            sector_data[symbol] = data
    
    # Calculate features
    feature = RotationIndicatorFeature()
    result = feature.calculate(spy, sector_data=sector_data)
    
    print(f"\n✅ Features calculated: {len(feature.get_feature_names())}")
    print(f"Features: {feature.get_feature_names()}")
    print(f"\nSample data (last 5 rows):")
    print(result[feature.get_feature_names()].tail())
