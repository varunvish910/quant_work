"""
Calendar and Seasonality Features

Features based on:
- Presidential cycle
- Month-of-year effects
- Quarter effects
- Options expiration cycles
- Day of week effects
"""

import pandas as pd
import numpy as np
from features.base import BaseFeature


class SeasonalityFeature(BaseFeature):
    """
    Calendar-based seasonality features for market prediction.
    """
    
    def __init__(self):
        super().__init__("Seasonality")
        self.feature_names = []
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate seasonality features.
        
        Args:
            data: DataFrame with DateTimeIndex
            
        Returns:
            DataFrame with seasonality features added
        """
        df = data.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 1. Presidential Cycle (1-4, where 1 = year after election)
        # Assuming 2024 is election year (cycle year 4)
        df['pres_cycle_year'] = ((df.index.year - 2024) % 4) + 1
        
        # Presidential cycle dummies
        df['pres_cycle_1'] = (df['pres_cycle_year'] == 1).astype(int)  # Post-election (weakest)
        df['pres_cycle_2'] = (df['pres_cycle_year'] == 2).astype(int)  # Mid-term
        df['pres_cycle_3'] = (df['pres_cycle_year'] == 3).astype(int)  # Pre-election (strongest)
        df['pres_cycle_4'] = (df['pres_cycle_year'] == 4).astype(int)  # Election year
        
        # 2. Month Effects
        df['month'] = df.index.month
        
        # Seasonal patterns
        df['is_january'] = (df['month'] == 1).astype(int)  # January effect
        df['is_september'] = (df['month'] == 9).astype(int)  # Historically weak
        df['is_october'] = (df['month'] == 10).astype(int)  # Crash month
        df['is_december'] = (df['month'] == 12).astype(int)  # Santa rally
        
        # Sell in May and go away
        df['summer_months'] = ((df['month'] >= 5) & (df['month'] <= 9)).astype(int)
        
        # 3. Quarter Effects
        df['quarter'] = df.index.quarter
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_q2'] = (df['quarter'] == 2).astype(int)
        df['is_q3'] = (df['quarter'] == 3).astype(int)  # Historically most volatile
        df['is_q4'] = (df['quarter'] == 4).astype(int)
        
        # Days until quarter end (earnings season proxy)
        df['days_in_quarter'] = df.index.day
        quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}
        df['days_to_quarter_end'] = df.apply(
            lambda row: (pd.Timestamp(row.name.year, quarter_end_month[row['quarter']], 1) + 
                        pd.offsets.MonthEnd(0) - row.name).days,
            axis=1
        )
        
        # Earnings season (last 2 weeks of quarter + first 3 weeks after)
        df['is_earnings_season'] = ((df['days_to_quarter_end'] <= 14) | 
                                    (df['days_to_quarter_end'] >= 77)).astype(int)
        
        # 4. Options Expiration (OpEx) Cycles
        # Monthly OpEx: 3rd Friday of each month
        # Calculate days until next OpEx
        df['day_of_month'] = df.index.day
        df['day_of_week'] = df.index.dayofweek  # Monday=0, Friday=4
        
        # Find 3rd Friday of current month
        def days_to_opex(date):
            # Find 3rd Friday of current month
            first_day = pd.Timestamp(date.year, date.month, 1)
            first_friday = first_day + pd.Timedelta(days=(4 - first_day.dayofweek) % 7)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            
            # If we're past 3rd Friday, look to next month
            if date > third_friday:
                if date.month == 12:
                    next_month = pd.Timestamp(date.year + 1, 1, 1)
                else:
                    next_month = pd.Timestamp(date.year, date.month + 1, 1)
                first_friday_next = next_month + pd.Timedelta(days=(4 - next_month.dayofweek) % 7)
                third_friday = first_friday_next + pd.Timedelta(weeks=2)
            
            return (third_friday - date).days
        
        df['days_to_opex'] = df.index.map(days_to_opex)
        
        # OpEx week (3 days before and day of)
        df['is_opex_week'] = (df['days_to_opex'] <= 3).astype(int)
        
        # Quarterly OpEx (March, June, September, December)
        df['is_quarterly_opex_month'] = df['month'].isin([3, 6, 9, 12]).astype(int)
        
        # 5. Day of Week Effects
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # 6. Turn-of-Month Effect (last 4 days + first 3 days)
        days_in_month = df.index.to_series().dt.days_in_month
        df['is_turn_of_month'] = ((df['day_of_month'] <= 3) | 
                                  (df['day_of_month'] >= days_in_month - 3)).astype(int)
        
        # 7. Holiday Proximity
        # Simplified: Days to year-end (holiday season)
        df['days_to_year_end'] = (pd.Timestamp(df.index.year[0], 12, 31) - df.index).days
        df['is_holiday_season'] = ((df['month'] == 12) & (df['day_of_month'] >= 15)).astype(int)
        
        # Store feature names
        self.feature_names = [
            'pres_cycle_year', 'pres_cycle_1', 'pres_cycle_2', 'pres_cycle_3', 'pres_cycle_4',
            'month', 'is_january', 'is_september', 'is_october', 'is_december', 'summer_months',
            'quarter', 'is_q1', 'is_q2', 'is_q3', 'is_q4',
            'days_to_quarter_end', 'is_earnings_season',
            'days_to_opex', 'is_opex_week', 'is_quarterly_opex_month',
            'is_monday', 'is_friday', 'is_turn_of_month',
            'days_to_year_end', 'is_holiday_season'
        ]
        
        return df


if __name__ == "__main__":
    # Test seasonality features
    import yfinance as yf
    
    print("Testing SeasonalityFeature...")
    spy = yf.download('SPY', start='2024-01-01', end='2024-12-31', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    feature = SeasonalityFeature()
    result = feature.calculate(spy)
    
    print(f"\n✅ Created {len(feature.feature_names)} seasonality features:")
    for feat in feature.feature_names:
        print(f"   - {feat}")
    
    print(f"\nSample data:")
    print(result[feature.feature_names].head(10))
