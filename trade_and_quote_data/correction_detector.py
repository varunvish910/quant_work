#!/usr/bin/env python3
"""
Market Correction Detection - Identifies correction events in price data
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

class CorrectionDetector:
    """Detects market correction events in price data"""
    
    def __init__(self, correction_threshold: float = 0.04, lookback_days: int = 20):
        self.correction_threshold = correction_threshold
        self.lookback_days = lookback_days
    
    def identify_corrections(self, price_data: pd.DataFrame) -> List[Dict]:
        """Identify all correction events in the price data"""
        if price_data is None or len(price_data) < self.lookback_days:
            return []
        
        corrections = []
        prices = price_data['underlying_price'].values
        dates = price_data['date'].values
        
        # Find local peaks using rolling maximum
        rolling_max = pd.Series(prices).rolling(window=self.lookback_days, min_periods=1).max()
        peak_mask = prices == rolling_max.values
        
        # Find corrections by looking for significant drawdowns from peaks
        for i in range(len(prices)):
            if not peak_mask[i]:
                continue
                
            peak_price = prices[i]
            peak_date = dates[i]
            
            # Look forward for the next significant trough
            for j in range(i + 1, len(prices)):
                current_price = prices[j]
                drawdown = (peak_price - current_price) / peak_price
                
                # Check if this is a significant correction
                if drawdown >= self.correction_threshold:
                    # Find the actual trough (lowest point in this correction)
                    trough_idx = j
                    trough_price = current_price
                    
                    # Continue looking for the actual bottom
                    for k in range(j + 1, len(prices)):
                        if prices[k] < trough_price:
                            trough_price = prices[k]
                            trough_idx = k
                        # Stop if we hit a new peak (recovery)
                        elif prices[k] >= peak_price * 0.98:  # 98% recovery threshold
                            break
                    
                    # Calculate final drawdown
                    final_drawdown = (peak_price - trough_price) / peak_price
                    
                    if final_drawdown >= self.correction_threshold:
                        correction = {
                            'peak_date': peak_date,
                            'trough_date': dates[trough_idx],
                            'peak_price': peak_price,
                            'trough_price': trough_price,
                            'magnitude': final_drawdown,
                            'duration_days': (pd.to_datetime(dates[trough_idx]) - pd.to_datetime(peak_date)).days,
                            'peak_idx': i,
                            'trough_idx': trough_idx
                        }
                        corrections.append(correction)
                        
                        # Skip ahead to avoid overlapping corrections
                        i = trough_idx
                    break
        
        # Remove overlapping corrections (keep larger ones)
        filtered_corrections = self._filter_overlapping_corrections(corrections)
        
        print(f"🔍 Found {len(filtered_corrections)} correction events (≥{self.correction_threshold*100:.1f}%)")
        return filtered_corrections
    
    def _filter_overlapping_corrections(self, corrections: List[Dict]) -> List[Dict]:
        """Remove overlapping corrections, keeping larger ones"""
        filtered_corrections = []
        for correction in corrections:
            is_overlapping = False
            for existing in filtered_corrections:
                # Check if this correction overlaps with existing ones
                if (correction['peak_idx'] <= existing['trough_idx'] and 
                    correction['trough_idx'] >= existing['peak_idx']):
                    # Keep the larger correction
                    if correction['magnitude'] > existing['magnitude']:
                        filtered_corrections.remove(existing)
                        filtered_corrections.append(correction)
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_corrections.append(correction)
        
        return filtered_corrections