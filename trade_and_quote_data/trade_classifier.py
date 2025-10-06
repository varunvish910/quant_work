#!/usr/bin/env python3
"""
SPX Weekly Options Dealer Positioning Analysis - Phase 2: Trade Classification
Classifies options trades as BTO/STO/BTC/STC and determines dealer positioning
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeClassification:
    """Result of trade classification"""
    customer_action: str  # BTO, STO, BTC, STC
    dealer_position: str  # long, short
    gamma_sign: int  # +1 or -1 from dealer perspective
    confidence: float  # 0-1 confidence in classification


class TradeClassifier:
    """Classifies options trades and determines dealer positioning"""
    
    def __init__(self, mid_market_threshold: float = 0.05):
        """
        Args:
            mid_market_threshold: Percentage of spread to consider mid-market
        """
        self.mid_market_threshold = mid_market_threshold
    
    def classify_trade(self, trade: Dict, quote: Dict, 
                      oi_change: Optional[float] = None) -> TradeClassification:
        """
        Classify a single trade as BTO/STO/BTC/STC
        
        Classification Logic:
        - If trade_price >= ask_price:
            - If OI increases: BTO (customer buys to open)
            - If OI decreases: BTC (customer buys to close short)
        - If trade_price <= bid_price:
            - If OI increases: STO (customer sells to open)  
            - If OI decreases: STC (customer sells to close long)
        - Mid-market trades use heuristics
        """
        trade_price = trade['price']
        bid = quote.get('bid', np.nan)
        ask = quote.get('ask', np.nan)
        option_type = trade['option_type']
        
        # Handle missing quote data
        if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
            return self._classify_without_quotes(trade, oi_change)
        
        spread = ask - bid
        mid_price = (bid + ask) / 2
        spread_pct = spread / mid_price if mid_price > 0 else 1.0
        
        # Classify based on trade price relative to bid/ask
        if trade_price >= ask * (1 - 0.01):  # At or above ask (small tolerance)
            if oi_change is None or oi_change >= 0:
                customer_action = "BTO"  # Customer buys to open
            else:
                customer_action = "BTC"  # Customer buys to close short
            confidence = 0.9
            
        elif trade_price <= bid * (1 + 0.01):  # At or below bid (small tolerance)
            if oi_change is None or oi_change >= 0:
                customer_action = "STO"  # Customer sells to open
            else:
                customer_action = "STC"  # Customer sells to close long
            confidence = 0.9
            
        else:  # Mid-market trade
            customer_action, confidence = self._classify_mid_market(
                trade, quote, oi_change, spread_pct
            )
        
        # Convert to dealer position
        dealer_position, gamma_sign = self._get_dealer_position(
            customer_action, option_type
        )
        
        return TradeClassification(
            customer_action=customer_action,
            dealer_position=dealer_position,
            gamma_sign=gamma_sign,
            confidence=confidence
        )
    
    def _classify_mid_market(self, trade: Dict, quote: Dict, 
                           oi_change: Optional[float], 
                           spread_pct: float) -> Tuple[str, float]:
        """Classify mid-market trades using heuristics"""
        trade_price = trade['price']
        bid = quote['bid']
        ask = quote['ask']
        mid_price = (bid + ask) / 2
        
        # Use OI change if available
        if oi_change is not None:
            if oi_change > 0:
                # OI increased - new position opened
                if trade_price > mid_price:
                    return "BTO", 0.7  # Above mid, likely customer buy
                else:
                    return "STO", 0.7  # Below mid, likely customer sell
            else:
                # OI decreased - position closed
                if trade_price > mid_price:
                    return "BTC", 0.6  # Above mid, closing short
                else:
                    return "STC", 0.6  # Below mid, closing long
        
        # Fallback heuristics without OI data
        price_position = (trade_price - bid) / (ask - bid)
        
        # Additional heuristics based on trade characteristics
        trade_size = trade.get('size', 1)
        time_factor = self._get_time_factor(trade)
        
        # Large trades more likely to be institutional (opening)
        size_factor = min(trade_size / 100, 2.0)  # Normalize trade size
        
        # Combine factors for classification
        open_probability = (
            price_position * 0.4 +  # Price relative to spread
            time_factor * 0.3 +     # Time of day effect
            size_factor * 0.3       # Size effect
        )
        
        if open_probability > 0.6:
            if price_position > 0.5:
                return "BTO", 0.5
            else:
                return "STO", 0.5
        else:
            if price_position > 0.5:
                return "BTC", 0.4
            else:
                return "STC", 0.4
    
    def _get_time_factor(self, trade: Dict) -> float:
        """Get time-of-day factor for classification"""
        try:
            timestamp = pd.to_datetime(trade['timestamp'], unit='ns')
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Market open (9:30-10:30) - more opening activity
            if 9 <= hour < 11:
                return 0.8
            # Mid-day (10:30-15:00) - mixed
            elif 11 <= hour < 15:
                return 0.5
            # Market close (15:00-16:00) - more closing activity
            else:
                return 0.2
                
        except Exception:
            return 0.5  # Default neutral
    
    def _classify_without_quotes(self, trade: Dict, 
                               oi_change: Optional[float]) -> TradeClassification:
        """Fallback classification when quotes unavailable"""
        option_type = trade['option_type']
        
        # Use OI change if available
        if oi_change is not None:
            if oi_change > 0:
                customer_action = "BTO"  # Assume opening when OI increases
            else:
                customer_action = "STC"  # Assume closing when OI decreases
            confidence = 0.3
        else:
            # Very low confidence without quote or OI data
            customer_action = "BTO"  # Default assumption
            confidence = 0.1
        
        dealer_position, gamma_sign = self._get_dealer_position(
            customer_action, option_type
        )
        
        return TradeClassification(
            customer_action=customer_action,
            dealer_position=dealer_position,
            gamma_sign=gamma_sign,
            confidence=confidence
        )
    
    def _get_dealer_position(self, customer_action: str, 
                           option_type: str) -> Tuple[str, int]:
        """
        Convert customer action to dealer position and gamma sign
        
        Dealer is the counterparty to customer:
        - Customer BTO Call → Dealer Short Call → Negative Gamma
        - Customer STO Call → Dealer Long Call → Positive Gamma
        - Customer BTO Put → Dealer Short Put → Positive Gamma (puts have negative gamma)
        - Customer STO Put → Dealer Long Put → Negative Gamma
        """
        if customer_action in ["BTO", "BTC"]:
            # Customer buying → Dealer selling/short
            dealer_position = "short"
            if option_type == "c":  # call
                gamma_sign = -1  # Short calls = negative gamma
            else:  # put
                gamma_sign = 1   # Short puts = positive gamma (from dealer perspective)
        else:  # STO, STC
            # Customer selling → Dealer buying/long
            dealer_position = "long"
            if option_type == "c":  # call
                gamma_sign = 1   # Long calls = positive gamma
            else:  # put
                gamma_sign = -1  # Long puts = negative gamma (from dealer perspective)
        
        return dealer_position, gamma_sign
    
    def classify_all_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Classify all trades in a DataFrame"""
        classified_trades = []
        
        print(f"Classifying {len(trades_df)} trades...")
        
        for idx, trade in trades_df.iterrows():
            # Prepare quote data
            quote = {
                'bid': trade.get('bid', np.nan),
                'ask': trade.get('ask', np.nan),
                'bid_size': trade.get('bid_size', np.nan),
                'ask_size': trade.get('ask_size', np.nan)
            }
            
            # Classify the trade
            classification = self.classify_trade(
                trade.to_dict(), 
                quote, 
                oi_change=None  # OI data not available from Polygon trades
            )
            
            # Add classification results to trade data
            trade_dict = trade.to_dict()
            trade_dict.update({
                'customer_action': classification.customer_action,
                'dealer_position': classification.dealer_position,
                'dealer_gamma_sign': classification.gamma_sign,
                'classification_confidence': classification.confidence
            })
            
            classified_trades.append(trade_dict)
            
            if idx % 1000 == 0:
                print(f"Classified {idx}/{len(trades_df)} trades")
        
        classified_df = pd.DataFrame(classified_trades)
        
        # Add summary statistics
        self._print_classification_summary(classified_df)
        
        return classified_df
    
    def _print_classification_summary(self, df: pd.DataFrame):
        """Print summary of trade classifications"""
        print("\n=== Trade Classification Summary ===")
        print(f"Total trades classified: {len(df):,}")
        print(f"Average confidence: {df['classification_confidence'].mean():.3f}")
        
        print("\nCustomer Action Distribution:")
        action_counts = df['customer_action'].value_counts()
        for action, count in action_counts.items():
            pct = count / len(df) * 100
            print(f"  {action}: {count:,} ({pct:.1f}%)")
        
        print("\nDealer Position Distribution:")
        position_counts = df['dealer_position'].value_counts()
        for position, count in position_counts.items():
            pct = count / len(df) * 100
            print(f"  {position}: {count:,} ({pct:.1f}%)")
        
        print("\nDealer Gamma Exposure:")
        gamma_long = (df['dealer_gamma_sign'] == 1).sum()
        gamma_short = (df['dealer_gamma_sign'] == -1).sum()
        print(f"  Positive gamma trades: {gamma_long:,} ({gamma_long/len(df)*100:.1f}%)")
        print(f"  Negative gamma trades: {gamma_short:,} ({gamma_short/len(df)*100:.1f}%)")
        
        print(f"\nConfidence Distribution:")
        conf_bins = pd.cut(df['classification_confidence'], 
                          bins=[0, 0.3, 0.6, 0.9, 1.0], 
                          labels=['Low', 'Medium', 'High', 'Very High'])
        conf_counts = conf_bins.value_counts()
        for conf, count in conf_counts.items():
            pct = count / len(df) * 100
            print(f"  {conf}: {count:,} ({pct:.1f}%)")


def main():
    """Example usage"""
    # Load sample trades data
    try:
        trades_df = pd.read_parquet("data/spx_options/trades/2025-10-05_enriched_trades.parquet")
        print(f"Loaded {len(trades_df)} trades for classification")
        
        # Initialize classifier
        classifier = TradeClassifier()
        
        # Classify all trades
        classified_df = classifier.classify_all_trades(trades_df)
        
        # Save classified trades
        output_file = "data/spx_options/classified/2025-10-05_classified_trades.parquet"
        classified_df.to_parquet(output_file, index=False)
        print(f"\nSaved classified trades to {output_file}")
        
        # Show sample results
        print("\nSample classified trades:")
        sample_cols = ['ticker', 'price', 'size', 'bid', 'ask', 
                      'customer_action', 'dealer_position', 'classification_confidence']
        print(classified_df[sample_cols].head(10))
        
    except FileNotFoundError:
        print("No trades data found. Run spx_trades_downloader.py first.")


if __name__ == "__main__":
    main()