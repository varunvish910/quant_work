#!/usr/bin/env python3
"""
SPX Weekly Options Dealer Positioning Analysis - Phase 7: Market Structure Analysis
Analyzes positioning patterns, identifies key levels, and classifies market regimes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import optimize
from sklearn.cluster import KMeans


@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # pinning, breakout, directional, volatility
    confidence: float  # 0-1
    characteristics: List[str]
    risk_level: str  # low, medium, high


@dataclass
class KeyLevels:
    """Important strike levels identified in positioning"""
    gamma_centroid: float
    upside_pivot: Optional[float]
    downside_pivot: Optional[float]
    max_gamma_strike: float
    max_negative_gamma_strike: Optional[float]
    put_wall: Optional[float]
    call_wall: Optional[float]
    neutral_zone: Tuple[float, float]


class MarketStructureAnalyzer:
    """Analyzes dealer positioning to identify market structure and key levels"""
    
    def __init__(self, spot_price: float):
        self.spot_price = spot_price
    
    def analyze_full_structure(self, greeks_df: pd.DataFrame) -> Dict:
        """Perform comprehensive market structure analysis"""
        if len(greeks_df) == 0:
            return {"error": "No Greeks data provided"}
        
        analysis = {}
        
        # 1. Identify key levels
        analysis['key_levels'] = self.identify_key_levels(greeks_df)
        
        # 2. Analyze positioning patterns
        analysis['positioning_patterns'] = self.analyze_positioning_patterns(greeks_df)
        
        # 3. Calculate market metrics
        analysis['market_metrics'] = self.calculate_market_metrics(greeks_df)
        
        # 4. Classify market regime
        analysis['market_regime'] = self.classify_market_regime(greeks_df, analysis)
        
        # 5. Assess risks
        analysis['risk_assessment'] = self.assess_risks(greeks_df, analysis)
        
        # 6. Generate trading insights
        analysis['trading_insights'] = self.generate_trading_insights(analysis)
        
        return analysis
    
    def identify_key_levels(self, greeks_df: pd.DataFrame) -> KeyLevels:
        """Identify critical strike levels from positioning"""
        sorted_df = greeks_df.sort_values('strike')
        
        # Gamma centroid (volume-weighted center)
        gamma_weights = abs(sorted_df['dealer_gamma'])
        total_weight = gamma_weights.sum()
        
        if total_weight > 0:
            gamma_centroid = (sorted_df['strike'] * gamma_weights).sum() / total_weight
        else:
            gamma_centroid = self.spot_price
        
        # Find max gamma strike
        max_gamma_idx = sorted_df['dealer_gamma'].abs().idxmax()
        max_gamma_strike = sorted_df.loc[max_gamma_idx, 'strike']
        
        # Find max negative gamma strike
        negative_gamma = sorted_df[sorted_df['dealer_gamma'] < 0]
        if len(negative_gamma) > 0:
            max_neg_gamma_idx = negative_gamma['dealer_gamma'].idxmin()
            max_negative_gamma_strike = negative_gamma.loc[max_neg_gamma_idx, 'strike']
        else:
            max_negative_gamma_strike = None
        
        # Find pivot points
        upside_pivot, downside_pivot = self._find_gamma_pivots(sorted_df)
        
        # Find gamma walls (large concentrations)
        put_wall, call_wall = self._find_gamma_walls(sorted_df)
        
        # Define neutral zone
        neutral_zone = self._define_neutral_zone(sorted_df, gamma_centroid)
        
        return KeyLevels(
            gamma_centroid=gamma_centroid,
            upside_pivot=upside_pivot,
            downside_pivot=downside_pivot,
            max_gamma_strike=max_gamma_strike,
            max_negative_gamma_strike=max_negative_gamma_strike,
            put_wall=put_wall,
            call_wall=call_wall,
            neutral_zone=neutral_zone
        )
    
    def _find_gamma_pivots(self, sorted_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Find where gamma changes sign (pivot points)"""
        upside_pivot = None
        downside_pivot = None
        
        for i in range(len(sorted_df) - 1):
            current_gamma = sorted_df.iloc[i]['dealer_gamma']
            next_gamma = sorted_df.iloc[i + 1]['dealer_gamma']
            current_strike = sorted_df.iloc[i]['strike']
            
            # Upside pivot: positive to negative gamma (above spot)
            if (current_gamma > 0 and next_gamma < 0 and 
                current_strike > self.spot_price and upside_pivot is None):
                upside_pivot = current_strike
            
            # Downside pivot: negative to positive gamma (below spot)
            if (current_gamma < 0 and next_gamma > 0 and 
                current_strike < self.spot_price and downside_pivot is None):
                downside_pivot = current_strike
        
        return upside_pivot, downside_pivot
    
    def _find_gamma_walls(self, sorted_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Find large gamma concentrations (walls)"""
        gamma_threshold = sorted_df['dealer_gamma'].abs().quantile(0.9)
        
        # Put wall: large positive gamma below spot
        put_candidates = sorted_df[
            (sorted_df['strike'] < self.spot_price) & 
            (sorted_df['dealer_gamma'] > gamma_threshold)
        ]
        put_wall = put_candidates['strike'].max() if len(put_candidates) > 0 else None
        
        # Call wall: large negative gamma above spot
        call_candidates = sorted_df[
            (sorted_df['strike'] > self.spot_price) & 
            (sorted_df['dealer_gamma'] < -gamma_threshold)
        ]
        call_wall = call_candidates['strike'].min() if len(call_candidates) > 0 else None
        
        return put_wall, call_wall
    
    def _define_neutral_zone(self, sorted_df: pd.DataFrame, 
                           gamma_centroid: float) -> Tuple[float, float]:
        """Define the range where gamma effects are minimal"""
        # Find strikes where abs(gamma) is below median
        median_gamma = sorted_df['dealer_gamma'].abs().median()
        low_gamma_strikes = sorted_df[sorted_df['dealer_gamma'].abs() < median_gamma]['strike']
        
        if len(low_gamma_strikes) > 0:
            lower_bound = low_gamma_strikes.min()
            upper_bound = low_gamma_strikes.max()
        else:
            # Fallback: +/- 2% around centroid
            lower_bound = gamma_centroid * 0.98
            upper_bound = gamma_centroid * 1.02
        
        return (lower_bound, upper_bound)
    
    def analyze_positioning_patterns(self, greeks_df: pd.DataFrame) -> Dict:
        """Identify common positioning patterns"""
        patterns = {}
        
        # 1. Butterfly pattern detection
        patterns['butterfly'] = self.identify_butterfly_pattern(greeks_df)
        
        # 2. Calendar spread analysis
        patterns['calendar_spreads'] = self.analyze_calendar_spreads(greeks_df)
        
        # 3. Directional bias analysis
        patterns['directional_bias'] = self.analyze_directional_bias(greeks_df)
        
        # 4. Wing analysis (tail positioning)
        patterns['wing_analysis'] = self.analyze_tail_positioning(greeks_df)
        
        return patterns
    
    def identify_butterfly_pattern(self, greeks_df: pd.DataFrame) -> Dict:
        """Detect butterfly positioning patterns"""
        sorted_df = greeks_df.sort_values('strike')
        
        # Look for short/long fly patterns in gamma
        gamma_profile = sorted_df['dealer_gamma'].values
        strikes = sorted_df['strike'].values
        
        # Find local minima and maxima in gamma
        gamma_diffs = np.diff(gamma_profile)
        
        # Count sign changes (indication of butterfly patterns)
        sign_changes = np.sum(np.diff(np.sign(gamma_diffs)) != 0)
        
        # Determine pattern type
        if sign_changes >= 4:
            pattern_type = "complex_butterfly"
        elif sign_changes >= 2:
            if gamma_profile[len(gamma_profile)//2] < 0:
                pattern_type = "short_butterfly"  # Negative gamma at center
            else:
                pattern_type = "long_butterfly"   # Positive gamma at center
        else:
            pattern_type = "none"
        
        # Calculate pattern strength
        gamma_range = gamma_profile.max() - gamma_profile.min()
        pattern_strength = gamma_range / abs(gamma_profile).mean() if abs(gamma_profile).mean() > 0 else 0
        
        return {
            'pattern_type': pattern_type,
            'strength': pattern_strength,
            'sign_changes': sign_changes,
            'gamma_range': gamma_range
        }
    
    def analyze_calendar_spreads(self, greeks_df: pd.DataFrame) -> Dict:
        """Analyze time decay patterns indicating calendar spreads"""
        if 'dealer_charm' not in greeks_df.columns:
            return {'detected': False, 'reason': 'No charm data available'}
        
        # Look for significant charm concentrations
        charm_concentrations = greeks_df[greeks_df['dealer_charm'].abs() > 
                                       greeks_df['dealer_charm'].abs().quantile(0.8)]
        
        if len(charm_concentrations) > 0:
            charm_strikes = charm_concentrations['strike'].tolist()
            total_charm = charm_concentrations['dealer_charm'].sum()
            
            return {
                'detected': True,
                'charm_strikes': charm_strikes,
                'total_charm_exposure': total_charm,
                'charm_concentration_level': 'high' if len(charm_concentrations) > 5 else 'moderate'
            }
        else:
            return {'detected': False, 'reason': 'No significant charm concentrations'}
    
    def analyze_directional_bias(self, greeks_df: pd.DataFrame) -> Dict:
        """Analyze skew and directional positioning"""
        # Calculate net delta exposure
        net_delta = greeks_df['dealer_delta'].sum()
        
        # Calculate vanna skew
        if 'dealer_vanna' in greeks_df.columns:
            upside_vanna = greeks_df[greeks_df['strike'] > self.spot_price]['dealer_vanna'].sum()
            downside_vanna = greeks_df[greeks_df['strike'] < self.spot_price]['dealer_vanna'].sum()
            vanna_skew = upside_vanna - downside_vanna
        else:
            vanna_skew = 0
        
        # Determine bias
        if abs(net_delta) < 0.1 * greeks_df['dealer_gamma'].abs().sum():
            bias = "neutral"
        elif net_delta > 0:
            bias = "bullish"
        else:
            bias = "bearish"
        
        return {
            'net_delta': net_delta,
            'vanna_skew': vanna_skew,
            'directional_bias': bias,
            'bias_strength': abs(net_delta) / greeks_df['dealer_gamma'].abs().sum()
        }
    
    def analyze_tail_positioning(self, greeks_df: pd.DataFrame) -> Dict:
        """Analyze positioning in the wings (25-delta levels)"""
        # Approximate 25-delta levels (rough estimate)
        delta_25_call = self.spot_price * 1.1  # ~10% OTM call
        delta_25_put = self.spot_price * 0.9   # ~10% OTM put
        
        # Wing positioning
        upside_wing = greeks_df[greeks_df['strike'] >= delta_25_call]
        downside_wing = greeks_df[greeks_df['strike'] <= delta_25_put]
        
        upside_gamma = upside_wing['dealer_gamma'].sum() if len(upside_wing) > 0 else 0
        downside_gamma = downside_wing['dealer_gamma'].sum() if len(downside_wing) > 0 else 0
        
        return {
            'upside_wing_gamma': upside_gamma,
            'downside_wing_gamma': downside_gamma,
            'wing_gamma_ratio': upside_gamma / downside_gamma if downside_gamma != 0 else np.inf,
            'wing_positioning': 'long' if upside_gamma + downside_gamma > 0 else 'short'
        }
    
    def calculate_market_metrics(self, greeks_df: pd.DataFrame) -> Dict:
        """Calculate key market metrics"""
        metrics = {}
        
        # Speed convexity
        if 'dealer_speed' in greeks_df.columns:
            metrics['net_speed'] = greeks_df['dealer_speed'].sum()
            metrics['speed_convexity'] = abs(metrics['net_speed'])
        
        # Vomma exposure  
        if 'dealer_vomma' in greeks_df.columns:
            metrics['net_vomma'] = greeks_df['dealer_vomma'].sum()
            metrics['vol_convexity'] = abs(metrics['net_vomma'])
        
        # Gamma dispersion
        gamma_std = greeks_df['dealer_gamma'].std()
        gamma_mean = greeks_df['dealer_gamma'].abs().mean()
        metrics['gamma_dispersion'] = gamma_std / gamma_mean if gamma_mean > 0 else 0
        
        # Pin risk measure
        atm_strikes = greeks_df[
            (greeks_df['strike'] >= self.spot_price * 0.99) & 
            (greeks_df['strike'] <= self.spot_price * 1.01)
        ]
        metrics['pin_risk'] = atm_strikes['dealer_gamma'].abs().sum()
        
        return metrics
    
    def classify_market_regime(self, greeks_df: pd.DataFrame, analysis: Dict) -> MarketRegime:
        """Classify the current market regime"""
        key_levels = analysis['key_levels']
        patterns = analysis['positioning_patterns']
        metrics = analysis['market_metrics']
        
        characteristics = []
        confidence = 0.5
        
        # Check for pinning regime
        if (key_levels.upside_pivot and key_levels.downside_pivot and 
            abs(key_levels.gamma_centroid - self.spot_price) < self.spot_price * 0.02):
            regime_type = "pinning"
            characteristics.extend(["tight gamma boundaries", "low volatility expected"])
            confidence += 0.3
        
        # Check for breakout regime
        elif patterns['butterfly']['pattern_type'] == "long_butterfly":
            regime_type = "breakout"
            characteristics.extend(["long gamma at center", "volatility expansion potential"])
            confidence += 0.2
        
        # Check for directional regime
        elif patterns['directional_bias']['directional_bias'] != "neutral":
            regime_type = "directional"
            bias = patterns['directional_bias']['directional_bias']
            characteristics.extend([f"{bias} bias", "asymmetric positioning"])
            confidence += 0.2
        
        # Check for volatility regime
        elif metrics.get('vol_convexity', 0) > greeks_df['dealer_gamma'].abs().sum() * 0.1:
            regime_type = "volatility"
            characteristics.extend(["high vomma exposure", "vol-sensitive"])
            confidence += 0.2
        
        else:
            regime_type = "mixed"
            characteristics.append("no clear regime")
        
        # Determine risk level
        gamma_concentration = analysis['market_metrics'].get('gamma_dispersion', 1)
        if gamma_concentration > 2:
            risk_level = "high"
        elif gamma_concentration > 1:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return MarketRegime(
            regime_type=regime_type,
            confidence=min(confidence, 1.0),
            characteristics=characteristics,
            risk_level=risk_level
        )
    
    def assess_risks(self, greeks_df: pd.DataFrame, analysis: Dict) -> Dict:
        """Assess various risk factors"""
        risks = {}
        
        # Gamma risk
        total_gamma = greeks_df['dealer_gamma'].sum()
        risks['gamma_risk'] = {
            'level': 'high' if abs(total_gamma) > 1000 else 'medium' if abs(total_gamma) > 500 else 'low',
            'net_exposure': total_gamma,
            'concentration': analysis['market_metrics'].get('gamma_dispersion', 0)
        }
        
        # Vanna risk
        if 'dealer_vanna' in greeks_df.columns:
            total_vanna = greeks_df['dealer_vanna'].sum()
            risks['vanna_risk'] = {
                'level': 'high' if abs(total_vanna) > 500 else 'medium' if abs(total_vanna) > 250 else 'low',
                'net_exposure': total_vanna
            }
        
        # Pin risk
        pin_risk_value = analysis['market_metrics'].get('pin_risk', 0)
        risks['pin_risk'] = {
            'level': 'high' if pin_risk_value > 500 else 'medium' if pin_risk_value > 200 else 'low',
            'atm_gamma_concentration': pin_risk_value
        }
        
        return risks
    
    def generate_trading_insights(self, analysis: Dict) -> List[str]:
        """Generate actionable trading insights"""
        insights = []
        
        key_levels = analysis['key_levels']
        regime = analysis['market_regime']
        patterns = analysis['positioning_patterns']
        
        # Regime-based insights
        if regime.regime_type == "pinning":
            insights.append(f"Expect price consolidation between {key_levels.downside_pivot:.0f} and {key_levels.upside_pivot:.0f}")
            insights.append("Consider short vol strategies or range-bound plays")
        
        elif regime.regime_type == "breakout":
            insights.append("Long gamma positioning suggests potential for sharp moves")
            insights.append("Consider long vol strategies on directional breaks")
        
        elif regime.regime_type == "directional":
            bias = patterns['directional_bias']['directional_bias']
            insights.append(f"Positioning shows {bias} bias - consider directional strategies")
        
        # Level-based insights
        if key_levels.upside_pivot:
            insights.append(f"Key resistance at {key_levels.upside_pivot:.0f} from negative gamma")
        
        if key_levels.downside_pivot:
            insights.append(f"Key support at {key_levels.downside_pivot:.0f} from positive gamma")
        
        # Risk warnings
        risks = analysis['risk_assessment']
        if risks['gamma_risk']['level'] == 'high':
            insights.append("WARNING: High gamma exposure - expect accelerated moves")
        
        if 'vanna_risk' in risks and risks['vanna_risk']['level'] == 'high':
            insights.append("WARNING: High vanna exposure - vol/spot correlation risk")
        
        return insights


def main():
    """Example usage"""
    try:
        # Load aggregated Greeks
        greeks_df = pd.read_parquet("data/spx_options/greeks/aggregated_greeks.parquet")
        print(f"Loaded Greeks data for {len(greeks_df)} strikes")
        
        # Current SPX level
        current_spx = 5800
        
        # Initialize analyzer
        analyzer = MarketStructureAnalyzer(spot_price=current_spx)
        
        # Perform full analysis
        analysis = analyzer.analyze_full_structure(greeks_df)
        
        # Print results
        print("\n=== Market Structure Analysis ===")
        
        print("\nKey Levels:")
        key_levels = analysis['key_levels']
        print(f"  Gamma Centroid: {key_levels.gamma_centroid:.0f}")
        print(f"  Upside Pivot: {key_levels.upside_pivot}")
        print(f"  Downside Pivot: {key_levels.downside_pivot}")
        print(f"  Neutral Zone: {key_levels.neutral_zone[0]:.0f} - {key_levels.neutral_zone[1]:.0f}")
        
        print(f"\nMarket Regime: {analysis['market_regime'].regime_type}")
        print(f"  Confidence: {analysis['market_regime'].confidence:.2f}")
        print(f"  Risk Level: {analysis['market_regime'].risk_level}")
        print(f"  Characteristics: {', '.join(analysis['market_regime'].characteristics)}")
        
        print("\nTrading Insights:")
        for insight in analysis['trading_insights']:
            print(f"  • {insight}")
        
        # Save analysis
        import json
        with open("data/spx_options/market_structure_analysis.json", "w") as f:
            # Convert dataclass objects to dict for JSON serialization
            analysis_json = analysis.copy()
            analysis_json['key_levels'] = key_levels.__dict__
            analysis_json['market_regime'] = analysis['market_regime'].__dict__
            json.dump(analysis_json, f, indent=2, default=str)
        
        print(f"\nAnalysis saved to data/spx_options/market_structure_analysis.json")
        
    except FileNotFoundError:
        print("No Greeks data found. Run greeks_calculator.py first.")


if __name__ == "__main__":
    main()