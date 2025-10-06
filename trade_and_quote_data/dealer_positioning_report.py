#!/usr/bin/env python3
"""
SPX Weekly Options Dealer Positioning Analysis - Phase 9: Report Generation
Generates comprehensive written analysis and trading recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import json
from pathlib import Path


class DealerPositioningReport:
    """Generates comprehensive dealer positioning analysis reports"""
    
    def __init__(self, spot_price: float):
        self.spot_price = spot_price
        self.analysis_date = datetime.now().strftime("%Y-%m-%d")
    
    def generate_full_report(self, greeks_df: pd.DataFrame, 
                           trade_greeks_df: pd.DataFrame,
                           analysis: Dict, 
                           output_file: str = "dealer_positioning_report.md") -> str:
        """Generate complete analysis report"""
        
        report_sections = []
        
        # Header
        report_sections.append(self._generate_header())
        
        # Executive Summary
        report_sections.append(self._generate_executive_summary(analysis))
        
        # Market Structure Analysis
        report_sections.append(self._generate_market_structure_section(analysis))
        
        # Greeks Analysis
        report_sections.append(self._generate_greeks_analysis(greeks_df, analysis))
        
        # Positioning Patterns
        report_sections.append(self._generate_positioning_patterns(analysis))
        
        # Risk Assessment
        report_sections.append(self._generate_risk_assessment(analysis))
        
        # Trading Implications
        report_sections.append(self._generate_trading_implications(analysis))
        
        # Quantitative Summary
        report_sections.append(self._generate_quantitative_summary(greeks_df, analysis))
        
        # Scenarios Analysis
        report_sections.append(self._generate_scenarios_analysis(analysis))
        
        # Appendix
        report_sections.append(self._generate_appendix(greeks_df, trade_greeks_df))
        
        # Combine all sections
        full_report = "\n\n".join(report_sections)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(full_report)
        
        print(f"Full report saved to {output_file}")
        return full_report
    
    def _generate_header(self) -> str:
        """Generate report header"""
        return f"""# SPX Weekly Options Dealer Positioning Analysis
## Analysis Date: {self.analysis_date}
## Spot Price: {self.spot_price:.2f}

---

*This report provides a comprehensive analysis of dealer positioning in SPX weekly options, including gamma exposure, higher-order Greeks, and market structure insights for informed trading decisions.*"""
    
    def _generate_executive_summary(self, analysis: Dict) -> str:
        """Generate executive summary section"""
        
        key_levels = analysis.get('key_levels', {})
        regime = analysis.get('market_regime', {})
        insights = analysis.get('trading_insights', [])
        
        # Extract key metrics
        gamma_centroid = getattr(key_levels, 'gamma_centroid', self.spot_price)
        upside_pivot = getattr(key_levels, 'upside_pivot', None)
        downside_pivot = getattr(key_levels, 'downside_pivot', None)
        regime_type = getattr(regime, 'regime_type', 'unknown')
        confidence = getattr(regime, 'confidence', 0.5)
        
        summary = f"""## Executive Summary

**Market Regime:** {regime_type.title()} (Confidence: {confidence:.0%})

**Key Positioning Insights:**
- Gamma centroid located at **{gamma_centroid:.0f}** ({self._distance_from_spot(gamma_centroid)})
- Current spot price: **{self.spot_price:.0f}**"""
        
        if upside_pivot:
            summary += f"\n- Upside resistance at **{upside_pivot:.0f}** ({self._distance_from_spot(upside_pivot)})"
        
        if downside_pivot:
            summary += f"\n- Downside support at **{downside_pivot:.0f}** ({self._distance_from_spot(downside_pivot)})"
        
        summary += f"""

**Primary Trading Theme:** {insights[0] if insights else 'Mixed positioning signals'}

**Risk Level:** {getattr(regime, 'risk_level', 'Medium').title()}"""
        
        return summary
    
    def _generate_market_structure_section(self, analysis: Dict) -> str:
        """Generate market structure analysis section"""
        
        patterns = analysis.get('positioning_patterns', {})
        key_levels = analysis.get('key_levels', {})
        
        section = """## Market Structure Analysis

### Dealer Positioning Overview"""
        
        # Butterfly analysis
        butterfly = patterns.get('butterfly', {})
        if butterfly.get('pattern_type') != 'none':
            section += f"""

**Butterfly Pattern Detected:** {butterfly.get('pattern_type', 'Unknown').replace('_', ' ').title()}
- Pattern strength: {butterfly.get('strength', 0):.2f}
- Sign changes in gamma profile: {butterfly.get('sign_changes', 0)}
- This suggests {"pinning behavior" if "short" in butterfly.get('pattern_type', '') else "volatility expansion potential"}"""
        
        # Directional bias
        directional = patterns.get('directional_bias', {})
        bias = directional.get('directional_bias', 'neutral')
        if bias != 'neutral':
            section += f"""

**Directional Bias:** {bias.title()}
- Net delta exposure: {directional.get('net_delta', 0):.0f}
- Vanna skew: {directional.get('vanna_skew', 0):.0f}
- Bias strength: {directional.get('bias_strength', 0):.2f}"""
        
        # Wing analysis
        wings = patterns.get('wing_analysis', {})
        section += f"""

**Wing Positioning (Tail Risk):**
- Upside wing gamma: {wings.get('upside_wing_gamma', 0):.0f}
- Downside wing gamma: {wings.get('downside_wing_gamma', 0):.0f}
- Overall wing positioning: {wings.get('wing_positioning', 'unknown').title()}"""
        
        return section
    
    def _generate_greeks_analysis(self, greeks_df: pd.DataFrame, analysis: Dict) -> str:
        """Generate Greeks analysis section"""
        
        # Calculate totals
        total_gamma = greeks_df['dealer_gamma'].sum()
        total_delta = greeks_df['dealer_delta'].sum()
        total_vega = greeks_df.get('dealer_vega', pd.Series([0])).sum()
        total_theta = greeks_df.get('dealer_theta', pd.Series([0])).sum()
        
        section = f"""## Greeks Exposure Analysis

### First-Order Greeks
- **Total Dealer Gamma:** {total_gamma:,.0f}
  - {"Dealers are net long gamma (supportive of mean reversion)" if total_gamma > 0 else "Dealers are net short gamma (accelerative of moves)"}
- **Total Dealer Delta:** {total_delta:,.0f}
  - {"Bullish delta bias" if total_delta > 0 else "Bearish delta bias" if total_delta < 0 else "Delta neutral"}
- **Total Dealer Vega:** {total_vega:,.0f}
  - {"Long volatility exposure" if total_vega > 0 else "Short volatility exposure" if total_vega < 0 else "Volatility neutral"}
- **Total Dealer Theta:** {total_theta:,.0f}
  - {"Positive time decay (benefits from time passage)" if total_theta > 0 else "Negative time decay (hurt by time passage)"}"""
        
        # Higher-order Greeks if available
        if 'dealer_vanna' in greeks_df.columns:
            total_vanna = greeks_df['dealer_vanna'].sum()
            section += f"""

### Higher-Order Greeks
- **Total Dealer Vanna:** {total_vanna:,.0f}
  - Cross-sensitivity between volatility and spot movements"""
        
        if 'dealer_charm' in greeks_df.columns:
            total_charm = greeks_df['dealer_charm'].sum()
            section += f"""
- **Total Dealer Charm:** {total_charm:,.0f}
  - Delta decay over time"""
        
        if 'dealer_vomma' in greeks_df.columns:
            total_vomma = greeks_df['dealer_vomma'].sum()
            section += f"""
- **Total Dealer Vomma:** {total_vomma:,.0f}
  - Volatility convexity exposure"""
        
        # Gamma distribution analysis
        positive_gamma_strikes = greeks_df[greeks_df['dealer_gamma'] > 0]
        negative_gamma_strikes = greeks_df[greeks_df['dealer_gamma'] < 0]
        
        section += f"""

### Gamma Distribution
- **Positive gamma strikes:** {len(positive_gamma_strikes)} ({len(positive_gamma_strikes)/len(greeks_df)*100:.0f}% of strikes)
- **Negative gamma strikes:** {len(negative_gamma_strikes)} ({len(negative_gamma_strikes)/len(greeks_df)*100:.0f}% of strikes)
- **Largest positive gamma:** {greeks_df['dealer_gamma'].max():.0f} at strike {greeks_df.loc[greeks_df['dealer_gamma'].idxmax(), 'strike']:.0f}
- **Largest negative gamma:** {greeks_df['dealer_gamma'].min():.0f} at strike {greeks_df.loc[greeks_df['dealer_gamma'].idxmin(), 'strike']:.0f}"""
        
        return section
    
    def _generate_positioning_patterns(self, analysis: Dict) -> str:
        """Generate positioning patterns section"""
        
        patterns = analysis.get('positioning_patterns', {})
        
        section = """## Positioning Patterns Analysis"""
        
        # Calendar spreads
        calendar = patterns.get('calendar_spreads', {})
        if calendar.get('detected', False):
            section += f"""

### Calendar Spread Activity
- **Detected:** Yes
- **Charm concentration level:** {calendar.get('charm_concentration_level', 'unknown').title()}
- **Total charm exposure:** {calendar.get('total_charm_exposure', 0):.0f}
- **Key strikes:** {', '.join(map(str, calendar.get('charm_strikes', [])[:5]))}

*This suggests active time decay strategies and potential range-bound expectations.*"""
        else:
            section += f"""

### Calendar Spread Activity
- **Detected:** No
- **Reason:** {calendar.get('reason', 'Unknown')}"""
        
        # Market making vs directional flow
        directional = patterns.get('directional_bias', {})
        if directional.get('directional_bias') != 'neutral':
            section += f"""

### Flow Analysis
- **Predominant flow:** {directional.get('directional_bias', 'unknown').title()}
- **Flow strength:** {directional.get('bias_strength', 0)*100:.1f}% of gamma exposure
- **Interpretation:** {"Strong directional conviction" if directional.get('bias_strength', 0) > 0.3 else "Moderate directional lean"}"""
        
        return section
    
    def _generate_risk_assessment(self, analysis: Dict) -> str:
        """Generate risk assessment section"""
        
        risks = analysis.get('risk_assessment', {})
        
        section = """## Risk Assessment

### Primary Risk Factors"""
        
        # Gamma risk
        gamma_risk = risks.get('gamma_risk', {})
        section += f"""

**Gamma Risk:** {gamma_risk.get('level', 'unknown').title()}
- Net exposure: {gamma_risk.get('net_exposure', 0):,.0f}
- Concentration: {gamma_risk.get('concentration', 0):.2f}
- **Implication:** {"High acceleration risk on directional moves" if gamma_risk.get('level') == 'high' else "Moderate gamma effects expected"}"""
        
        # Vanna risk
        vanna_risk = risks.get('vanna_risk', {})
        if vanna_risk:
            section += f"""

**Vanna Risk:** {vanna_risk.get('level', 'unknown').title()}
- Net exposure: {vanna_risk.get('net_exposure', 0):,.0f}
- **Implication:** {"Significant vol-spot correlation effects" if vanna_risk.get('level') == 'high' else "Moderate cross-effects"}"""
        
        # Pin risk
        pin_risk = risks.get('pin_risk', {})
        section += f"""

**Pin Risk:** {pin_risk.get('level', 'unknown').title()}
- ATM gamma concentration: {pin_risk.get('atm_gamma_concentration', 0):,.0f}
- **Implication:** {"Strong expiration effects expected" if pin_risk.get('level') == 'high' else "Limited pinning pressure"}"""
        
        # Market stress scenarios
        section += """

### Stress Scenarios
1. **Upside Break:** High gamma zones may provide resistance, but negative gamma above could accelerate moves
2. **Downside Break:** Positive gamma support levels could provide bounce opportunities
3. **Volatility Spike:** Vanna effects could amplify or dampen spot moves depending on positioning
4. **Time Decay:** Charm exposure suggests [acceleration/deceleration] of delta changes approaching expiration"""
        
        return section
    
    def _generate_trading_implications(self, analysis: Dict) -> str:
        """Generate trading implications section"""
        
        insights = analysis.get('trading_insights', [])
        regime = analysis.get('market_regime', {})
        key_levels = analysis.get('key_levels', {})
        
        section = """## Trading Implications

### Strategic Recommendations"""
        
        for i, insight in enumerate(insights, 1):
            section += f"\n{i}. {insight}"
        
        # Regime-specific strategies
        regime_type = getattr(regime, 'regime_type', 'unknown')
        
        section += f"""

### Regime-Specific Strategies ({regime_type.title()})"""
        
        if regime_type == "pinning":
            section += """
- **Iron Condor/Butterfly:** Sell volatility in expected range
- **Time Decay Plays:** Theta strategies likely profitable
- **Avoid:** Long vol/gamma strategies
- **Risk:** Sudden regime change leading to breakout"""
        
        elif regime_type == "breakout":
            section += """
- **Long Straddles/Strangles:** Benefit from volatility expansion
- **Gamma Scalping:** Long gamma positions for directional moves
- **Avoid:** Short vol strategies
- **Risk:** False breakouts in range-bound market"""
        
        elif regime_type == "directional":
            bias = analysis.get('positioning_patterns', {}).get('directional_bias', {}).get('directional_bias', 'unknown')
            section += f"""
- **Directional Plays:** Consider {bias} strategies
- **Risk Reversals:** Trade the skew
- **Avoid:** Range-bound strategies
- **Risk:** Reversal of directional bias"""
        
        # Key levels for trading
        gamma_centroid = getattr(key_levels, 'gamma_centroid', self.spot_price)
        upside_pivot = getattr(key_levels, 'upside_pivot', None)
        downside_pivot = getattr(key_levels, 'downside_pivot', None)
        
        section += f"""

### Key Trading Levels
- **Gamma Centroid ({gamma_centroid:.0f}):** Center of positioning, expect mean reversion
- **Current Spot ({self.spot_price:.0f}):** {self._relative_position_analysis(gamma_centroid)}"""
        
        if upside_pivot:
            section += f"\n- **Upside Pivot ({upside_pivot:.0f}):** Key resistance, negative gamma above"
        
        if downside_pivot:
            section += f"\n- **Downside Pivot ({downside_pivot:.0f}):** Key support, positive gamma below"
        
        return section
    
    def _generate_quantitative_summary(self, greeks_df: pd.DataFrame, analysis: Dict) -> str:
        """Generate quantitative summary section"""
        
        key_levels = analysis.get('key_levels', {})
        market_metrics = analysis.get('market_metrics', {})
        
        # Calculate additional metrics
        implied_move = self._calculate_implied_move(greeks_df)
        expected_range = self._calculate_expected_range(analysis)
        
        section = f"""## Quantitative Summary

### Key Metrics
| Metric | Value | Interpretation |
|--------|--------|----------------|
| **Spot Price** | {self.spot_price:.0f} | Current market level |
| **Gamma Centroid** | {getattr(key_levels, 'gamma_centroid', 0):.0f} | Center of dealer positioning |
| **Implied Weekly Move** | {implied_move:.1f}% | Expected price movement |
| **Expected Range** | {expected_range[0]:.0f} - {expected_range[1]:.0f} | 1-sigma range estimate |
| **Net Dealer Gamma** | {greeks_df['dealer_gamma'].sum():,.0f} | Overall gamma exposure |
| **Gamma Dispersion** | {market_metrics.get('gamma_dispersion', 0):.2f} | Concentration measure |
| **Pin Risk** | {market_metrics.get('pin_risk', 0):,.0f} | ATM gamma concentration |"""
        
        if 'net_speed' in market_metrics:
            section += f"""
| **Speed Convexity** | {market_metrics.get('speed_convexity', 0):,.0f} | Third-order effects |"""
        
        if 'vol_convexity' in market_metrics:
            section += f"""
| **Vol Convexity** | {market_metrics.get('vol_convexity', 0):,.0f} | Volatility sensitivity |"""
        
        # Probability assessments
        section += f"""

### Probability Assessments
- **Range Bound Trading:** {self._calculate_range_probability(analysis):.0%}
- **Upside Breakout:** {self._calculate_breakout_probability(analysis, 'upside'):.0%}
- **Downside Breakout:** {self._calculate_breakout_probability(analysis, 'downside'):.0%}
- **Volatility Expansion:** {self._calculate_vol_expansion_probability(analysis):.0%}"""
        
        return section
    
    def _generate_scenarios_analysis(self, analysis: Dict) -> str:
        """Generate scenarios analysis section"""
        
        key_levels = analysis.get('key_levels', {})
        
        section = """## Scenario Analysis

### Upside Scenario (+2% move)"""
        
        upside_target = self.spot_price * 1.02
        section += f"""
- **Target:** {upside_target:.0f}
- **Gamma Environment:** {self._assess_gamma_environment(upside_target, analysis)}
- **Expected Behavior:** {self._describe_expected_behavior(upside_target, analysis)}
- **Trading Strategy:** {self._suggest_strategy('upside', analysis)}"""
        
        section += """

### Downside Scenario (-2% move)"""
        
        downside_target = self.spot_price * 0.98
        section += f"""
- **Target:** {downside_target:.0f}
- **Gamma Environment:** {self._assess_gamma_environment(downside_target, analysis)}
- **Expected Behavior:** {self._describe_expected_behavior(downside_target, analysis)}
- **Trading Strategy:** {self._suggest_strategy('downside', analysis)}"""
        
        section += """

### Volatility Spike Scenario (+50% IV)"""
        section += f"""
- **Vanna Effects:** {self._assess_vanna_effects(analysis)}
- **Positioning Impact:** Dealer hedging could amplify or dampen moves
- **Strategy:** Focus on cross-effects and correlation trades"""
        
        return section
    
    def _generate_appendix(self, greeks_df: pd.DataFrame, trade_greeks_df: pd.DataFrame) -> str:
        """Generate appendix with detailed data"""
        
        section = f"""## Appendix

### Data Summary
- **Total Strikes Analyzed:** {len(greeks_df)}
- **Total Trades Processed:** {len(trade_greeks_df) if len(trade_greeks_df) > 0 else 'N/A'}
- **Strike Range:** {greeks_df['strike'].min():.0f} - {greeks_df['strike'].max():.0f}
- **Analysis Date:** {self.analysis_date}
- **Spot at Analysis:** {self.spot_price:.2f}

### Top 10 Gamma Exposures by Strike"""
        
        # Top gamma exposures
        top_gamma = greeks_df.nlargest(10, 'dealer_gamma')[['strike', 'dealer_gamma', 'dealer_delta']]
        section += f"""
```
{top_gamma.to_string(index=False, formatters={'dealer_gamma': '{:,.0f}'.format, 'dealer_delta': '{:,.0f}'.format})}
```"""
        
        section += """

### Methodology Notes
- **Trade Classification:** Algorithmic classification based on bid/ask spread analysis
- **Greeks Calculation:** Black-Scholes with estimated implied volatilities
- **Dealer Perspective:** All Greeks shown from dealer counterparty viewpoint
- **Aggregation:** Volume-weighted aggregation by strike and expiry
- **Risk Assessment:** Based on historical volatility and gamma concentration patterns

### Disclaimers
- This analysis is for educational and informational purposes only
- Past performance does not guarantee future results
- Options trading involves significant risk of loss
- Consult with qualified professionals before making trading decisions"""
        
        return section
    
    # Helper methods
    def _distance_from_spot(self, level: float) -> str:
        """Calculate distance from spot as percentage"""
        pct_diff = (level - self.spot_price) / self.spot_price * 100
        direction = "above" if pct_diff > 0 else "below"
        return f"{abs(pct_diff):.1f}% {direction} spot"
    
    def _relative_position_analysis(self, gamma_centroid: float) -> str:
        """Analyze relative position to gamma centroid"""
        if abs(self.spot_price - gamma_centroid) < gamma_centroid * 0.01:
            return "Near gamma centroid, balanced positioning"
        elif self.spot_price > gamma_centroid:
            return "Above gamma centroid, upside resistance likely"
        else:
            return "Below gamma centroid, downside support likely"
    
    def _calculate_implied_move(self, greeks_df: pd.DataFrame) -> float:
        """Calculate implied weekly move"""
        # Simplified calculation based on vega exposure
        total_vega = greeks_df.get('dealer_vega', pd.Series([0])).abs().sum()
        if total_vega > 0:
            return min(total_vega / self.spot_price * 100 * 0.1, 10.0)  # Cap at 10%
        return 2.0  # Default 2% weekly move
    
    def _calculate_expected_range(self, analysis: Dict) -> Tuple[float, float]:
        """Calculate expected trading range"""
        key_levels = analysis.get('key_levels', {})
        
        # Use pivots if available, otherwise use +/- implied move
        upside_pivot = getattr(key_levels, 'upside_pivot', None)
        downside_pivot = getattr(key_levels, 'downside_pivot', None)
        
        if upside_pivot and downside_pivot:
            return (downside_pivot, upside_pivot)
        else:
            implied_move = 0.02  # 2% default
            return (self.spot_price * (1 - implied_move), self.spot_price * (1 + implied_move))
    
    def _calculate_range_probability(self, analysis: Dict) -> float:
        """Calculate probability of range-bound trading"""
        regime = analysis.get('market_regime', {})
        if getattr(regime, 'regime_type', '') == 'pinning':
            return 0.7
        elif getattr(regime, 'regime_type', '') == 'breakout':
            return 0.3
        else:
            return 0.5
    
    def _calculate_breakout_probability(self, analysis: Dict, direction: str) -> float:
        """Calculate probability of breakout in given direction"""
        patterns = analysis.get('positioning_patterns', {})
        bias = patterns.get('directional_bias', {}).get('directional_bias', 'neutral')
        
        if direction == 'upside' and bias == 'bullish':
            return 0.4
        elif direction == 'downside' and bias == 'bearish':
            return 0.4
        else:
            return 0.2
    
    def _calculate_vol_expansion_probability(self, analysis: Dict) -> float:
        """Calculate probability of volatility expansion"""
        regime = analysis.get('market_regime', {})
        if getattr(regime, 'regime_type', '') == 'breakout':
            return 0.6
        else:
            return 0.3
    
    def _assess_gamma_environment(self, target_price: float, analysis: Dict) -> str:
        """Assess gamma environment at target price"""
        key_levels = analysis.get('key_levels', {})
        upside_pivot = getattr(key_levels, 'upside_pivot', None)
        downside_pivot = getattr(key_levels, 'downside_pivot', None)
        
        if upside_pivot and target_price > upside_pivot:
            return "Negative gamma zone - accelerative"
        elif downside_pivot and target_price < downside_pivot:
            return "Negative gamma zone - accelerative"
        else:
            return "Positive gamma zone - supportive"
    
    def _describe_expected_behavior(self, target_price: float, analysis: Dict) -> str:
        """Describe expected price behavior at target"""
        gamma_env = self._assess_gamma_environment(target_price, analysis)
        
        if "Negative" in gamma_env:
            return "Momentum continuation, reduced resistance"
        else:
            return "Mean reversion pressure, increased support/resistance"
    
    def _suggest_strategy(self, direction: str, analysis: Dict) -> str:
        """Suggest trading strategy for scenario"""
        regime = analysis.get('market_regime', {})
        regime_type = getattr(regime, 'regime_type', 'unknown')
        
        if direction == 'upside':
            if regime_type == 'breakout':
                return "Long calls, long gamma strategies"
            else:
                return "Short puts, sell resistance rallies"
        else:  # downside
            if regime_type == 'breakout':
                return "Long puts, long gamma strategies"
            else:
                return "Short calls, buy support bounces"
    
    def _assess_vanna_effects(self, analysis: Dict) -> str:
        """Assess vanna effects for volatility scenario"""
        patterns = analysis.get('positioning_patterns', {})
        directional = patterns.get('directional_bias', {})
        vanna_skew = directional.get('vanna_skew', 0)
        
        if abs(vanna_skew) < 100:
            return "Minimal cross-effects expected"
        elif vanna_skew > 0:
            return "Positive vanna skew - vol up/spot up correlation"
        else:
            return "Negative vanna skew - vol up/spot down correlation"


def main():
    """Example usage"""
    try:
        # Load data
        greeks_df = pd.read_parquet("data/spx_options/greeks/aggregated_greeks.parquet")
        trade_greeks_df = pd.read_parquet("data/spx_options/greeks/trade_level_greeks.parquet")
        
        # Load analysis
        with open("data/spx_options/market_structure_analysis.json", "r") as f:
            analysis = json.load(f)
        
        print(f"Loaded data for report generation")
        
        # Current SPX
        current_spx = 5800
        
        # Create report generator
        report_gen = DealerPositioningReport(spot_price=current_spx)
        
        # Generate full report
        report = report_gen.generate_full_report(
            greeks_df, trade_greeks_df, analysis,
            output_file="outputs/dealer_positioning/comprehensive_report.md"
        )
        
        print("\n=== Report Generation Complete ===")
        print("Report saved to: outputs/dealer_positioning/comprehensive_report.md")
        
        # Also create a summary version
        summary_sections = [
            report_gen._generate_header(),
            report_gen._generate_executive_summary(analysis),
            report_gen._generate_trading_implications(analysis),
            report_gen._generate_quantitative_summary(greeks_df, analysis)
        ]
        
        summary_report = "\n\n".join(summary_sections)
        
        with open("outputs/dealer_positioning/executive_summary.md", 'w') as f:
            f.write(summary_report)
        
        print("Executive summary saved to: outputs/dealer_positioning/executive_summary.md")
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Run the previous analysis steps first.")


if __name__ == "__main__":
    main()