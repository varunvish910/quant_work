# SPY Weekly Options Dealer Positioning Analysis Plan

## Overview
This document outlines a comprehensive plan to analyze SPY weekly options trades for expiries during October 6-10, 2025 (Monday through Friday), focusing on dealer positioning, gamma exposure, and higher-order Greeks to understand market dynamics.

## Key Dates & Context
- **Analysis Period**: October 6-10, 2025
  - Monday, Oct 6: Weekly expiry
  - Wednesday, Oct 8: Weekly expiry  
  - Friday, Oct 10: Weekly expiry
- **Data Collection Period**: September 22 - October 10, 2025 (capture position buildup)
- **Current Date**: October 6, 2025 (Monday)
- **Underlying Ticker**: SPY (SPDR S&P 500 ETF)
- **Contract Multiplier**: 100 shares

## Phase 1: Data Collection

### 1.1 Trades Data Collection
- **Data Source**: Polygon.io Trades endpoint
- **Instrument**: SPY weekly options (ticker format: O:SPY251006C00550000)
- **Target Expiries**: 
  - October 6, 2025 (Monday) - O:SPY251006*
  - October 8, 2025 (Wednesday) - O:SPY251008*
  - October 10, 2025 (Friday) - O:SPY251010*
- **Trade Data Period**: September 22 - October 6, 2025
- **Fields Required**:
  - Trade timestamp
  - Strike price
  - Option type (call/put)
  - Trade price
  - Trade size (contracts)
  - Trade conditions
  - Exchange

### 1.2 Quotes Data Collection
- **Data Source**: Polygon.io Quotes endpoint
- **Time Window**: ±1 second around each trade timestamp
- **Fields Required**:
  - Bid price
  - Ask price
  - Bid size
  - Ask size
  - Quote timestamp
  - NBBO indicators

### 1.3 Implementation Steps
1. Use Polygon flat files to download all options trades for each day
2. Filter trades for SPY options with our target expiries (Oct 6, 8, 10)
3. Download corresponding quotes flat files to get bid/ask spreads
4. Match trades with quotes based on timestamp
5. Store filtered SPY trades in parquet format for efficient processing

### 1.3.1 Polygon Flat Files Structure
- **Trades Flat Files**: `s3://flatfiles/us_options_opra/trades_v1/{year}/{month}/{date}.csv.gz`
  - Contains all option trades for the day (millions of records)
  - Fields: ticker, sip_timestamp, participant_timestamp, price, size, conditions, exchange
  - Example ticker: O:SPY251006C00550000 (SPY Oct 6 2025 $550 Call)
- **Quotes Flat Files**: `s3://flatfiles/us_options_opra/quotes_v1/{year}/{month}/{date}.csv.gz`
  - Contains all option quotes for the day
  - Fields: ticker, sip_timestamp, participant_timestamp, bid_price, ask_price, bid_size, ask_size, bid_exchange, ask_exchange

### 1.3.2 Data Processing Pipeline
1. **Download**: Use AWS CLI with Polygon credentials to download compressed files
2. **Filter**: Extract only SPY options with target expiries (O:SPY251006*, O:SPY251008*, O:SPY251010*)
3. **Enrich**: Match trades with quotes within ±1 second window
4. **Store**: Save filtered SPY trades with quote context to parquet

### 1.3.3 Advantages of Flat File Approach
- **Efficiency**: Download all trades at once, then filter
- **Completeness**: Ensures we capture all SPY trades
- **Cost-effective**: Fewer API calls compared to per-ticker requests
- **Reusability**: Can analyze multiple tickers from same files

### 1.4 Code Structure for SPY Trades Downloader
```python
# spy_trades_downloader.py
class SPYTradesDownloader:
    def __init__(self, output_dir='data/spy_trades'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # AWS credentials for Polygon flat files
        self.aws_key = "86959ae1-29bc-4433-be13-1a41b935d9d1"
        self.aws_secret = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.endpoint = "https://files.polygon.io"
        
        # Target expiries for filtering
        self.target_expiries = ['251006', '251008', '251010']  # YYMMDD format
    
    def download_trades_flatfile(self, date_str: str):
        """Download trades flat file for a specific date"""
        # s3://flatfiles/us_options_opra/trades_v1/{year}/{month}/{date}.csv.gz
        
    def download_quotes_flatfile(self, date_str: str):
        """Download quotes flat file for a specific date"""
        # s3://flatfiles/us_options_opra/quotes_v1/{year}/{month}/{date}.csv.gz
        
    def filter_spy_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Filter trades for SPY options with target expiries"""
        # Filter: ticker starts with 'O:SPY' AND contains target expiry dates
        # Example: O:SPY251006C00550000 (Oct 6, 2025 $550 Call)
        
    def filter_spy_quotes(self, quotes_df: pd.DataFrame, spy_tickers: List[str]) -> pd.DataFrame:
        """Filter quotes for relevant SPY options"""
        
    def match_trades_with_quotes(self, trades_df: pd.DataFrame, quotes_df: pd.DataFrame):
        """Match each trade with surrounding quotes for bid/ask context"""
        # For each trade, find quotes within ±1 second window
        
    def get_spy_spot_price(self, date: str):
        """Get SPY spot price for the date"""
        # Use yfinance or calculate from ATM options
```

## Phase 2: Trade Classification

### 2.1 Classification Algorithm
Classify each trade into one of four categories:
1. **SOLD TO OPEN (STO)**: Customer sells to open new position
2. **BOUGHT TO CLOSE (BTC)**: Customer buys to close existing short
3. **BOUGHT TO OPEN (BTO)**: Customer buys to open new position
4. **SOLD TO CLOSE (STC)**: Customer sells to close existing long

### 2.2 Classification Logic
```
If trade_price >= ask_price:
    If open_interest_change > 0: BTO
    Else: BTC
Else if trade_price <= bid_price:
    If open_interest_change > 0: STO
    Else: STC
Else (mid-market trades):
    Use additional heuristics:
    - Trade size patterns
    - Time of day
    - Historical OI changes
```

### 2.3 Dealer Perspective Translation
- Customer BTO → Dealer sells call/put (short gamma for calls, long gamma for puts)
- Customer STO → Dealer buys call/put (long gamma for calls, short gamma for puts)
- Customer BTC → Dealer buys back short
- Customer STC → Dealer sells back long

### 2.4 Code Structure for Trade Classifier
```python
# trade_classifier.py
class TradeClassifier:
    def classify_trade(self, trade: Dict, quote: Dict, oi_change: float) -> str:
        """Classify trade as BTO/STO/BTC/STC"""
        
    def get_dealer_position(self, customer_action: str, option_type: str) -> Dict:
        """Convert customer action to dealer position"""
        # Returns: {position: 'long'/'short', gamma_sign: +1/-1}
```

## Phase 3: Implied Move Calculation

### 3.1 Weekly Implied Move
Calculate using at-the-money (ATM) straddle pricing:
```
Implied Move = ATM_Straddle_Price / Spot_Price
```

### 3.2 Implementation
1. Identify ATM strike (closest to current spot)
2. Sum call and put prices at ATM strike
3. Calculate percentage move expectation
4. Analyze skew to determine directional bias

## Phase 4: Gamma Exposure Analysis

### 4.1 Aggregate Gamma by Strike
For each strike:
```
Dealer_Gamma = Σ(Trade_Size × Option_Gamma × Dealer_Position_Sign)

Where Dealer_Position_Sign:
- Call BTO: -1 (dealer short)
- Call STO: +1 (dealer long)
- Put BTO: +1 (dealer long)
- Put STO: -1 (dealer short)
```

### 4.2 Gamma Profile Construction
1. Calculate individual option gammas using Black-Scholes
2. Aggregate by strike from dealer perspective
3. Create gamma exposure chart showing:
   - Positive gamma zones (supportive)
   - Negative gamma zones (accelerative)
   - Zero gamma line (flip points)

### 4.3 Code Structure for Greeks Calculator
```python
# greeks_calculator.py
class GreeksCalculator:
    def __init__(self, spot: float, rate: float = 0.05):
        self.spot = spot
        self.rate = rate
    
    def calculate_all_greeks(self, strike: float, expiry: float, 
                           vol: float, option_type: str) -> Dict:
        """Calculate all Greeks for a single option"""
        # Returns: {delta, gamma, theta, vega, rho, vanna, charm, vomma, speed, zomma, color}
    
    def aggregate_dealer_greeks(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Greeks from dealer perspective"""
```

## Phase 5: Higher-Order Greeks Analysis

### 5.1 Greeks to Calculate (per expiry)
1. **Gamma (Γ)**: Rate of change of delta
2. **Vanna**: ∂Delta/∂σ or ∂Vega/∂S
3. **Charm (Delta Decay)**: -∂Delta/∂t
4. **Delta (Δ)**: Price sensitivity
5. **Vega (ν)**: Volatility sensitivity
6. **Vomma (Volga)**: ∂Vega/∂σ

### 5.2 Additional Greeks for Advanced Analysis
1. **Speed**: ∂Gamma/∂S (third derivative)
2. **Zomma**: ∂Gamma/∂σ
3. **Veta**: ∂Vega/∂t
4. **Color**: ∂Gamma/∂t

### 5.3 Calculation Framework
```python
For each expiry:
    For each strike:
        Calculate all Greeks using:
        - Current spot price
        - Strike price
        - Time to expiration
        - Implied volatility
        - Risk-free rate
        - Dividend yield
        
    Aggregate by dealer positioning
    Weight by open interest/volume
```

## Phase 6: Visualization

### 6.1 Primary Chart (Similar to Reference Image)
Create multi-panel chart showing:
- **Top Panel**: Charm exposure by strike
- **Middle Panel**: Gamma exposure by strike
- **Bottom Panels**: Vanna, Delta, Vega, Vomma

### 6.2 Chart Elements
- X-axis: Strike prices
- Y-axis: Greek exposure values
- Color coding: Long (positive) vs Short (negative) exposure
- Spot price indicator line
- Key levels marked (centroid, pivots, wings)

### 6.3 Additional Visualizations
1. 3D surface plot of gamma over time
2. Heatmap of dealer positioning changes
3. Volatility smile/skew analysis
4. Term structure comparison

### 6.4 Code Structure for Visualizer
```python
# dealer_positioning_visualizer.py
class DealerPositioningVisualizer:
    def create_greeks_panel(self, greeks_df: pd.DataFrame, spot: float):
        """Create multi-panel Greeks exposure chart"""
        
    def identify_key_levels(self, gamma_df: pd.DataFrame) -> Dict:
        """Find pivots, centroids, and support/resistance"""
        
    def generate_analysis_text(self, metrics: Dict) -> str:
        """Generate written analysis like reference example"""
```

## Phase 7: Market Structure Analysis

### 7.1 Positioning Patterns
Identify key patterns:
1. **Butterfly Structures**: Short/Long fly positioning
2. **Condor Patterns**: Risk reversal setups
3. **Calendar Spreads**: Time decay plays
4. **Directional Biases**: Skew analysis

### 7.2 Key Metrics to Calculate
1. **Speed Convexity**: Net speed across strikes (∂³Price/∂S³)
2. **Gamma Centroid**: Weighted center of gamma exposure
3. **Vanna Balance Point**: Where vanna effects neutralize
4. **Charm Decay Profile**: Time decay acceleration zones
5. **Wing Analysis**: Tail positioning (25-delta ranges)

### 7.3 Pivot Points Identification
```
Upside Pivot = Strike where positive gamma turns to negative gamma vacuum
Downside Pivot = Strike where negative gamma turns to positive gamma support
Neutral Zone = Range between pivots
```

### 7.4 Code Structure for Market Analyzer
```python
# market_structure_analyzer.py
class MarketStructureAnalyzer:
    def identify_butterfly_pattern(self, gamma_profile: pd.DataFrame) -> Dict:
        """Detect short/long fly positioning"""
        
    def calculate_speed_convexity(self, speed_profile: pd.DataFrame) -> float:
        """Net speed across all strikes"""
        
    def find_gamma_pivots(self, gamma_profile: pd.DataFrame) -> Dict:
        """Identify support/resistance transitions"""
        
    def analyze_tail_positioning(self, greeks_df: pd.DataFrame) -> Dict:
        """Analyze 25-delta wing positioning"""
```

## Phase 8: Interpretive Analysis

### 8.1 Market Regime Classification
Based on Greeks profile, classify market positioning:
1. **Pinning Regime**: Short gamma at center, long wings
2. **Breakout Regime**: Long gamma at center, short wings
3. **Directional Regime**: Asymmetric gamma/vanna profile
4. **Volatility Regime**: High vomma/veta exposure

### 8.2 Risk Assessment
Analyze key risks:
1. **Gamma Risk**: Acceleration zones and flip points
2. **Vanna Risk**: Vol-spot correlation effects
3. **Charm Risk**: Time decay acceleration
4. **Vomma Risk**: Volatility of volatility exposure
5. **Pin Risk**: Expiration effects

### 8.3 Predictive Insights
Generate actionable insights:
1. **Expected Range**: Based on gamma barriers
2. **Volatility Forecast**: From vega/vomma positioning
3. **Directional Bias**: From delta/vanna skew
4. **Time Decay Effects**: From charm/theta profile
5. **Reversion Probability**: From mean reversion models

## Phase 9: Report Generation

### 9.1 Written Analysis Structure
Following the reference example format:
1. **Opening Summary**: Key positioning pattern identified
2. **Speed/Convexity Analysis**: Third-order effects
3. **Gamma Profile Interpretation**: Support/resistance levels
4. **Vanna/Charm Dynamics**: Cross-effects analysis
5. **Risk Scenarios**: Upside/downside projections
6. **Trading Implications**: Risk/reward assessment

### 9.2 Quantitative Outputs
1. **Implied Weekly Move**: X.XX%
2. **Gamma Pivots**: Upside at XXXX, Downside at XXXX
3. **Expected Ranges**: Based on vomma boundaries
4. **Skew Metrics**: Fixed strike and 25-delta changes
5. **Reversion Probability**: Statistical likelihood

### 9.3 Visual Report
Combine all charts and analysis into comprehensive report:
1. Executive summary with key levels
2. Full Greeks exposure charts
3. Detailed written analysis
4. Risk scenarios and probabilities
5. Trading recommendations

### 9.4 Code Structure for Report Generator
```python
# dealer_positioning_report.py
class DealerPositioningReport:
    def __init__(self, analyzer: MarketStructureAnalyzer, 
                 visualizer: DealerPositioningVisualizer):
        self.analyzer = analyzer
        self.visualizer = visualizer
    
    def generate_full_report(self, date: str, trades_df: pd.DataFrame, 
                           greeks_df: pd.DataFrame) -> Dict:
        """Generate complete analysis report"""
        
    def create_executive_summary(self, metrics: Dict) -> str:
        """Create high-level summary with key levels"""
        
    def generate_trading_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable trading insights"""
```

## Implementation Timeline

### Week 1: Data Infrastructure
- Set up Polygon API connections
- Build trade/quote data collectors
- Create data storage structure

### Week 2: Classification Engine
- Implement trade classification logic
- Validate against known patterns
- Build dealer position translator

### Week 3: Greeks Calculation
- Implement Black-Scholes Greeks
- Build aggregation framework
- Validate calculations

### Week 4: Analysis & Visualization
- Create visualization framework
- Build analysis algorithms
- Generate first reports

## Technical Stack

### Required Libraries
```python
# Data Processing
import pandas as pd
import numpy as np
import polars as pl

# Options Pricing
from scipy.stats import norm
import QuantLib as ql
import py_vollib

# Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# API & Data
import polygon
import asyncio
import aiohttp

# Analysis
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
```

### Data Storage Structure
```
/data/
  /spx_options/
    /trades/
      - 2025-10-05_trades.parquet
      - 2025-10-06_trades.parquet
      ...
    /quotes/
      - 2025-10-05_quotes.parquet
      ...
    /classified/
      - weekly_classified_trades.parquet
    /greeks/
      - gamma_exposure.parquet
      - all_greeks_by_strike.parquet
```

## Key Formulas Reference

### Black-Scholes Greeks
```
d1 = (ln(S/K) + (r + σ²/2)t) / (σ√t)
d2 = d1 - σ√t

Call Delta = N(d1)
Put Delta = N(d1) - 1

Gamma = φ(d1) / (S·σ·√t)

Vega = S·φ(d1)·√t

Theta_Call = -(S·φ(d1)·σ)/(2√t) - r·K·e^(-r·t)·N(d2)

Vanna = -φ(d1)·d2/σ

Charm = -φ(d1)·(2·r·t - d2·σ·√t)/(2·t·σ·√t)

Vomma = Vega·d1·d2/σ

Speed = -Gamma·(d1/(σ·√t) + 1)/S

Zomma = Gamma·(d1·d2 - 1)/σ

Color = -φ(d1)/(2·S·t·σ·√t)·[2·r·t - 1 + d1/(σ·√t)·(2·r·t - d2·σ·√t)]
```

## Success Criteria

1. **Data Quality**: 99%+ trade classification accuracy
2. **Calculation Speed**: Full analysis in <5 minutes
3. **Predictive Value**: 70%+ accuracy on range predictions
4. **Report Clarity**: Actionable insights in plain language
5. **Visualization Quality**: Professional-grade charts

## Risk Considerations

1. **Data Limitations**: Polygon may have incomplete options data
2. **Classification Errors**: Mid-market trades difficult to classify
3. **Model Risk**: Black-Scholes assumptions may not hold
4. **Computational Load**: High-frequency data processing intensive
5. **Market Regime Changes**: Historical patterns may not persist

## Next Steps

1. Review and approve plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Schedule weekly progress reviews
5. Iterate based on initial results

## Main Entry Point Structure

```python
# main_dealer_positioning.py
def main():
    # Configuration
    expiry_dates = ['2025-10-06', '2025-10-08', '2025-10-10']
    data_period_start = '2025-09-22'
    data_period_end = '2025-10-06'
    analysis_date = '2025-10-06'  # Current date for analysis
    
    # 1. Download and process trades using flat files
    downloader = SPYTradesDownloader()
    
    # Get trading days in our period
    trading_days = pd.bdate_range(start=data_period_start, end=data_period_end)
    
    all_trades = []
    print(f"Downloading trades for {len(trading_days)} days...")
    
    for date in trading_days:
        date_str = date.strftime('%Y-%m-%d')
        
        # Download flat files for trades and quotes
        trades_file = downloader.download_trades_flatfile(date_str)
        quotes_file = downloader.download_quotes_flatfile(date_str)
        
        if trades_file and quotes_file:
            # Parse and filter for SPY options with target expiries
            trades_df = pd.read_csv(trades_file, compression='gzip')
            spy_trades = downloader.filter_spy_trades(trades_df)
            
            # Get corresponding quotes
            quotes_df = pd.read_csv(quotes_file, compression='gzip')
            spy_quotes = downloader.filter_spy_quotes(quotes_df, spy_trades['ticker'].unique())
            
            # Match trades with quotes
            enriched_trades = downloader.match_trades_with_quotes(spy_trades, spy_quotes)
            all_trades.append(enriched_trades)
            
            print(f"✅ {date_str}: {len(spy_trades)} SPY trades")
    
    # Combine all trades
    combined_trades = pd.concat(all_trades, ignore_index=True)
    print(f"\nTotal SPY trades collected: {len(combined_trades):,}")
    
    # 2. Get current SPY price
    spy_spot = downloader.get_spy_spot_price(analysis_date)
    
    # 3. Classify trades
    classifier = TradeClassifier()
    classified_trades = classifier.classify_all_trades(combined_trades)
    
    # 4. Calculate Greeks
    calculator = GreeksCalculator(spot=spy_spot)
    greeks_df = calculator.aggregate_dealer_greeks(classified_trades)
    
    # 5. Analyze market structure
    analyzer = MarketStructureAnalyzer()
    market_analysis = analyzer.analyze_full_structure(greeks_df)
    
    # 6. Create visualizations
    visualizer = DealerPositioningVisualizer()
    charts = visualizer.create_all_charts(greeks_df, market_analysis)
    
    # 7. Generate report
    report_gen = DealerPositioningReport(analyzer, visualizer)
    full_report = report_gen.generate_full_report(analysis_date, classified_trades, greeks_df)
    
    # 8. Save outputs
    save_results(full_report, charts, output_dir='outputs/dealer_positioning/')
```

---

*This plan provides a comprehensive framework for building a professional-grade SPY options dealer positioning analysis system. Each phase builds upon the previous, ensuring robust and reliable insights for trading decisions.*