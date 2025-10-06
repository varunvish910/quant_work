# SPY Gamma Positioning Dealer Dashboard Plan

## Executive Summary
Build a comprehensive gamma positioning dashboard for SPY options expiring this week (October 6-10, 2025) using the last 30 days of trades data from Polygon flat files. The system will leverage existing dealer positioning modules to analyze dealer positioning and identify key support/resistance levels and market dynamics.

## Existing Code Architecture

### Core Modules (Already Implemented)
1. **`dealer_positioning/spy_trades_downloader.py`** - Downloads SPY options trades from Polygon
2. **`dealer_positioning/trade_classifier.py`** - Classifies trades as BTO/STO/BTC/STC
3. **`dealer_positioning/market_structure_analyzer.py`** - Analyzes market structure patterns
4. **`dealer_positioning/dealer_positioning_visualizer.py`** - Creates interactive charts
5. **`dealer_positioning/dealer_positioning_report.py`** - Generates written analysis
6. **`dealer_positioning/main_spy_positioning.py`** - Main pipeline orchestrator

### Data Management (Reusable)
1. **`data_management/unified_options_downloader.py`** - Options data framework
2. **`config/api_config.py`** - Polygon API configuration
3. **`config/data_sources.yaml`** - Data source definitions

### No New Files Needed
The existing codebase already contains all necessary components. We only need to:
1. Create a Greeks calculator module (currently missing)
2. Use the existing `main_spy_positioning.py` with proper parameters

## 1. Data Collection Strategy

### Time Windows
- **Analysis Date**: October 5, 2025 (Sunday)
- **Target Expiries**: 
  - October 6, 2025 (Monday) - Next day expiry
  - October 8, 2025 (Wednesday) - Mid-week expiry
  - October 10, 2025 (Friday) - Standard weekly expiry
- **Historical Data Period**: September 5 - October 5, 2025 (30 days)

### Data Sources
- **Primary**: Polygon flat files (trades and quotes)
  - Trades: `s3://flatfiles/us_options_opra/trades_v1/{year}/{month}/{date}.csv.gz`
  - Quotes: `s3://flatfiles/us_options_opra/quotes_v1/{year}/{month}/{date}.csv.gz`
- **Spot Price**: Yahoo Finance for current SPY price
- **Volume**: ~20-30 million SPY options trades over 30 days

## 2. Processing Pipeline Architecture

### Phase 1: Data Acquisition
1. **Download Trade Flat Files Only**
   - Use AWS S3 credentials to access Polygon flat files
   - Download 30 days of compressed TRADE files only
   - Estimate: ~500MB-1GB compressed per day for trades

2. **Filter SPY Options**
   - Extract only SPY options (ticker pattern: `O:SPY*`)
   - Filter for target expiries (251006, 251008, 251010)
   - Expected: ~500K-1M relevant trades

3. **Real Quote Data Approach**
   - DO NOT use synthetic quotes from `spy_trades_downloader.py`
   - For each SPY trade, fetch real quotes using Polygon API
   - Use timestamp matching with `pandas.merge_asof()`
   - Query quotes within ±1 second window of each trade
   - This ensures accurate bid/ask for proper trade classification

### Phase 2: Trade Classification (Already Implemented)
The `trade_classifier.py` module handles all classification:
1. **Algorithm Logic**
   - Trades at/above ask → Customer buying (BTO/BTC)
   - Trades at/below bid → Customer selling (STO/STC)
   - Mid-market trades → Uses time-of-day and size heuristics
   - Confidence scoring for each classification

2. **Dealer Position Mapping** 
   - Customer BTO Call → Dealer short gamma (-1)
   - Customer STO Call → Dealer long gamma (+1)
   - Customer BTO Put → Dealer long gamma (+1)
   - Customer STO Put → Dealer short gamma (-1)

### Phase 3: Greeks Calculation
1. **Black-Scholes Greeks**
   - Calculate for each unique strike/expiry
   - Key Greeks: Delta, Gamma, Vanna, Charm, Vomma
   - Use implied volatility from market prices

2. **Aggregation by Strike**
   - Sum dealer gamma exposure by strike
   - Weight by trade size and direction
   - Identify net positioning

## 3. Dashboard Components

### Main Visualization Panel
1. **Gamma Profile Chart**
   - X-axis: Strike prices ($650-$690 range)
   - Y-axis: Net dealer gamma exposure
   - Color coding: Positive (green) vs Negative (red)
   - Current spot price indicator

2. **Multi-Greek Panel**
   - Stacked charts showing:
     - Charm (delta decay)
     - Vanna (vol-spot correlation)
     - Vomma (vol of vol)
     - Speed (gamma change rate)

3. **3D Surface Plot**
   - Gamma exposure over strike × time
   - Shows evolution of positioning
   - Identifies roll patterns

### Key Metrics Display
1. **Quantitative Summary**
   - Total gamma exposure
   - Gamma flip points
   - Weighted gamma centroid
   - Max pain calculation

2. **Market Regime Classification**
   - Pinning regime (short gamma at center)
   - Breakout regime (long gamma at center)
   - Directional bias indicators

3. **Support/Resistance Levels**
   - Upside gamma vacuum levels
   - Downside support levels
   - Zero gamma crossover points

## 4. Analysis Framework

### Positioning Patterns to Identify
1. **Butterfly Structures**
   - Short strikes at center, long wings
   - Indicates range-bound expectations

2. **Risk Reversals**
   - Skewed put/call positioning
   - Shows directional bias

3. **Calendar Effects**
   - Monday vs Wednesday vs Friday dynamics
   - Time decay acceleration zones

### Risk Metrics
1. **Gamma Risk**
   - Acceleration zones
   - Flip point stability
   - Hedging requirements

2. **Vanna Risk**
   - Volatility-spot correlation effects
   - Skew dynamics

3. **Pin Risk**
   - Concentration around strikes
   - Expiration day effects

## 5. Output Deliverables

### Visual Reports
1. **Interactive HTML Dashboard**
   - Main gamma profile chart
   - Greeks panel display
   - Clickable strike analysis

2. **Static Charts**
   - PNG exports for reports
   - Time-series heatmaps
   - Volatility smile analysis

### Written Analysis
1. **Executive Summary**
   - Key positioning themes
   - Critical levels to watch
   - Risk scenarios

2. **Detailed Report**
   - Strike-by-strike analysis
   - Historical comparison
   - Trading implications

3. **Quantitative Metrics**
   - CSV export of all calculations
   - JSON data for API consumption
   - Time-series database format

## 6. Implementation Steps (Using Existing Code)

### Step 1: Create Greeks Calculator Module
The only missing component is `greeks_calculator.py`. This module needs to:
- Calculate Black-Scholes Greeks (Delta, Gamma, Vanna, Charm, Vomma)
- Aggregate dealer Greeks by strike
- Interface with existing pipeline

### Step 2: Execute Analysis Pipeline
Run the existing pipeline with proper parameters:
```bash
cd dealer_positioning
python main_spy_positioning.py \
  --date 2025-10-05 \
  --expiry 2025-10-06 2025-10-08 2025-10-10 \
  --output outputs/spy_gamma_dashboard
```

### Step 3: Generated Outputs (Automatic)
The existing pipeline will create:
- `greeks_panel.html` - Multi-panel Greeks exposure chart
- `gamma_profile.html` - Interactive gamma profile
- `volatility_smile.html` - Volatility smile analysis
- `comprehensive_report.md` - Full written analysis
- `executive_summary.md` - Key findings
- `analysis_summary.json` - Machine-readable data

## 7. Technical Considerations

### Performance Optimization
- Use Polars for fast data processing
- Implement parallel file processing
- Cache calculated Greeks

### Data Quality
- Handle missing quotes
- Validate trade timestamps
- Check for data anomalies

### Scalability
- Modular architecture
- Database storage for historical data
- API endpoints for real-time updates

## 8. Success Metrics

1. **Accuracy**
   - 95%+ trade classification accuracy
   - Greeks validation against market prices
   - Backtested prediction accuracy

2. **Performance**
   - Process 30 days in < 10 minutes
   - Real-time dashboard updates
   - Sub-second chart rendering

3. **Insights**
   - Identify key support/resistance 80%+ accuracy
   - Predict intraday ranges
   - Early warning on positioning shifts

## 9. Risk Management

1. **Data Risks**
   - Incomplete trade data
   - Quote synchronization issues
   - API rate limits

2. **Model Risks**
   - Black-Scholes assumptions
   - Volatility smile effects
   - Early exercise considerations

3. **Operational Risks**
   - System downtime
   - Data storage costs
   - Processing delays

## 10. Code Usage Summary

### Primary Entry Point
```python
# main_spy_positioning.py already handles everything:
from dealer_positioning.main_spy_positioning import SPYPositioningPipeline

pipeline = SPYPositioningPipeline(
    api_key=api_key,
    spot_price=current_spy_price,
    output_dir="outputs/spy_gamma_dashboard"
)

results = pipeline.run_full_analysis(
    target_date="2025-10-05",
    expiry_dates=["2025-10-06", "2025-10-08", "2025-10-10"],
    skip_download=False
)
```

### Data Flow Through Existing Modules
1. **SPYTradesDownloader** → Downloads Polygon flat files
2. **TradeClassifier** → Classifies each trade (BTO/STO/BTC/STC)
3. **GreeksCalculator** → Calculates and aggregates Greeks (needs creation)
4. **MarketStructureAnalyzer** → Identifies patterns and regimes
5. **DealerPositioningVisualizer** → Creates all charts
6. **DealerPositioningReport** → Generates written analysis

### Configuration Files Used
- `config/api_config.py` - Polygon API credentials
- `config/data_sources.yaml` - Data source definitions
- No additional configuration needed

## 11. Minimal Implementation Path

Required modifications to existing code:

1. **Modify `spy_trades_downloader.py`**
   - Remove synthetic quote generation (_add_synthetic_quotes method)
   - Add real quote fetching via Polygon API
   - Implement batch quote fetching for efficiency
   - Use `merge_asof` for timestamp-based matching

2. **Create `dealer_positioning/greeks_calculator.py`**
   - Import numpy/scipy for Black-Scholes calculations
   - Calculate Greeks for each option (Gamma, Delta, Vanna, Charm, Vomma)
   - Aggregate by dealer positioning

3. **Run the existing pipeline**
   - Pipeline will use real quotes for accurate classification
   - All other modules work as-is
   - Outputs generated in specified directory

4. **View results**
   - Open HTML files in browser for interactive charts
   - Read markdown reports for analysis
   - Use JSON for programmatic access

This approach ensures accurate trade classification with real market quotes while maximizing reuse of existing infrastructure.

## 12. Execution Summary

### What's Actually Implemented:
- ✅ **SPYTradesDownloader** (`spy_trades_downloader.py`) - Downloads trades, but uses synthetic quotes
- ✅ **TradeClassifier** (`trade_classifier.py`) - Fully working
- ✅ **MarketStructureAnalyzer** (`market_structure_analyzer.py`) - Fully working
- ✅ **DealerPositioningVisualizer** (`dealer_positioning_visualizer.py`) - Fully working
- ✅ **DealerPositioningReport** (`dealer_positioning_report.py`) - Fully working
- ✅ **Main Pipeline** (`main_spy_positioning.py`) - Orchestrates everything
- ❌ **GreeksCalculator** - Referenced but NOT implemented!

### Two Things Need Implementation:

1. **Real Quote Fetching in `spy_trades_downloader.py`**
   - Replace the `_add_synthetic_quotes()` method
   - Add Polygon API quote fetching
   - Use `merge_asof` for timestamp matching

2. **Create `dealer_positioning/greeks_calculator.py`**
   - Implement Black-Scholes Greeks calculations
   - Include: Delta, Gamma, Vanna, Charm, Vomma, Speed, Zomma
   - Aggregate by dealer positioning

### Command to Run (After Implementation):
```bash
cd dealer_positioning
python main_spy_positioning.py \
  --date 2025-10-05 \
  --expiry 2025-10-06 2025-10-08 2025-10-10 \
  --output outputs/spy_gamma_dashboard
```

### Expected Outputs:
- **Interactive HTML Charts**:
  - `greeks_panel.html` - Multi-Greek visualization
  - `gamma_profile.html` - 3D gamma surface
  - `volatility_smile.html` - IV analysis
  
- **Reports**:
  - `comprehensive_report.md` - Full analysis
  - `executive_summary.md` - Key findings
  - `analysis_summary.json` - Data export

The pipeline is 90% complete - just needs real quotes and Greeks calculations!
