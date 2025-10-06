# SPX Weekly Options Dealer Positioning Analysis

A comprehensive system for analyzing dealer positioning in SPX weekly options, providing insights into gamma exposure, higher-order Greeks, and market structure for informed trading decisions.

## Features

- **Automated Data Collection**: Downloads SPX options trades and quotes from Polygon.io
- **Intelligent Trade Classification**: Classifies trades as BTO/STO/BTC/STC using bid/ask analysis
- **Complete Greeks Calculation**: All first, second, and third-order Greeks including vanna, charm, vomma, speed, zomma, and color
- **Market Structure Analysis**: Identifies positioning patterns, key levels, and market regimes
- **Professional Visualizations**: Multi-panel charts showing dealer exposure across all Greeks
- **Comprehensive Reports**: Written analysis with trading implications and risk assessment

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Get a Polygon.io API key and set it:

```bash
export POLYGON_API_KEY="your_api_key_here"
```

### 3. Run Analysis

```bash
python main_dealer_positioning.py --date 2025-10-05 --spot 5800
```

## Usage

### Basic Analysis

```bash
# Analyze SPX positioning for specific date
python main_dealer_positioning.py --date 2025-10-05 --expiry 2025-10-11 --spot 5800

# Use existing data (skip download)
python main_dealer_positioning.py --date 2025-10-05 --skip-download --spot 5800

# Custom output directory
python main_dealer_positioning.py --date 2025-10-05 --output custom_output/
```

### Command Line Options

- `--date`: Target analysis date (YYYY-MM-DD)
- `--expiry`: Option expiry dates (can specify multiple)
- `--spot`: Current SPX spot price
- `--api-key`: Polygon API key (or use POLYGON_API_KEY env var)
- `--skip-download`: Use existing data instead of downloading
- `--output`: Output directory for results

### Individual Modules

Each analysis phase can be run independently:

```python
# Data collection
from spx_trades_downloader import SPXTradesDownloader
downloader = SPXTradesDownloader(api_key)
trades = downloader.download_trades('2025-10-05', ['2025-10-11'])

# Trade classification
from trade_classifier import TradeClassifier
classifier = TradeClassifier()
classified = classifier.classify_all_trades(trades)

# Greeks calculation
from greeks_calculator import GreeksCalculator
calculator = GreeksCalculator(spot=5800)
greeks, trade_greeks = calculator.aggregate_dealer_greeks(classified)

# Market analysis
from market_structure_analyzer import MarketStructureAnalyzer
analyzer = MarketStructureAnalyzer(spot_price=5800)
analysis = analyzer.analyze_full_structure(greeks)

# Visualization
from dealer_positioning_visualizer import DealerPositioningVisualizer
visualizer = DealerPositioningVisualizer(spot_price=5800)
charts = visualizer.create_all_charts(greeks, trade_greeks, analysis)

# Report generation
from dealer_positioning_report import DealerPositioningReport
reporter = DealerPositioningReport(spot_price=5800)
report = reporter.generate_full_report(greeks, trade_greeks, analysis)
```

## Output Files

The analysis generates several output files:

### Visualizations
- `greeks_panel.html`: Multi-panel Greeks exposure chart
- `gamma_profile.html`: Interactive gamma profile chart
- `volatility_smile.html`: Volatility smile analysis
- `dashboard.html`: Comprehensive summary dashboard
- `positioning_heatmap.png`: Time-series heatmap of positioning

### Reports
- `comprehensive_report.md`: Complete written analysis
- `executive_summary.md`: Key findings and trading implications
- `analysis_summary.json`: Machine-readable summary

### Data Files
- `data/spx_options/trades/`: Raw and enriched trades data
- `data/spx_options/classified/`: Classified trades
- `data/spx_options/greeks/`: Greeks calculations
- `data/spx_options/market_structure_analysis.json`: Full analysis results

## Analysis Framework

### 1. Trade Classification Algorithm

Classifies each options trade into four categories from the customer perspective:

- **BTO (Bought to Open)**: Customer opens new long position
- **STO (Sold to Open)**: Customer opens new short position  
- **BTC (Bought to Close)**: Customer closes existing short position
- **STC (Sold to Close)**: Customer closes existing long position

Classification logic:
```
If trade_price >= ask_price:
    If OI_change > 0: BTO, else: BTC
Else if trade_price <= bid_price:
    If OI_change > 0: STO, else: STC
Else (mid-market):
    Use heuristics (time, size, price_position)
```

### 2. Dealer Position Translation

Converts customer actions to dealer perspective:

| Customer Action | Dealer Position | Gamma Sign |
|----------------|----------------|------------|
| Call BTO | Short Call | Negative |
| Call STO | Long Call | Positive |
| Put BTO | Short Put | Positive |
| Put STO | Long Put | Negative |

### 3. Greeks Calculation

Calculates all option Greeks using Black-Scholes formulas:

**First-order Greeks:**
- Delta (Δ): Price sensitivity
- Gamma (Γ): Delta sensitivity  
- Theta (Θ): Time decay
- Vega (ν): Volatility sensitivity
- Rho (ρ): Interest rate sensitivity

**Second-order Greeks:**
- Vanna: ∂Delta/∂σ (vol-spot cross effect)
- Charm: -∂Delta/∂t (delta decay)
- Vomma: ∂Vega/∂σ (vol convexity)

**Third-order Greeks:**
- Speed: ∂Gamma/∂S (gamma acceleration)
- Zomma: ∂Gamma/∂σ (gamma-vol cross)
- Color: ∂Gamma/∂t (gamma decay)

### 4. Market Regime Classification

Identifies market positioning regimes:

- **Pinning**: Short gamma at center, long wings → Range-bound expectations
- **Breakout**: Long gamma at center, short wings → Volatility expansion potential  
- **Directional**: Asymmetric positioning → Directional bias
- **Volatility**: High vomma/veta exposure → Vol-sensitive environment

### 5. Key Levels Identification

Automatically identifies critical price levels:

- **Gamma Centroid**: Volume-weighted center of positioning
- **Pivot Points**: Where gamma changes sign (support/resistance)
- **Gamma Walls**: Large concentrations of gamma exposure
- **Neutral Zone**: Range of minimal gamma effects

## Risk Considerations

1. **Data Limitations**: Polygon may have incomplete options data
2. **Classification Accuracy**: Mid-market trades difficult to classify precisely
3. **Model Assumptions**: Black-Scholes assumptions may not hold in all conditions
4. **Computational Load**: High-frequency data processing can be intensive
5. **Market Regime Changes**: Historical patterns may not persist

## API Rate Limits

The system includes built-in rate limiting for Polygon API:
- Trade downloads: 0.1 second delay between requests
- Quote enrichment: 0.05 second delay between requests
- Automatic batch processing to optimize API usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with your broker's terms of service and applicable regulations when using for trading decisions.

## Disclaimer

This analysis is for educational and informational purposes only. Past performance does not guarantee future results. Options trading involves significant risk of loss. Consult with qualified professionals before making trading decisions.

## Support

For issues and questions:
1. Check the existing documentation
2. Review the code comments for implementation details
3. Open an issue with detailed description and error logs