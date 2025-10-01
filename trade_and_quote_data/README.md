# Options Chain Data System

A comprehensive system for downloading, storing, and analyzing historical options chain data using the Polygon API.

## Overview

This system provides complete functionality for:
- Downloading complete historical options chains for any date
- Reconstructing point-in-time options chains from stored data  
- Advanced analytics: IV surfaces, Greeks calculation, skew analysis
- Backtest-ready data export with efficient storage
- High performance parallel processing with intelligent rate limiting

## Features

### Data Collection
- **Complete Historical Coverage**: 5+ years of options data via Polygon API
- **Comprehensive Chain Data**: All strikes, expirations, and contract types
- **Real-time Processing**: Parallel downloads with intelligent rate limiting
- **Quality Assurance**: Data validation and quality scoring

### Analytics
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, Rho using Black-Scholes
- **Implied Volatility**: Calculated using Brent's method
- **IV Surfaces**: 3D interpolated volatility surfaces
- **Chain Statistics**: Volume, open interest, put-call ratios
- **Moneyness Analysis**: ATM, ITM, OTM contract identification

### Storage & Export
- **Efficient Storage**: Parquet format with year/month partitioning
- **Fast Querying**: Optimized for date range and ticker filtering
- **Backtest Export**: Pre-processed data ready for strategy testing
- **Flexible Filtering**: Multiple criteria for data selection

## Installation

### Prerequisites
- Python 3.8+
- Polygon API key (Professional plan recommended for bulk downloads)

### Dependencies
```bash
pip install pandas numpy polygon-api-client scipy requests tqdm
```

## Quick Start

### Environment Setup
```bash
export POLYGON_API_KEY="your_polygon_api_key_here"
```

### Command Line Usage

#### Download Single Date
```bash
python options_anomaly_detection/downloaders/options_chain_downloader.py \
    --download --ticker SPY --date 2025-01-01
```

#### Download Date Range
```bash
python options_anomaly_detection/downloaders/options_chain_downloader.py \
    --download --ticker SPY \
    --start-date 2025-01-01 --end-date 2025-01-31
```

#### Analyze Stored Data
```bash
python options_anomaly_detection/downloaders/options_chain_downloader.py \
    --analyze --ticker SPY --date 2025-01-01
```

#### Export for Backtesting
```bash
python options_anomaly_detection/downloaders/options_chain_downloader.py \
    --export --ticker SPY \
    --start-date 2025-01-01 --end-date 2025-01-31
```

### Python API Usage

```python
from options_anomaly_detection.downloaders.options_chain_downloader import OptionsChainSystem

# Initialize system
system = OptionsChainSystem(api_key="your_polygon_key")

# Download complete options chain
chain = system.download_chain("SPY", "2025-01-01")
system.save_chain(chain, "SPY", "2025-01-01")

# Load and analyze historical data
chain = system.load_chain("SPY", "2025-01-01")
summary = system.analyze_chain(chain)
iv_surface = system.build_iv_surface(chain)
atm_contracts = system.get_atm_contracts(chain)

# Filter chains for specific criteria
from options_anomaly_detection.downloaders.options_chain_downloader import ChainFilter

filter_config = ChainFilter(
    contract_types=['call'],
    min_moneyness=0.95,
    max_moneyness=1.05,
    min_days_to_expiration=30,
    max_days_to_expiration=45,
    min_volume=100
)

filtered_chain = system.load_chain("SPY", "2025-01-01", filter_config)

# Load multiple dates for analysis
chains = system.load_chains_for_period("SPY", "2025-01-01", "2025-01-31")

# Export for backtesting
backtest_df = system.export_for_backtesting(chains, "SPY_backtest.parquet")
```

## Data Schema

Each options contract record contains:

### Core Data
- **Date & Identification**: date, underlying_ticker, contract_symbol
- **Contract Specs**: strike_price, expiration_date, contract_type, days_to_expiration
- **Market Data**: open, high, low, close, volume, vwap
- **Options Metrics**: open_interest, implied_volatility

### Calculated Fields
- **Greeks**: delta, gamma, theta, vega, rho (Black-Scholes)
- **Valuation**: moneyness, intrinsic_value, time_value, break_even
- **Quality**: data_quality_score, data_source, download_timestamp

## Storage Structure

```
data/
├── spy/
│   ├── options_chains/
│   │   ├── year=2025/
│   │   │   ├── month=01/
│   │   │   │   ├── chains_2025-01-01.parquet
│   │   │   │   ├── chains_2025-01-02.parquet
│   │   │   │   └── ...
│   │   │   └── month=02/
│   │   │       └── ...
│   │   └── year=2024/
│   │       └── ...
└── qqq/
    └── ...
```

## API Configuration

### Polygon API Limits
- **Free Tier**: 5 requests/minute
- **Basic Plan**: 100 requests/minute  
- **Professional Plans**: 5000+ requests/minute

### Rate Limiting
The system automatically handles rate limiting based on your plan:
- Professional: 0.12s delay (5000 req/min)
- Basic: 0.6s delay (100 req/min)
- Free: 12s delay (5 req/min)

## Performance

### Typical Download Performance
- **Complete SPY Chain**: ~2000 contracts per day
- **Professional API**: ~500 contracts/minute
- **Storage**: ~2-5MB per day (compressed Parquet)

### Optimization Features
- **Parallel Processing**: 20 concurrent workers
- **Smart Caching**: LRU cache for 100 chains
- **Incremental Downloads**: Skips existing files
- **Trading Day Filtering**: Excludes weekends and holidays

## Advanced Features

### Filtering Options
```python
filter_config = ChainFilter(
    min_volume=100,                    # Minimum daily volume
    min_open_interest=500,             # Minimum open interest
    min_days_to_expiration=7,          # Minimum DTE
    max_days_to_expiration=60,         # Maximum DTE
    min_moneyness=0.8,                 # Minimum moneyness
    max_moneyness=1.2,                 # Maximum moneyness
    contract_types=['call', 'put'],    # Contract types
    min_data_quality=0.8               # Minimum quality score
)
```

### IV Surface Construction
```python
# Build implied volatility surface
surfaces = system.build_iv_surface(chain)
call_surface = surfaces.get('call', {})
put_surface = surfaces.get('put', {})

# Access surface data
moneyness_grid = call_surface['moneyness_grid']
days_grid = call_surface['days_grid']
iv_surface = call_surface['iv_surface']
```

### Chain Analysis
```python
summary = system.analyze_chain(chain)
# Returns:
# - total_contracts, call_contracts, put_contracts
# - total_volume, total_open_interest
# - strike ranges, expiration ranges
# - IV statistics
# - put_call_volume_ratio, put_call_oi_ratio
```

## Error Handling

The system includes comprehensive error handling:
- **API Failures**: Automatic retries with exponential backoff
- **Data Validation**: Quality scoring and outlier detection
- **Missing Data**: Graceful handling of incomplete chains
- **Rate Limiting**: Automatic adjustment for API quotas

## Cost Considerations

### Polygon API Costs
- **Professional Plan**: $199/month for 5000 req/min
- **Complete SPY Chain**: ~2000 API calls per day
- **Monthly Cost**: ~$40/month for daily SPY chains
- **Historical Backfill**: One-time cost for multi-year data

## Troubleshooting

### Common Issues

**"No API key found"**
```bash
export POLYGON_API_KEY="your_key_here"
```

**"Rate limit exceeded"**
- Upgrade to Professional plan
- Reduce parallel workers in code

**"No data found"**
- Verify date is a trading day
- Check if data exists for that date on Polygon

**"Import errors"**
```bash
pip install -r options_anomaly_detection/requirements.txt
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your Polygon API plan and limits
3. Ensure all dependencies are installed
4. Check that the date is a valid trading day

## License

This project is for educational and research purposes. Please ensure compliance with Polygon API terms of service and any applicable regulations when using financial data.

---

## About This System

This options chain downloader was designed to provide institutional-quality historical options data for research, backtesting, and trading strategy development. It leverages the Polygon API's comprehensive options data coverage to reconstruct complete point-in-time options chains for any historical trading day.

The system includes advanced features like Greeks calculation, implied volatility surfaces, and quality scoring to ensure you have the most accurate and complete options data available for your analysis.

*For the complete system documentation and advanced usage examples, see the individual module documentation in the `archive/` directory.*