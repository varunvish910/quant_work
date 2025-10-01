# SPY Options Anomaly Detection System

**Last Updated**: October 1, 2025  
**Status**: Production Ready - Data Acquisition Complete

A comprehensive system for downloading, storing, and analyzing historical SPY options data with Open Interest proxy calculations for anomaly detection.

---

## 🎯 Project Overview

This system identifies unusual trading patterns in SPY options chains and generates actionable trading signals using historical data analysis and Open Interest proxy calculations.

## 📁 **PROJECT STRUCTURE**

```
/
├── README.md                 # Main project documentation  
├── requirements.txt          # Python dependencies
├── scripts/                  # Standalone analysis scripts
│   ├── hedging_activity_analyzer.py     # Hedging pattern analysis
│   └── hedging_buildup_analysis.py      # Pre-event buildup analysis
├── tools/                    # Core utilities
│   ├── analyze_date.py      # Date-specific analysis tool
│   ├── see_results.py       # Results viewer
│   ├── validate_data.py     # Data validation utility
│   └── download_2024.py     # Data downloader
├── analysis/                # Organized analysis scripts
│   ├── core_analysis/       # Main analysis engines (11 files)
│   ├── hedging_patterns/    # Hedging-specific analysis (8 files)
│   ├── correction_signals/  # Market correction prediction (6 files)
│   ├── temporal_comparisons/# Date/period comparisons (6 files)
│   ├── floor_analysis/      # Put floor analysis scripts
│   ├── historical_studies/  # Specialized research (4 files)
│   └── outputs/            # All CSV, PNG, and analysis output files
├── docs/                    # Project documentation
│   ├── TODO.md             # Development roadmap
│   └── ACTION_PLAN.md      # Current market analysis plan
├── data/                    # Historical options data
├── config/                  # Configuration files
├── options_anomaly_detection/  # ML anomaly detection engine
│   └── target_creator.py          # Correction target labeling
└── archive/                 # Reference implementations
```

### ✅ **COMPLETED FEATURES**

1. **SPY Options Downloader** (`tools/download_2024.py`)
   - Downloads historical SPY options data from Polygon flat files
   - Calculates Open Interest proxy from volume and transaction data
   - Command-line interface for easy automation
   - Supports any date range from 2016-2025

2. **Open Interest Proxy System**
   - Uses volume, transactions, ATM bias, and DTE to estimate OI
   - Multiple proxy versions (composite, volume-based, liquidity-based)
   - Realistic OI values for anomaly detection

3. **Data Management** (`data_management/`)
   - Optimized options chain downloader
   - OHLC data downloader
   - Cache management system

4. **Archive Reference** (`archive/`)
   - Complete proven anomaly detection system
   - Backtesting framework
   - Analysis tools and visualizations

---

## 🚀 Quick Start

### Download SPY Options Data

```bash
# Download January 2016
python3 archive/spy_options_downloader.py 2016-01-01 2016-01-31

# Download any date range
python3 archive/spy_options_downloader.py 2020-01-01 2020-12-31

# Save individual date files too
python3 archive/spy_options_downloader.py 2016-01-01 2016-01-31 --individual
```

### Data Output

The downloader creates Parquet files with:
- **Contract details**: ticker, strike, expiration, DTE, option type
- **Market data**: OHLC, volume, transactions, VWAP
- **OI Proxy**: Multiple versions calculated from volume/transactions
- **Analysis features**: moneyness, ATM score, liquidity score

---

## 📊 Data Validation (January 2016)

- **36,413 contracts** across 19 trading days
- **Date range**: 2016-01-04 to 2016-01-29
- **Strike range**: $10 to $320
- **DTE range**: 0 to 1,082 days
- **Calls**: 15,341 | **Puts**: 21,072
- **OI Proxy range**: 537 to 8,905 (realistic values)

---

## 🏗️ System Architecture

```
trade_and_quote_data/
├── archive/                       # Reference implementations
│   └── spy_options_downloader.py # Main downloader tool
├── data_management/               # Core data processing
│   ├── options_chain_downloader.py
│   ├── ohlc_data_downloader.py
│   └── cache_manager.py
├── options_anomaly_detection/     # Analysis framework
├── data/                         # Downloaded datasets
│   └── spy_options/              # SPY options data with OI proxy
├── TODO.md                       # Development roadmap
└── README.md                     # This file
```

---

## 🔧 Technical Details

### Open Interest Proxy Calculation

The system calculates OI proxy using:
- **Volume** (30% weight) - Primary indicator of activity
- **Transactions** (20% weight) - Indicates institutional activity
- **Liquidity Score** (20% weight) - sqrt(volume × transactions)
- **ATM Score** (20% weight) - Favors at-the-money options
- **DTE Weight** (10% weight) - Longer-dated options accumulate more OI

### Data Sources

- **Polygon Flat Files**: Historical OHLCV data (2014-2025)
- **No API Rate Limits**: Uses bulk flat file downloads
- **Comprehensive Coverage**: All SPY options contracts

---

## 📈 Next Steps

1. **Expand Historical Dataset**: Download 2016-2025 data
2. **Anomaly Detection**: Implement pattern recognition algorithms
3. **Signal Generation**: Create trading signals from anomalies
4. **Backtesting**: Validate signals against historical performance
5. **Real-time Monitoring**: Live anomaly detection system

---

## 🛠️ Development

### Requirements

- Python 3.8+
- pandas, numpy
- AWS CLI (for Polygon flat files)
- Polygon API credentials

### Setup

```bash
# Install dependencies
pip install pandas numpy

# Configure AWS credentials for Polygon
export AWS_ACCESS_KEY_ID="your_polygon_aws_key"
export AWS_SECRET_ACCESS_KEY="your_polygon_aws_secret"
```

---

## 📝 Notes

- **Historical OI Limitation**: Polygon flat files don't include historical Open Interest
- **Proxy Solution**: OI proxy provides realistic estimates for anomaly detection
- **Data Quality**: Validated against known market patterns and volume distributions
- **Scalability**: Designed to handle multi-year datasets efficiently

---

*This system provides a solid foundation for options anomaly detection with realistic OI proxy calculations and comprehensive historical data coverage.*