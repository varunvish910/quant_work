# SPY Options Anomaly Detection TODO

## 1. Core Anomaly Detection Pipeline

### Feature Engineering
- [x] Calculate daily aggregated metrics from raw options data:
  - [x] Total volume and notional value
  - [x] Put/Call volume ratio  
  - [x] Strike concentration metrics (% volume at top strikes)
  - [x] Volume-weighted average premiums
  - [x] Expiry distribution patterns (weekly vs monthly concentration)
  - [x] Intraday patterns (opening/closing volume concentration)

### Statistical Anomaly Detection
- [x] Implement Z-score based detection (2-3 standard deviations)
- [x] Calculate rolling baselines (20-30 day windows)
- [x] Detect volume spikes (2x-3x baseline)
- [x] Identify unusual P/C ratios (<0.8 or >1.5)
- [x] Flag extreme strike concentrations (>30% at single strike)
- [x] Track day-over-day and week-over-week changes

### Machine Learning Detection
- [x] Use Isolation Forest for multivariate anomaly detection
- [x] Train on historical patterns
- [x] Combine with statistical methods for composite scoring
- [x] Implement contamination parameter tuning
- [x] Additional ML models (One-Class SVM, DBSCAN)

## 2. Technical Indicators Integration

### Price-Based Indicators
- [ ] RSI (Relative Strength Index) - identify overbought/oversold conditions
- [ ] MACD (Moving Average Convergence Divergence) - trend changes
- [ ] Bollinger Bands - volatility expansion/contraction
- [ ] ATR (Average True Range) - volatility measurement
- [ ] Support/Resistance levels - key price zones

### Volume Indicators  
- [ ] OBV (On-Balance Volume) - volume flow direction
- [ ] Volume Rate of Change
- [ ] VWAP (Volume Weighted Average Price)
- [ ] Accumulation/Distribution Line

### Options-Specific Technical Indicators
- [ ] Put/Call ratio moving averages (5, 10, 20-day)
- [ ] Implied Volatility rank and percentile
- [ ] Skew index (OTM puts vs OTM calls IV)
- [ ] Term structure analysis (near vs far expiry IV)
- [ ] Greeks anomalies (unusual gamma/vega concentrations)

### Market Breadth Indicators
- [ ] Options volume breadth (advancing vs declining strikes)
- [ ] Strike dispersion index
- [ ] Moneyness distribution shifts

## 3. Signal Generation & Alerts

### Composite Scoring
- [x] Weight statistical anomalies (40%)
- [x] Weight ML anomalies (30%)
- [ ] Weight technical indicators (30%)
- [x] Create tiered alert system (low/medium/high confidence)

### Trading Signals
- [x] Bullish anomaly patterns (call volume spikes + technical support)
- [x] Bearish anomaly patterns (put accumulation + technical resistance)
- [x] Volatility expansion signals (IV spike + volume surge)
- [x] Mean reversion opportunities (extreme P/C + oversold/overbought)

## 4. Implementation Steps

### Phase 1: Data Preparation
- [x] Load existing SPY options data (parquet files)
- [x] Merge with underlying OHLC data
- [x] Clean and validate data quality
- [x] Create reusable data loading pipeline

### Phase 2: Core Detection
- [x] Implement feature engineering module
- [x] Build statistical anomaly detector
- [x] Add ML-based detection (if >100 days of data)
- [x] Create composite anomaly scoring

### Phase 3: Technical Analysis
- [ ] Calculate price-based technical indicators
- [ ] Add volume-based indicators
- [ ] Implement options-specific metrics
- [ ] Integrate with anomaly scoring

### Phase 4: Output & Visualization
- [x] Generate daily anomaly reports (CSV/JSON)
- [ ] Create visualization dashboard
- [ ] Build alert notification system
- [x] Export actionable trading signals

## 5. Backtesting & Validation

### Performance Metrics
- [x] Accuracy of anomaly detection (precision/recall)
- [ ] Trading signal profitability
- [ ] Risk-adjusted returns (Sharpe ratio)
- [ ] Maximum drawdown analysis
- [ ] Win rate and profit factor

### Optimization
- [ ] Parameter tuning (thresholds, windows, weights)
- [ ] Feature selection and importance ranking
- [ ] Model ensemble optimization
- [ ] Walk-forward analysis

## 6. Production Considerations

### Real-time Processing
- [ ] Stream processing for live data
- [ ] Incremental feature calculation
- [ ] Alert latency optimization
- [ ] Database storage for historical anomalies

### Risk Management
- [ ] Position sizing based on anomaly confidence
- [ ] Stop-loss and take-profit levels
- [ ] Maximum daily exposure limits
- [ ] Correlation with market regime

## Data Structure Reference
Current data fields available:
- `ticker`: Options contract ticker
- `underlying_ticker`: SPY
- `expiration_date`: Contract expiration
- `strike_price`: Strike price
- `contract_type`: call/put
- `shares_per_contract`: 100
- `date`: Trading date
- `underlying_price`: SPY closing price
- `open`, `high`, `low`, `close`: Options OHLC
- `volume`: Options volume
- `vwap`: Volume-weighted average price
- `timestamp`: Unix timestamp

## Notes
- Leverage existing code from `archive/option_anomaly_detection/` directory
- Consider market regime (bull/bear/sideways) in anomaly interpretation
- Account for known events (FOMC, earnings, expiry dates)
- Validate against historical market events for accuracy