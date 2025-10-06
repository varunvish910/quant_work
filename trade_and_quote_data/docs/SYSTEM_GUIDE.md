# Options Anomaly Detection System - Complete Guide

## 🎯 How to Use the System

### Quick Start
```bash
# 1. Get today's predictions
python3 user_interface.py

# 2. Run interactive demo
python3 quick_demo.py

# 3. Analyze specific periods
python3 -c "
from user_interface import OptionsAnomalyInterface
interface = OptionsAnomalyInterface()
interface.load_models()
interface.get_latest_data(30)
interface.analyze_period('2024-07-01', '2024-07-31')
"
```

### Daily Workflow
1. **Morning Check**: Run `python3 user_interface.py` → type `today`
2. **Risk Assessment**: Look at probability percentage and risk level
3. **Historical Context**: Check recent trends with `period` command
4. **Decision Making**: Use predictions to inform trading decisions

### Programmatic Usage
```python
from user_interface import OptionsAnomalyInterface

# Initialize
interface = OptionsAnomalyInterface()
interface.load_models()
interface.get_latest_data(60)  # Last 60 days

# Get today's prediction
prediction = interface.predict_today()

# Analyze July 2024 (example period)
interface.analyze_period('2024-07-01', '2024-07-31')
```

## 📊 Understanding the Outputs

### Risk Levels
- **🔴 HIGH (>50%)**: Strong correction signal - consider hedging
- **🟡 MEDIUM (30-50%)**: Elevated risk - monitor closely  
- **🟢 LOW (<30%)**: Normal market conditions

### Ensemble Signals
- **🚨 WARNING**: Multiple detection methods agree
- **✅ NORMAL**: No significant anomalies detected
- **Confidence Levels**: High/Medium/Low based on signal strength

## 📈 Technical Metrics Explained

### Brier Score
**What it is**: Measures how well probability predictions match actual outcomes.

**Formula**: `Brier = (1/N) × Σ(predicted_probability - actual_outcome)²`

**Example**:
- If you predict 70% chance of correction and it happens: `(0.7 - 1)² = 0.09`
- If you predict 70% chance but nothing happens: `(0.7 - 0)² = 0.49`
- Lower scores are better (perfect score = 0.0)

**In our system**: 
- Before calibration: 0.1625
- After calibration: 0.1407 ✅ (14% improvement)

**Why it matters**: Better calibrated probabilities mean when the model says "30% chance", it's actually a 30% chance, not overconfident or underconfident.

### F1 Score
**What it is**: Harmonic mean of precision and recall.

**Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Example**:
- Precision = 31.8% (when model predicts correction, it's right 32% of time)
- Recall = 35.0% (model catches 35% of actual corrections)
- F1 = 2 × (0.318 × 0.350) / (0.318 + 0.350) = 33.3%

**Why use F1**: Balances catching events (recall) vs avoiding false alarms (precision).

### Precision
**What it is**: Of all the times the model predicted a correction, how often was it right?

**Formula**: `Precision = True Positives / (True Positives + False Positives)`

**Example**: Model predicts correction 100 times, 32 are correct → 32% precision

**In trading terms**: "When I get a sell signal, what's the chance I should actually sell?"

### Recall (Sensitivity)
**What it is**: Of all actual corrections, how many did the model catch?

**Formula**: `Recall = True Positives / (True Positives + False Negatives)`

**Example**: 100 actual corrections happen, model catches 35 → 35% recall

**In trading terms**: "When the market crashes, what's the chance I got a warning?"

### Precision-Recall Trade-off
**The dilemma**: You can't optimize both simultaneously
- **Higher precision**: Fewer false alarms, but miss more real events
- **Higher recall**: Catch more events, but more false alarms

**Our results**:
- Enhanced Model: 31.8% precision, 35.0% recall
- Calibrated Model: 19.4% precision, 35.0% recall (fewer false alarms)

## 🔧 System Architecture

### Models Available
1. **Enhanced Model**: 65 features, trained for July 2024 detection
2. **Ensemble System**: Multi-day signal aggregation 
3. **Calibrated Model**: Reduced false positives via probability calibration

### Feature Categories (65 total)
1. **Baseline (8)**: Price, volume, volatility basics
2. **Currency (14)**: USD/JPY, EUR/USD, DXY for carry trade detection
3. **Volatility (25)**: VIX, VVIX, VIX9D term structure
4. **Enhanced Yen (18)**: Specialized carry trade unwind detection

### Data Sources
- **SPY**: Primary equity data
- **Sector ETFs**: XLF, XLK, XLE for sector rotation
- **Currency**: JPY, EUR, DXY from Yahoo Finance
- **Volatility**: VIX family from CBOE

## 🎯 Real-World Applications

### For Day Traders
```python
# Check every morning
if probability > 0.5:
    print("🔴 HIGH RISK - Consider reducing position size")
elif probability > 0.3:
    print("🟡 MEDIUM RISK - Monitor intraday volatility")
else:
    print("🟢 LOW RISK - Normal trading")
```

### For Portfolio Managers
```python
# Weekly risk assessment
high_risk_days = count_days_above_threshold(week_data, 0.4)
if high_risk_days >= 3:
    print("📉 Consider hedging portfolio this week")
```

### For Options Traders
```python
# Volatility regime detection
if vix_expansion_signal and yen_carry_risk > 0.6:
    print("📈 Consider volatility plays - potential regime change")
```

## ⚠️ Limitations & Considerations

### Model Limitations
- **July 2024 Miss**: Current models didn't catch the yen carry trade unwind
- **False Positives**: 68-80% of signals are false alarms
- **Regime Changes**: May need retraining for new market conditions

### Data Limitations
- **Real-time lag**: Using daily close data, not intraday
- **Survivorship bias**: Only analyzing SPY, not broader market
- **Feature stability**: Currency/volatility data availability varies

### Usage Guidelines
1. **Don't rely solely**: Use as one input among many
2. **Backtest first**: Test on your specific use case
3. **Monitor performance**: Track prediction accuracy over time
4. **Regular updates**: Retrain models quarterly

## 🔄 Maintenance & Updates

### Monthly Tasks
- Check model performance metrics
- Review false positive/negative rates
- Update feature importance analysis

### Quarterly Tasks
- Retrain models with new data
- Evaluate new feature candidates
- Calibrate probability thresholds

### Annual Tasks
- Full model architecture review
- Regime change analysis
- Feature engineering updates

## 📞 Getting Help

### Common Issues
1. **"No models found"**: Run training scripts first
2. **"Data loading error"**: Check internet connection for Yahoo Finance
3. **"Prediction error"**: Ensure all feature columns are available

### Model Files Needed
- `enhanced_model_results.pkl`: Main prediction model
- `ensemble_early_warning_results.pkl`: Multi-signal system
- `calibrated_model_results.pkl`: Reduced false positives

### Next Steps
1. Run `python3 user_interface.py` for hands-on experience
2. Try different time periods to understand model behavior
3. Integrate predictions into your existing trading workflow
4. Monitor real-world performance vs historical backtests