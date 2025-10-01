# Quick Usage Guide

## 🚀 Getting Started

### **1. Run Quick Analysis**
```bash
# See results for 4 sample dates
python3 see_results.py
```

### **2. Analyze Specific Date**
```bash
# Analyze a specific date (format: YYYYMMDD)
python3 analyze_date.py 20240109
python3 analyze_date.py 20250116
```

### **3. Interactive Analysis**
```bash
# Full interactive tool
python3 options_anomaly_detection/run_analysis.py
```

## 📊 Understanding Results

### **Key Numbers to Watch**
- **Anomaly Rate**: 7-9% = Normal, 10%+ = Unusual
- **Confidence**: 0.7+ = Good, 0.8+ = Excellent
- **Quality**: High = Reliable, Medium = Caution, Low = Avoid
- **Direction**: Bearish = Puts active, Bullish = Calls active

### **Quick Decision Matrix**
| Anomaly Rate | Quality | Confidence | Action |
|--------------|---------|------------|--------|
| 8-12% | High | >0.8 | Strong signal - Trade |
| 8-12% | Medium | >0.7 | Good signal - Monitor |
| 12%+ | Any | Any | High stress - Defensive |
| <8% | Any | Any | Normal - Wait |

## 🎯 Common Scenarios

### **Scenario 1: High Quality Bearish Signal**
```
Direction: bearish
Quality: high
Confidence: 0.85
Anomaly Rate: 11%
```
**Action**: Consider put strategies, defensive positioning

### **Scenario 2: Medium Quality Bullish Signal**
```
Direction: bullish
Quality: medium
Confidence: 0.72
Anomaly Rate: 9%
```
**Action**: Monitor closely, small positions only

### **Scenario 3: Low Quality Signal**
```
Direction: bearish
Quality: low
Confidence: 0.45
Anomaly Rate: 6%
```
**Action**: Ignore, wait for better signal

### **Scenario 4: High Anomaly Rate**
```
Anomaly Rate: 15%
Quality: high
Confidence: 0.90
```
**Action**: Major event likely - Reduce risk, increase hedging

## 🔧 Troubleshooting

### **Common Issues**
1. **"File not found"** → Check date format (YYYYMMDD)
2. **"No features prepared"** → Data missing required columns
3. **"Import error"** → Run `pip install -r requirements.txt`

### **Getting Help**
- Check the main README.md for detailed explanations
- See RESULT_INTERPRETATION.md for signal meanings
- Verify your data format and file paths

## 📈 Pro Tips

1. **Start with `see_results.py`** to get familiar
2. **Use specific dates** with `analyze_date.py` for detailed analysis
3. **Look for patterns** across multiple days
4. **Combine with other analysis** - don't rely on single signals
5. **Keep records** of signal performance

---

**Happy Trading! 🚀**
