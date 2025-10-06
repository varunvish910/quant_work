# 2024 Target Events - What We're Trying to Predict

## Target Definition
**Predict 2%+ pullbacks that occur 3-5 days in advance**

---

## Major Pullback Periods in 2024

### 1. **April Pullback** (April 15 - May 2)
- **Peak**: March 27, 2024 at $513.65
- **Trough**: $486.15
- **Max Drawdown**: -5.35%
- **Signal Window**: April 8-12 (we should have signaled here)
- **What happened**: Tech sector rotation, rising bond yields

### 2. **August Crash** (July 24 - August 14) ⚠️ MAJOR EVENT
- **Peak**: July 16, 2024 at $556.37
- **Trough**: $509.60
- **Max Drawdown**: -8.41% (largest of 2024)
- **Signal Window**: July 31 - August 2 (critical signals)
- **What happened**: Yen carry trade unwind, global market selloff

### 3. **September Pullback** (September 6-9)
- **Peak**: July 16, 2024 at $556.37 (still from July high)
- **Trough**: $532.24
- **Max Drawdown**: -4.34%
- **Signal Window**: August 30 (should have signaled)
- **What happened**: Labor market concerns, Fed uncertainty

### 4. **December Pullback** (December 18-19)
- **Peak**: December 6, 2024 at $600.51
- **Trough**: $579.06
- **Max Drawdown**: -3.57%
- **Signal Window**: December 11-16 (should have signaled)
- **What happened**: Hawkish Fed, rate cut expectations reset

---

## All 25 Pullback Events (2%+ within 3-5 days)

### Q1 2024: 0 events
*No significant pullbacks in Q1*

### Q2 2024: 7 events

#### April Cluster (5 events)
1. **April 8** → Low April 15 (-2.92%)
2. **April 9** → Low April 16 (-3.29%)
3. **April 10** → Low April 17 (-2.92%)
4. **April 11** → Low April 18 (-3.75%)
5. **April 12** → Low April 19 (-3.33%)

#### Late April/May (2 events)
6. **April 29** → Low May 2 (-2.06%)
7. **May 28** → Low May 31 (-2.16%)

### Q3 2024: 11 events (MOST VOLATILE QUARTER)

#### July Cluster (5 events leading to August crash)
8. **July 16** → Low July 19 (-3.00%)
9. **July 17** → Low July 24 (-2.99%)
10. **July 22** → Low July 25 (-3.10%)
11. **July 23** → Low July 30 (-2.76%)
12. **July 26** → Low August 2 (-2.91%)

#### August Crash (3 events)
13. **July 31** → Low August 5 (-7.36%) ⚠️ **CRITICAL**
14. **August 1** → Low August 6 (-4.63%)
15. **August 2** → Low August 7 (-2.79%)

#### Late August/September (3 events)
16. **August 26** → Low September 3 (-2.01%)
17. **August 27** → Low September 4 (-2.15%)
18. **August 28** → Low September 5 (-2.01%)
19. **August 30** → Low September 6 (-4.30%)

### Q4 2024: 7 events

#### October/November (3 events)
20. **October 29** → Low November 4 (-2.39%)
21. **November 11** → Low November 15 (-2.49%)
22. **November 13** → Low November 19 (-2.20%)

#### December Cluster (4 events)
23. **December 11** → Low December 18 (-3.55%)
24. **December 12** → Low December 19 (-3.06%)
25. **December 16** → Low December 20 (-3.94%)

---

## Model Performance Requirements

### Critical Events to Catch (High Priority)
1. **July 31 - August 2**: August crash warning (-7.36% max)
2. **April 8-12**: April pullback (-5.35% max)
3. **August 30**: September pullback (-4.34%)
4. **December 11-16**: December pullback (-3.57%)

**If we catch these 4 clusters, we catch the major risk periods.**

### Current Model Performance (Regularized)
- **Recall**: 32.8% → Catching ~8 out of 25 events
- **Precision**: 57.6% → ~42% false positives

### What This Means
- **Missing ~17 out of 25 events** (67% miss rate)
- **Most critical**: Are we catching the big ones (July 31, April 8-12)?

---

## Key Insights for Improvement

### 1. **Clustering Pattern**
Most pullbacks come in clusters:
- April 8-12: 5 consecutive signals
- July 16-26: 5 consecutive signals
- August 26-30: 4 consecutive signals
- December 11-16: 3 consecutive signals

**Implication**: We don't need to catch every single day, just need to signal ONCE per cluster.

### 2. **Lead Time Reality**
Many events show 7-8 days lead time, not 3-5:
- April cluster: 7-8 days
- September: 7-8 days
- December: 7 days

**Implication**: Our 3-5 day window might be too narrow. Consider 3-7 days.

### 3. **Severity Matters**
The biggest events had the strongest signals:
- **July 31**: -7.36% (August crash)
- **April 11**: -3.75% (April pullback)
- **August 30**: -4.30% (September)
- **December 16**: -3.94% (December)

**Implication**: Focus on predicting SEVERITY, not just occurrence.

### 4. **Quarterly Pattern**
- Q1: 0 events (strong bull market)
- Q2: 7 events (rotation/consolidation)
- Q3: 11 events (peak volatility)
- Q4: 7 events (year-end volatility)

**Implication**: Model should be more sensitive in Q3/Q4.

---

## Success Criteria (Realistic)

### Minimum Viable Performance
- **Catch 3 out of 4 major clusters**: 75% hit rate on critical events
- **Precision**: 60%+ (reduce false positives)
- **Lead time**: 3-7 days (expand window slightly)

### Stretch Goal
- **Catch all 4 major clusters**: 100% hit rate
- **Catch 15+ out of 25 total events**: 60% recall
- **Precision**: 70%+ (high confidence signals)

---

## Next Steps Based on This Analysis

### Immediate Actions
1. **Expand target window to 3-7 days** (not 3-5)
2. **Add severity prediction** (predict drawdown magnitude)
3. **Focus on cluster detection** (don't need every day, just first signal)
4. **Add Q3/Q4 seasonality** (higher sensitivity in volatile quarters)

### Feature Engineering Priorities
1. **Yen carry trade indicators** (for August crash)
2. **Bond yield momentum** (for April pullback)
3. **Fed meeting proximity** (for December)
4. **Volatility regime shifts** (for all clusters)

Would you like me to implement these improvements?
