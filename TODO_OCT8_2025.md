# TODO for October 8, 2025

## Quote Fetching Tasks

### 1. Fetch ATM Quotes (10 minutes)
```bash
source venv/bin/activate
python3 scripts/options_analysis/fetch_atm_quotes.py 2025-10-06
```

**Purpose:** Get true VIX term structure from ATM options
- ~1,161 unique ATM tickers (SPY: 1,050, VIX: 111)
- Runtime: ~10 minutes at 2 calls/sec
- Output: `trade_and_quote_data/cache/atm_quotes_2025-10-06.json`

### 2. Analyze with ATM Quotes (instant)
```bash
python3 scripts/options_analysis/analyze_with_quotes.py 2025-10-06
```

**Expected Output:**
- True VIX term structure (not volume-weighted approximation)
- ATM volatility levels
- Initial trade direction analysis

### 3. (Optional) Fetch Smart Quotes for Trade Direction (45 minutes)
```bash
python3 scripts/options_analysis/fetch_smart_quotes.py 2025-10-06
```

**Purpose:** Get trade direction (BTO/STO) for dealer positioning
- ~15,000-20,000 API calls with smart bucketing
- Runtime: 30-45 minutes
- High-volume tickers prioritized

### 4. Final Analysis
```bash
python3 scripts/options_analysis/analyze_with_quotes.py 2025-10-06
```

**Expected Output:**
- Complete VIX term structure
- Trade direction breakdown (BTO vs STO)
- Dealer positioning (gamma exposure)
- Updated risk assessment

---

## Scripts Created

✅ `scripts/options_analysis/fetch_atm_quotes.py` - ATM quote fetcher
✅ `scripts/options_analysis/fetch_smart_quotes.py` - Smart quote fetcher with prioritization
✅ `scripts/options_analysis/analyze_with_quotes.py` - Analysis using cached quotes

## Current Status

- Have trades data for Oct 6
- Have 10 test quotes cached
- Ready to fetch full ATM quote set
- Will enable true VIX term structure calculation

## Why This Matters

Current analysis uses volume-weighted strikes as proxy - not true forward VIX.
With real quotes, we can:
1. Calculate actual VIX term structure
2. Determine if market pricing stress (backwardation)
3. Classify trade direction for dealer positioning
4. Update risk assessment with proper data