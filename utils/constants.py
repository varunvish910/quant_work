"""
Global constants for the SPX Early Warning System

All critical thresholds, dates, and configuration values centralized here.
"""

# =============================================================================
# DATA SPLIT DATES - STRICT TEMPORAL SEPARATION
# =============================================================================

TRAIN_START_DATE = '2000-01-01'
TRAIN_END_DATE = '2022-12-31'
VAL_END_DATE = '2023-12-31'
TEST_END_DATE = '2024-12-31'

# =============================================================================
# TARGET DEFINITION PARAMETERS
# =============================================================================

# Early warning system thresholds
DRAWDOWN_THRESHOLD = 0.05  # 5% drawdown to qualify as "correction"
EARLY_WARNING_DAYS = 3  # Must signal 3-5 days before event
LOOKFORWARD_DAYS = 10  # Look 10 days ahead for potential drawdowns

# Mean reversion thresholds
MEAN_REVERSION_THRESHOLD = 0.03  # 3% bounce threshold
MEAN_REVERSION_DAYS = 5  # Days to detect bounce

# =============================================================================
# APPROVED DATA SOURCES
# =============================================================================

APPROVED_DATA_SOURCES = ['yfinance', 'polygon', 'fred', 'cboe']

FORBIDDEN_PATTERNS = [
    'np.random',
    'make_classification',
    'synthetic',
    'simulated',
    'fake_data'
]

# =============================================================================
# MARKET SYMBOLS
# =============================================================================

# Main index
SPY_SYMBOL = 'SPY'

# Sector ETFs
SECTOR_ETFS = {
    'XLU': 'Utilities',
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLB': 'Materials',
    'XLRE': 'Real Estate'
}

# Rotation Indicator ETFs (for concentration risk analysis)
ROTATION_ETFS = {
    'MAGS': 'Magnificent 7',  # AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
    'RSP': 'Equal Weight S&P 500',  # Breadth indicator
    'QQQ': 'Nasdaq 100',  # Tech concentration
    'QQQE': 'Equal Weight Nasdaq 100'  # Tech breadth
}

# Currency pairs
CURRENCY_SYMBOLS = {
    'USDJPY': 'JPY=X',      # Critical for Yen carry trade detection
    'EURUSD': 'EURUSD=X',   # Euro strength
    'DXY': 'UUP'            # Dollar Index (ETF proxy since DX-Y.NYC unavailable)
}

# Volatility indices
VOLATILITY_SYMBOLS = {
    'VIX': '^VIX',          # VIX Fear & Greed Index
    'VIX9D': '^VIX9D',      # 9-day VIX for term structure
    'VVIX': '^VVIX'         # VIX of VIX (volatility of volatility)
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost parameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 10,
    'random_state': 42,
    'n_jobs': -1
}

# Ensemble voting weights
ENSEMBLE_WEIGHTS = [1, 2]  # [RF weight, XGB weight]

# =============================================================================
# CRITICAL EVENTS FOR VALIDATION (2024)
# =============================================================================

MUST_CATCH_2024_EVENTS = {
    'July Yen Carry Unwind': {
        'date': '2024-08-05',
        'description': 'Yen carry trade unwinding, 3% SPY drop',
        'early_warning_window': ('2024-07-29', '2024-08-02')  # 3-5 days before
    },
    'August VIX Spike': {
        'date': '2024-08-05',
        'description': 'VIX spike to 65+',
        'early_warning_window': ('2024-07-29', '2024-08-02')
    },
    'October Correction': {
        'date': '2024-10-01',
        'description': '5%+ correction',
        'early_warning_window': ('2024-09-24', '2024-09-28')
    }
}

# =============================================================================
# DATA VALIDATION PARAMETERS
# =============================================================================

# Price validation ranges (relaxed for historical data)
PRICE_VALIDATION = {
    'SPY': {'min': 50, 'max': 1000},
    'SECTOR_ETF': {'min': 3, 'max': 500},  # Relaxed min from 10 to 3 for historical XLF, XLU data
    'ROTATION_ETF': {'min': 10, 'max': 500},  # MAGS, RSP, QQQ, QQQE
    'VIX': {'min': 5, 'max': 120},
    'VVIX': {'min': 40, 'max': 300},
    'CURRENCY': {'min': 0.5, 'max': 200}
}

# Data quality thresholds
MIN_DATA_POINTS = 100  # Minimum data points required
MAX_MISSING_PCT = 0.1  # Maximum 10% missing data allowed

# =============================================================================
# FEATURE ENGINEERING PARAMETERS
# =============================================================================

# Rolling window sizes
VOLATILITY_WINDOW = 20  # 20-day rolling volatility
SHORT_MA_WINDOW = 10    # 10-day moving average
MEDIUM_MA_WINDOW = 50   # 50-day moving average
LONG_MA_WINDOW = 200    # 200-day moving average

# Momentum windows
MOMENTUM_SHORT = 5      # 5-day momentum
MOMENTUM_MEDIUM = 10    # 10-day momentum
MOMENTUM_LONG = 20      # 20-day momentum

# =============================================================================
# MODEL REGISTRY PATHS
# =============================================================================

MODEL_REGISTRY_PATH = 'models/trained'
MODEL_METADATA_FILE = 'model_metadata.json'
FEATURE_COLUMNS_FILE = 'feature_columns.json'

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

