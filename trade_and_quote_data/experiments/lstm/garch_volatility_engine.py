#!/usr/bin/env python3
"""
GARCH Volatility Engine
Phase 2: Advanced volatility modeling and regime detection

Author: AI Assistant
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import warnings
import logging
from datetime import datetime
from pathlib import Path
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GARCHVolatilityEngine:
    """Advanced GARCH volatility modeling and regime detection"""
    
    def __init__(self):
        self.models = {}
        self.volatility_forecasts = {}
        self.regime_states = {}
        self.performance_metrics = {}
        
    def download_extended_data(self):
        """Download comprehensive market data for GARCH modeling"""
        logger.info("Downloading extended market data for GARCH modeling...")
        
        # Extended asset universe for volatility modeling
        symbols = ['SPY', '^VIX', '^TNX', 'GLD', 'TLT', 'QQQ', 'IWM']
        
        data = {}
        for symbol in symbols:
            try:
                df = yf.download(symbol, start='2016-01-01', end='2025-01-01', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(symbol, axis=1, level=1)
                data[symbol] = df
                logger.info(f"Downloaded {symbol}: {len(df)} days")
            except Exception as e:
                logger.warning(f"Failed to download {symbol}: {e}")
        
        return data
    
    def prepare_returns_data(self, data):
        """Prepare returns data for GARCH modeling"""
        logger.info("Preparing returns data for GARCH modeling...")
        
        returns = {}
        for symbol, df in data.items():
            if 'Close' in df.columns:
                # Calculate returns
                ret = df['Close'].pct_change().dropna()
                
                # Remove extreme outliers (beyond 5 standard deviations)
                ret = ret[np.abs(ret - ret.mean()) <= (5 * ret.std())]
                
                # Convert to percentage for better numerical stability
                ret = ret * 100
                
                returns[symbol] = ret
                logger.info(f"{symbol} returns prepared: {len(ret)} observations")
        
        return returns
    
    def fit_garch_models(self, returns):
        """Fit GARCH(1,1) models to return series"""
        logger.info("Fitting GARCH models...")
        
        for symbol, ret_series in returns.items():
            try:
                logger.info(f"Fitting GARCH model for {symbol}...")
                
                # Fit GARCH(1,1) model
                model = arch_model(ret_series, vol='Garch', p=1, q=1, rescale=False)
                fitted_model = model.fit(disp='off')
                
                self.models[symbol] = {
                    'model': model,
                    'fitted': fitted_model,
                    'returns': ret_series
                }
                
                # Extract volatility forecasts
                volatility = fitted_model.conditional_volatility
                self.volatility_forecasts[symbol] = volatility
                
                logger.info(f"GARCH model fitted for {symbol}")
                logger.info(f"  AIC: {fitted_model.aic:.2f}")
                logger.info(f"  Log-likelihood: {fitted_model.loglikelihood:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to fit GARCH model for {symbol}: {e}")
    
    def detect_volatility_regimes(self, returns):
        """Detect volatility regimes using rolling statistics"""
        logger.info("Detecting volatility regimes...")
        
        for symbol, ret_series in returns.items():
            if symbol in self.volatility_forecasts:
                vol_forecast = self.volatility_forecasts[symbol]
                
                # Calculate rolling statistics
                vol_ma_short = vol_forecast.rolling(20).mean()
                vol_ma_long = vol_forecast.rolling(60).mean()
                vol_percentile = vol_forecast.rolling(252).rank(pct=True)
                
                # Define regimes
                regimes = pd.Series('Normal', index=vol_forecast.index)
                
                # Low volatility regime
                regimes[vol_percentile < 0.25] = 'Low_Vol'
                
                # High volatility regime
                regimes[vol_percentile > 0.75] = 'High_Vol'
                
                # Crisis regime (extreme volatility)
                regimes[vol_percentile > 0.95] = 'Crisis'
                
                # Trending volatility
                vol_trend = vol_ma_short / vol_ma_long
                regimes[(vol_trend > 1.2) & (vol_percentile > 0.5)] = 'Vol_Expansion'
                regimes[(vol_trend < 0.8) & (vol_percentile < 0.5)] = 'Vol_Contraction'
                
                self.regime_states[symbol] = regimes
                
                # Calculate regime statistics
                regime_stats = regimes.value_counts(normalize=True)
                logger.info(f"{symbol} regime distribution:")
                for regime, pct in regime_stats.items():
                    logger.info(f"  {regime}: {pct:.1%}")
    
    def create_garch_features(self, returns):
        """Create GARCH-based features for predictive modeling"""
        logger.info("Creating GARCH-based features...")
        
        garch_features = {}
        
        for symbol, ret_series in returns.items():
            if symbol in self.volatility_forecasts:
                vol_forecast = self.volatility_forecasts[symbol]
                regimes = self.regime_states.get(symbol, pd.Series('Normal', index=vol_forecast.index))
                
                features = pd.DataFrame(index=vol_forecast.index)
                
                # Volatility level features
                features[f'{symbol}_vol_forecast'] = vol_forecast
                features[f'{symbol}_vol_zscore'] = (vol_forecast - vol_forecast.rolling(252).mean()) / vol_forecast.rolling(252).std()
                features[f'{symbol}_vol_percentile'] = vol_forecast.rolling(252).rank(pct=True)
                
                # Volatility momentum
                features[f'{symbol}_vol_momentum_5d'] = vol_forecast.pct_change(5)
                features[f'{symbol}_vol_momentum_20d'] = vol_forecast.pct_change(20)
                
                # Volatility moving averages
                features[f'{symbol}_vol_ma_ratio'] = vol_forecast / vol_forecast.rolling(20).mean()
                features[f'{symbol}_vol_trend'] = vol_forecast.rolling(20).mean() / vol_forecast.rolling(60).mean()
                
                # Regime indicators
                for regime in regimes.unique():
                    features[f'{symbol}_regime_{regime}'] = (regimes == regime).astype(int)
                
                # Volatility term structure (for VIX)
                if symbol == '^VIX':
                    features[f'{symbol}_term_structure'] = vol_forecast / vol_forecast.rolling(5).mean()
                    features[f'{symbol}_mean_reversion'] = (vol_forecast.rolling(20).mean() - vol_forecast) / vol_forecast.rolling(20).std()
                
                garch_features[symbol] = features.dropna()
                logger.info(f"Created {len(features.columns)} GARCH features for {symbol}")
        
        return garch_features
    
    def forecast_volatility(self, symbol, horizon=5):
        """Generate volatility forecasts"""
        if symbol in self.models:
            fitted_model = self.models[symbol]['fitted']
            
            # Generate forecasts
            forecasts = fitted_model.forecast(horizon=horizon, reindex=False)
            vol_forecast = np.sqrt(forecasts.variance.values[-1, :])
            
            return vol_forecast
        return None
    
    def calculate_volatility_signals(self, garch_features):
        """Calculate volatility-based trading signals"""
        logger.info("Calculating volatility signals...")
        
        signals = {}
        
        for symbol, features in garch_features.items():
            signal_df = pd.DataFrame(index=features.index)
            
            # VIX contango/backwardation signal
            if symbol == '^VIX':
                signal_df['vix_spike_signal'] = (
                    (features[f'{symbol}_vol_zscore'] > 1.5) & 
                    (features[f'{symbol}_vol_momentum_5d'] > 0.2)
                ).astype(int)
                
                signal_df['vix_mean_reversion_signal'] = (
                    (features[f'{symbol}_vol_percentile'] > 0.8) &
                    (features[f'{symbol}_vol_momentum_5d'] < -0.1)
                ).astype(int)
            
            # SPY volatility regime signals
            if symbol == 'SPY':
                signal_df['spy_vol_breakout'] = (
                    features[f'{symbol}_vol_ma_ratio'] > 1.3
                ).astype(int)
                
                signal_df['spy_vol_compression'] = (
                    (features[f'{symbol}_vol_percentile'] < 0.2) &
                    (features[f'{symbol}_vol_trend'] < 0.9)
                ).astype(int)
            
            signals[symbol] = signal_df
        
        return signals
    
    def backtest_volatility_signals(self, signals, returns):
        """Backtest volatility-based signals"""
        logger.info("Backtesting volatility signals...")
        
        backtest_results = {}
        
        for symbol, signal_df in signals.items():
            if symbol in returns:
                ret_series = returns[symbol]
                
                results = {}
                for signal_name in signal_df.columns:
                    signal = signal_df[signal_name]
                    
                    # Align signals with returns (forward-looking)
                    aligned_signal = signal.shift(1).reindex(ret_series.index, fill_value=0)
                    aligned_returns = ret_series.reindex(aligned_signal.index)
                    
                    # Calculate signal performance
                    signal_returns = aligned_returns[aligned_signal == 1]
                    
                    if len(signal_returns) > 5:
                        results[signal_name] = {
                            'n_signals': len(signal_returns),
                            'hit_rate': (signal_returns > 0).mean(),
                            'avg_return': signal_returns.mean(),
                            'sharpe_ratio': signal_returns.mean() / signal_returns.std() if signal_returns.std() > 0 else 0,
                            'max_return': signal_returns.max(),
                            'min_return': signal_returns.min()
                        }
                
                backtest_results[symbol] = results
        
        return backtest_results
    
    def create_composite_volatility_features(self, garch_features):
        """Create composite features across all assets"""
        logger.info("Creating composite volatility features...")
        
        # Find common index
        indices = [df.index for df in garch_features.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
        
        composite = pd.DataFrame(index=common_index)
        
        # Cross-asset volatility relationships
        if 'SPY' in garch_features and '^VIX' in garch_features:
            spy_vol = garch_features['SPY']['SPY_vol_forecast'].reindex(common_index)
            vix_vol = garch_features['^VIX']['^VIX_vol_forecast'].reindex(common_index)
            
            composite['spy_vix_vol_ratio'] = spy_vol / vix_vol
            composite['spy_vix_vol_correlation'] = spy_vol.rolling(20).corr(vix_vol)
        
        # Volatility dispersion across assets
        vol_columns = []
        for symbol, features in garch_features.items():
            vol_col = f'{symbol}_vol_forecast'
            if vol_col in features.columns:
                composite[vol_col] = features[vol_col].reindex(common_index)
                vol_columns.append(vol_col)
        
        if len(vol_columns) > 2:
            composite['vol_dispersion'] = composite[vol_columns].std(axis=1)
            composite['vol_momentum_average'] = composite[vol_columns].pct_change(5).mean(axis=1)
        
        logger.info(f"Created {len(composite.columns)} composite volatility features")
        return composite
    
    def run_garch_analysis(self):
        """Run complete GARCH analysis pipeline"""
        logger.info("Starting GARCH volatility analysis...")
        
        # Download data
        data = self.download_extended_data()
        
        # Prepare returns
        returns = self.prepare_returns_data(data)
        
        # Fit GARCH models
        self.fit_garch_models(returns)
        
        # Detect regimes
        self.detect_volatility_regimes(returns)
        
        # Create features
        garch_features = self.create_garch_features(returns)
        
        # Calculate signals
        signals = self.calculate_volatility_signals(garch_features)
        
        # Backtest signals
        backtest_results = self.backtest_volatility_signals(signals, returns)
        
        # Create composite features
        composite_features = self.create_composite_volatility_features(garch_features)
        
        return {
            'garch_features': garch_features,
            'signals': signals,
            'backtest_results': backtest_results,
            'composite_features': composite_features,
            'models': self.models,
            'regimes': self.regime_states
        }
    
    def save_results(self, results):
        """Save GARCH analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('analysis/outputs/garch_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save composite features
        if 'composite_features' in results:
            features_file = output_dir / f'garch_features_{timestamp}.csv'
            results['composite_features'].to_csv(features_file)
            logger.info(f"GARCH features saved to {features_file}")
        
        # Save backtest results
        if 'backtest_results' in results:
            backtest_file = output_dir / f'garch_backtest_{timestamp}.json'
            with open(backtest_file, 'w') as f:
                json.dump(results['backtest_results'], f, indent=2)
            logger.info(f"Backtest results saved to {backtest_file}")
        
        return output_dir


def main():
    """Main execution function"""
    print("🔥 Starting GARCH Volatility Engine")
    print("Phase 2: Advanced Volatility Modeling")
    print("=" * 60)
    
    # Initialize engine
    garch_engine = GARCHVolatilityEngine()
    
    # Run complete analysis
    results = garch_engine.run_garch_analysis()
    
    # Save results
    output_dir = garch_engine.save_results(results)
    
    # Print summary
    print("\n✅ GARCH Analysis Complete!")
    print(f"Models fitted: {len(results['models'])}")
    print(f"Features created: {len(results['composite_features'].columns)}")
    print(f"Signals analyzed: {sum(len(signals) for signals in results['signals'].values())}")
    print(f"Results saved to: {output_dir}")
    
    # Print performance summary
    print("\n📊 Signal Performance Summary:")
    for symbol, symbol_results in results['backtest_results'].items():
        print(f"\n{symbol}:")
        for signal, metrics in symbol_results.items():
            print(f"  {signal}: {metrics['n_signals']} signals, "
                  f"Hit Rate: {metrics['hit_rate']:.1%}, "
                  f"Avg Return: {metrics['avg_return']:.3f}%")
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n🎉 Phase 2 GARCH Development Complete!")