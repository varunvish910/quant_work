#!/usr/bin/env python3
"""
Analyze Floor Prediction Signals July 16 - August 5, 2024
=========================================================

This analyzes the period from July 16 through the August 5 crash to identify
signals that could have predicted where the floor would be.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('options_anomaly_detection')

from feature_engineering import OptionsFeatureEngine
from anomaly_detection import OptionsAnomalyDetector

def analyze_decline_period():
    """Analyze the decline period for floor prediction signals"""
    
    print(f"🔍 FLOOR PREDICTION ANALYSIS: July 16 - August 5, 2024")
    print("=" * 70)
    
    # Key dates during the decline (weekdays only)
    dates = [
        '20240716',  # Tuesday - start point
        '20240717',  # Wednesday
        '20240718',  # Thursday  
        '20240719',  # Friday
        # Weekend
        '20240722',  # Monday
        '20240723',  # Tuesday
        '20240724',  # Wednesday
        '20240725',  # Thursday
        '20240726',  # Friday
        # Weekend
        '20240729',  # Monday
        '20240730',  # Tuesday
        '20240731',  # Wednesday
        '20240801',  # Thursday
        '20240802',  # Friday
        # Weekend
        '20240805'   # Monday - the crash day
    ]
    
    results = []
    
    for date_str in dates:
        print(f"\n📅 Analyzing {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}...")
        
        try:
            # Load data
            year = date_str[:4]
            month = date_str[4:6]
            file_path = Path(f"data/options_chains/SPY/{year}/{month}/SPY_options_snapshot_{date_str}.parquet")
            
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                continue
                
            df = pd.read_parquet(file_path)
            df['date'] = pd.to_datetime(date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8])
            
            # Process features
            fe = OptionsFeatureEngine()
            df_processed = fe.calculate_oi_features(df)
            df_processed = fe.calculate_volume_features(df_processed)
            df_processed = fe.calculate_price_features(df_processed)
            df_processed = fe.calculate_temporal_features(df_processed)
            df_processed = fe.calculate_anomaly_features(df_processed)
            
            # Detect anomalies
            detector = OptionsAnomalyDetector()
            features = detector.prepare_features(df_processed)
            
            if len(features) == 0:
                continue
                
            detector.fit_models(features, contamination=0.1)
            anomaly_results = detector.ensemble_detection(features)
            signals = detector.generate_signals(df_processed, anomaly_results)
            
            # Extract results
            ensemble_anomalies = anomaly_results.get('ensemble_anomaly', [])
            high_conf_anomalies = anomaly_results.get('high_confidence', [])
            
            spy_price = df['underlying_price'].iloc[0]
            
            # Floor prediction analysis
            result = {
                'date': date_str,
                'date_formatted': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                'spy_price': spy_price,
                'contracts': len(df),
                'anomaly_count': ensemble_anomalies.sum(),
                'anomaly_rate': ensemble_anomalies.mean(),
                'high_conf_count': high_conf_anomalies.sum(),
                'high_conf_rate': high_conf_anomalies.mean(),
                'direction': signals.get('direction', 'neutral'),
                'strength': signals.get('strength', 0),
                'confidence': signals.get('confidence', 0),
                
                # Floor prediction metrics
                'put_concentration': 0,
                'support_strikes': [],
                'max_pain_estimate': 0,
                'put_wall_strength': 0,
                'panic_selling_score': 0,
                'institutional_support_score': 0
            }
            
            # Analyze anomalous contracts for floor signals
            if ensemble_anomalies.sum() > 0:
                anomaly_mask = ensemble_anomalies.astype(bool)
                anomaly_contracts = df_processed[anomaly_mask].copy()
                
                # Put concentration analysis
                anomaly_puts = anomaly_contracts[anomaly_contracts['option_type'] == 'P']
                result['put_concentration'] = len(anomaly_puts) / len(anomaly_contracts) if len(anomaly_contracts) > 0 else 0
                
                # Support strike analysis (high OI put strikes)
                if len(anomaly_puts) > 0:
                    # Find strikes with unusually high put OI
                    strike_oi = anomaly_puts.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
                    top_strikes = strike_oi.head(5)
                    result['support_strikes'] = list(top_strikes.index)
                    result['put_wall_strength'] = top_strikes.iloc[0] if len(top_strikes) > 0 else 0
                
                # Max pain estimation (simplified)
                all_puts = df_processed[df_processed['option_type'] == 'P']
                all_calls = df_processed[df_processed['option_type'] == 'C']
                
                if len(all_puts) > 0 and len(all_calls) > 0:
                    # Find strike with maximum total OI (simplified max pain)
                    strike_totals = df_processed.groupby('strike')['oi_proxy'].sum().sort_values(ascending=False)
                    result['max_pain_estimate'] = strike_totals.index[0] if len(strike_totals) > 0 else spy_price
                
                # Panic selling indicators
                high_vol_anomalies = anomaly_contracts[anomaly_contracts['volume'] > anomaly_contracts['volume'].quantile(0.9)]
                result['panic_selling_score'] = len(high_vol_anomalies) / len(anomaly_contracts) if len(anomaly_contracts) > 0 else 0
                
                # Institutional support indicators (high OI, low volume = support)
                high_oi_low_vol = anomaly_contracts[
                    (anomaly_contracts['oi_proxy'] > anomaly_contracts['oi_proxy'].quantile(0.8)) &
                    (anomaly_contracts['volume'] < anomaly_contracts['volume'].quantile(0.5))
                ]
                result['institutional_support_score'] = len(high_oi_low_vol) / len(anomaly_contracts) if len(anomaly_contracts) > 0 else 0
            
            results.append(result)
            print(f"  ✅ SPY: ${spy_price:.2f}, Anomalies: {result['anomaly_count']} ({result['anomaly_rate']:.1%})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if results:
        print(f"\n{'='*90}")
        print(f"  DECLINE PERIOD ANALYSIS")
        print(f"{'='*90}")
        
        # Summary table
        print(f"{'Date':<12} {'SPY':<8} {'Change':<8} {'Anomalies':<10} {'Put%':<6} {'PutWall':<8} {'Panic':<6} {'Support':<8}")
        print("-" * 90)
        
        prev_price = None
        for result in results:
            change = ""
            if prev_price:
                pct_change = (result['spy_price'] - prev_price) / prev_price * 100
                change = f"{pct_change:+.1f}%"
            
            print(f"{result['date_formatted']:<12} "
                  f"${result['spy_price']:<7.2f} "
                  f"{change:<8} "
                  f"{result['anomaly_count']:<10} "
                  f"{result['put_concentration']:<6.1%} "
                  f"{result['put_wall_strength']:<8,.0f} "
                  f"{result['panic_selling_score']:<6.1%} "
                  f"{result['institutional_support_score']:<8.1%}")
            
            prev_price = result['spy_price']
        
        # Floor prediction analysis
        print(f"\n🎯 FLOOR PREDICTION SIGNALS:")
        
        # Track key metrics over time
        dates_formatted = [r['date_formatted'] for r in results]
        spy_prices = [r['spy_price'] for r in results]
        put_concentrations = [r['put_concentration'] for r in results]
        panic_scores = [r['panic_selling_score'] for r in results]
        support_scores = [r['institutional_support_score'] for r in results]
        
        # Find the actual bottom
        min_price = min(spy_prices)
        min_date_idx = spy_prices.index(min_price)
        bottom_date = dates_formatted[min_date_idx]
        
        print(f"  • Actual bottom: ${min_price:.2f} on {bottom_date}")
        
        # Look for signals before the bottom
        if min_date_idx > 0:
            pre_bottom_results = results[:min_date_idx+1]
            
            # Support level analysis
            print(f"\n📊 SUPPORT LEVEL SIGNALS:")
            for i, result in enumerate(pre_bottom_results[-5:]):  # Last 5 days before bottom
                if result['support_strikes']:
                    strongest_support = min(result['support_strikes'])  # Lowest strike with high OI
                    distance_to_support = result['spy_price'] - strongest_support
                    print(f"    {result['date_formatted']}: SPY ${result['spy_price']:.2f}, "
                          f"Support at ${strongest_support:.0f} "
                          f"({distance_to_support:.0f} points below)")
            
            # Capitulation signals
            print(f"\n🔥 CAPITULATION SIGNALS:")
            max_panic = max(panic_scores[:min_date_idx+1])
            max_panic_idx = panic_scores[:min_date_idx+1].index(max_panic)
            print(f"    Peak panic: {max_panic:.1%} on {dates_formatted[max_panic_idx]}")
            
            max_support = max(support_scores[:min_date_idx+1])
            max_support_idx = support_scores[:min_date_idx+1].index(max_support)
            print(f"    Peak institutional support: {max_support:.1%} on {dates_formatted[max_support_idx]}")
            
            # Recovery signals
            if min_date_idx < len(results) - 1:
                post_bottom = results[min_date_idx+1:]
                print(f"\n🔄 RECOVERY SIGNALS:")
                for result in post_bottom[:3]:  # First 3 days after bottom
                    if result['institutional_support_score'] > 0.3:
                        print(f"    {result['date_formatted']}: Strong institutional support ({result['institutional_support_score']:.1%})")
        
        # Summary insights
        print(f"\n💡 KEY INSIGHTS:")
        avg_put_conc = np.mean(put_concentrations)
        avg_panic = np.mean(panic_scores)
        avg_support = np.mean(support_scores)
        
        print(f"  • Average put concentration: {avg_put_conc:.1%}")
        print(f"  • Average panic selling score: {avg_panic:.1%}")
        print(f"  • Average institutional support: {avg_support:.1%}")
        
        # Look for divergences
        if len(results) > 5:
            recent_support = np.mean(support_scores[-5:])
            early_support = np.mean(support_scores[:5])
            support_trend = recent_support - early_support
            
            print(f"  • Support trend: {support_trend:+.1%} (positive = increasing institutional support)")
    
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    analyze_decline_period()