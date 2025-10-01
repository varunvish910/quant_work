"""
SPY Hedging Risk Summary
========================

This script provides a focused summary of hedging risk concentration
and key findings from the 2025 analysis.

Author: AI Assistant
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_risk_concentration_summary():
    """Analyze and summarize hedging risk concentration patterns"""
    
    # Load the results
    try:
        df = pd.read_csv('hedging_buildup_results.csv')
        print("📊 SPY HEDGING RISK CONCENTRATION SUMMARY")
        print("=" * 50)
        print(f"Analysis Period: {df['date'].min()} to {df['date'].max()}")
        print(f"Total Trading Days: {len(df)}")
        print()
        
        # Current risk status
        latest = df.iloc[-1]
        print("🔍 CURRENT RISK STATUS (September 30, 2025):")
        print("-" * 45)
        print(f"Overall Risk Level: {latest.get('risk_level', 'UNKNOWN')}")
        print(f"Buildup Score: {latest.get('buildup_score', 0):.1f}/100")
        print(f"SPY Price: ${latest.get('spy_price', 0):.2f}")
        print()
        
        # Risk concentration by strike ranges
        print("🎯 RISK CONCENTRATION BY STRIKE RANGES:")
        print("-" * 40)
        
        range_analysis = []
        ranges = [
            ("Near-the-Money", "Near-the-Money"),
            ("5-10% OTM", "5-10% OTM"),
            ("10-20% OTM", "10-20% OTM"),
            ("Deep OTM (>20%)", "Deep OTM (>20%)")
        ]
        
        for display_name, key_prefix in ranges:
            oi_key = f'{key_prefix}_oi'
            oi_pct_key = f'{key_prefix}_oi_pct'
            vol_oi_key = f'{key_prefix}_vol_oi'
            
            if oi_key in latest:
                oi = latest.get(oi_key, 0)
                oi_pct = latest.get(oi_pct_key, 0) * 100
                vol_oi = latest.get(vol_oi_key, 0)
                
                range_analysis.append({
                    'range': display_name,
                    'oi': oi,
                    'oi_pct': oi_pct,
                    'vol_oi': vol_oi
                })
                
                print(f"{display_name:15}: {oi:>10,.0f} OI ({oi_pct:>5.1f}% of total, V/OI: {vol_oi:>5.2f})")
        
        print()
        
        # Key findings
        print("🔍 KEY FINDINGS:")
        print("-" * 15)
        
        # Find the range with highest concentration
        if range_analysis:
            max_range = max(range_analysis, key=lambda x: x['oi_pct'])
            print(f"• Highest concentration: {max_range['range']} ({max_range['oi_pct']:.1f}% of total OI)")
        
        # Check for hedging-like activity (low V/OI ratios)
        hedging_ranges = [r for r in range_analysis if r['vol_oi'] < 0.5]
        if hedging_ranges:
            print(f"• Hedging-like activity detected in {len(hedging_ranges)} strike ranges (V/OI < 0.5)")
            for r in hedging_ranges:
                print(f"  - {r['range']}: V/OI = {r['vol_oi']:.2f}")
        
        # Deep OTM analysis
        deep_otm_oi = latest.get('deep_otm_oi', 0)
        avg_deep_otm = df['deep_otm_oi'].mean()
        deep_otm_pct_change = ((deep_otm_oi - avg_deep_otm) / avg_deep_otm) * 100
        
        print(f"• Deep OTM puts: {deep_otm_oi:,.0f} OI ({deep_otm_pct_change:+.1f}% vs average)")
        
        if deep_otm_pct_change > 20:
            print("  ⚠️  Elevated deep OTM put activity - potential hedging buildup")
        elif deep_otm_pct_change < -20:
            print("  📉 Below average deep OTM put activity")
        else:
            print("  ✅ Normal deep OTM put activity")
        
        print()
        
        # Historical trends
        print("📈 HISTORICAL TRENDS:")
        print("-" * 20)
        
        if 'buildup_score' in df.columns:
            avg_score = df['buildup_score'].mean()
            max_score = df['buildup_score'].max()
            min_score = df['buildup_score'].min()
            recent_avg = df['buildup_score'].tail(10).mean()
            
            print(f"• Average buildup score: {avg_score:.1f}/100")
            print(f"• Range: {min_score:.1f} - {max_score:.1f}")
            print(f"• Recent average (last 10 days): {recent_avg:.1f}")
            
            if recent_avg > avg_score * 1.2:
                print("  📈 RISING trend in hedging activity")
            elif recent_avg < avg_score * 0.8:
                print("  📉 DECLINING trend in hedging activity")
            else:
                print("  ➡️  STABLE trend in hedging activity")
        
        print()
        
        # Risk assessment
        print("⚠️  RISK ASSESSMENT:")
        print("-" * 20)
        
        current_score = latest.get('buildup_score', 0)
        if current_score >= 75:
            risk_level = "HIGH"
            risk_color = "🔴"
            risk_desc = "Significant hedging buildup detected - monitor for potential market weakness"
        elif current_score >= 50:
            risk_level = "MEDIUM"
            risk_color = "🟡"
            risk_desc = "Moderate hedging activity - watch for acceleration"
        elif current_score >= 25:
            risk_level = "LOW"
            risk_color = "🟢"
            risk_desc = "Minimal hedging buildup - normal market conditions"
        else:
            risk_level = "MINIMAL"
            risk_color = "✅"
            risk_desc = "Very low hedging activity - market appears stable"
        
        print(f"{risk_color} Risk Level: {risk_level}")
        print(f"   {risk_desc}")
        
        # Specific concerns
        concerns = []
        
        # Check for unusual concentration
        max_concentration = latest.get('max_strike_oi_pct', 0)
        if max_concentration > 0.20:
            concerns.append(f"High single-strike concentration ({max_concentration:.1%})")
        
        # Check for low V/OI ratios (hedging activity)
        low_vol_oi_ranges = [r for r in range_analysis if r['vol_oi'] < 0.3]
        if len(low_vol_oi_ranges) >= 2:
            concerns.append(f"Multiple ranges showing hedging-like activity (V/OI < 0.3)")
        
        # Check for recent acceleration
        if 'buildup_trend' in df.columns:
            recent_trend = df['buildup_trend'].tail(5).mean()
            if recent_trend > 5:
                concerns.append("Recent acceleration in hedging activity")
        
        if concerns:
            print("\n🚨 SPECIFIC CONCERNS:")
            for concern in concerns:
                print(f"   • {concern}")
        else:
            print("\n✅ No specific concerns identified")
        
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 20)
        
        if current_score >= 50:
            print("• Monitor options flow for signs of institutional hedging")
            print("• Watch for acceleration in put buying activity")
            print("• Consider defensive positioning if trend continues")
        elif current_score >= 25:
            print("• Continue monitoring hedging activity")
            print("• Watch for changes in strike concentration")
        else:
            print("• Current hedging levels are normal")
            print("• Continue regular monitoring")
        
        return df
        
    except FileNotFoundError:
        print("❌ Results file not found. Please run the hedging buildup analyzer first.")
        return None
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None

def create_risk_visualization(df):
    """Create risk concentration visualization"""
    if df is None or df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SPY Hedging Risk Concentration Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Buildup score over time
    if 'buildup_score' in df.columns:
        axes[0, 0].plot(pd.to_datetime(df['date']), df['buildup_score'], 
                       linewidth=2, color='red', alpha=0.7)
        axes[0, 0].set_title('Hedging Buildup Score Over Time')
        axes[0, 0].set_ylabel('Buildup Score (0-100)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add risk level zones
        axes[0, 0].axhspan(0, 25, alpha=0.1, color='green', label='Minimal Risk')
        axes[0, 0].axhspan(25, 50, alpha=0.1, color='yellow', label='Low Risk')
        axes[0, 0].axhspan(50, 75, alpha=0.1, color='orange', label='Medium Risk')
        axes[0, 0].axhspan(75, 100, alpha=0.1, color='red', label='High Risk')
    
    # Plot 2: Strike range concentrations (latest data)
    latest = df.iloc[-1]
    ranges = ['Near-the-Money', '5-10% OTM', '10-20% OTM', 'Deep OTM (>20%)']
    oi_values = []
    oi_pct_values = []
    
    for range_name in ranges:
        oi_key = f'{range_name}_oi'
        oi_pct_key = f'{range_name}_oi_pct'
        oi_values.append(latest.get(oi_key, 0))
        oi_pct_values.append(latest.get(oi_pct_key, 0) * 100)
    
    bars = axes[0, 1].bar(ranges, oi_pct_values, color=['red', 'orange', 'yellow', 'green'])
    axes[0, 1].set_title('Current OI Concentration by Strike Range')
    axes[0, 1].set_ylabel('Percentage of Total OI')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, oi_pct_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Deep OTM put activity over time
    if 'deep_otm_oi' in df.columns:
        axes[1, 0].plot(pd.to_datetime(df['date']), df['deep_otm_oi'], 
                       linewidth=2, color='purple')
        axes[1, 0].set_title('Deep OTM Put Open Interest Over Time')
        axes[1, 0].set_ylabel('Open Interest')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Volume/OI ratios by strike range
    vol_oi_values = []
    for range_name in ranges:
        vol_oi_key = f'{range_name}_vol_oi'
        vol_oi_values.append(latest.get(vol_oi_key, 0))
    
    bars2 = axes[1, 1].bar(ranges, vol_oi_values, color=['red', 'orange', 'yellow', 'green'])
    axes[1, 1].set_title('Volume/OI Ratios by Strike Range')
    axes[1, 1].set_ylabel('V/OI Ratio')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Hedging Threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('hedging_risk_concentration.png', dpi=300, bbox_inches='tight')
    print("📊 Risk concentration visualization saved to 'hedging_risk_concentration.png'")
    plt.show()

if __name__ == "__main__":
    df = analyze_risk_concentration_summary()
    if df is not None:
        create_risk_visualization(df)
