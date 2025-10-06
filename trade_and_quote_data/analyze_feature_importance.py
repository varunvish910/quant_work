#!/usr/bin/env python3
"""
Analyze Feature Importance - Check if volatility is overweighted
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
import json

print("=" * 80)
print("🔍 ANALYZING FEATURE IMPORTANCE")
print("=" * 80)
print()

# Load the trained model
model = lgb.Booster(model_file='models/trained/early_warning_2pct_3to5d.txt')
print("✅ Loaded model")

# Load feature names
with open('models/trained/early_warning_2pct_3to5d_features.json', 'r') as f:
    feature_config = json.load(f)

feature_names = feature_config['features']
print(f"✅ Loaded {len(feature_names)} feature names")
print()

# Get feature importance
importance = model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("=" * 80)
print("📊 TOP 20 MOST IMPORTANT FEATURES")
print("=" * 80)
print()
print(feature_importance.head(20).to_string(index=False))
print()

# Categorize features
volatility_features = [f for f in feature_names if 'vix' in f.lower() or 'vvix' in f.lower() or 'vol' in f.lower()]
technical_features = [f for f in feature_names if any(x in f.lower() for x in ['ma', 'rsi', 'macd', 'adx', 'bb', 'sma', 'ema', 'momentum', 'regime'])]
options_features = [f for f in feature_names if any(x in f.lower() for x in ['iv', 'skew', 'put_call', 'implied', 'option'])]
sector_features = [f for f in feature_names if any(x in f.lower() for x in ['xl', 'sector', 'rotation', 'mags', 'rsp', 'qqqe'])]
currency_features = [f for f in feature_names if any(x in f.lower() for x in ['usd', 'jpy', 'eur', 'dxy', 'currency'])]

# Calculate importance by category
def get_category_importance(features):
    return feature_importance[feature_importance['feature'].isin(features)]['importance'].sum()

volatility_importance = get_category_importance(volatility_features)
technical_importance = get_category_importance(technical_features)
options_importance = get_category_importance(options_features)
sector_importance = get_category_importance(sector_features)
currency_importance = get_category_importance(currency_features)

total_importance = feature_importance['importance'].sum()

print("=" * 80)
print("📊 FEATURE IMPORTANCE BY CATEGORY")
print("=" * 80)
print()
print(f"Volatility Features ({len(volatility_features)} features):")
print(f"  Total Importance: {volatility_importance:,.0f}")
print(f"  % of Total: {volatility_importance/total_importance*100:.1f}%")
print()
print(f"Technical Features ({len(technical_features)} features):")
print(f"  Total Importance: {technical_importance:,.0f}")
print(f"  % of Total: {technical_importance/total_importance*100:.1f}%")
print()
print(f"Options Features ({len(options_features)} features):")
print(f"  Total Importance: {options_importance:,.0f}")
print(f"  % of Total: {options_importance/total_importance*100:.1f}%")
print()
print(f"Sector Features ({len(sector_features)} features):")
print(f"  Total Importance: {sector_importance:,.0f}")
print(f"  % of Total: {sector_importance/total_importance*100:.1f}%")
print()
print(f"Currency Features ({len(currency_features)} features):")
print(f"  Total Importance: {currency_importance:,.0f}")
print(f"  % of Total: {currency_importance/total_importance*100:.1f}%")
print()

# Check if volatility is overweighted
if volatility_importance / total_importance > 0.4:
    print("🚨 WARNING: Volatility features account for >40% of importance!")
    print("   This may cause the model to:")
    print("   - Flag risk whenever VIX spikes (even if it's already correcting)")
    print("   - Miss pullbacks that happen in low-volatility environments")
    print("   - Generate false positives during volatility expansions")
    print()
elif volatility_importance / total_importance > 0.3:
    print("⚠️  CAUTION: Volatility features account for >30% of importance")
    print("   Consider rebalancing feature weights")
    print()
else:
    print("✅ Volatility features are reasonably weighted")
    print()

# Show top volatility features
print("=" * 80)
print("📊 TOP VOLATILITY FEATURES")
print("=" * 80)
print()
vol_features_df = feature_importance[feature_importance['feature'].isin(volatility_features)].head(10)
print(vol_features_df.to_string(index=False))
print()

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Top 20 Features
ax1 = axes[0, 0]
top_20 = feature_importance.head(20)
colors = ['red' if f in volatility_features else 
          'blue' if f in technical_features else
          'green' if f in options_features else
          'orange' if f in sector_features else
          'purple' for f in top_20['feature']]

ax1.barh(range(len(top_20)), top_20['importance'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['feature'], fontsize=9)
ax1.set_xlabel('Importance (Gain)', fontsize=11, fontweight='bold')
ax1.set_title('Top 20 Most Important Features', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, label='Volatility'),
    Patch(facecolor='blue', alpha=0.7, label='Technical'),
    Patch(facecolor='green', alpha=0.7, label='Options'),
    Patch(facecolor='orange', alpha=0.7, label='Sector'),
    Patch(facecolor='purple', alpha=0.7, label='Currency')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Plot 2: Category Importance Pie Chart
ax2 = axes[0, 1]
categories = ['Volatility', 'Technical', 'Options', 'Sector', 'Currency']
importances = [volatility_importance, technical_importance, options_importance, 
               sector_importance, currency_importance]
colors_pie = ['red', 'blue', 'green', 'orange', 'purple']

wedges, texts, autotexts = ax2.pie(importances, labels=categories, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90, 
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Feature Importance by Category', fontsize=13, fontweight='bold')

# Plot 3: Feature Count vs Importance
ax3 = axes[1, 0]
category_counts = [len(volatility_features), len(technical_features), len(options_features),
                   len(sector_features), len(currency_features)]
importance_pcts = [imp/total_importance*100 for imp in importances]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, category_counts, width, label='# Features', 
               color='lightblue', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x + width/2, importance_pcts, width, label='% Importance', 
               color='lightcoral', alpha=0.7, edgecolor='black')

ax3.set_ylabel('Count / Percentage', fontsize=11, fontweight='bold')
ax3.set_title('Feature Count vs Importance Share', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=10)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# Plot 4: Cumulative Importance
ax4 = axes[1, 1]
feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)
cumulative_importance = np.cumsum(feature_importance_sorted['importance']) / total_importance * 100

ax4.plot(range(len(cumulative_importance)), cumulative_importance, 'b-', linewidth=2)
ax4.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% importance')
ax4.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='80% importance')
ax4.axhline(y=95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% importance')

# Find how many features needed for 50%, 80%, 95%
n_50 = np.argmax(cumulative_importance >= 50) + 1
n_80 = np.argmax(cumulative_importance >= 80) + 1
n_95 = np.argmax(cumulative_importance >= 95) + 1

ax4.scatter([n_50, n_80, n_95], [50, 80, 95], s=100, c=['red', 'orange', 'green'], 
           zorder=5, edgecolors='black', linewidth=2)
ax4.text(n_50, 50, f' {n_50} features', fontsize=10, va='center')
ax4.text(n_80, 80, f' {n_80} features', fontsize=10, va='center')
ax4.text(n_95, 95, f' {n_95} features', fontsize=10, va='center')

ax4.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cumulative Importance (%)', fontsize=11, fontweight='bold')
ax4.set_title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xlim([0, len(feature_names)])
ax4.set_ylim([0, 100])

plt.suptitle('Feature Importance Analysis: Is Volatility Overweighted?', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save
output_dir = Path('output')
fig_path = output_dir / 'feature_importance_analysis.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization: {fig_path}")

print()
print("=" * 80)
print("🎯 RECOMMENDATIONS")
print("=" * 80)
print()

if volatility_importance / total_importance > 0.35:
    print("RECOMMENDATION: Reduce volatility feature weight")
    print()
    print("Options:")
    print("  1. Remove redundant volatility features")
    print("     - Keep: VIX, VIX term structure")
    print("     - Remove: VVIX, VIX percentiles, VIX changes")
    print()
    print("  2. Add feature constraints in LightGBM")
    print("     - Use feature_fraction=0.8 (random feature sampling)")
    print("     - Use max_depth=5 (prevent deep trees on single features)")
    print()
    print("  3. Manually reweight features")
    print("     - Multiply volatility features by 0.5-0.7")
    print("     - Increase weight on technical/sector features")
    print()
    print("  4. Train separate models")
    print("     - Model A: Without volatility features")
    print("     - Model B: With volatility features")
    print("     - Ensemble with equal weights")
    print()
else:
    print("✅ Feature weights look reasonable")
    print()
    print("However, you can still improve by:")
    print("  - Adding more technical features (momentum divergence)")
    print("  - Adding market breadth features")
    print("  - Using feature selection to remove noise")
    print()

print(f"Top {n_50} features account for 50% of importance")
print(f"Top {n_80} features account for 80% of importance")
print(f"Consider training with only top {n_80} features to reduce noise")
print()
