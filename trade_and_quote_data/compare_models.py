#!/usr/bin/env python3
import pandas as pd
import joblib
from pathlib import Path

print("="*80)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*80)

models_dir = Path('models/trained')
models = list(models_dir.glob('*.pkl'))

print(f"\nFound {len(models)} models:\n")

results = []
for model_path in sorted(models):
    try:
        model = joblib.load(model_path)
        results.append({
            'Model': model_path.stem,
            'Type': getattr(model, 'model_type', 'unknown'),
            'Size': f"{model_path.stat().st_size / 1024:.1f} KB"
        })
    except:
        pass

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

print("\n" + "="*80)
print("✅ Comparison complete")
print("="*80)
