#!/usr/bin/env python3
from training.multi_target_trainer import MultiTargetTrainer
from core.lightgbm_model import LightGBMModel

print("Training LightGBM model...")
trainer = MultiTargetTrainer(
    model_type='lightgbm',
    feature_sets=['baseline', 'currency', 'volatility'],
    start_date='2000-01-01',
    end_date='2024-12-31'
)

results = trainer.train_all_targets(
    targets=['early_warning'],
    save_models=True
)
print(f"✅ LightGBM trained: {results}")
