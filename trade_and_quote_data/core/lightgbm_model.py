"""LightGBM model implementation"""
import lightgbm as lgb
from core.models import EarlyWarningModel

class LightGBMModel(EarlyWarningModel):
    """LightGBM classifier for early warning"""
    
    def __init__(self):
        super().__init__(model_type='lightgbm')
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
