"""
Core modules for the SPX Early Warning System
"""

from .data_loader import DataLoader, DataIntegrityValidator
from .targets import TargetCreator
from .features import FeatureEngine
from .models import EarlyWarningModel

