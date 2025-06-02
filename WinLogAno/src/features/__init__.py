# src/features/__init__.py

"""
Feature engineering package for transforming processed log data into model-ready features.

This package includes:
- AdvancedFeatureEngineer: For calculating sophisticated, EDA-driven malware features.
- MalwareFeatureExtractor: A wrapper using AdvancedFeatureEngineer, potentially for risk assessment.
- generate_session_level_features: Main function in build_features.py to create a comprehensive
                                   session-based feature set for modeling.
"""

from .advanced_features import AdvancedFeatureEngineer, MalwareFeatureExtractor
from .build_features import generate_session_level_features

__all__ = [
    'AdvancedFeatureEngineer',
    'MalwareFeatureExtractor',
    'generate_session_level_features'
]

print("Features package (advanced_features, build_features) initialized.")