"""
Módulo de predicción de triples (3P) NBA
=======================================

Contiene modelos y features especializados para predicción de triples.
"""

from .model_triples import XGBoost3PTModel, Stacking3PTModel
from .features_triples import ThreePointsFeatureEngineer

__all__ = [
    'XGBoost3PTModel',
    'Stacking3PTModel', 
    'ThreePointsFeatureEngineer'
] 