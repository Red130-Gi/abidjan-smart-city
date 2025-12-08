"""
ML Module for Traffic Predictions
"""
from .prediction_service import TrafficPredictionService, prediction_service

try:
    from .real_predictor import RealMLPredictor
    from .ensemble_model import TrafficEnsemble, ensemble_predictor
except ImportError:
    pass

__all__ = ['TrafficPredictionService', 'prediction_service', 'RealMLPredictor', 'TrafficEnsemble', 'ensemble_predictor']
