"""
Data models for Project Agri-Safe

This module includes:
- Weather data models (Pydantic)
- Flood risk prediction models (v1: rule-based, v2: ML)
- Training and prediction pipelines

Author: AgriSafe Development Team
"""

from .weather import (
    WeatherForecast,
    WeatherCondition,
    TyphoonAlert,
    FloodRiskAssessment as WeatherFloodRiskAssessment
)

from .flood_risk_v1 import (
    RuleBasedFloodModel,
    FloodRiskLevel,
    FloodRiskAssessment,
    WeatherFeatures
)

from .flood_risk_v2 import MLFloodModel

from .training_pipeline import FloodModelTrainingPipeline

from .batch_predictions import FloodRiskBatchPredictor

__all__ = [
    # Weather models
    'WeatherForecast',
    'WeatherCondition',
    'TyphoonAlert',
    'WeatherFloodRiskAssessment',

    # Flood risk models
    'RuleBasedFloodModel',
    'MLFloodModel',
    'FloodRiskLevel',
    'FloodRiskAssessment',
    'WeatherFeatures',

    # Pipelines
    'FloodModelTrainingPipeline',
    'FloodRiskBatchPredictor'
]
