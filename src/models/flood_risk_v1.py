"""
Rule-Based Flood Risk Prediction Model (Version 1)

This module implements a baseline flood risk assessment using expert-defined rules
and thresholds. It provides quick, interpretable predictions based on:
- Rainfall thresholds (1-day, 7-day accumulations)
- Region elevation
- Historical flood patterns
- Weather conditions

This serves as a baseline model before ML-based approaches.

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import logging
from datetime import datetime, date
from typing import Dict, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FloodRiskLevel(str, Enum):
    """Flood risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FloodRiskAssessment:
    """
    Comprehensive flood risk assessment result

    Attributes:
        region_id: Region identifier
        assessment_date: Date of assessment
        risk_level: Overall flood risk level
        confidence_score: Confidence in prediction (0-1)
        risk_score: Numerical risk score (0-100)
        contributing_factors: Dictionary of factors and their weights
        recommendation: Actionable recommendation text
        triggered_rules: List of rule names that triggered
        model_version: Model version identifier
    """
    region_id: str
    assessment_date: date
    risk_level: FloodRiskLevel
    confidence_score: float
    risk_score: float
    contributing_factors: Dict[str, float]
    recommendation: str
    triggered_rules: List[str] = field(default_factory=list)
    model_version: str = "v1.0_rule_based"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        return {
            "region_id": self.region_id,
            "assessment_date": self.assessment_date.isoformat() if isinstance(self.assessment_date, date) else self.assessment_date,
            "risk_level": self.risk_level.value,
            "confidence_score": self.confidence_score,
            "risk_score": self.risk_score,
            "contributing_factors": self.contributing_factors,
            "recommendation": self.recommendation,
            "triggered_rules": self.triggered_rules,
            "model_version": self.model_version,
            "metadata": self.metadata
        }


class WeatherFeatures(BaseModel):
    """
    Input features for flood risk prediction

    All features are optional to handle missing data gracefully
    """
    # Required identifiers
    region_id: str = Field(..., description="Region identifier")
    assessment_date: date = Field(default_factory=date.today, description="Date of assessment")

    # Rainfall features
    rainfall_1d: float = Field(0.0, description="Today's rainfall (mm)")
    rainfall_3d: float = Field(0.0, description="3-day accumulated rainfall (mm)")
    rainfall_7d: float = Field(0.0, description="7-day accumulated rainfall (mm)")
    rainfall_14d: float = Field(0.0, description="14-day accumulated rainfall (mm)")

    # Temperature features
    temperature_avg: float = Field(28.0, description="Average temperature (°C)")
    temperature_high: Optional[float] = Field(None, description="High temperature (°C)")
    temperature_low: Optional[float] = Field(None, description="Low temperature (°C)")

    # Wind features
    wind_speed: float = Field(0.0, description="Wind speed (km/h)")
    wind_speed_max_7d: Optional[float] = Field(None, description="Max wind in 7 days (km/h)")

    # Geographic features
    elevation: float = Field(100.0, description="Region elevation (m)")
    latitude: Optional[float] = Field(None, description="Latitude")
    longitude: Optional[float] = Field(None, description="Longitude")

    # Historical features
    historical_flood_count: int = Field(0, description="Historical floods in region")
    region_vulnerability_score: float = Field(0.0, description="Regional vulnerability (0-100)")

    # Derived features
    rainy_days_7d: int = Field(0, description="Number of rainy days in past 7 days")
    heavy_rain_days_7d: int = Field(0, description="Number of heavy rain days in past 7 days")
    soil_moisture_proxy: float = Field(0.0, description="Soil moisture estimate")

    # Seasonal features
    is_typhoon_season: bool = Field(False, description="Is it typhoon season")
    month: int = Field(1, description="Month of year (1-12)")

    @field_validator('rainfall_1d', 'rainfall_3d', 'rainfall_7d', 'rainfall_14d')
    @classmethod
    def validate_rainfall(cls, v: float) -> float:
        """Ensure rainfall is non-negative"""
        return max(0.0, v)

    @field_validator('elevation')
    @classmethod
    def validate_elevation(cls, v: float) -> float:
        """Ensure elevation is reasonable"""
        return max(0.0, min(v, 3000.0))

    class Config:
        use_enum_values = True


class RuleBasedFloodModel:
    """
    Rule-based flood risk prediction model

    This model uses expert-defined thresholds and rules to assess flood risk.
    It's designed to be:
    - Fast and lightweight
    - Interpretable and explainable
    - Useful as a baseline for ML models

    Rules are based on Philippine meteorological standards and flood patterns.
    """

    # Rainfall thresholds (mm) - Based on PAGASA rainfall advisories
    CRITICAL_RAINFALL_1D = 150.0  # Torrential rain
    HIGH_RAINFALL_1D = 100.0      # Heavy rain
    MEDIUM_RAINFALL_1D = 50.0     # Moderate rain

    CRITICAL_RAINFALL_7D = 400.0  # Extreme accumulation
    HIGH_RAINFALL_7D = 250.0      # High accumulation
    MEDIUM_RAINFALL_7D = 150.0    # Moderate accumulation

    # Elevation thresholds (meters)
    LOW_ELEVATION = 50.0          # Highly flood-prone
    MEDIUM_ELEVATION = 100.0      # Moderately flood-prone

    # Wind speed thresholds (km/h)
    STORM_WIND = 100.0            # Tropical storm
    HIGH_WIND = 60.0              # Strong winds

    def __init__(self):
        """Initialize the rule-based model"""
        self.version = "v1.0"
        logger.info(f"Initialized RuleBasedFloodModel {self.version}")

    def _evaluate_rainfall_rules(self, features: WeatherFeatures) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Evaluate rainfall-based rules

        Returns:
            Tuple of (score, factors, triggered_rules)
        """
        score = 0.0
        factors = {}
        triggered = []

        # Rule 1: Critical daily rainfall
        if features.rainfall_1d >= self.CRITICAL_RAINFALL_1D:
            score += 35
            factors['critical_daily_rainfall'] = 0.85
            triggered.append('RULE_CRITICAL_DAILY_RAIN')
        elif features.rainfall_1d >= self.HIGH_RAINFALL_1D:
            score += 25
            factors['high_daily_rainfall'] = 0.65
            triggered.append('RULE_HIGH_DAILY_RAIN')
        elif features.rainfall_1d >= self.MEDIUM_RAINFALL_1D:
            score += 12
            factors['moderate_daily_rainfall'] = 0.40
            triggered.append('RULE_MODERATE_DAILY_RAIN')

        # Rule 2: Critical accumulated rainfall
        if features.rainfall_7d >= self.CRITICAL_RAINFALL_7D:
            score += 30
            factors['critical_accumulated_rainfall'] = 0.80
            triggered.append('RULE_CRITICAL_7D_RAIN')
        elif features.rainfall_7d >= self.HIGH_RAINFALL_7D:
            score += 20
            factors['high_accumulated_rainfall'] = 0.60
            triggered.append('RULE_HIGH_7D_RAIN')
        elif features.rainfall_7d >= self.MEDIUM_RAINFALL_7D:
            score += 10
            factors['moderate_accumulated_rainfall'] = 0.40
            triggered.append('RULE_MODERATE_7D_RAIN')

        # Rule 3: Rainfall intensity (high daily relative to weekly)
        if features.rainfall_7d > 0:
            intensity = features.rainfall_1d / features.rainfall_7d
            if intensity > 0.5:  # >50% of weekly rain in one day
                score += 10
                factors['rainfall_intensity'] = 0.50
                triggered.append('RULE_HIGH_INTENSITY')

        # Rule 4: Heavy rain days
        if features.heavy_rain_days_7d >= 3:
            score += 8
            factors['persistent_heavy_rain'] = 0.45
            triggered.append('RULE_PERSISTENT_HEAVY_RAIN')

        return score, factors, triggered

    def _evaluate_geographic_rules(self, features: WeatherFeatures) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Evaluate geography-based rules

        Returns:
            Tuple of (score, factors, triggered_rules)
        """
        score = 0.0
        factors = {}
        triggered = []

        # Rule 5: Low elevation (flood-prone)
        if features.elevation < self.LOW_ELEVATION:
            score += 15
            factors['very_low_elevation'] = 0.70
            triggered.append('RULE_VERY_LOW_ELEVATION')
        elif features.elevation < self.MEDIUM_ELEVATION:
            score += 7
            factors['low_elevation'] = 0.40
            triggered.append('RULE_LOW_ELEVATION')

        # Rule 6: Combined elevation and rainfall
        if features.elevation < self.MEDIUM_ELEVATION and features.rainfall_7d > self.MEDIUM_RAINFALL_7D:
            score += 5
            factors['lowland_with_rain'] = 0.35
            triggered.append('RULE_LOWLAND_RAIN_COMBO')

        return score, factors, triggered

    def _evaluate_historical_rules(self, features: WeatherFeatures) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Evaluate historical pattern rules

        Returns:
            Tuple of (score, factors, triggered_rules)
        """
        score = 0.0
        factors = {}
        triggered = []

        # Rule 7: High historical flood frequency
        if features.historical_flood_count > 5:
            score += 10
            factors['flood_prone_area'] = 0.60
            triggered.append('RULE_FLOOD_PRONE_AREA')
        elif features.historical_flood_count > 2:
            score += 5
            factors['some_flood_history'] = 0.30
            triggered.append('RULE_SOME_FLOOD_HISTORY')

        # Rule 8: Region vulnerability score
        if features.region_vulnerability_score > 70:
            score += 8
            factors['high_vulnerability'] = 0.50
            triggered.append('RULE_HIGH_VULNERABILITY')
        elif features.region_vulnerability_score > 40:
            score += 4
            factors['moderate_vulnerability'] = 0.25
            triggered.append('RULE_MODERATE_VULNERABILITY')

        return score, factors, triggered

    def _evaluate_seasonal_rules(self, features: WeatherFeatures) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Evaluate seasonal and weather pattern rules

        Returns:
            Tuple of (score, factors, triggered_rules)
        """
        score = 0.0
        factors = {}
        triggered = []

        # Rule 9: Typhoon season + high rainfall
        if features.is_typhoon_season and features.rainfall_7d > self.MEDIUM_RAINFALL_7D:
            score += 7
            factors['typhoon_season_risk'] = 0.40
            triggered.append('RULE_TYPHOON_SEASON')

        # Rule 10: High winds (potential storm)
        if features.wind_speed_max_7d and features.wind_speed_max_7d >= self.STORM_WIND:
            score += 12
            factors['storm_conditions'] = 0.55
            triggered.append('RULE_STORM_CONDITIONS')
        elif features.wind_speed >= self.HIGH_WIND:
            score += 5
            factors['high_winds'] = 0.30
            triggered.append('RULE_HIGH_WINDS')

        # Rule 11: Soil saturation (high moisture proxy)
        if features.soil_moisture_proxy > 15:
            score += 6
            factors['soil_saturation'] = 0.35
            triggered.append('RULE_SOIL_SATURATION')

        return score, factors, triggered

    def _determine_risk_level(self, total_score: float) -> Tuple[FloodRiskLevel, str]:
        """
        Determine risk level and recommendation from score

        Args:
            total_score: Cumulative risk score

        Returns:
            Tuple of (risk_level, recommendation)
        """
        if total_score >= 70:
            return (
                FloodRiskLevel.CRITICAL,
                "URGENT: High flood risk detected. Harvest crops immediately if possible. "
                "Prepare emergency evacuation plans. Secure livestock and equipment."
            )
        elif total_score >= 50:
            return (
                FloodRiskLevel.HIGH,
                "WARNING: Elevated flood risk. Prepare for early harvest within 24-48 hours. "
                "Monitor weather updates closely. Secure valuable equipment to higher ground."
            )
        elif total_score >= 30:
            return (
                FloodRiskLevel.MEDIUM,
                "CAUTION: Moderate flood risk. Monitor weather conditions closely. "
                "Plan contingency measures. Be prepared to accelerate harvest if conditions worsen."
            )
        else:
            return (
                FloodRiskLevel.LOW,
                "Normal conditions. Continue regular farming schedule. "
                "Maintain awareness of weather forecasts."
            )

    def predict(self, features: WeatherFeatures) -> FloodRiskAssessment:
        """
        Predict flood risk based on rules

        Args:
            features: Weather and geographic features

        Returns:
            FloodRiskAssessment with risk level and details
        """
        logger.debug(f"Predicting flood risk for region {features.region_id} on {features.assessment_date}")

        # Evaluate all rule categories
        rainfall_score, rainfall_factors, rainfall_rules = self._evaluate_rainfall_rules(features)
        geo_score, geo_factors, geo_rules = self._evaluate_geographic_rules(features)
        hist_score, hist_factors, hist_rules = self._evaluate_historical_rules(features)
        season_score, season_factors, season_rules = self._evaluate_seasonal_rules(features)

        # Combine scores and factors
        total_score = rainfall_score + geo_score + hist_score + season_score
        all_factors = {**rainfall_factors, **geo_factors, **hist_factors, **season_factors}
        all_triggered_rules = rainfall_rules + geo_rules + hist_rules + season_rules

        # Determine risk level and recommendation
        risk_level, recommendation = self._determine_risk_level(total_score)

        # Calculate confidence (based on number of triggered rules and data completeness)
        confidence = min(0.5 + (len(all_triggered_rules) * 0.08), 1.0)

        # Create assessment
        assessment = FloodRiskAssessment(
            region_id=features.region_id,
            assessment_date=features.assessment_date,
            risk_level=risk_level,
            confidence_score=round(confidence, 3),
            risk_score=round(min(total_score, 100.0), 2),
            contributing_factors=all_factors,
            recommendation=recommendation,
            triggered_rules=all_triggered_rules,
            model_version=f"v1.0_rule_based",
            metadata={
                "rainfall_score": rainfall_score,
                "geographic_score": geo_score,
                "historical_score": hist_score,
                "seasonal_score": season_score,
                "rules_triggered": len(all_triggered_rules)
            }
        )

        logger.info(
            f"Flood risk assessment for {features.region_id}: "
            f"{risk_level.value.upper()} (score: {total_score:.1f}, confidence: {confidence:.2f})"
        )

        return assessment

    def batch_predict(self, features_list: List[WeatherFeatures]) -> List[FloodRiskAssessment]:
        """
        Predict flood risk for multiple regions

        Args:
            features_list: List of WeatherFeatures for different regions

        Returns:
            List of FloodRiskAssessment results
        """
        logger.info(f"Running batch prediction for {len(features_list)} regions")

        assessments = []
        for features in features_list:
            try:
                assessment = self.predict(features)
                assessments.append(assessment)
            except Exception as e:
                logger.error(f"Failed to predict for region {features.region_id}: {str(e)}")
                continue

        logger.info(f"Completed {len(assessments)} predictions")
        return assessments

    def explain_prediction(self, assessment: FloodRiskAssessment) -> str:
        """
        Generate human-readable explanation of prediction

        Args:
            assessment: FloodRiskAssessment to explain

        Returns:
            Formatted explanation string
        """
        explanation = [
            f"\nFlood Risk Assessment - Region {assessment.region_id}",
            f"{'='*60}",
            f"Date: {assessment.assessment_date}",
            f"Risk Level: {assessment.risk_level.value.upper()}",
            f"Risk Score: {assessment.risk_score}/100",
            f"Confidence: {assessment.confidence_score*100:.1f}%",
            f"\nTriggered Rules ({len(assessment.triggered_rules)}):",
        ]

        for rule in assessment.triggered_rules:
            explanation.append(f"  - {rule}")

        explanation.append(f"\nContributing Factors:")
        for factor, weight in sorted(assessment.contributing_factors.items(), key=lambda x: x[1], reverse=True):
            explanation.append(f"  - {factor}: {weight:.2f}")

        explanation.append(f"\nRecommendation:")
        explanation.append(f"  {assessment.recommendation}")
        explanation.append(f"{'='*60}\n")

        return "\n".join(explanation)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Critical risk scenario
    print("Example 1: Critical Flood Risk")
    critical_features = WeatherFeatures(
        region_id="REG001",
        rainfall_1d=160.0,
        rainfall_7d=450.0,
        elevation=35.0,
        historical_flood_count=8,
        is_typhoon_season=True,
        wind_speed=85.0
    )

    model = RuleBasedFloodModel()
    assessment = model.predict(critical_features)
    print(model.explain_prediction(assessment))

    # Example 2: Low risk scenario
    print("\nExample 2: Low Flood Risk")
    low_features = WeatherFeatures(
        region_id="REG002",
        rainfall_1d=5.0,
        rainfall_7d=25.0,
        elevation=250.0,
        historical_flood_count=0,
        is_typhoon_season=False,
        wind_speed=15.0
    )

    assessment = model.predict(low_features)
    print(model.explain_prediction(assessment))
