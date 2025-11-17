"""
Weather data models for Project Agri-Safe
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class WeatherCondition(str, Enum):
    """Weather condition types"""
    SUNNY = "sunny"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    LIGHT_RAIN = "light_rain"
    MODERATE_RAIN = "moderate_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    TYPHOON = "typhoon"
    UNKNOWN = "unknown"


class FloodRiskLevel(str, Enum):
    """Flood risk levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class WeatherForecast(BaseModel):
    """
    Weather forecast data model
    """
    region_id: int = Field(..., description="ID of the region")
    forecast_date: date = Field(..., description="Date of the forecast")
    forecast_created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this forecast was created"
    )
    temperature_min: Optional[float] = Field(None, description="Minimum temperature in Celsius")
    temperature_max: Optional[float] = Field(None, description="Maximum temperature in Celsius")
    temperature_avg: Optional[float] = Field(None, description="Average temperature in Celsius")
    humidity_percent: Optional[float] = Field(None, description="Humidity percentage")
    rainfall_mm: Optional[float] = Field(None, description="Expected rainfall in millimeters")
    wind_speed_kph: Optional[float] = Field(None, description="Wind speed in km/h")
    weather_condition: Optional[WeatherCondition] = Field(
        WeatherCondition.UNKNOWN,
        description="General weather condition"
    )
    weather_description: Optional[str] = Field(None, description="Detailed weather description")
    data_source: str = Field(default="PAGASA", description="Source of the data")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw API response data")

    @field_validator('temperature_min', 'temperature_max', 'temperature_avg')
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        """Validate temperature is within reasonable range"""
        if v is not None and (v < -10 or v > 60):
            raise ValueError(f"Temperature {v}°C is outside reasonable range (-10 to 60)")
        return v

    @field_validator('humidity_percent')
    @classmethod
    def validate_humidity(cls, v: Optional[float]) -> Optional[float]:
        """Validate humidity is between 0 and 100"""
        if v is not None and (v < 0 or v > 100):
            raise ValueError(f"Humidity {v}% must be between 0 and 100")
        return v

    @field_validator('rainfall_mm')
    @classmethod
    def validate_rainfall(cls, v: Optional[float]) -> Optional[float]:
        """Validate rainfall is non-negative"""
        if v is not None and v < 0:
            raise ValueError(f"Rainfall {v}mm cannot be negative")
        return v

    class Config:
        use_enum_values = True


class TyphoonAlert(BaseModel):
    """
    Typhoon alert data model
    """
    typhoon_name: Optional[str] = Field(None, description="Name of the typhoon")
    alert_level: int = Field(..., description="Alert level (1-5 for TCWS)")
    affected_regions: List[int] = Field(..., description="List of affected region IDs")
    alert_start_date: datetime = Field(..., description="When the alert started")
    alert_end_date: Optional[datetime] = Field(None, description="When the alert ended")
    max_wind_speed_kph: Optional[float] = Field(None, description="Maximum wind speed in km/h")
    expected_rainfall_mm: Optional[float] = Field(None, description="Expected rainfall in mm")
    description: Optional[str] = Field(None, description="Alert description")
    advisory_text: Optional[str] = Field(None, description="Official advisory text")
    data_source: str = Field(default="PAGASA", description="Source of the data")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw API response data")

    @field_validator('alert_level')
    @classmethod
    def validate_alert_level(cls, v: int) -> int:
        """Validate alert level is between 0 and 5"""
        if v < 0 or v > 5:
            raise ValueError(f"Alert level {v} must be between 0 and 5")
        return v

    class Config:
        use_enum_values = True


class FloodRiskAssessment(BaseModel):
    """
    Flood risk assessment data model
    """
    region_id: int = Field(..., description="ID of the region")
    assessment_date: date = Field(..., description="Date of the assessment")
    risk_level: FloodRiskLevel = Field(..., description="Overall flood risk level")
    risk_score: float = Field(..., description="Risk score (0-100)")
    rainfall_forecast_mm: Optional[float] = Field(None, description="Forecasted rainfall in mm")
    historical_flood_probability: Optional[float] = Field(
        None,
        description="Historical probability of flooding (0-100)"
    )
    soil_saturation_index: Optional[float] = Field(
        None,
        description="Soil saturation index (0-100)"
    )
    river_level_status: Optional[str] = Field(None, description="River level status")
    model_version: str = Field(..., description="Version of the risk model used")
    confidence_score: float = Field(..., description="Confidence in the assessment (0-100)")
    factors: Optional[Dict[str, Any]] = Field(None, description="Detailed risk factors")

    @field_validator('risk_score', 'confidence_score', 'historical_flood_probability', 'soil_saturation_index')
    @classmethod
    def validate_percentage(cls, v: Optional[float]) -> Optional[float]:
        """Validate percentage is between 0 and 100"""
        if v is not None and (v < 0 or v > 100):
            raise ValueError(f"Value {v} must be between 0 and 100")
        return v

    class Config:
        use_enum_values = True


class PAGASAResponse(BaseModel):
    """
    Raw PAGASA API response model
    """
    issued_at: Optional[datetime] = Field(None, description="When the forecast was issued")
    synopsis: Optional[str] = Field(None, description="Weather synopsis")
    weather_conditions: Optional[Dict[str, Any]] = Field(None, description="Weather conditions by region")
    wind_conditions: Optional[Dict[str, Any]] = Field(None, description="Wind conditions")
    temperature: Optional[Dict[str, Any]] = Field(None, description="Temperature data")
    humidity: Optional[Dict[str, Any]] = Field(None, description="Humidity data")
    tides: Optional[Dict[str, Any]] = Field(None, description="Tidal information")
    astronomical: Optional[Dict[str, Any]] = Field(None, description="Sunrise/sunset data")
    raw_json: Optional[Dict[str, Any]] = Field(None, description="Complete raw JSON response")

    class Config:
        use_enum_values = True
