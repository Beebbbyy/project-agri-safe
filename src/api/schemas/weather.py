from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
import uuid


class WeatherDailyResponse(BaseModel):
    """Schema for daily weather statistics"""
    region_id: str
    date: date
    avg_temp: Optional[float]
    min_temp: Optional[float]
    max_temp: Optional[float]
    total_rainfall: Optional[float]
    avg_humidity: Optional[float]
    avg_wind_speed: Optional[float]

    class Config:
        from_attributes = True


class WeatherRollingResponse(BaseModel):
    """Schema for rolling window features"""
    region_id: str
    date: date
    rainfall_7d: Optional[float]
    rainfall_14d: Optional[float]
    rainfall_30d: Optional[float]
    temp_avg_7d: Optional[float]
    temp_avg_14d: Optional[float]
    temp_avg_30d: Optional[float]

    class Config:
        from_attributes = True


# Enhanced Forecast Schemas for Forecast Router

class DailyForecastResponse(BaseModel):
    """Schema for single day forecast"""
    id: uuid.UUID
    forecast_date: date
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    temperature_avg: Optional[float] = None
    humidity_percent: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_speed_kph: Optional[float] = None
    weather_condition: Optional[str] = None
    weather_description: Optional[str] = None
    data_source: Optional[str] = None
    forecast_created_at: datetime

    class Config:
        from_attributes = True


class WeatherDailyStatsResponse(BaseModel):
    """Schema for weather daily statistics"""
    id: uuid.UUID
    region_id: int
    stat_date: date
    temp_high_avg: Optional[float] = None
    temp_low_avg: Optional[float] = None
    temp_range: Optional[float] = None
    rainfall_total: Optional[float] = None
    rainfall_max: Optional[float] = None
    rainfall_min: Optional[float] = None
    wind_speed_max: Optional[float] = None
    wind_speed_avg: Optional[float] = None
    dominant_condition: Optional[str] = None
    forecast_count: Optional[int] = None
    data_quality_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class RiskIndicatorsResponse(BaseModel):
    """Schema for region risk indicators"""
    id: uuid.UUID
    region_id: int
    indicator_date: date
    flood_season_score: Optional[float] = None
    typhoon_probability: Optional[float] = None
    harvest_suitability: Optional[float] = None
    drought_risk_score: Optional[float] = None
    risk_factors: Optional[dict] = None
    model_version: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ComprehensiveForecastResponse(BaseModel):
    """Schema for comprehensive forecast with all related data"""
    forecast: DailyForecastResponse
    region_name: str
    province: Optional[str] = None
    daily_stats: Optional[WeatherDailyStatsResponse] = None
    risk_indicators: Optional[RiskIndicatorsResponse] = None


class MultidayForecastResponse(BaseModel):
    """Schema for multi-day forecast for a region"""
    region_id: int
    region_name: str
    province: Optional[str] = None
    forecasts: list[DailyForecastResponse]
    total_days: int
    date_range: dict = Field(description="Start and end dates of the forecast period")


class ForecastWithRiskResponse(BaseModel):
    """Schema for forecast combined with risk assessment"""
    forecast_date: date
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    rainfall_mm: Optional[float] = None
    weather_condition: Optional[str] = None
    flood_risk_level: Optional[str] = None
    flood_risk_score: Optional[float] = None
    typhoon_probability: Optional[float] = None
    harvest_suitability: Optional[float] = None
    recommendations: list[str] = []


class RegionForecastSummary(BaseModel):
    """Schema for region forecast summary"""
    region_id: int
    region_name: str
    province: Optional[str] = None
    summary_date: date
    forecast_days: int
    avg_temperature: Optional[float] = None
    total_rainfall: Optional[float] = None
    max_wind_speed: Optional[float] = None
    dominant_condition: Optional[str] = None
    overall_flood_risk: Optional[str] = None
    overall_harvest_suitability: Optional[float] = None
    daily_forecasts: list[ForecastWithRiskResponse]


class ForecastList(BaseModel):
    """Schema for paginated forecast list"""
    forecasts: list[DailyForecastResponse]
    total: int
    page: int
    page_size: int


class ForecastResponse(BaseModel):
    """Schema for weather forecast (legacy - for backward compatibility)"""
    date: str
    rainfall_mm: Optional[float]
    temp_high: Optional[float]
    temp_low: Optional[float]
    humidity_pct: Optional[float]
    condition: Optional[str]
    flood_risk: Optional[str]


class RegionForecastResponse(BaseModel):
    """Schema for region forecast with multiple days (legacy)"""
    region_id: str
    region_name: str
    forecasts: list[ForecastResponse]
    metadata: dict


class FloodRiskResponse(BaseModel):
    """Schema for flood risk assessment"""
    region_id: str
    risk_level: str
    confidence_score: float
    contributing_factors: dict
    recommendation: str
    generated_at: str
