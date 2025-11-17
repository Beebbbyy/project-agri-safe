from pydantic import BaseModel
from typing import Optional
from datetime import date


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


class ForecastResponse(BaseModel):
    """Schema for weather forecast"""
    date: str
    rainfall_mm: Optional[float]
    temp_high: Optional[float]
    temp_low: Optional[float]
    humidity_pct: Optional[float]
    condition: Optional[str]
    flood_risk: Optional[str]


class RegionForecastResponse(BaseModel):
    """Schema for region forecast with multiple days"""
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
