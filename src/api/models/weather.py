from sqlalchemy import Column, String, Integer, Numeric, Date, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from src.api.core.database import Base


class WeatherForecast(Base):
    __tablename__ = "weather_forecasts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_id = Column(Integer, ForeignKey("regions.id", ondelete="CASCADE"), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    forecast_created_at = Column(DateTime(timezone=True), nullable=False)
    temperature_min = Column(Numeric(5, 2))
    temperature_max = Column(Numeric(5, 2))
    temperature_avg = Column(Numeric(5, 2))
    humidity_percent = Column(Numeric(5, 2))
    rainfall_mm = Column(Numeric(8, 2))
    wind_speed_kph = Column(Numeric(6, 2))
    weather_condition = Column(String(100))
    weather_description = Column(Text)
    data_source = Column(String(100), default="PAGASA")
    raw_data = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<WeatherForecast {self.region_id} - {self.forecast_date}>"


class FloodRiskAssessment(Base):
    __tablename__ = "flood_risk_assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_id = Column(Integer, ForeignKey("regions.id", ondelete="CASCADE"), nullable=False, index=True)
    assessment_date = Column(Date, nullable=False, index=True)
    risk_level = Column(String(50), nullable=False, index=True)
    risk_score = Column(Numeric(5, 2))
    rainfall_forecast_mm = Column(Numeric(8, 2))
    historical_flood_probability = Column(Numeric(5, 2))
    soil_saturation_index = Column(Numeric(5, 2))
    river_level_status = Column(String(50))
    model_version = Column(String(50))
    confidence_score = Column(Numeric(5, 2))
    factors = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<FloodRiskAssessment {self.region_id} - {self.assessment_date} - {self.risk_level}>"
