from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class ChatRequest(BaseModel):
    """Schema for chat request"""
    message: str = Field(..., min_length=1, max_length=1000)
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    """Schema for chat response"""
    message: str
    timestamp: datetime


class RecommendationRequest(BaseModel):
    """Schema for harvest recommendation request"""
    crop: str
    region: str
    planting_date: datetime
    growth_stage: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Schema for harvest recommendation response"""
    crop: str
    region: str
    planting_date: str
    days_since_planting: int
    expected_harvest_date: str
    days_until_harvest: int
    current_growth_stage: Optional[str]
    recommendation: str
    weather_forecast: dict
    flood_risk: dict
    generated_at: str
