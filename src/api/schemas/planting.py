from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class PlantingCreate(BaseModel):
    """Schema for creating a planting record"""
    crop_id: str
    region_id: str
    planting_date: datetime
    expected_harvest_date: Optional[datetime] = None
    area_hectares: Optional[float] = None
    notes: Optional[str] = None


class PlantingUpdate(BaseModel):
    """Schema for updating a planting record"""
    expected_harvest_date: Optional[datetime] = None
    actual_harvest_date: Optional[datetime] = None
    area_hectares: Optional[float] = None
    notes: Optional[str] = None
    current_growth_stage: Optional[str] = None


class PlantingResponse(BaseModel):
    """Schema for planting response"""
    id: UUID
    user_id: UUID
    crop_id: str
    region_id: str
    planting_date: datetime
    expected_harvest_date: Optional[datetime]
    actual_harvest_date: Optional[datetime]
    area_hectares: Optional[float]
    current_growth_stage: Optional[str]
    notes: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class PlantingList(BaseModel):
    """Schema for paginated planting list"""
    plantings: list[PlantingResponse]
    total: int
    page: int
    page_size: int
