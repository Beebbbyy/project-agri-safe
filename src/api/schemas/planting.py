from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date, datetime
from uuid import UUID
from decimal import Decimal

# Valid planting status values
PlantingStatus = Literal["active", "harvested", "failed", "planned", "abandoned"]


class PlantingCreate(BaseModel):
    """Schema for creating a planting record"""
    farm_id: UUID
    crop_type_id: int
    planting_date: date
    expected_harvest_date: Optional[date] = None
    area_planted_hectares: Optional[Decimal] = Field(None, ge=0)
    status: Optional[PlantingStatus] = Field(default="active")
    notes: Optional[str] = None


class PlantingUpdate(BaseModel):
    """Schema for updating a planting record"""
    expected_harvest_date: Optional[date] = None
    actual_harvest_date: Optional[date] = None
    area_planted_hectares: Optional[Decimal] = Field(None, ge=0)
    status: Optional[PlantingStatus] = None
    notes: Optional[str] = None


class PlantingResponse(BaseModel):
    """Schema for planting response"""
    id: UUID
    farm_id: UUID
    crop_type_id: int
    planting_date: date
    expected_harvest_date: Optional[date]
    actual_harvest_date: Optional[date]
    area_planted_hectares: Optional[Decimal]
    status: str
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
