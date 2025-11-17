from pydantic import BaseModel, Field
from typing import Optional


class CropBase(BaseModel):
    """Base schema for crop"""
    name: str
    category: str
    avg_maturity_days: int
    min_maturity_days: Optional[int] = None
    max_maturity_days: Optional[int] = None
    optimal_harvest_moisture_pct: Optional[float] = None


class CropResponse(CropBase):
    """Schema for crop response"""
    id: str

    class Config:
        from_attributes = True


class CropList(BaseModel):
    """Schema for paginated crop list"""
    crops: list[CropResponse]
    total: int
    page: int
    page_size: int
