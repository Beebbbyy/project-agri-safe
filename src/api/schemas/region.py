from pydantic import BaseModel, Field
from typing import Optional


class RegionBase(BaseModel):
    """Base schema for region"""
    name: str
    province: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None


class RegionResponse(RegionBase):
    """Schema for region response"""
    id: str

    class Config:
        from_attributes = True


class RegionList(BaseModel):
    """Schema for paginated region list"""
    regions: list[RegionResponse]
    total: int
    page: int
    page_size: int
