from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional

from src.api.core.database import get_db
from src.api.models.region import Region
from src.api.schemas.region import RegionResponse, RegionList
from src.api.config import settings

router = APIRouter()


@router.get("/", response_model=RegionList)
async def list_regions(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    province: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all regions with pagination

    Optionally filter by province
    """
    # Build query
    query = select(Region)
    if province:
        query = query.where(Region.province.ilike(f"%{province}%"))

    # Get total count
    count_query = select(func.count()).select_from(Region)
    if province:
        count_query = count_query.where(Region.province.ilike(f"%{province}%"))

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Region.region_name)

    # Execute query
    result = await db.execute(query)
    regions = result.scalars().all()

    return RegionList(
        regions=regions,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{region_id}", response_model=RegionResponse)
async def get_region(
    region_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific region
    """
    result = await db.execute(
        select(Region).where(Region.id == region_id)
    )
    region = result.scalar_one_or_none()

    if not region:
        raise HTTPException(status_code=404, detail="Region not found")

    return region


@router.get("/search/", response_model=RegionList)
async def search_regions(
    q: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    db: AsyncSession = Depends(get_db)
):
    """
    Search regions by name or province
    """
    search_term = f"%{q}%"
    query = select(Region).where(
        (Region.region_name.ilike(search_term)) |
        (Region.province.ilike(search_term)) |
        (Region.municipality.ilike(search_term))
    )

    # Get total count
    count_query = select(func.count()).select_from(Region).where(
        (Region.region_name.ilike(search_term)) |
        (Region.province.ilike(search_term)) |
        (Region.municipality.ilike(search_term))
    )

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Region.region_name)

    result = await db.execute(query)
    regions = result.scalars().all()

    return RegionList(
        regions=regions,
        total=total,
        page=page,
        page_size=page_size
    )
