from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from typing import Optional

from src.api.core.database import get_db
from src.api.models.crop import CropType
from src.api.schemas.crop import CropResponse, CropList
from src.api.config import settings

router = APIRouter()


@router.get("/", response_model=CropList)
async def list_crops(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all crops with pagination

    Optionally filter by category
    """
    # Build query
    query = select(CropType)
    if category:
        query = query.where(CropType.crop_category.ilike(f"%{category}%"))

    # Get total count
    count_query = select(func.count()).select_from(CropType)
    if category:
        count_query = count_query.where(CropType.crop_category.ilike(f"%{category}%"))

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(CropType.crop_name)

    # Execute query
    result = await db.execute(query)
    crops = result.scalars().all()

    return CropList(
        crops=crops,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/categories", response_model=list[str])
async def get_categories(db: AsyncSession = Depends(get_db)):
    """
    Get all unique crop categories
    """
    result = await db.execute(
        select(distinct(CropType.crop_category))
        .where(CropType.crop_category.isnot(None))
        .order_by(CropType.crop_category)
    )
    categories = result.scalars().all()
    return categories


@router.get("/{crop_id}", response_model=CropResponse)
async def get_crop(
    crop_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific crop
    """
    result = await db.execute(
        select(CropType).where(CropType.id == crop_id)
    )
    crop = result.scalar_one_or_none()

    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    return crop


@router.get("/search/", response_model=CropList)
async def search_crops(
    q: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    db: AsyncSession = Depends(get_db)
):
    """
    Search crops by name
    """
    search_term = f"%{q}%"
    query = select(CropType).where(
        (CropType.crop_name.ilike(search_term)) |
        (CropType.crop_category.ilike(search_term))
    )

    # Get total count
    count_query = select(func.count()).select_from(CropType).where(
        (CropType.crop_name.ilike(search_term)) |
        (CropType.crop_category.ilike(search_term))
    )

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(CropType.crop_name)

    result = await db.execute(query)
    crops = result.scalars().all()

    return CropList(
        crops=crops,
        total=total,
        page=page,
        page_size=page_size
    )
