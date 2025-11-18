from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import Optional
from uuid import UUID

from src.api.core.database import get_db
from src.api.core.security import get_current_user_id
from src.api.models.planting import Planting
from src.api.models.farm import Farm
from src.api.schemas.planting import PlantingCreate, PlantingUpdate, PlantingResponse, PlantingList
from src.api.config import settings

router = APIRouter()


@router.get("/", response_model=PlantingList)
async def list_plantings(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    farm_id: Optional[UUID] = None,
    crop_type_id: Optional[int] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    List all plantings for the authenticated user with pagination and optional filtering

    Filters:
    - farm_id: Filter by specific farm
    - crop_type_id: Filter by specific crop type
    - status: Filter by planting status (active, harvested, failed, etc.)
    """
    # Build query - join with farms to ensure user owns the plantings
    query = (
        select(Planting)
        .join(Farm, Planting.farm_id == Farm.id)
        .where(Farm.user_id == UUID(user_id))
    )

    # Apply filters
    filters = []
    if farm_id:
        filters.append(Planting.farm_id == farm_id)
    if crop_type_id:
        filters.append(Planting.crop_type_id == crop_type_id)
    if status_filter:
        # Use exact case-insensitive match to prevent LIKE wildcard injection
        # Users should provide exact status values (e.g., "active", "harvested")
        filters.append(Planting.status.ilike(status_filter))

    if filters:
        query = query.where(and_(*filters))

    # Get total count
    count_query = (
        select(func.count())
        .select_from(Planting)
        .join(Farm, Planting.farm_id == Farm.id)
        .where(Farm.user_id == UUID(user_id))
    )
    if filters:
        count_query = count_query.where(and_(*filters))

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination and ordering
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Planting.planting_date.desc())

    # Execute query
    result = await db.execute(query)
    plantings = result.scalars().all()

    return PlantingList(
        plantings=plantings,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{planting_id}", response_model=PlantingResponse)
async def get_planting(
    planting_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific planting

    Only returns planting if it belongs to the authenticated user
    """
    result = await db.execute(
        select(Planting)
        .join(Farm, Planting.farm_id == Farm.id)
        .where(
            and_(
                Planting.id == planting_id,
                Farm.user_id == UUID(user_id)
            )
        )
    )
    planting = result.scalar_one_or_none()

    if not planting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Planting not found"
        )

    return planting


@router.post("/", response_model=PlantingResponse, status_code=status.HTTP_201_CREATED)
async def create_planting(
    planting_data: PlantingCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new planting record

    Requires authentication. Farm must belong to the authenticated user.
    """
    # Verify farm belongs to user
    farm_result = await db.execute(
        select(Farm).where(
            and_(
                Farm.id == planting_data.farm_id,
                Farm.user_id == UUID(user_id)
            )
        )
    )
    farm = farm_result.scalar_one_or_none()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found or does not belong to user"
        )

    # Create planting
    planting = Planting(
        farm_id=planting_data.farm_id,
        crop_type_id=planting_data.crop_type_id,
        planting_date=planting_data.planting_date,
        expected_harvest_date=planting_data.expected_harvest_date,
        area_planted_hectares=planting_data.area_planted_hectares,
        status=planting_data.status or "active",
        notes=planting_data.notes
    )

    db.add(planting)
    await db.commit()
    await db.refresh(planting)

    return planting


@router.put("/{planting_id}", response_model=PlantingResponse)
async def update_planting(
    planting_id: UUID,
    planting_data: PlantingUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing planting record

    Only updates plantings that belong to the authenticated user
    """
    # Get planting and verify ownership
    result = await db.execute(
        select(Planting)
        .join(Farm, Planting.farm_id == Farm.id)
        .where(
            and_(
                Planting.id == planting_id,
                Farm.user_id == UUID(user_id)
            )
        )
    )
    planting = result.scalar_one_or_none()

    if not planting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Planting not found"
        )

    # Update fields if provided
    update_data = planting_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(planting, field, value)

    await db.commit()
    await db.refresh(planting)

    return planting


@router.delete("/{planting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_planting(
    planting_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a planting record

    Only deletes plantings that belong to the authenticated user
    """
    # Get planting and verify ownership
    result = await db.execute(
        select(Planting)
        .join(Farm, Planting.farm_id == Farm.id)
        .where(
            and_(
                Planting.id == planting_id,
                Farm.user_id == UUID(user_id)
            )
        )
    )
    planting = result.scalar_one_or_none()

    if not planting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Planting not found"
        )

    await db.delete(planting)
    await db.commit()


@router.get("/farm/{farm_id}", response_model=PlantingList)
async def list_plantings_by_farm(
    farm_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    List all plantings for a specific farm

    Only returns plantings if the farm belongs to the authenticated user
    """
    # Verify farm belongs to user
    farm_result = await db.execute(
        select(Farm).where(
            and_(
                Farm.id == farm_id,
                Farm.user_id == UUID(user_id)
            )
        )
    )
    farm = farm_result.scalar_one_or_none()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found or does not belong to user"
        )

    # Build query
    query = select(Planting).where(Planting.farm_id == farm_id)

    # Get total count
    count_query = select(func.count()).select_from(Planting).where(Planting.farm_id == farm_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Planting.planting_date.desc())

    # Execute query
    result = await db.execute(query)
    plantings = result.scalars().all()

    return PlantingList(
        plantings=plantings,
        total=total,
        page=page,
        page_size=page_size
    )
