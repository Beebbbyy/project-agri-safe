from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from typing import Optional
from datetime import date, datetime, timedelta
import uuid

from src.api.core.database import get_db
from src.api.models.weather import WeatherForecast, FloodRiskAssessment
from src.api.models.region import Region
from src.api.config import settings

router = APIRouter()


# Pydantic schemas for responses
from pydantic import BaseModel, Field


class WeatherForecastResponse(BaseModel):
    """Schema for weather forecast response"""
    id: uuid.UUID
    region_id: int
    forecast_date: date
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    temperature_avg: Optional[float] = None
    humidity_percent: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_speed_kph: Optional[float] = None
    weather_condition: Optional[str] = None
    weather_description: Optional[str] = None
    data_source: Optional[str] = None
    forecast_created_at: datetime

    class Config:
        from_attributes = True


class WeatherForecastList(BaseModel):
    """Schema for paginated weather forecast list"""
    forecasts: list[WeatherForecastResponse]
    total: int
    page: int
    page_size: int


class FloodRiskResponse(BaseModel):
    """Schema for flood risk assessment response"""
    id: uuid.UUID
    region_id: int
    assessment_date: date
    risk_level: str
    risk_score: Optional[float] = None
    rainfall_forecast_mm: Optional[float] = None
    historical_flood_probability: Optional[float] = None
    soil_saturation_index: Optional[float] = None
    river_level_status: Optional[str] = None
    model_version: Optional[str] = None
    confidence_score: Optional[float] = None
    factors: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class FloodRiskList(BaseModel):
    """Schema for paginated flood risk list"""
    assessments: list[FloodRiskResponse]
    total: int
    page: int
    page_size: int


class RegionWeatherSummary(BaseModel):
    """Schema for region weather summary"""
    region_id: int
    region_name: str
    province: Optional[str] = None
    latest_forecast: Optional[WeatherForecastResponse] = None
    latest_flood_risk: Optional[FloodRiskResponse] = None
    forecast_count: int
    date_range: Optional[dict] = None


# Weather Forecast Endpoints

@router.get("/forecasts", response_model=WeatherForecastList)
async def list_weather_forecasts(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(
        settings.DEFAULT_PAGE_SIZE,
        ge=1,
        le=settings.MAX_PAGE_SIZE,
        description="Number of items per page"
    ),
    region_id: Optional[int] = Query(None, description="Filter by region ID"),
    start_date: Optional[date] = Query(None, description="Filter forecasts from this date"),
    end_date: Optional[date] = Query(None, description="Filter forecasts until this date"),
    data_source: Optional[str] = Query(None, description="Filter by data source (e.g., PAGASA)"),
    db: AsyncSession = Depends(get_db)
):
    """
    List weather forecasts with pagination and filtering options.

    - **region_id**: Filter forecasts for a specific region
    - **start_date**: Get forecasts from this date onwards
    - **end_date**: Get forecasts up to this date
    - **data_source**: Filter by data source (e.g., PAGASA)
    """
    # Build query with filters
    query = select(WeatherForecast)
    filters = []

    if region_id:
        filters.append(WeatherForecast.region_id == region_id)
    if start_date:
        filters.append(WeatherForecast.forecast_date >= start_date)
    if end_date:
        filters.append(WeatherForecast.forecast_date <= end_date)
    if data_source:
        filters.append(WeatherForecast.data_source.ilike(f"%{data_source}%"))

    if filters:
        query = query.where(and_(*filters))

    # Count total records
    count_query = select(func.count()).select_from(WeatherForecast)
    if filters:
        count_query = count_query.where(and_(*filters))

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination and ordering
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(
        desc(WeatherForecast.forecast_date),
        desc(WeatherForecast.forecast_created_at)
    )

    result = await db.execute(query)
    forecasts = result.scalars().all()

    return WeatherForecastList(
        forecasts=forecasts,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/forecasts/{forecast_id}", response_model=WeatherForecastResponse)
async def get_weather_forecast(
    forecast_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific weather forecast by its ID.
    """
    result = await db.execute(
        select(WeatherForecast).where(WeatherForecast.id == forecast_id)
    )
    forecast = result.scalar_one_or_none()

    if not forecast:
        raise HTTPException(status_code=404, detail="Weather forecast not found")

    return forecast


@router.get("/forecasts/region/{region_id}", response_model=WeatherForecastList)
async def get_region_forecasts(
    region_id: int,
    days: int = Query(7, ge=1, le=30, description="Number of days to forecast"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get weather forecasts for a specific region.

    - **region_id**: The ID of the region
    - **days**: Number of days to forecast (default: 7, max: 30)

    Returns the most recent forecasts for the specified number of days.
    """
    # Verify region exists
    region_result = await db.execute(
        select(Region).where(Region.id == region_id)
    )
    region = region_result.scalar_one_or_none()

    if not region:
        raise HTTPException(status_code=404, detail="Region not found")

    # Get forecasts for the next N days
    today = date.today()
    end_date = today + timedelta(days=days)

    query = select(WeatherForecast).where(
        and_(
            WeatherForecast.region_id == region_id,
            WeatherForecast.forecast_date >= today,
            WeatherForecast.forecast_date <= end_date
        )
    ).order_by(
        WeatherForecast.forecast_date,
        desc(WeatherForecast.forecast_created_at)
    )

    result = await db.execute(query)
    all_forecasts = result.scalars().all()

    # Get only the most recent forecast for each date
    seen_dates = set()
    unique_forecasts = []
    for forecast in all_forecasts:
        if forecast.forecast_date not in seen_dates:
            unique_forecasts.append(forecast)
            seen_dates.add(forecast.forecast_date)

    return WeatherForecastList(
        forecasts=unique_forecasts,
        total=len(unique_forecasts),
        page=1,
        page_size=len(unique_forecasts)
    )


@router.get("/forecasts/region/{region_id}/current", response_model=WeatherForecastResponse)
async def get_current_weather(
    region_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the current/latest weather forecast for a specific region.

    Returns the most recent forecast for today or the nearest available date.
    """
    # Verify region exists
    region_result = await db.execute(
        select(Region).where(Region.id == region_id)
    )
    region = region_result.scalar_one_or_none()

    if not region:
        raise HTTPException(status_code=404, detail="Region not found")

    # Get the most recent forecast for today or later
    today = date.today()

    result = await db.execute(
        select(WeatherForecast)
        .where(
            and_(
                WeatherForecast.region_id == region_id,
                WeatherForecast.forecast_date >= today
            )
        )
        .order_by(
            WeatherForecast.forecast_date,
            desc(WeatherForecast.forecast_created_at)
        )
        .limit(1)
    )
    forecast = result.scalar_one_or_none()

    if not forecast:
        raise HTTPException(
            status_code=404,
            detail=f"No current weather forecast available for region {region_id}"
        )

    return forecast


# Flood Risk Assessment Endpoints

@router.get("/flood-risk", response_model=FloodRiskList)
async def list_flood_risk_assessments(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(
        settings.DEFAULT_PAGE_SIZE,
        ge=1,
        le=settings.MAX_PAGE_SIZE,
        description="Number of items per page"
    ),
    region_id: Optional[int] = Query(None, description="Filter by region ID"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (Low, Moderate, High, Critical)"),
    start_date: Optional[date] = Query(None, description="Filter assessments from this date"),
    end_date: Optional[date] = Query(None, description="Filter assessments until this date"),
    db: AsyncSession = Depends(get_db)
):
    """
    List flood risk assessments with pagination and filtering options.

    - **region_id**: Filter assessments for a specific region
    - **risk_level**: Filter by risk level (e.g., Low, Moderate, High, Critical)
    - **start_date**: Get assessments from this date onwards
    - **end_date**: Get assessments up to this date
    """
    # Build query with filters
    query = select(FloodRiskAssessment)
    filters = []

    if region_id:
        filters.append(FloodRiskAssessment.region_id == region_id)
    if risk_level:
        filters.append(FloodRiskAssessment.risk_level.ilike(f"%{risk_level}%"))
    if start_date:
        filters.append(FloodRiskAssessment.assessment_date >= start_date)
    if end_date:
        filters.append(FloodRiskAssessment.assessment_date <= end_date)

    if filters:
        query = query.where(and_(*filters))

    # Count total records
    count_query = select(func.count()).select_from(FloodRiskAssessment)
    if filters:
        count_query = count_query.where(and_(*filters))

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination and ordering
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(
        desc(FloodRiskAssessment.assessment_date),
        desc(FloodRiskAssessment.created_at)
    )

    result = await db.execute(query)
    assessments = result.scalars().all()

    return FloodRiskList(
        assessments=assessments,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/flood-risk/{assessment_id}", response_model=FloodRiskResponse)
async def get_flood_risk_assessment(
    assessment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific flood risk assessment by its ID.
    """
    result = await db.execute(
        select(FloodRiskAssessment).where(FloodRiskAssessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="Flood risk assessment not found")

    return assessment


@router.get("/flood-risk/region/{region_id}", response_model=FloodRiskResponse)
async def get_region_flood_risk(
    region_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the latest flood risk assessment for a specific region.

    Returns the most recent flood risk assessment for the region.
    """
    # Verify region exists
    region_result = await db.execute(
        select(Region).where(Region.id == region_id)
    )
    region = region_result.scalar_one_or_none()

    if not region:
        raise HTTPException(status_code=404, detail="Region not found")

    # Get the most recent flood risk assessment
    result = await db.execute(
        select(FloodRiskAssessment)
        .where(FloodRiskAssessment.region_id == region_id)
        .order_by(
            desc(FloodRiskAssessment.assessment_date),
            desc(FloodRiskAssessment.created_at)
        )
        .limit(1)
    )
    assessment = result.scalar_one_or_none()

    if not assessment:
        raise HTTPException(
            status_code=404,
            detail=f"No flood risk assessment available for region {region_id}"
        )

    return assessment


# Combined Weather Summary Endpoint

@router.get("/summary/region/{region_id}", response_model=RegionWeatherSummary)
async def get_region_weather_summary(
    region_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a comprehensive weather summary for a specific region.

    Includes:
    - Region information
    - Latest weather forecast
    - Latest flood risk assessment
    - Forecast availability statistics
    """
    # Verify region exists and get details
    region_result = await db.execute(
        select(Region).where(Region.id == region_id)
    )
    region = region_result.scalar_one_or_none()

    if not region:
        raise HTTPException(status_code=404, detail="Region not found")

    # Get latest forecast
    forecast_result = await db.execute(
        select(WeatherForecast)
        .where(WeatherForecast.region_id == region_id)
        .order_by(
            desc(WeatherForecast.forecast_date),
            desc(WeatherForecast.forecast_created_at)
        )
        .limit(1)
    )
    latest_forecast = forecast_result.scalar_one_or_none()

    # Get latest flood risk
    flood_risk_result = await db.execute(
        select(FloodRiskAssessment)
        .where(FloodRiskAssessment.region_id == region_id)
        .order_by(
            desc(FloodRiskAssessment.assessment_date),
            desc(FloodRiskAssessment.created_at)
        )
        .limit(1)
    )
    latest_flood_risk = flood_risk_result.scalar_one_or_none()

    # Get forecast count and date range
    count_result = await db.execute(
        select(func.count()).select_from(WeatherForecast)
        .where(WeatherForecast.region_id == region_id)
    )
    forecast_count = count_result.scalar()

    # Get date range of available forecasts
    date_range = None
    if forecast_count > 0:
        min_date_result = await db.execute(
            select(func.min(WeatherForecast.forecast_date))
            .where(WeatherForecast.region_id == region_id)
        )
        max_date_result = await db.execute(
            select(func.max(WeatherForecast.forecast_date))
            .where(WeatherForecast.region_id == region_id)
        )
        min_date = min_date_result.scalar()
        max_date = max_date_result.scalar()

        if min_date and max_date:
            date_range = {
                "earliest": min_date.isoformat(),
                "latest": max_date.isoformat()
            }

    return RegionWeatherSummary(
        region_id=region.id,
        region_name=region.region_name,
        province=region.province,
        latest_forecast=latest_forecast,
        latest_flood_risk=latest_flood_risk,
        forecast_count=forecast_count,
        date_range=date_range
    )


# Health check endpoint for weather service
@router.get("/health")
async def weather_service_health(
    db: AsyncSession = Depends(get_db)
):
    """
    Health check endpoint for the weather service.

    Returns statistics about available weather data.
    """
    # Count total forecasts
    forecast_count_result = await db.execute(
        select(func.count()).select_from(WeatherForecast)
    )
    total_forecasts = forecast_count_result.scalar()

    # Count total flood risk assessments
    flood_count_result = await db.execute(
        select(func.count()).select_from(FloodRiskAssessment)
    )
    total_flood_assessments = flood_count_result.scalar()

    # Get latest forecast date
    latest_forecast_result = await db.execute(
        select(func.max(WeatherForecast.forecast_date))
    )
    latest_forecast_date = latest_forecast_result.scalar()

    # Get latest flood risk assessment date
    latest_flood_result = await db.execute(
        select(func.max(FloodRiskAssessment.assessment_date))
    )
    latest_flood_date = latest_flood_result.scalar()

    return {
        "status": "healthy",
        "service": "weather",
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_forecasts": total_forecasts,
            "total_flood_assessments": total_flood_assessments,
            "latest_forecast_date": latest_forecast_date.isoformat() if latest_forecast_date else None,
            "latest_flood_assessment_date": latest_flood_date.isoformat() if latest_flood_date else None
        }
    }
