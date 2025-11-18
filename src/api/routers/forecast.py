from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, or_
from typing import Optional
from datetime import date, datetime, timedelta
import uuid

from src.api.core.database import get_db
from src.api.models.weather import WeatherForecast, WeatherDailyStats, RegionRiskIndicators, FloodRiskAssessment
from src.api.models.region import Region
from src.api.config import settings
from src.api.schemas.weather import (
    DailyForecastResponse,
    ForecastList,
    MultidayForecastResponse,
    RegionForecastSummary,
    ForecastWithRiskResponse,
    ComprehensiveForecastResponse,
    WeatherDailyStatsResponse,
    RiskIndicatorsResponse
)

router = APIRouter()


# Helper function to generate recommendations based on risk indicators
def generate_recommendations(
    rainfall_mm: Optional[float],
    flood_risk_level: Optional[str],
    typhoon_prob: Optional[float],
    harvest_suit: Optional[float]
) -> list[str]:
    """Generate actionable recommendations based on weather and risk data"""
    recommendations = []

    if rainfall_mm and rainfall_mm > 50:
        recommendations.append("Heavy rainfall expected. Ensure proper drainage in fields.")
    elif rainfall_mm and rainfall_mm > 20:
        recommendations.append("Moderate rainfall expected. Monitor soil moisture levels.")

    if flood_risk_level in ["High", "Critical"]:
        recommendations.append("High flood risk. Consider postponing field operations.")
        recommendations.append("Prepare emergency drainage and evacuation plans.")
    elif flood_risk_level == "Moderate":
        recommendations.append("Moderate flood risk. Monitor weather updates closely.")

    if typhoon_prob and typhoon_prob > 0.7:
        recommendations.append("High typhoon probability. Secure crops and equipment.")
        recommendations.append("Avoid planting or harvesting operations.")
    elif typhoon_prob and typhoon_prob > 0.4:
        recommendations.append("Moderate typhoon probability. Stay updated on weather advisories.")

    if harvest_suit and harvest_suit < 0.3:
        recommendations.append("Poor harvest conditions. Consider delaying harvest if possible.")
    elif harvest_suit and harvest_suit > 0.7:
        recommendations.append("Good harvest conditions. Proceed with planned operations.")

    if not recommendations:
        recommendations.append("Weather conditions are normal. Continue regular farm operations.")

    return recommendations


@router.get("/list", response_model=ForecastList)
async def list_forecasts(
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
    db: AsyncSession = Depends(get_db)
):
    """
    List all forecasts with pagination and filtering options.

    - **region_id**: Filter forecasts for a specific region
    - **start_date**: Get forecasts from this date onwards
    - **end_date**: Get forecasts up to this date
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

    return ForecastList(
        forecasts=forecasts,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{forecast_id}", response_model=ComprehensiveForecastResponse)
async def get_forecast_by_id(
    forecast_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive forecast details by forecast ID.

    Returns the forecast along with daily stats and risk indicators for the same date and region.
    """
    # Get the forecast
    result = await db.execute(
        select(WeatherForecast).where(WeatherForecast.id == forecast_id)
    )
    forecast = result.scalar_one_or_none()

    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not found")

    # Get region details
    region_result = await db.execute(
        select(Region).where(Region.id == forecast.region_id)
    )
    region = region_result.scalar_one_or_none()

    if not region:
        raise HTTPException(status_code=404, detail="Region not found")

    # Get daily stats for the same date
    stats_result = await db.execute(
        select(WeatherDailyStats).where(
            and_(
                WeatherDailyStats.region_id == forecast.region_id,
                WeatherDailyStats.stat_date == forecast.forecast_date
            )
        ).order_by(desc(WeatherDailyStats.created_at)).limit(1)
    )
    daily_stats = stats_result.scalar_one_or_none()

    # Get risk indicators for the same date (if table exists)
    risk_indicators = None
    try:
        risk_result = await db.execute(
            select(RegionRiskIndicators).where(
                and_(
                    RegionRiskIndicators.region_id == forecast.region_id,
                    RegionRiskIndicators.indicator_date == forecast.forecast_date
                )
            ).order_by(desc(RegionRiskIndicators.created_at)).limit(1)
        )
        risk_indicators = risk_result.scalar_one_or_none()
    except Exception:
        # Table doesn't exist yet - gracefully continue without risk indicators
        pass

    return ComprehensiveForecastResponse(
        forecast=DailyForecastResponse.from_orm(forecast),
        region_name=region.region_name,
        province=region.province,
        daily_stats=WeatherDailyStatsResponse.from_orm(daily_stats) if daily_stats else None,
        risk_indicators=RiskIndicatorsResponse.from_orm(risk_indicators) if risk_indicators else None
    )


@router.get("/region/{region_id}", response_model=MultidayForecastResponse)
async def get_region_forecast(
    region_id: int,
    days: int = Query(7, ge=1, le=30, description="Number of days to forecast"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get multi-day forecast for a specific region.

    - **region_id**: The ID of the region
    - **days**: Number of days to forecast (default: 7, max: 30)

    Returns forecasts for the specified number of days starting from today.
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
            WeatherForecast.forecast_date < end_date
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

    return MultidayForecastResponse(
        region_id=region.id,
        region_name=region.region_name,
        province=region.province,
        forecasts=[DailyForecastResponse.from_orm(f) for f in unique_forecasts],
        total_days=len(unique_forecasts),
        date_range={
            "start": today.isoformat(),
            "end": end_date.isoformat()
        }
    )


@router.get("/region/{region_id}/summary", response_model=RegionForecastSummary)
async def get_region_forecast_summary(
    region_id: int,
    days: int = Query(7, ge=1, le=30, description="Number of days to include in summary"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive forecast summary for a region including risk indicators.

    - **region_id**: The ID of the region
    - **days**: Number of days to include (default: 7, max: 30)

    Returns aggregated statistics and daily forecasts with risk assessments.
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

    forecast_query = select(WeatherForecast).where(
        and_(
            WeatherForecast.region_id == region_id,
            WeatherForecast.forecast_date >= today,
            WeatherForecast.forecast_date < end_date
        )
    ).order_by(
        WeatherForecast.forecast_date,
        desc(WeatherForecast.forecast_created_at)
    )

    forecast_result = await db.execute(forecast_query)
    all_forecasts = forecast_result.scalars().all()

    # Get unique forecasts (most recent for each date)
    seen_dates = set()
    unique_forecasts = []
    for forecast in all_forecasts:
        if forecast.forecast_date not in seen_dates:
            unique_forecasts.append(forecast)
            seen_dates.add(forecast.forecast_date)

    # Calculate summary statistics
    if unique_forecasts:
        temps = [f.temperature_avg for f in unique_forecasts if f.temperature_avg is not None]
        avg_temp = sum(temps) / len(temps) if temps else None

        rainfall = [f.rainfall_mm for f in unique_forecasts if f.rainfall_mm is not None]
        total_rainfall = sum(rainfall) if rainfall else None

        wind_speeds = [f.wind_speed_kph for f in unique_forecasts if f.wind_speed_kph is not None]
        max_wind = max(wind_speeds) if wind_speeds else None

        # Find most common weather condition
        conditions = [f.weather_condition for f in unique_forecasts if f.weather_condition]
        dominant_condition = max(set(conditions), key=conditions.count) if conditions else None
    else:
        avg_temp = None
        total_rainfall = None
        max_wind = None
        dominant_condition = None

    # Get flood risk assessments for the period
    flood_risk_query = select(FloodRiskAssessment).where(
        and_(
            FloodRiskAssessment.region_id == region_id,
            FloodRiskAssessment.assessment_date >= today,
            FloodRiskAssessment.assessment_date < end_date
        )
    ).order_by(desc(FloodRiskAssessment.assessment_date))

    flood_result = await db.execute(flood_risk_query)
    flood_risks = flood_result.scalars().all()

    # Create a map of flood risks by date
    flood_risk_map = {fr.assessment_date: fr for fr in flood_risks}

    # Get risk indicators for the period (if table exists)
    risk_map = {}
    try:
        risk_query = select(RegionRiskIndicators).where(
            and_(
                RegionRiskIndicators.region_id == region_id,
                RegionRiskIndicators.indicator_date >= today,
                RegionRiskIndicators.indicator_date < end_date
            )
        ).order_by(RegionRiskIndicators.indicator_date, desc(RegionRiskIndicators.created_at))

        risk_result = await db.execute(risk_query)
        all_risks = risk_result.scalars().all()

        # Get unique risk indicators (most recent for each date)
        for risk in all_risks:
            if risk.indicator_date not in risk_map:
                risk_map[risk.indicator_date] = risk
    except Exception:
        # Table doesn't exist yet - continue without risk indicators
        pass

    # Build daily forecasts with risk data
    daily_forecasts = []
    for forecast in unique_forecasts:
        flood_risk = flood_risk_map.get(forecast.forecast_date)
        risk_indicators = risk_map.get(forecast.forecast_date)

        recommendations = generate_recommendations(
            rainfall_mm=forecast.rainfall_mm,
            flood_risk_level=flood_risk.risk_level if flood_risk else None,
            typhoon_prob=risk_indicators.typhoon_probability if risk_indicators else None,
            harvest_suit=risk_indicators.harvest_suitability if risk_indicators else None
        )

        daily_forecasts.append(ForecastWithRiskResponse(
            forecast_date=forecast.forecast_date,
            temperature_min=forecast.temperature_min,
            temperature_max=forecast.temperature_max,
            rainfall_mm=forecast.rainfall_mm,
            weather_condition=forecast.weather_condition,
            flood_risk_level=flood_risk.risk_level if flood_risk else None,
            flood_risk_score=flood_risk.risk_score if flood_risk else None,
            typhoon_probability=risk_indicators.typhoon_probability if risk_indicators else None,
            harvest_suitability=risk_indicators.harvest_suitability if risk_indicators else None,
            recommendations=recommendations
        ))

    # Determine overall flood risk (highest risk level in the period)
    if flood_risks:
        risk_levels = [fr.risk_level for fr in flood_risks]
        if "Critical" in risk_levels:
            overall_flood_risk = "Critical"
        elif "High" in risk_levels:
            overall_flood_risk = "High"
        elif "Moderate" in risk_levels:
            overall_flood_risk = "Moderate"
        else:
            overall_flood_risk = "Low"
    else:
        overall_flood_risk = None

    # Calculate average harvest suitability
    if risk_map:
        harvest_suits = [r.harvest_suitability for r in risk_map.values() if r.harvest_suitability is not None]
        avg_harvest_suit = sum(harvest_suits) / len(harvest_suits) if harvest_suits else None
    else:
        avg_harvest_suit = None

    return RegionForecastSummary(
        region_id=region.id,
        region_name=region.region_name,
        province=region.province,
        summary_date=today,
        forecast_days=len(unique_forecasts),
        avg_temperature=avg_temp,
        total_rainfall=total_rainfall,
        max_wind_speed=max_wind,
        dominant_condition=dominant_condition,
        overall_flood_risk=overall_flood_risk,
        overall_harvest_suitability=avg_harvest_suit,
        daily_forecasts=daily_forecasts
    )


@router.get("/region/{region_id}/current", response_model=DailyForecastResponse)
async def get_current_forecast(
    region_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the current day's forecast for a specific region.

    Returns the most recent forecast for today or the nearest future date.
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
            detail=f"No current forecast available for region {region_id}"
        )

    return DailyForecastResponse.from_orm(forecast)


@router.get("/health")
async def forecast_health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint for forecast service.

    Returns statistics about available forecasts.
    """
    today = date.today()

    # Count total forecasts
    total_result = await db.execute(select(func.count()).select_from(WeatherForecast))
    total_forecasts = total_result.scalar()

    # Count forecasts for today and future
    future_result = await db.execute(
        select(func.count())
        .select_from(WeatherForecast)
        .where(WeatherForecast.forecast_date >= today)
    )
    future_forecasts = future_result.scalar()

    # Count unique regions with forecasts
    regions_result = await db.execute(
        select(func.count(func.distinct(WeatherForecast.region_id)))
        .select_from(WeatherForecast)
        .where(WeatherForecast.forecast_date >= today)
    )
    regions_with_forecasts = regions_result.scalar()

    # Get date range
    date_range_result = await db.execute(
        select(
            func.min(WeatherForecast.forecast_date),
            func.max(WeatherForecast.forecast_date)
        ).select_from(WeatherForecast)
    )
    min_date, max_date = date_range_result.first()

    return {
        "status": "healthy",
        "total_forecasts": total_forecasts,
        "future_forecasts": future_forecasts,
        "regions_with_forecasts": regions_with_forecasts,
        "date_range": {
            "earliest": min_date.isoformat() if min_date else None,
            "latest": max_date.isoformat() if max_date else None
        },
        "last_updated": datetime.utcnow().isoformat()
    }
