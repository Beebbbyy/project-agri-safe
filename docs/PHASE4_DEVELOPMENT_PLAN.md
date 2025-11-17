# Phase 4: Backend API Development Plan
**Duration**: Weeks 7-8
**Status**: 📋 Planning
**Dependencies**: Phase 1 ✅, Phase 2 ✅, Phase 3 ✅

---

## Table of Contents
1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Architecture](#architecture)
4. [Component 1: FastAPI Backend](#component-1-fastapi-backend)
5. [Component 2: LLM Integration](#component-2-llm-integration)
6. [Component 3: Authentication System](#component-3-authentication-system)
7. [Component 4: Caching Layer](#component-4-caching-layer)
8. [Implementation Timeline](#implementation-timeline)
9. [Testing Strategy](#testing-strategy)
10. [Dependencies & Setup](#dependencies--setup)
11. [Success Criteria](#success-criteria)

---

## Overview

Phase 4 builds the backend API layer that serves as the bridge between Phase 3's data processing layer and Phase 5's frontend interface. This phase delivers the core business logic, LLM-powered harvest recommendations, and secure API endpoints.

### Current State
- ✅ PostgreSQL database with weather, regions, and crop data
- ✅ Daily weather statistics and rolling features
- ✅ Flood risk indicators calculated and stored
- ✅ Redis cache with weather features
- ✅ Airflow orchestration running ETL pipelines

### Target State
- 🎯 RESTful API with FastAPI serving all data
- 🎯 LLM-powered harvest advisor chatbot
- 🎯 JWT-based authentication system
- 🎯 API documentation (OpenAPI/Swagger)
- 🎯 Redis caching for API responses
- 🎯 Rate limiting and security middleware
- 🎯 Comprehensive test coverage (>80%)

---

## Objectives

### Primary Goals
1. **Build production-ready FastAPI backend** with RESTful endpoints
2. **Integrate LLM** (OpenAI/Anthropic) for intelligent harvest recommendations
3. **Implement secure authentication** with JWT tokens
4. **Create caching strategy** for optimal performance
5. **Develop comprehensive API documentation** with examples

### Key Deliverables
- [ ] FastAPI application with structured endpoints
- [ ] LLM integration service (harvest advisor)
- [ ] User authentication and authorization
- [ ] API documentation (auto-generated + custom guides)
- [ ] Redis caching layer for API responses
- [ ] Rate limiting and security middleware
- [ ] Unit and integration tests (>80% coverage)
- [ ] API deployment configuration

---

## Architecture

### Technology Stack
```
API Framework:
- FastAPI 0.109+ (async, high-performance)
- Pydantic 2.5+ (data validation)
- SQLAlchemy 2.0+ (ORM)
- Alembic (migrations)

LLM Integration:
- OpenAI Python SDK (GPT-4)
- Anthropic Python SDK (Claude)
- LangChain (optional, for advanced RAG)

Authentication:
- python-jose (JWT tokens)
- passlib + bcrypt (password hashing)
- python-multipart (form data)

Caching & Performance:
- Redis 7+ (response caching)
- aiocache (async caching library)

Testing:
- pytest + pytest-asyncio
- httpx (async HTTP client for testing)
- factory-boy (test data generation)
```

### API Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATIONS                         │
│        (Phase 5: Streamlit, Future: Mobile App)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS/REST
┌────────────────────────▼────────────────────────────────────────┐
│                   FASTAPI APPLICATION                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              API GATEWAY LAYER                           │  │
│  │  - CORS Middleware                                       │  │
│  │  - Rate Limiting                                         │  │
│  │  - Request Validation                                    │  │
│  │  - Error Handling                                        │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │           AUTHENTICATION LAYER                           │  │
│  │  - JWT Token Validation                                  │  │
│  │  - User Authorization                                    │  │
│  │  - Session Management                                    │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │              BUSINESS LOGIC LAYER                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │  │
│  │  │  Weather   │  │  Harvest   │  │   User     │         │  │
│  │  │  Service   │  │  Advisor   │  │  Service   │         │  │
│  │  └────────────┘  └────────────┘  └────────────┘         │  │
│  └──────┬──────────────────┬──────────────────┬─────────────┘  │
│         │                  │                  │                 │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│   PostgreSQL    │  │  LLM Service │  │    Redis     │
│   Database      │  │  (GPT/Claude)│  │    Cache     │
│  - Regions      │  │              │  │              │
│  - Weather      │  │  + Context   │  │  - Forecasts │
│  - Crops        │  │    Builder   │  │  - LLM Cache │
│  - Users        │  │              │  │  - Sessions  │
└─────────────────┘  └──────────────┘  └──────────────┘
```

### Request Flow Example
```
1. Client Request: GET /api/v1/forecast/region-1?days=5
                          ↓
2. Middleware: CORS, Rate Limit Check
                          ↓
3. Auth Layer: Validate JWT Token (if protected)
                          ↓
4. Cache Check: Redis lookup for cached response
                          ↓
5. If Cache Miss:
   a. Business Logic: Query Database
   b. Transform Data (Pydantic Models)
   c. Store in Redis (TTL: 6 hours)
                          ↓
6. Response: JSON with forecast data + metadata
```

---

## Component 1: FastAPI Backend

### 1.1 Project Structure

```
src/api/
├── __init__.py
├── main.py                    # FastAPI application entry point
├── config.py                  # Settings and configuration
├── dependencies.py            # Dependency injection
├── models/                    # SQLAlchemy ORM models
│   ├── __init__.py
│   ├── user.py
│   ├── region.py
│   ├── crop.py
│   ├── weather.py
│   └── planting.py
├── schemas/                   # Pydantic schemas (request/response)
│   ├── __init__.py
│   ├── user.py
│   ├── region.py
│   ├── crop.py
│   ├── weather.py
│   ├── planting.py
│   └── chat.py
├── routers/                   # API route handlers
│   ├── __init__.py
│   ├── auth.py               # /api/v1/auth/*
│   ├── regions.py            # /api/v1/regions/*
│   ├── crops.py              # /api/v1/crops/*
│   ├── weather.py            # /api/v1/weather/*
│   ├── forecast.py           # /api/v1/forecast/*
│   ├── plantings.py          # /api/v1/plantings/*
│   └── advisor.py            # /api/v1/advisor/*
├── services/                  # Business logic services
│   ├── __init__.py
│   ├── auth_service.py
│   ├── weather_service.py
│   ├── forecast_service.py
│   ├── advisor_service.py
│   └── cache_service.py
├── core/                      # Core utilities
│   ├── __init__.py
│   ├── security.py           # JWT, password hashing
│   ├── database.py           # DB connection, session
│   ├── cache.py              # Redis connection
│   └── llm.py                # LLM client wrapper
└── middleware/
    ├── __init__.py
    ├── rate_limit.py
    └── error_handler.py
```

### 1.2 Main Application Setup

**File**: `src/api/main.py`

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from src.api.routers import (
    auth, regions, crops, weather,
    forecast, plantings, advisor
)
from src.api.core.database import engine, Base
from src.api.core.cache import redis_client
from src.api.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting AgriSafe API...")

    # Test database connection
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

    # Test Redis connection
    try:
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down AgriSafe API...")
    await redis_client.close()
    await engine.dispose()


# Initialize FastAPI application
app = FastAPI(
    title="AgriSafe API",
    description="Weather forecast and harvest advisor API for Filipino farmers",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "database": "connected",
        "cache": "connected"
    }


# API version prefix
API_V1_PREFIX = "/api/v1"

# Include routers
app.include_router(
    auth.router,
    prefix=f"{API_V1_PREFIX}/auth",
    tags=["Authentication"]
)
app.include_router(
    regions.router,
    prefix=f"{API_V1_PREFIX}/regions",
    tags=["Regions"]
)
app.include_router(
    crops.router,
    prefix=f"{API_V1_PREFIX}/crops",
    tags=["Crops"]
)
app.include_router(
    weather.router,
    prefix=f"{API_V1_PREFIX}/weather",
    tags=["Weather"]
)
app.include_router(
    forecast.router,
    prefix=f"{API_V1_PREFIX}/forecast",
    tags=["Forecast"]
)
app.include_router(
    plantings.router,
    prefix=f"{API_V1_PREFIX}/plantings",
    tags=["Plantings"]
)
app.include_router(
    advisor.router,
    prefix=f"{API_V1_PREFIX}/advisor",
    tags=["Harvest Advisor"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to AgriSafe API",
        "docs": "/api/docs",
        "version": "1.0.0",
        "endpoints": {
            "regions": f"{API_V1_PREFIX}/regions",
            "crops": f"{API_V1_PREFIX}/crops",
            "weather": f"{API_V1_PREFIX}/weather",
            "forecast": f"{API_V1_PREFIX}/forecast",
            "plantings": f"{API_V1_PREFIX}/plantings",
            "advisor": f"{API_V1_PREFIX}/advisor"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
```

### 1.3 Configuration Management

**File**: `src/api/config.py`

```python
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "AgriSafe API"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    # Database
    POSTGRES_USER: str = "agrisafe"
    POSTGRES_PASSWORD: str = "agrisafe"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "agrisafe"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit default
        "http://localhost:8000"
    ]

    # LLM Configuration
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: str = "openai"  # "openai" or "anthropic"
    LLM_MODEL: str = "gpt-4-turbo-preview"  # or "claude-3-sonnet-20240229"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000

    # Cache TTL (in seconds)
    CACHE_TTL_WEATHER: int = 3600  # 1 hour
    CACHE_TTL_FORECAST: int = 21600  # 6 hours
    CACHE_TTL_LLM: int = 1800  # 30 minutes

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


settings = get_settings()
```

### 1.4 Database Connection

**File**: `src/api/core/database.py`

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator

from src.api.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base class for ORM models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions

    Usage in routes:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### 1.5 API Endpoints Overview

#### Authentication Endpoints (`/api/v1/auth`)
```python
POST   /register          # Create new user account
POST   /login             # Login and get JWT tokens
POST   /refresh           # Refresh access token
POST   /logout            # Invalidate refresh token
GET    /me                # Get current user info
PUT    /me                # Update user profile
POST   /change-password   # Change password
```

#### Regions Endpoints (`/api/v1/regions`)
```python
GET    /                  # List all regions (paginated)
GET    /{region_id}       # Get region details
GET    /search            # Search regions by name/province
```

#### Crops Endpoints (`/api/v1/crops`)
```python
GET    /                  # List all crops (paginated)
GET    /{crop_id}         # Get crop details
GET    /categories        # Get crop categories
GET    /search            # Search crops by name
```

#### Weather Endpoints (`/api/v1/weather`)
```python
GET    /daily/{region_id}        # Daily weather statistics
GET    /rolling/{region_id}      # Rolling window features
GET    /current/{region_id}      # Current weather conditions
```

#### Forecast Endpoints (`/api/v1/forecast`)
```python
GET    /{region_id}              # Get forecast for region (default 5 days)
GET    /{region_id}/risk         # Get flood risk indicators
GET    /map                      # Get forecast map data (all regions)
```

#### Plantings Endpoints (`/api/v1/plantings`) - Protected
```python
GET    /                         # List user's plantings
POST   /                         # Create new planting record
GET    /{planting_id}            # Get planting details
PUT    /{planting_id}            # Update planting
DELETE /{planting_id}            # Delete planting
GET    /{planting_id}/forecast   # Get forecast for planting location
```

#### Harvest Advisor Endpoints (`/api/v1/advisor`)
```python
POST   /chat                     # Send message to harvest advisor
POST   /recommend                # Get harvest recommendation
GET    /history                  # Get chat history (if authenticated)
```

---

## Component 2: LLM Integration

### 2.1 LLM Client Wrapper

**File**: `src/api/core/llm.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from src.api.config import settings


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client wrapper"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using OpenAI"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS
        )

        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic (Claude) API client wrapper"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using Anthropic Claude"""

        # Convert messages format (OpenAI -> Anthropic)
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                conversation_messages.append(msg)

        response = await self.client.messages.create(
            model=self.model,
            system=system_message,
            messages=conversation_messages,
            temperature=temperature or settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS
        )

        return response.content[0].text


def get_llm_client() -> LLMClient:
    """Factory function to get configured LLM client"""

    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        return OpenAIClient(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL
        )

    elif settings.LLM_PROVIDER == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        return AnthropicClient(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.LLM_MODEL
        )

    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
```

### 2.2 Harvest Advisor Service

**File**: `src/api/services/advisor_service.py`

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.core.llm import get_llm_client
from src.api.models.crop import Crop
from src.api.models.region import Region
from src.api.services.forecast_service import ForecastService
from src.api.services.cache_service import CacheService


class HarvestAdvisorService:
    """Service for LLM-powered harvest recommendations"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.llm_client = get_llm_client()
        self.forecast_service = ForecastService(db)
        self.cache_service = CacheService()

    async def get_recommendation(
        self,
        crop_name: str,
        region_name: str,
        planting_date: datetime,
        current_growth_stage: Optional[str] = None
    ) -> Dict:
        """
        Get harvest recommendation for a specific planting

        Args:
            crop_name: Name of the crop (e.g., "Rice", "Corn")
            region_name: Name of the region
            planting_date: Date when crop was planted
            current_growth_stage: Optional current growth stage

        Returns:
            Dict with recommendation, harvest_window, risk_assessment
        """

        # 1. Get crop information
        crop = await self._get_crop_by_name(crop_name)
        if not crop:
            raise ValueError(f"Crop '{crop_name}' not found")

        # 2. Get region information
        region = await self._get_region_by_name(region_name)
        if not region:
            raise ValueError(f"Region '{region_name}' not found")

        # 3. Calculate expected harvest window
        days_since_planting = (datetime.now() - planting_date).days
        expected_harvest_date = planting_date + timedelta(days=crop.avg_maturity_days)
        days_until_harvest = (expected_harvest_date - datetime.now()).days

        # 4. Get weather forecast for harvest window
        forecast_data = await self.forecast_service.get_forecast(
            region_id=str(region.id),
            days=7  # Get 7-day forecast
        )

        # 5. Get flood risk indicators
        flood_risk = await self.forecast_service.get_flood_risk(
            region_id=str(region.id)
        )

        # 6. Build context for LLM
        context = self._build_context(
            crop=crop,
            region=region,
            planting_date=planting_date,
            days_since_planting=days_since_planting,
            days_until_harvest=days_until_harvest,
            expected_harvest_date=expected_harvest_date,
            forecast_data=forecast_data,
            flood_risk=flood_risk,
            current_growth_stage=current_growth_stage
        )

        # 7. Generate LLM recommendation
        recommendation = await self._generate_recommendation(context)

        # 8. Structure response
        return {
            "crop": crop_name,
            "region": region_name,
            "planting_date": planting_date.isoformat(),
            "days_since_planting": days_since_planting,
            "expected_harvest_date": expected_harvest_date.isoformat(),
            "days_until_harvest": days_until_harvest,
            "current_growth_stage": current_growth_stage,
            "recommendation": recommendation,
            "weather_forecast": forecast_data,
            "flood_risk": flood_risk,
            "generated_at": datetime.now().isoformat()
        }

    async def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Interactive chat with harvest advisor

        Args:
            user_message: User's question
            conversation_history: Previous conversation messages

        Returns:
            Assistant's response
        """

        # Check cache for common questions
        cache_key = f"chat:{user_message}"
        cached_response = await self.cache_service.get(cache_key)
        if cached_response:
            return cached_response

        # Build messages for LLM
        messages = self._build_chat_messages(user_message, conversation_history)

        # Generate response
        response = await self.llm_client.chat_completion(messages)

        # Cache response
        await self.cache_service.set(cache_key, response, ttl=1800)  # 30 min

        return response

    def _build_context(
        self,
        crop: Crop,
        region: Region,
        planting_date: datetime,
        days_since_planting: int,
        days_until_harvest: int,
        expected_harvest_date: datetime,
        forecast_data: Dict,
        flood_risk: Dict,
        current_growth_stage: Optional[str]
    ) -> str:
        """Build context string for LLM prompt"""

        context = f"""
# Harvest Advisory Context

## Crop Information
- Crop: {crop.name}
- Category: {crop.category}
- Average Maturity: {crop.avg_maturity_days} days
- Maturity Range: {crop.min_maturity_days}-{crop.max_maturity_days} days
- Optimal Harvest Moisture: {crop.optimal_harvest_moisture_pct}%

## Planting Information
- Planting Date: {planting_date.strftime('%Y-%m-%d')}
- Days Since Planting: {days_since_planting} days
- Expected Harvest Date: {expected_harvest_date.strftime('%Y-%m-%d')}
- Days Until Harvest: {days_until_harvest} days
- Current Growth Stage: {current_growth_stage or 'Not specified'}

## Location
- Region: {region.name}
- Province: {region.province}
- Elevation: {region.elevation}m

## Weather Forecast (Next 7 Days)
"""

        # Add forecast details
        for forecast in forecast_data.get('forecasts', []):
            context += f"""
- {forecast['date']}:
  - Rainfall: {forecast.get('rainfall_mm', 0):.1f}mm
  - Temp: {forecast.get('temp_high', 0):.1f}°C - {forecast.get('temp_low', 0):.1f}°C
  - Condition: {forecast.get('condition', 'N/A')}
"""

        # Add flood risk assessment
        context += f"""
## Flood Risk Assessment
- Risk Level: {flood_risk.get('risk_level', 'Unknown')}
- Confidence: {flood_risk.get('confidence_score', 0):.0%}
- Contributing Factors: {', '.join(flood_risk.get('contributing_factors', {}).keys())}
- Alert: {flood_risk.get('recommendation', '')}
"""

        return context

    async def _generate_recommendation(self, context: str) -> str:
        """Generate harvest recommendation using LLM"""

        system_prompt = """You are an expert agricultural advisor specializing in Philippine farming.
Your role is to provide practical, actionable harvest timing recommendations based on:
- Crop maturity stages
- Weather forecasts
- Flood risk assessments
- Regional conditions

Provide clear, concise advice in a friendly tone. Consider:
1. Optimal harvest timing
2. Weather-related risks
3. Practical steps farmers should take
4. Alternative scenarios (early vs. delayed harvest)

Keep recommendations under 250 words and use simple language."""

        user_prompt = f"""{context}

Based on the above information, provide a harvest recommendation. Include:
1. Recommended action (harvest now, wait, or prepare for early harvest)
2. Reasoning based on weather and crop maturity
3. Specific steps the farmer should take
4. Any risks or concerns to monitor"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return await self.llm_client.chat_completion(messages)

    def _build_chat_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]]
    ) -> List[Dict[str, str]]:
        """Build message list for chat completion"""

        system_prompt = """You are AgriSafe's harvest advisor, an AI assistant helping Filipino farmers
optimize their harvest timing. You have access to weather forecasts, flood risk data, and crop information.

Be helpful, practical, and empathetic. Provide actionable advice in simple language.
If asked about specific crops or regions, suggest using the /recommend endpoint for detailed analysis."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def _get_crop_by_name(self, crop_name: str) -> Optional[Crop]:
        """Get crop by name (case-insensitive)"""
        result = await self.db.execute(
            select(Crop).where(Crop.name.ilike(crop_name))
        )
        return result.scalar_one_or_none()

    async def _get_region_by_name(self, region_name: str) -> Optional[Region]:
        """Get region by name (case-insensitive)"""
        result = await self.db.execute(
            select(Region).where(Region.name.ilike(region_name))
        )
        return result.scalar_one_or_none()
```

### 2.3 Advisor Router (API Endpoints)

**File**: `src/api/routers/advisor.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict
from datetime import datetime

from src.api.core.database import get_db
from src.api.services.advisor_service import HarvestAdvisorService
from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    RecommendationRequest,
    RecommendationResponse
)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_advisor(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with the harvest advisor AI

    Send a natural language question about farming, weather, or harvest timing.
    The AI will provide helpful guidance based on available data.

    Example questions:
    - "When should I harvest my rice in Bulacan?"
    - "Is it safe to harvest corn this week?"
    - "What's the weather forecast for Central Luzon?"
    """

    service = HarvestAdvisorService(db)

    try:
        response = await service.chat(
            user_message=request.message,
            conversation_history=request.history
        )

        return ChatResponse(
            message=response,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/recommend", response_model=RecommendationResponse)
async def get_harvest_recommendation(
    request: RecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed harvest timing recommendation

    Provide crop type, location, and planting date to receive:
    - Optimal harvest window
    - Weather forecast for harvest period
    - Flood risk assessment
    - Actionable recommendations

    This endpoint uses AI to analyze current conditions and provide
    personalized advice for your specific planting.
    """

    service = HarvestAdvisorService(db)

    try:
        recommendation = await service.get_recommendation(
            crop_name=request.crop,
            region_name=request.region,
            planting_date=request.planting_date,
            current_growth_stage=request.growth_stage
        )

        return RecommendationResponse(**recommendation)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendation: {str(e)}"
        )
```

---

## Component 3: Authentication System

### 3.1 Security Utilities

**File**: `src/api/core/security.py`

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.api.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token

    Args:
        data: Data to encode in token (typically {"sub": user_id})
        expires_delta: Token expiration time (default: 30 minutes)

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire, "type": "access"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create JWT refresh token

    Args:
        data: Data to encode in token

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({"exp": expire, "type": "refresh"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Dependency to get current user ID from JWT token

    Usage:
        @app.get("/protected")
        async def protected_route(user_id: str = Depends(get_current_user_id)):
            ...
    """
    token = credentials.credentials
    payload = decode_token(token)

    # Verify token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    return user_id
```

### 3.2 User Model

**File**: `src/api/models/user.py`

```python
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from src.api.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(200))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<User {self.username}>"
```

### 3.3 Authentication Router

**File**: `src/api/routers/auth.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.core.database import get_db
from src.api.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    get_current_user_id
)
from src.api.models.user import User
from src.api.schemas.user import (
    UserCreate,
    UserLogin,
    TokenResponse,
    UserResponse
)

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account

    Creates a new user with email and password authentication.
    Email and username must be unique.
    """

    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    # Create new user
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Login with email and password

    Returns JWT access and refresh tokens for authentication.
    """

    # Find user by email
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalar_one_or_none()

    # Verify credentials
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )

    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current authenticated user information

    Requires valid JWT access token in Authorization header.
    """

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user
```

---

## Component 4: Caching Layer

### 4.1 Redis Cache Service

**File**: `src/api/core/cache.py`

```python
import redis.asyncio as redis
from typing import Optional, Any
import json
from src.api.config import settings

# Create Redis client
redis_client = redis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True
)


class CacheService:
    """Redis caching service"""

    def __init__(self):
        self.client = redis_client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache (will be JSON-serialized)
            ttl: Time-to-live in seconds
        """
        serialized = json.dumps(value) if not isinstance(value, str) else value

        if ttl:
            return await self.client.setex(key, ttl, serialized)
        else:
            return await self.client.set(key, serialized)

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return await self.client.delete(key) > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return await self.client.exists(key) > 0

    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern

        Args:
            pattern: Redis pattern (e.g., "forecast:*")

        Returns:
            Number of keys deleted
        """
        keys = []
        async for key in self.client.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            return await self.client.delete(*keys)
        return 0
```

### 4.2 Caching Decorator

**File**: `src/api/services/cache_service.py`

```python
from functools import wraps
from typing import Callable, Optional
import hashlib
import json

from src.api.core.cache import CacheService
from src.api.config import settings


def cached(
    ttl: Optional[int] = None,
    prefix: str = "cache",
    key_builder: Optional[Callable] = None
):
    """
    Decorator to cache function results in Redis

    Args:
        ttl: Cache TTL in seconds (None = use default)
        prefix: Cache key prefix
        key_builder: Custom function to build cache key from args

    Usage:
        @cached(ttl=3600, prefix="forecast")
        async def get_forecast(region_id: str):
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = CacheService()

            # Build cache key
            if key_builder:
                cache_key = f"{prefix}:{key_builder(*args, **kwargs)}"
            else:
                # Default: hash function name + args
                args_str = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
                args_hash = hashlib.md5(args_str.encode()).hexdigest()
                cache_key = f"{prefix}:{func.__name__}:{args_hash}"

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Example usage in service
class ForecastService:
    """Service for weather forecast operations"""

    def __init__(self, db):
        self.db = db

    @cached(ttl=settings.CACHE_TTL_FORECAST, prefix="forecast")
    async def get_forecast(self, region_id: str, days: int = 5):
        """Get weather forecast (cached for 6 hours)"""
        # Database query here...
        pass
```

---

## Implementation Timeline

### Week 7: Core Backend Development

#### Day 1-2: Project Setup & Core Infrastructure
- [ ] Set up FastAPI project structure
- [ ] Configure database connections (async SQLAlchemy)
- [ ] Set up Redis cache client
- [ ] Create configuration management
- [ ] Docker setup for API service
- [ ] Basic health check endpoint

#### Day 3-4: Database Models & Basic Endpoints
- [ ] Create SQLAlchemy models (User, Region, Crop, etc.)
- [ ] Implement regions router (GET /regions, /regions/{id})
- [ ] Implement crops router (GET /crops, /crops/{id})
- [ ] Add pagination utilities
- [ ] Create response schemas (Pydantic)

#### Day 5-6: Weather & Forecast Endpoints
- [ ] Implement weather service
- [ ] Create forecast endpoints
- [ ] Add flood risk endpoint
- [ ] Implement caching for weather data
- [ ] Add search and filtering capabilities

#### Day 7: Authentication System
- [ ] Implement JWT authentication
- [ ] Create user registration endpoint
- [ ] Create login endpoint
- [ ] Add password hashing utilities
- [ ] Test authentication flow

### Week 8: LLM Integration & Advanced Features

#### Day 1-2: LLM Client Setup
- [ ] Create LLM client wrapper (OpenAI/Anthropic)
- [ ] Implement prompt templates
- [ ] Build context retrieval system
- [ ] Test LLM integration

#### Day 3-4: Harvest Advisor Service
- [ ] Implement advisor service class
- [ ] Create recommendation endpoint
- [ ] Build chat endpoint
- [ ] Add conversation history support
- [ ] Implement LLM response caching

#### Day 5: Plantings Management
- [ ] Create plantings model
- [ ] Implement CRUD endpoints for plantings
- [ ] Add user-specific planting queries
- [ ] Link plantings with forecast data

#### Day 6-7: Testing & Documentation
- [ ] Write unit tests for all services
- [ ] Write integration tests for API endpoints
- [ ] Add API documentation examples
- [ ] Create postman/thunder collection
- [ ] Performance testing

---

## Testing Strategy

### 5.1 Unit Tests

**File**: `tests/api/test_advisor_service.py`

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.api.services.advisor_service import HarvestAdvisorService
from src.api.models.crop import Crop
from src.api.models.region import Region


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_crop():
    return Crop(
        id="crop-1",
        name="Rice",
        category="Grain",
        avg_maturity_days=120,
        min_maturity_days=110,
        max_maturity_days=130
    )


@pytest.fixture
def mock_region():
    return Region(
        id="region-1",
        name="Central Luzon",
        province="Bulacan",
        elevation=10
    )


@pytest.mark.asyncio
async def test_get_recommendation_success(mock_db, mock_crop, mock_region):
    """Test successful harvest recommendation generation"""

    service = HarvestAdvisorService(mock_db)

    # Mock database queries
    service._get_crop_by_name = AsyncMock(return_value=mock_crop)
    service._get_region_by_name = AsyncMock(return_value=mock_region)

    # Mock forecast service
    service.forecast_service.get_forecast = AsyncMock(return_value={
        "forecasts": [
            {"date": "2025-01-18", "rainfall_mm": 5.0, "temp_high": 32.0}
        ]
    })
    service.forecast_service.get_flood_risk = AsyncMock(return_value={
        "risk_level": "Low",
        "confidence_score": 0.85
    })

    # Mock LLM response
    with patch.object(service.llm_client, 'chat_completion', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "Harvest in 10-14 days when moisture is optimal."

        result = await service.get_recommendation(
            crop_name="Rice",
            region_name="Central Luzon",
            planting_date=datetime.now() - timedelta(days=100)
        )

        assert result["crop"] == "Rice"
        assert result["region"] == "Central Luzon"
        assert "recommendation" in result
        assert result["days_since_planting"] == 100


@pytest.mark.asyncio
async def test_chat_with_caching(mock_db):
    """Test chat with cache hit"""

    service = HarvestAdvisorService(mock_db)

    # Mock cache
    service.cache_service.get = AsyncMock(return_value="Cached response")

    response = await service.chat("When should I harvest?")

    assert response == "Cached response"
    service.cache_service.get.assert_called_once()
```

### 5.2 Integration Tests

**File**: `tests/api/test_advisor_endpoints.py`

```python
import pytest
from httpx import AsyncClient
from datetime import datetime

from src.api.main import app


@pytest.mark.asyncio
async def test_chat_endpoint():
    """Test chat endpoint integration"""

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/advisor/chat",
            json={
                "message": "When should I harvest rice?",
                "history": []
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_recommend_endpoint():
    """Test recommendation endpoint integration"""

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/advisor/recommend",
            json={
                "crop": "Rice",
                "region": "Central Luzon",
                "planting_date": "2024-09-01",
                "growth_stage": "Flowering"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["crop"] == "Rice"
        assert "recommendation" in data
        assert "weather_forecast" in data
```

### 5.3 Load Testing

**File**: `tests/load/locustfile.py`

```python
from locust import HttpUser, task, between


class AgriSafeAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def get_regions(self):
        """Test regions endpoint"""
        self.client.get("/api/v1/regions")

    @task(2)
    def get_forecast(self):
        """Test forecast endpoint"""
        self.client.get("/api/v1/forecast/region-1?days=5")

    @task(1)
    def chat_advisor(self):
        """Test chat endpoint"""
        self.client.post(
            "/api/v1/advisor/chat",
            json={"message": "When should I harvest?", "history": []}
        )
```

---

## Dependencies & Setup

### 6.1 Python Dependencies

**Add to `requirements.txt`:**

```
# Core Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# Database
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
alembic==1.13.1
psycopg2-binary==2.9.9

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# LLM Integration
openai==1.10.0
anthropic==0.8.1

# Caching
redis[hiredis]==5.0.1
aiocache==0.12.2

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
httpx==0.26.0
factory-boy==3.3.0
locust==2.20.0

# Utilities
python-dotenv==1.0.0
```

### 6.2 Environment Variables

**File**: `.env.example`

```bash
# Application
DEBUG=false
SECRET_KEY=your-secret-key-here-change-in-production

# Database
POSTGRES_USER=agrisafe
POSTGRES_PASSWORD=agrisafe
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=agrisafe

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# LLM Provider (openai or anthropic)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Security
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8501
```

### 6.3 Docker Configuration

**File**: `docker/api/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Update `docker-compose.yml`:**

```yaml
services:
  # ... existing services ...

  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    container_name: agrisafe-api
    ports:
      - "8000:8000"
    environment:
      - DEBUG=${DEBUG:-false}
      - SECRET_KEY=${SECRET_KEY}
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./src:/app/src
      - ./alembic:/app/alembic
    networks:
      - agrisafe-network
    restart: unless-stopped
```

### 6.4 Makefile Commands

**Add to `Makefile`:**

```makefile
# Phase 4 Commands

.PHONY: api-up
api-up:
	docker-compose up -d api

.PHONY: api-down
api-down:
	docker-compose stop api

.PHONY: api-logs
api-logs:
	docker-compose logs -f api

.PHONY: api-shell
api-shell:
	docker exec -it agrisafe-api bash

.PHONY: api-test
api-test:
	docker exec agrisafe-api pytest tests/api -v --cov=src/api

.PHONY: api-docs
api-docs:
	@echo "API Documentation: http://localhost:8000/api/docs"
	@echo "ReDoc: http://localhost:8000/api/redoc"

.PHONY: db-migrate
db-migrate:
	docker exec agrisafe-api alembic upgrade head

.PHONY: db-revision
db-revision:
	docker exec agrisafe-api alembic revision --autogenerate -m "$(message)"
```

---

## Success Criteria

### Functional Requirements
- [ ] All API endpoints implemented and documented
- [ ] LLM integration working with both OpenAI and Anthropic
- [ ] JWT authentication fully functional
- [ ] Redis caching operational for all services
- [ ] User registration and login working
- [ ] Harvest recommendations generating successfully

### Performance Requirements
- [ ] API response time < 200ms (p95) for cached requests
- [ ] API response time < 500ms (p95) for database queries
- [ ] LLM responses < 3 seconds
- [ ] Support 100+ concurrent users
- [ ] Cache hit rate > 70% for forecast endpoints

### Quality Requirements
- [ ] Unit test coverage > 80%
- [ ] Integration test coverage for all endpoints
- [ ] API documentation complete with examples
- [ ] Error handling comprehensive
- [ ] Logging configured properly
- [ ] Security best practices followed

### Documentation Requirements
- [ ] OpenAPI/Swagger documentation auto-generated
- [ ] API usage guide with examples
- [ ] Authentication flow documented
- [ ] Deployment guide
- [ ] Troubleshooting guide

---

## Next Steps (Phase 5)

After completing Phase 4, we'll be ready for:
1. **Streamlit Frontend**: User-facing web application
2. **Dashboard Visualizations**: Charts and maps
3. **Chat Interface**: Interactive harvest advisor UI
4. **User Planting Tracker**: Personal farming records

---

## Appendix

### A. API Endpoint Summary

```
Authentication:
POST   /api/v1/auth/register
POST   /api/v1/auth/login
GET    /api/v1/auth/me

Data Endpoints:
GET    /api/v1/regions
GET    /api/v1/regions/{id}
GET    /api/v1/crops
GET    /api/v1/crops/{id}

Weather & Forecast:
GET    /api/v1/weather/daily/{region_id}
GET    /api/v1/forecast/{region_id}
GET    /api/v1/forecast/{region_id}/risk

Plantings (Protected):
GET    /api/v1/plantings
POST   /api/v1/plantings
GET    /api/v1/plantings/{id}
PUT    /api/v1/plantings/{id}
DELETE /api/v1/plantings/{id}

Harvest Advisor:
POST   /api/v1/advisor/chat
POST   /api/v1/advisor/recommend
```

### B. Sample API Responses

**GET /api/v1/forecast/{region_id}**
```json
{
  "region_id": "region-1",
  "region_name": "Central Luzon",
  "forecasts": [
    {
      "date": "2025-01-18",
      "rainfall_mm": 5.2,
      "temp_high": 32.5,
      "temp_low": 24.0,
      "condition": "Partly Cloudy",
      "flood_risk": "Low"
    }
  ],
  "metadata": {
    "generated_at": "2025-01-17T10:00:00Z",
    "cache_hit": true
  }
}
```

**POST /api/v1/advisor/recommend**
```json
{
  "crop": "Rice",
  "region": "Central Luzon",
  "planting_date": "2024-09-01",
  "days_since_planting": 138,
  "expected_harvest_date": "2025-01-29",
  "days_until_harvest": 12,
  "recommendation": "Your rice is approaching optimal harvest maturity...",
  "weather_forecast": {...},
  "flood_risk": {...}
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-17
**Author**: AgriSafe Development Team
**Status**: Ready for Implementation 🚀
