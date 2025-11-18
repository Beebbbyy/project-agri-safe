from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

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
            # Note: We're not creating tables here since they already exist
            # await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        # Don't raise - allow API to start even if DB is temporarily unavailable

    # Test Redis connection
    try:
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        # Don't raise - allow API to start even if Redis is temporarily unavailable

    yield

    # Shutdown
    logger.info("Shutting down AgriSafe API...")
    try:
        await redis_client.close()
    except Exception:
        pass
    try:
        await engine.dispose()
    except Exception:
        pass


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
    allow_origins=settings.cors_origins_list,
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
    db_status = "connected"
    redis_status = "connected"

    # Test database
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
    except Exception:
        db_status = "disconnected"

    # Test Redis
    try:
        await redis_client.ping()
    except Exception:
        redis_status = "disconnected"

    return {
        "status": "healthy" if (db_status == "connected" and redis_status == "connected") else "degraded",
        "version": "1.0.0",
        "database": db_status,
        "cache": redis_status
    }


# API version prefix
API_V1_PREFIX = "/api/v1"

# Import routers
from src.api.routers import auth, regions, crops, weather, forecast, plantings

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


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to AgriSafe API",
        "docs": "/api/docs",
        "version": "1.0.0",
        "status": "Phase 4 - Backend API Development",
        "endpoints": {
            "health": "/health",
            "docs": "/api/docs",
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
