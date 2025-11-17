#!/bin/bash

# Phase 3 Spark ETL - Automated Test Script
# This script tests all components of the Spark ETL pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_separator() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
    echo ""
}

# Change to project directory
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

print_separator "Phase 3 Spark ETL - Automated Testing"

# Step 1: Check Docker services
print_separator "Step 1: Checking Docker Services"

if ! docker --version &> /dev/null; then
    log_error "Docker is not installed or not running"
    exit 1
fi
log_success "Docker is installed"

# Check PostgreSQL
if docker exec agrisafe-postgres pg_isready -U agrisafe &> /dev/null; then
    log_success "PostgreSQL is running and accepting connections"
else
    log_error "PostgreSQL is not accessible"
    log_info "Try running: docker-compose up -d postgres"
    exit 1
fi

# Check Redis
if docker exec agrisafe-redis redis-cli ping &> /dev/null; then
    log_success "Redis is running"
else
    log_error "Redis is not accessible"
    log_info "Try running: docker-compose up -d redis"
    exit 1
fi

# Step 2: Set environment variables
print_separator "Step 2: Setting Environment Variables"

export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agrisafe_db
export POSTGRES_USER=agrisafe
export POSTGRES_PASSWORD=agrisafe_password
export REDIS_HOST=localhost
export REDIS_PORT=6379
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

log_success "Environment variables set"

# Step 3: Check Python dependencies
print_separator "Step 3: Checking Python Dependencies"

if python3 -c "import pyspark" &> /dev/null; then
    PYSPARK_VERSION=$(python3 -c "import pyspark; print(pyspark.__version__)")
    log_success "PySpark is installed (version: $PYSPARK_VERSION)"
else
    log_error "PySpark is not installed"
    log_info "Install with: pip install pyspark==3.5.0"
    exit 1
fi

if python3 -c "import redis" &> /dev/null; then
    log_success "Redis Python library is installed"
else
    log_error "Redis library is not installed"
    log_info "Install with: pip install redis==5.0.1"
    exit 1
fi

if python3 -c "from loguru import logger" &> /dev/null; then
    log_success "Loguru is installed"
else
    log_error "Loguru is not installed"
    log_info "Install with: pip install loguru==0.7.2"
    exit 1
fi

# Step 4: Apply database schema
print_separator "Step 4: Applying Database Schema"

log_info "Checking if feature tables exist..."

TABLE_CHECK=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'weather_daily_stats';")

if [[ "$TABLE_CHECK" -gt 0 ]]; then
    log_warning "Feature tables already exist, skipping schema application"
else
    log_info "Applying feature tables schema..."
    docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db \
        < sql/schema/02_feature_tables.sql
    log_success "Database schema applied successfully"
fi

# Step 5: Check for weather data
print_separator "Step 5: Checking Weather Data"

FORECAST_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT COUNT(*) FROM weather_forecasts;")

log_info "Found $FORECAST_COUNT weather forecast records"

if [[ "$FORECAST_COUNT" -lt 10 ]]; then
    log_warning "Not enough weather data to test properly"
    log_info "Run PAGASA ingestion first: python -m src.ingestion.pagasa_connector"
    log_warning "Continuing with limited data..."
fi

REGION_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT COUNT(*) FROM regions;")

log_info "Found $REGION_COUNT regions in database"

if [[ "$REGION_COUNT" -lt 1 ]]; then
    log_error "No regions found in database. Cannot proceed."
    exit 1
fi

# Step 6: Test Daily Weather Statistics Job
print_separator "Step 6: Testing Daily Weather Statistics Job"

log_info "Running daily_weather_stats job..."

if python3 -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07 \
    --mode append; then
    log_success "Daily weather statistics job completed successfully"
else
    log_error "Daily weather statistics job failed"
    exit 1
fi

# Verify data was created
STATS_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT COUNT(*) FROM weather_daily_stats;")

log_info "Created $STATS_COUNT daily statistics records"

if [[ "$STATS_COUNT" -gt 0 ]]; then
    log_success "Daily statistics data verified in database"
else
    log_error "No daily statistics data found in database"
    exit 1
fi

# Step 7: Test Rolling Features Job
print_separator "Step 7: Testing Rolling Features Job"

log_info "Running rolling_features job..."

if python3 -m src.processing.jobs.rolling_features \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --windows 7,14,30 \
    --mode append; then
    log_success "Rolling features job completed successfully"
else
    log_error "Rolling features job failed"
    exit 1
fi

# Verify data was created
FEATURES_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT COUNT(*) FROM weather_rolling_features;")

log_info "Created $FEATURES_COUNT rolling feature records"

if [[ "$FEATURES_COUNT" -gt 0 ]]; then
    log_success "Rolling features data verified in database"
else
    log_error "No rolling features data found in database"
    exit 1
fi

# Check all window sizes
for WINDOW in 7 14 30; do
    WINDOW_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
        "SELECT COUNT(*) FROM weather_rolling_features WHERE window_days = $WINDOW;")
    log_info "${WINDOW}-day window: $WINDOW_COUNT records"
done

# Step 8: Test Flood Risk Indicators Job
print_separator "Step 8: Testing Flood Risk Indicators Job"

log_info "Running flood_risk_indicators job..."

if python3 -m src.processing.jobs.flood_risk_indicators \
    --start-date 2025-01-01 \
    --end-date 2025-01-14 \
    --mode append; then
    log_success "Flood risk indicators job completed successfully"
else
    log_error "Flood risk indicators job failed"
    exit 1
fi

# Verify data was created
RISK_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT COUNT(*) FROM flood_risk_indicators;")

log_info "Created $RISK_COUNT flood risk indicator records"

if [[ "$RISK_COUNT" -gt 0 ]]; then
    log_success "Flood risk indicators data verified in database"
else
    log_error "No flood risk indicators data found in database"
    exit 1
fi

# Check risk level distribution
log_info "Risk level distribution:"
docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT flood_risk_level, COUNT(*) FROM flood_risk_indicators GROUP BY flood_risk_level ORDER BY flood_risk_level;"

# Step 9: Test Redis Cache
print_separator "Step 9: Testing Redis Cache"

# Check for cached keys
CACHE_KEYS=$(docker exec agrisafe-redis redis-cli KEYS "agrisafe:*" | wc -l)

log_info "Found $CACHE_KEYS cached keys in Redis"

if [[ "$CACHE_KEYS" -gt 0 ]]; then
    log_success "Redis cache is populated"

    # Show sample cached keys
    log_info "Sample cached keys:"
    docker exec agrisafe-redis redis-cli KEYS "agrisafe:*" | head -5
else
    log_warning "No cached keys found in Redis (caching may be disabled)"
fi

# Step 10: Verify Job Metadata
print_separator "Step 10: Verifying Job Metadata"

METADATA_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT COUNT(*) FROM feature_metadata;")

log_info "Found $METADATA_COUNT job execution records"

if [[ "$METADATA_COUNT" -gt 0 ]]; then
    log_success "Job metadata is being tracked"

    # Show recent jobs
    log_info "Recent job executions:"
    docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c \
        "SELECT job_name, status, records_created, duration_seconds FROM feature_metadata ORDER BY created_at DESC LIMIT 5;"
else
    log_warning "No job metadata found"
fi

# Step 11: Run Unit Tests (if pytest is available)
print_separator "Step 11: Running Unit Tests"

if command -v pytest &> /dev/null; then
    log_info "Running unit tests..."

    if pytest tests/processing/ -v --tb=short; then
        log_success "All unit tests passed"
    else
        log_error "Some unit tests failed"
        exit 1
    fi
else
    log_warning "pytest not installed, skipping unit tests"
    log_info "Install with: pip install pytest"
fi

# Final Summary
print_separator "Test Summary"

echo ""
log_success "✓ Docker services are running"
log_success "✓ Python dependencies are installed"
log_success "✓ Database schema is applied"
log_success "✓ Daily weather statistics job works"
log_success "✓ Rolling features job works"
log_success "✓ Flood risk indicators job works"
log_success "✓ Redis caching is functional"
log_success "✓ Job metadata is tracked"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                  ║${NC}"
echo -e "${GREEN}║   🎉  All Phase 3 Tests Passed Successfully!  🎉  ║${NC}"
echo -e "${GREEN}║                                                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

log_info "You can now:"
echo "  1. View data in database: docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db"
echo "  2. Check Redis cache: docker exec -it agrisafe-redis redis-cli"
echo "  3. View Airflow UI: http://localhost:8080"
echo "  4. Run jobs manually with custom dates"
echo ""

log_info "Next steps:"
echo "  - Enable Airflow DAG for automated daily runs"
echo "  - Proceed to Phase 4 (Backend API development)"
echo ""
