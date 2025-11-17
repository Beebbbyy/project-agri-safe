#!/bin/bash
# ==================================================
# Phase 2 Testing Script - PAGASA API Connector
# ==================================================

set -e  # Exit on error

echo "=========================================="
echo "Phase 2: PAGASA API Connector - Test Suite"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        ((TESTS_FAILED++))
    fi
}

# Function to print section header
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# ==================================================
# Test 1: Check Docker Services
# ==================================================
print_header "Test 1: Docker Services Health Check"

echo "Checking if PostgreSQL is running..."
if docker-compose ps postgres | grep -q "healthy\|Up"; then
    print_status 0 "PostgreSQL service is running"
else
    print_status 1 "PostgreSQL service is not running"
fi

echo "Checking if Redis is running..."
if docker-compose ps redis | grep -q "healthy\|Up"; then
    print_status 0 "Redis service is running"
else
    print_status 1 "Redis service is not running"
fi

echo "Checking if Airflow Webserver is running..."
if docker-compose ps airflow-webserver | grep -q "healthy\|Up"; then
    print_status 0 "Airflow Webserver service is running"
else
    print_status 1 "Airflow Webserver service is not running"
fi

# ==================================================
# Test 2: Check Database Schema
# ==================================================
print_header "Test 2: Database Schema Verification"

echo "Checking if weather_forecasts table exists..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'weather_forecasts');" | grep -q "t"; then
    print_status 0 "weather_forecasts table exists"
else
    print_status 1 "weather_forecasts table does not exist"
fi

echo "Checking if api_logs table exists..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_logs');" | grep -q "t"; then
    print_status 0 "api_logs table exists"
else
    print_status 1 "api_logs table does not exist"
fi

echo "Checking if regions table has data..."
REGION_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -tAc "SELECT COUNT(*) FROM regions;")
if [ "$REGION_COUNT" -gt 0 ]; then
    print_status 0 "Regions table has $REGION_COUNT regions"
else
    print_status 1 "Regions table is empty"
fi

# ==================================================
# Test 3: Check File Structure
# ==================================================
print_header "Test 3: File Structure Verification"

FILES=(
    "src/utils/__init__.py"
    "src/utils/database.py"
    "src/utils/logger.py"
    "src/models/__init__.py"
    "src/models/weather.py"
    "src/ingestion/__init__.py"
    "src/ingestion/pagasa_connector.py"
    "airflow/dags/pagasa_daily_ingestion.py"
    "tests/ingestion/test_pagasa_connector.py"
    "docs/PHASE2_PAGASA_CONNECTOR.md"
    "docs/PHASE2_QUICK_START.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        print_status 0 "File exists: $file"
    else
        print_status 1 "File missing: $file"
    fi
done

# ==================================================
# Test 4: Python Syntax Check
# ==================================================
print_header "Test 4: Python Syntax Check"

echo "Checking Python syntax for connector..."
if python3 -m py_compile src/ingestion/pagasa_connector.py 2>/dev/null; then
    print_status 0 "PAGASA connector syntax is valid"
else
    print_status 1 "PAGASA connector has syntax errors"
fi

echo "Checking Python syntax for models..."
if python3 -m py_compile src/models/weather.py 2>/dev/null; then
    print_status 0 "Weather models syntax is valid"
else
    print_status 1 "Weather models have syntax errors"
fi

echo "Checking Python syntax for DAG..."
if python3 -m py_compile airflow/dags/pagasa_daily_ingestion.py 2>/dev/null; then
    print_status 0 "Airflow DAG syntax is valid"
else
    print_status 1 "Airflow DAG has syntax errors"
fi

# ==================================================
# Test 5: External API Connectivity
# ==================================================
print_header "Test 5: External API Connectivity"

echo "Testing PAGASA Vercel API..."
if curl -s --max-time 10 https://pagasa-forecast-api.vercel.app/api/pagasa-forecast | grep -q "issued\|synopsis"; then
    print_status 0 "PAGASA Vercel API is reachable and responding"
else
    echo -e "${YELLOW}⚠ WARNING${NC}: PAGASA Vercel API may be unreachable (this is non-critical)"
fi

# ==================================================
# Test 6: Airflow DAG Detection
# ==================================================
print_header "Test 6: Airflow DAG Detection"

# Wait a moment for Airflow to detect DAGs
echo "Waiting for Airflow to detect DAGs..."
sleep 3

echo "Checking if pagasa_daily_ingestion DAG is loaded..."
if docker exec agrisafe-airflow-webserver airflow dags list 2>/dev/null | grep -q "pagasa_daily_ingestion"; then
    print_status 0 "Airflow DAG 'pagasa_daily_ingestion' is loaded"
else
    echo -e "${YELLOW}⚠ WARNING${NC}: DAG may not be loaded yet (check Airflow UI)"
fi

# ==================================================
# Summary
# ==================================================
print_header "Test Summary"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))

echo ""
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "✓ ALL TESTS PASSED!"
    echo "==========================================${NC}"
    echo ""
    echo "Phase 2 implementation is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Access Airflow UI: http://localhost:8080"
    echo "  2. Enable the 'pagasa_daily_ingestion' DAG"
    echo "  3. Trigger a manual run to test ingestion"
    echo "  4. Check database for weather forecast data"
    echo ""
    exit 0
else
    echo -e "${RED}=========================================="
    echo "✗ SOME TESTS FAILED"
    echo "==========================================${NC}"
    echo ""
    echo "Please review the failed tests above and fix any issues."
    echo ""
    exit 1
fi
