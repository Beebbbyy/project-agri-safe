#!/bin/bash

# Test script for Forecast Router endpoints
# Make sure the API server is running on port 8000 before running this script

BASE_URL="http://localhost:8000/api/v1"

echo "========================================"
echo "Testing Forecast Router Endpoints"
echo "========================================"
echo ""

# 1. Health Check
echo "1. Testing Forecast Health Check"
echo "GET ${BASE_URL}/forecast/health"
curl -X GET "${BASE_URL}/forecast/health" | jq .
echo -e "\n"

# 2. List all forecasts (paginated)
echo "2. Testing List All Forecasts"
echo "GET ${BASE_URL}/forecast/list?page=1&page_size=5"
curl -X GET "${BASE_URL}/forecast/list?page=1&page_size=5" | jq .
echo -e "\n"

# 3. Get forecasts for a specific region
echo "3. Testing Get Region Forecast (7 days)"
echo "GET ${BASE_URL}/forecast/region/1?days=7"
curl -X GET "${BASE_URL}/forecast/region/1?days=7" | jq .
echo -e "\n"

# 4. Get current forecast for a region
echo "4. Testing Get Current Forecast"
echo "GET ${BASE_URL}/forecast/region/1/current"
curl -X GET "${BASE_URL}/forecast/region/1/current" | jq .
echo -e "\n"

# 5. Get comprehensive summary for a region
echo "5. Testing Get Region Forecast Summary"
echo "GET ${BASE_URL}/forecast/region/1/summary?days=7"
curl -X GET "${BASE_URL}/forecast/region/1/summary?days=7" | jq .
echo -e "\n"

# 6. Filter forecasts by date range
echo "6. Testing Filter Forecasts by Date Range"
echo "GET ${BASE_URL}/forecast/list?region_id=1&start_date=2025-01-01&end_date=2025-01-31"
curl -X GET "${BASE_URL}/forecast/list?region_id=1&start_date=2025-01-01&end_date=2025-01-31" | jq .
echo -e "\n"

# 7. Get specific forecast by ID (you'll need a real forecast ID)
echo "7. Testing Get Forecast by ID"
echo "Note: Replace {forecast_id} with an actual UUID from the list endpoint"
echo "GET ${BASE_URL}/forecast/{forecast_id}"
echo "(Skipping this test - requires actual forecast ID)"
echo -e "\n"

echo "========================================"
echo "All tests completed!"
echo "========================================"
