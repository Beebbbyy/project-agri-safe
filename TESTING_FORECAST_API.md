# Testing the Forecast Router API

This guide shows you how to test the newly implemented Forecast Router endpoints.

## Quick Verification

### 1. Check Code Structure (No Server Required)

Verify that all files are properly structured:

```bash
# Check Python syntax
python3 -m py_compile src/api/routers/forecast.py
python3 -m py_compile src/api/models/weather.py
python3 -m py_compile src/api/schemas/weather.py

# Verify imports work
python3 -c "from src.api.routers import forecast; print('✓ Forecast router loads successfully')"
```

### 2. View API Documentation

Once the server is running, access the interactive API docs:

**Swagger UI:** http://localhost:8000/api/docs
**ReDoc:** http://localhost:8000/api/redoc

---

## Available Endpoints

All forecast endpoints are prefixed with `/api/v1/forecast`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/list` | List all forecasts with pagination |
| GET | `/{forecast_id}` | Get comprehensive forecast by ID |
| GET | `/region/{region_id}` | Get multi-day forecast for region |
| GET | `/region/{region_id}/current` | Get current forecast for region |
| GET | `/region/{region_id}/summary` | Get comprehensive summary with risks |
| GET | `/health` | Health check and statistics |

---

## Testing Methods

### Method 1: Interactive API Docs (Recommended)

1. **Start the server:**
   ```bash
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Open your browser:**
   - Navigate to: http://localhost:8000/api/docs
   - Find the "Forecast" section
   - Click "Try it out" on any endpoint
   - Fill in parameters and click "Execute"

### Method 2: Using the Shell Script

```bash
# Make sure server is running first
./test_forecast_endpoints.sh
```

### Method 3: Using Python Test Script

```bash
# Make sure server is running first
python test_forecast_api.py
```

### Method 4: Using curl

```bash
# 1. Health check
curl http://localhost:8000/api/v1/forecast/health | jq .

# 2. List forecasts (first page, 5 items)
curl "http://localhost:8000/api/v1/forecast/list?page=1&page_size=5" | jq .

# 3. Get 7-day forecast for region 1
curl "http://localhost:8000/api/v1/forecast/region/1?days=7" | jq .

# 4. Get current forecast for region 1
curl "http://localhost:8000/api/v1/forecast/region/1/current" | jq .

# 5. Get comprehensive summary with risk indicators
curl "http://localhost:8000/api/v1/forecast/region/1/summary?days=7" | jq .

# 6. Filter forecasts by date range
curl "http://localhost:8000/api/v1/forecast/list?region_id=1&start_date=2025-01-01&end_date=2025-01-31" | jq .
```

### Method 5: Using pytest

```bash
# Run unit tests
pytest tests/test_forecast_router.py -v

# Run with coverage
pytest tests/test_forecast_router.py -v --cov=src.api.routers.forecast
```

---

## Example Responses

### Health Check Response
```json
{
  "status": "healthy",
  "total_forecasts": 1500,
  "future_forecasts": 210,
  "regions_with_forecasts": 15,
  "date_range": {
    "earliest": "2025-01-01",
    "latest": "2025-02-15"
  },
  "last_updated": "2025-01-17T10:30:00"
}
```

### Region Summary Response
```json
{
  "region_id": 1,
  "region_name": "Metro Manila",
  "province": "NCR",
  "summary_date": "2025-01-17",
  "forecast_days": 7,
  "avg_temperature": 28.5,
  "total_rainfall": 45.2,
  "max_wind_speed": 25.0,
  "dominant_condition": "Partly Cloudy",
  "overall_flood_risk": "Moderate",
  "overall_harvest_suitability": 0.75,
  "daily_forecasts": [
    {
      "forecast_date": "2025-01-17",
      "temperature_min": 24.0,
      "temperature_max": 32.0,
      "rainfall_mm": 5.2,
      "weather_condition": "Partly Cloudy",
      "flood_risk_level": "Low",
      "flood_risk_score": 0.2,
      "typhoon_probability": 0.1,
      "harvest_suitability": 0.8,
      "recommendations": [
        "Weather conditions are normal. Continue regular farm operations.",
        "Good harvest conditions. Proceed with planned operations."
      ]
    }
  ]
}
```

---

## Common Issues & Solutions

### Issue: "No module named 'fastapi'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Could not connect to database"
**Solution:** Check database configuration in `.env` or `src/api/config.py`

### Issue: "Region not found (404)"
**Solution:** Make sure you have regions in your database. Check available regions:
```bash
curl http://localhost:8000/api/v1/regions | jq .
```

### Issue: "No current forecast available"
**Solution:** The database needs forecast data. Check if forecasts exist:
```bash
curl "http://localhost:8000/api/v1/forecast/list?page=1&page_size=1" | jq .
```

---

## Query Parameters

### Pagination (for `/list` endpoint)
- `page` (int, default: 1) - Page number (min: 1)
- `page_size` (int, default: 20) - Items per page (min: 1, max: 100)

### Filtering (for `/list` endpoint)
- `region_id` (int) - Filter by specific region
- `start_date` (date, format: YYYY-MM-DD) - Start of date range
- `end_date` (date, format: YYYY-MM-DD) - End of date range

### Days (for region endpoints)
- `days` (int, default: 7) - Number of days to forecast (min: 1, max: 30)

---

## Next Steps

1. **Add Integration Tests:** Create tests with actual database data
2. **Add Caching:** Implement Redis caching for frequently accessed forecasts
3. **Add Rate Limiting:** Protect endpoints from abuse
4. **Add Authentication:** Secure endpoints with JWT tokens
5. **Monitor Performance:** Track response times and optimize queries

---

## Support

- API Documentation: http://localhost:8000/api/docs
- OpenAPI Schema: http://localhost:8000/api/openapi.json
- GitHub Issues: https://github.com/Beebbbyy/project-agri-safe/issues
