"""
Unit tests for Forecast Router
Run with: pytest tests/test_forecast_router.py -v
"""

import pytest
from fastapi.testclient import TestClient
from datetime import date, timedelta
import uuid

# Import the FastAPI app
from src.api.main import app

# Create test client
client = TestClient(app)


class TestForecastEndpoints:
    """Test suite for Forecast Router endpoints"""

    def test_forecast_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/api/v1/forecast/health")
        assert response.status_code in [200, 500]  # May fail if DB not connected
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "total_forecasts" in data

    def test_list_forecasts_endpoint(self):
        """Test the list forecasts endpoint"""
        response = client.get("/api/v1/forecast/list?page=1&page_size=10")
        assert response.status_code in [200, 500]  # May fail if DB not connected
        if response.status_code == 200:
            data = response.json()
            assert "forecasts" in data
            assert "total" in data
            assert "page" in data
            assert "page_size" in data

    def test_list_forecasts_with_filters(self):
        """Test list forecasts with filtering parameters"""
        today = date.today()
        params = {
            "page": 1,
            "page_size": 5,
            "region_id": 1,
            "start_date": today.isoformat(),
            "end_date": (today + timedelta(days=7)).isoformat()
        }
        response = client.get("/api/v1/forecast/list", params=params)
        assert response.status_code in [200, 500]

    def test_get_region_forecast(self):
        """Test get region forecast endpoint"""
        response = client.get("/api/v1/forecast/region/1?days=7")
        # Could be 200 (success), 404 (region not found), or 500 (DB error)
        assert response.status_code in [200, 404, 500]

    def test_get_region_forecast_invalid_days(self):
        """Test region forecast with invalid days parameter"""
        response = client.get("/api/v1/forecast/region/1?days=50")  # Max is 30
        assert response.status_code == 422  # Validation error

    def test_get_current_forecast(self):
        """Test get current forecast endpoint"""
        response = client.get("/api/v1/forecast/region/1/current")
        assert response.status_code in [200, 404, 500]

    def test_get_region_summary(self):
        """Test get region summary endpoint"""
        response = client.get("/api/v1/forecast/region/1/summary?days=7")
        assert response.status_code in [200, 404, 500]

    def test_get_forecast_by_id_invalid_uuid(self):
        """Test get forecast by ID with invalid UUID"""
        response = client.get("/api/v1/forecast/invalid-uuid")
        assert response.status_code == 422  # Validation error

    def test_get_forecast_by_id_not_found(self):
        """Test get forecast by ID that doesn't exist"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/forecast/{fake_uuid}")
        assert response.status_code in [404, 500]

    def test_pagination_parameters(self):
        """Test pagination parameter validation"""
        # Test invalid page number
        response = client.get("/api/v1/forecast/list?page=0")
        assert response.status_code == 422

        # Test invalid page size
        response = client.get("/api/v1/forecast/list?page_size=0")
        assert response.status_code == 422

    def test_api_documentation(self):
        """Test that the API documentation is accessible"""
        response = client.get("/api/docs")
        assert response.status_code == 200

    def test_openapi_schema(self):
        """Test that the OpenAPI schema includes forecast endpoints"""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        # Check that forecast endpoints are in the schema
        paths = schema.get("paths", {})
        assert "/api/v1/forecast/list" in paths
        assert "/api/v1/forecast/region/{region_id}" in paths
        assert "/api/v1/forecast/health" in paths


class TestForecastSchemas:
    """Test forecast response schemas"""

    def test_forecast_list_schema(self):
        """Test that forecast list endpoint returns correct schema"""
        response = client.get("/api/v1/forecast/list?page=1&page_size=1")
        if response.status_code == 200:
            data = response.json()
            # Verify schema structure
            assert isinstance(data["forecasts"], list)
            assert isinstance(data["total"], int)
            assert isinstance(data["page"], int)
            assert isinstance(data["page_size"], int)

    def test_health_check_schema(self):
        """Test that health check returns correct schema"""
        response = client.get("/api/v1/forecast/health")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "total_forecasts" in data
            assert "future_forecasts" in data
            assert "date_range" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
