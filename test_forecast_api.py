"""
Test script for Forecast Router API endpoints
Run this after starting the API server: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import requests
import json
from datetime import date, timedelta

BASE_URL = "http://localhost:8000/api/v1"


def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response:\n{json.dumps(response.json(), indent=2, default=str)}")
    else:
        print(f"Error: {response.text}")
    print()


def test_forecast_health():
    """Test forecast health check endpoint"""
    response = requests.get(f"{BASE_URL}/forecast/health")
    print_response("1. Forecast Health Check", response)
    return response.json() if response.status_code == 200 else None


def test_list_forecasts(page=1, page_size=5, region_id=None):
    """Test list all forecasts endpoint"""
    params = {"page": page, "page_size": page_size}
    if region_id:
        params["region_id"] = region_id

    response = requests.get(f"{BASE_URL}/forecast/list", params=params)
    print_response(f"2. List Forecasts (page={page}, page_size={page_size})", response)
    return response.json() if response.status_code == 200 else None


def test_get_region_forecast(region_id=1, days=7):
    """Test get region forecast endpoint"""
    response = requests.get(f"{BASE_URL}/forecast/region/{region_id}", params={"days": days})
    print_response(f"3. Get Region {region_id} Forecast ({days} days)", response)
    return response.json() if response.status_code == 200 else None


def test_get_current_forecast(region_id=1):
    """Test get current forecast endpoint"""
    response = requests.get(f"{BASE_URL}/forecast/region/{region_id}/current")
    print_response(f"4. Get Current Forecast for Region {region_id}", response)
    return response.json() if response.status_code == 200 else None


def test_get_region_summary(region_id=1, days=7):
    """Test get region forecast summary endpoint"""
    response = requests.get(f"{BASE_URL}/forecast/region/{region_id}/summary", params={"days": days})
    print_response(f"5. Get Region {region_id} Summary ({days} days)", response)
    return response.json() if response.status_code == 200 else None


def test_get_forecast_by_id(forecast_id):
    """Test get forecast by ID endpoint"""
    response = requests.get(f"{BASE_URL}/forecast/{forecast_id}")
    print_response(f"6. Get Forecast by ID: {forecast_id}", response)
    return response.json() if response.status_code == 200 else None


def test_filter_by_date_range(region_id=1, days_ahead=30):
    """Test filtering forecasts by date range"""
    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    params = {
        "region_id": region_id,
        "start_date": today.isoformat(),
        "end_date": end_date.isoformat(),
        "page_size": 10
    }

    response = requests.get(f"{BASE_URL}/forecast/list", params=params)
    print_response(f"7. Filter Forecasts by Date Range ({today} to {end_date})", response)
    return response.json() if response.status_code == 200 else None


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FORECAST ROUTER API TESTS")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print("="*60 + "\n")

    try:
        # Test 1: Health check
        health = test_forecast_health()

        # Test 2: List forecasts
        forecast_list = test_list_forecasts(page=1, page_size=5)

        # Test 3: Get region forecast
        region_forecast = test_get_region_forecast(region_id=1, days=7)

        # Test 4: Get current forecast
        current = test_get_current_forecast(region_id=1)

        # Test 5: Get region summary with recommendations
        summary = test_get_region_summary(region_id=1, days=7)

        # Test 6: If we got forecasts, test getting by ID
        if forecast_list and forecast_list.get('forecasts'):
            first_forecast = forecast_list['forecasts'][0]
            forecast_id = first_forecast['id']
            test_get_forecast_by_id(forecast_id)
        else:
            print("\nSkipping Test 6: No forecasts available to test by ID")

        # Test 7: Filter by date range
        test_filter_by_date_range(region_id=1, days_ahead=30)

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the server is running:")
        print("  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        print()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")


if __name__ == "__main__":
    main()
