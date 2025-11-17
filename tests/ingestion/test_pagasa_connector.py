"""
Tests for PAGASA Connector
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from src.ingestion.pagasa_connector import PAGASAConnector, PAGASAIngestionService
from src.models.weather import WeatherForecast, PAGASAResponse, WeatherCondition


class TestPAGASAConnector:
    """Test cases for PAGASAConnector class"""

    @pytest.fixture
    def connector(self):
        """Create a connector instance for testing"""
        return PAGASAConnector(timeout=10, max_retries=2)

    @pytest.fixture
    def mock_pagasa_response(self):
        """Mock PAGASA API response"""
        return {
            "issued": "2025-11-17T06:00:00+08:00",
            "synopsis": "The Northeast Monsoon affecting Luzon. Partly cloudy to cloudy skies with isolated light rains.",
            "weather_conditions": {
                "Luzon": "Partly cloudy to cloudy skies with isolated light rains",
                "Visayas": "Partly cloudy to cloudy skies with isolated rainshowers",
                "Mindanao": "Partly cloudy to cloudy skies with isolated rainshowers"
            },
            "wind_conditions": {
                "Luzon": "Moderate to strong winds from the Northeast",
                "Visayas": "Light to moderate winds from the Northeast",
                "Mindanao": "Light to moderate winds from the Northeast"
            },
            "temperature": {
                "Metro Manila": {"min": 24, "max": 31},
                "Baguio City": {"min": 16, "max": 23},
                "Tuguegarao": {"min": 23, "max": 32}
            },
            "humidity": {
                "Metro Manila": {"value": 70}
            }
        }

    def test_connector_initialization(self, connector):
        """Test that connector initializes correctly"""
        assert connector.timeout == 10
        assert connector.max_retries == 2
        assert connector.session is not None

    @patch('src.ingestion.pagasa_connector.requests.Session')
    def test_create_session_with_retry(self, mock_session_class, connector):
        """Test session creation with retry logic"""
        # The session should be created with retry configuration
        assert connector.session is not None

    @patch('src.ingestion.pagasa_connector.requests.Session.get')
    def test_fetch_forecast_data_success(self, mock_get, connector, mock_pagasa_response):
        """Test successful data fetch from API"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_pagasa_response
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_get.return_value = mock_response

        # Fetch data
        result = connector.fetch_forecast_data()

        # Assertions
        assert result is not None
        assert result['synopsis'] == mock_pagasa_response['synopsis']
        mock_get.assert_called_once()

    @patch('src.ingestion.pagasa_connector.requests.Session.get')
    def test_fetch_forecast_data_failure(self, mock_get, connector):
        """Test handling of API fetch failure"""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")

        # Fetch data
        result = connector.fetch_forecast_data()

        # Should return None on failure
        assert result is None

    def test_parse_forecast_response(self, connector, mock_pagasa_response):
        """Test parsing of PAGASA API response"""
        # Parse response
        parsed = connector.parse_forecast_response(mock_pagasa_response)

        # Assertions
        assert isinstance(parsed, PAGASAResponse)
        assert parsed.synopsis == mock_pagasa_response['synopsis']
        assert parsed.weather_conditions == mock_pagasa_response['weather_conditions']
        assert parsed.raw_json == mock_pagasa_response

    def test_extract_regional_forecasts(self, connector, mock_pagasa_response):
        """Test extraction of regional forecasts"""
        # Parse response first
        pagasa_response = connector.parse_forecast_response(mock_pagasa_response)

        # Mock regions
        regions = [
            {'id': 1, 'region_name': 'Metro Manila', 'province': 'NCR'},
            {'id': 2, 'region_name': 'Central Luzon', 'province': 'Bulacan'}
        ]

        # Extract forecasts
        forecasts = connector.extract_regional_forecasts(pagasa_response, regions)

        # Assertions
        assert len(forecasts) == 10  # 2 regions * 5 days
        assert all(isinstance(f, WeatherForecast) for f in forecasts)

        # Check first forecast
        first_forecast = forecasts[0]
        assert first_forecast.region_id in [1, 2]
        assert first_forecast.temperature_min == 24
        assert first_forecast.temperature_max == 31

    @pytest.mark.parametrize("description,expected_condition", [
        ("Typhoon warning in effect", WeatherCondition.TYPHOON),
        ("Thunderstorms expected", WeatherCondition.THUNDERSTORM),
        ("Heavy rainfall", WeatherCondition.HEAVY_RAIN),
        ("Moderate rain showers", WeatherCondition.MODERATE_RAIN),
        ("Light rain", WeatherCondition.LIGHT_RAIN),
        ("Cloudy skies", WeatherCondition.CLOUDY),
        ("Partly cloudy", WeatherCondition.PARTLY_CLOUDY),
        ("Sunny weather", WeatherCondition.SUNNY),
        ("", WeatherCondition.UNKNOWN),
    ])
    def test_determine_weather_condition(self, connector, description, expected_condition):
        """Test weather condition determination from description"""
        result = connector._determine_weather_condition(description)
        assert result == expected_condition


class TestPAGASAIngestionService:
    """Test cases for PAGASAIngestionService class"""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing"""
        with patch('src.ingestion.pagasa_connector.get_db_connection'):
            return PAGASAIngestionService()

    @pytest.fixture
    def mock_regions(self):
        """Mock regions from database"""
        return [
            {'id': 1, 'region_name': 'Metro Manila', 'province': 'NCR', 'municipality': None},
            {'id': 2, 'region_name': 'Central Luzon', 'province': 'Bulacan', 'municipality': 'Malolos'}
        ]

    @pytest.fixture
    def mock_forecasts(self):
        """Mock weather forecasts"""
        return [
            WeatherForecast(
                region_id=1,
                forecast_date=date.today(),
                temperature_min=24.0,
                temperature_max=31.0,
                weather_condition=WeatherCondition.PARTLY_CLOUDY,
                data_source="PAGASA"
            ),
            WeatherForecast(
                region_id=2,
                forecast_date=date.today(),
                temperature_min=23.0,
                temperature_max=30.0,
                weather_condition=WeatherCondition.CLOUDY,
                data_source="PAGASA"
            )
        ]

    @patch('src.ingestion.pagasa_connector.get_db_connection')
    def test_get_regions(self, mock_db, service, mock_regions):
        """Test getting regions from database"""
        # Mock database response
        mock_db_instance = Mock()
        mock_db_instance.execute_query.return_value = mock_regions
        mock_db.return_value = mock_db_instance

        # Reinitialize service with mocked db
        service.db = mock_db_instance

        # Get regions
        regions = service.get_regions()

        # Assertions
        assert len(regions) == 2
        assert regions[0]['region_name'] == 'Metro Manila'

    @patch('src.ingestion.pagasa_connector.get_db_connection')
    def test_save_weather_forecasts(self, mock_db, service, mock_forecasts):
        """Test saving weather forecasts to database"""
        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.execute_many.return_value = None
        mock_db.return_value = mock_db_instance

        # Reinitialize service with mocked db
        service.db = mock_db_instance

        # Save forecasts
        saved_count = service.save_weather_forecasts(mock_forecasts)

        # Assertions
        assert saved_count == 2
        mock_db_instance.execute_many.assert_called_once()

    def test_save_empty_forecasts(self, service):
        """Test saving empty forecast list"""
        result = service.save_weather_forecasts([])
        assert result == 0

    @patch.object(PAGASAConnector, 'fetch_forecast_data')
    @patch.object(PAGASAIngestionService, 'get_regions')
    @patch.object(PAGASAIngestionService, 'save_weather_forecasts')
    def test_run_ingestion_success(
        self,
        mock_save,
        mock_get_regions,
        mock_fetch,
        service,
        mock_regions
    ):
        """Test successful ingestion run"""
        # Mock data
        mock_fetch.return_value = {
            "issued": "2025-11-17T06:00:00+08:00",
            "synopsis": "Test synopsis",
            "temperature": {"Metro Manila": {"min": 24, "max": 31}}
        }
        mock_get_regions.return_value = mock_regions
        mock_save.return_value = 10

        # Run ingestion
        result = service.run_ingestion()

        # Assertions
        assert result['success'] is True
        assert result['forecasts_saved'] == 10
        assert result['regions_processed'] == 2
        assert 'duration_seconds' in result

    @patch.object(PAGASAConnector, 'fetch_forecast_data')
    def test_run_ingestion_fetch_failure(self, mock_fetch, service):
        """Test ingestion when fetch fails"""
        # Mock fetch failure
        mock_fetch.return_value = None

        # Run ingestion
        result = service.run_ingestion()

        # Assertions
        assert result['success'] is False
        assert 'error' in result

    @patch.object(PAGASAConnector, 'fetch_forecast_data')
    @patch.object(PAGASAIngestionService, 'get_regions')
    def test_run_ingestion_no_regions(self, mock_get_regions, mock_fetch, service):
        """Test ingestion when no regions found"""
        # Mock data
        mock_fetch.return_value = {"synopsis": "Test"}
        mock_get_regions.return_value = []

        # Run ingestion
        result = service.run_ingestion()

        # Assertions
        assert result['success'] is False
        assert 'No regions found' in result['error']


class TestWeatherForecastModel:
    """Test cases for WeatherForecast model"""

    def test_valid_forecast_creation(self):
        """Test creating a valid weather forecast"""
        forecast = WeatherForecast(
            region_id=1,
            forecast_date=date.today(),
            temperature_min=20.0,
            temperature_max=30.0,
            humidity_percent=70.0,
            rainfall_mm=5.5,
            weather_condition=WeatherCondition.PARTLY_CLOUDY
        )

        assert forecast.region_id == 1
        assert forecast.temperature_min == 20.0
        assert forecast.weather_condition == WeatherCondition.PARTLY_CLOUDY

    def test_invalid_temperature(self):
        """Test validation of invalid temperature"""
        with pytest.raises(ValueError):
            WeatherForecast(
                region_id=1,
                forecast_date=date.today(),
                temperature_min=100.0,  # Invalid: too high
                temperature_max=30.0
            )

    def test_invalid_humidity(self):
        """Test validation of invalid humidity"""
        with pytest.raises(ValueError):
            WeatherForecast(
                region_id=1,
                forecast_date=date.today(),
                humidity_percent=150.0  # Invalid: > 100
            )

    def test_invalid_rainfall(self):
        """Test validation of invalid rainfall"""
        with pytest.raises(ValueError):
            WeatherForecast(
                region_id=1,
                forecast_date=date.today(),
                rainfall_mm=-5.0  # Invalid: negative
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
