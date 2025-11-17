"""
PAGASA API Connector for weather data ingestion
"""

import time
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

from ..models.weather import WeatherForecast, PAGASAResponse, WeatherCondition
from ..utils.database import get_db_connection
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PAGASAConnector:
    """
    Connector for PAGASA weather data

    This connector can fetch data from multiple sources:
    1. Community-hosted PAGASA API (Vercel)
    2. Direct PAGASA website scraping (future)
    """

    # API endpoints
    VERCEL_API_URL = "https://pagasa-forecast-api.vercel.app/api/pagasa-forecast"
    PAGASA_OFFICIAL_URL = "https://www.pagasa.dost.gov.ph/weather"

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        """
        Initialize PAGASA connector

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff factor for retries
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session()
        logger.info("PAGASA Connector initialized")

    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic

        Returns:
            Configured requests session
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def fetch_forecast_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch weather forecast data from PAGASA sources

        Returns:
            Raw forecast data as dictionary, or None if fetch fails
        """
        logger.info("Fetching weather forecast from PAGASA...")

        # Try Vercel API first
        try:
            response = self.session.get(
                self.VERCEL_API_URL,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            logger.info("Successfully fetched data from Vercel API")

            # Log API call to database
            self._log_api_call(
                api_name="PAGASA_Vercel",
                endpoint=self.VERCEL_API_URL,
                status=response.status_code,
                response_time=response.elapsed.total_seconds() * 1000
            )

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch from Vercel API: {e}")

            # Log failed API call
            self._log_api_call(
                api_name="PAGASA_Vercel",
                endpoint=self.VERCEL_API_URL,
                status=0,
                error_message=str(e)
            )

            # TODO: Add fallback to direct PAGASA scraping
            logger.warning("No fallback source available yet")
            return None

    def parse_forecast_response(self, raw_data: Dict[str, Any]) -> PAGASAResponse:
        """
        Parse raw PAGASA API response into structured model

        Args:
            raw_data: Raw API response dictionary

        Returns:
            Parsed PAGASA response model
        """
        try:
            # Parse issued timestamp
            issued_at = None
            if 'issued' in raw_data:
                try:
                    issued_at = datetime.fromisoformat(raw_data['issued'])
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse issued timestamp: {raw_data.get('issued')}")

            pagasa_response = PAGASAResponse(
                issued_at=issued_at,
                synopsis=raw_data.get('synopsis'),
                weather_conditions=raw_data.get('weather_conditions'),
                wind_conditions=raw_data.get('wind_conditions'),
                temperature=raw_data.get('temperature'),
                humidity=raw_data.get('humidity'),
                tides=raw_data.get('tides'),
                astronomical=raw_data.get('astronomical'),
                raw_json=raw_data
            )

            logger.info("Successfully parsed PAGASA response")
            return pagasa_response

        except Exception as e:
            logger.error(f"Error parsing PAGASA response: {e}")
            raise

    def extract_regional_forecasts(
        self,
        pagasa_response: PAGASAResponse,
        regions: List[Dict[str, Any]]
    ) -> List[WeatherForecast]:
        """
        Extract regional weather forecasts from PAGASA response

        Args:
            pagasa_response: Parsed PAGASA response
            regions: List of region dictionaries from database

        Returns:
            List of WeatherForecast objects
        """
        forecasts = []

        # PAGASA typically provides national/regional overview
        # We'll create forecasts for all regions with the general data
        # In a real scenario, you'd parse region-specific data from the response

        issued_at = pagasa_response.issued_at or datetime.now()

        # Extract temperature data
        temp_data = pagasa_response.temperature or {}
        temp_min = None
        temp_max = None

        # Parse temperature from the response structure
        # This is a simplified version - actual parsing depends on PAGASA response format
        if isinstance(temp_data, dict):
            for location, temps in temp_data.items():
                if isinstance(temps, dict):
                    if 'min' in temps:
                        temp_min = float(temps['min'])
                    if 'max' in temps:
                        temp_max = float(temps['max'])
                    break  # Use first location as general reference

        # Extract weather condition
        weather_desc = pagasa_response.synopsis or ""
        weather_condition = self._determine_weather_condition(weather_desc)

        # Create forecasts for each region
        # For now, we'll create 5-day forecasts with general national data
        for region in regions:
            for day_offset in range(5):
                forecast_date = (datetime.now() + timedelta(days=day_offset)).date()

                forecast = WeatherForecast(
                    region_id=region['id'],
                    forecast_date=forecast_date,
                    forecast_created_at=issued_at,
                    temperature_min=temp_min,
                    temperature_max=temp_max,
                    temperature_avg=(temp_min + temp_max) / 2 if temp_min and temp_max else None,
                    weather_condition=weather_condition,
                    weather_description=weather_desc,
                    data_source="PAGASA",
                    raw_data=pagasa_response.raw_json
                )

                forecasts.append(forecast)

        logger.info(f"Extracted {len(forecasts)} regional forecasts")
        return forecasts

    def _determine_weather_condition(self, description: str) -> WeatherCondition:
        """
        Determine weather condition from description text

        Args:
            description: Weather description string

        Returns:
            WeatherCondition enum value
        """
        if not description:
            return WeatherCondition.UNKNOWN

        desc_lower = description.lower()

        if any(word in desc_lower for word in ['typhoon', 'tropical cyclone', 'tropical storm']):
            return WeatherCondition.TYPHOON
        elif any(word in desc_lower for word in ['thunderstorm', 'lightning']):
            return WeatherCondition.THUNDERSTORM
        elif any(word in desc_lower for word in ['heavy rain', 'heavy rainfall']):
            return WeatherCondition.HEAVY_RAIN
        elif any(word in desc_lower for word in ['moderate rain', 'rain showers']):
            return WeatherCondition.MODERATE_RAIN
        elif any(word in desc_lower for word in ['light rain', 'drizzle']):
            return WeatherCondition.LIGHT_RAIN
        elif 'overcast' in desc_lower:
            return WeatherCondition.OVERCAST
        elif any(word in desc_lower for word in ['cloudy', 'clouds']):
            return WeatherCondition.CLOUDY
        elif any(word in desc_lower for word in ['partly cloudy', 'partly sunny']):
            return WeatherCondition.PARTLY_CLOUDY
        elif any(word in desc_lower for word in ['sunny', 'clear', 'fair']):
            return WeatherCondition.SUNNY
        else:
            return WeatherCondition.UNKNOWN

    def _log_api_call(
        self,
        api_name: str,
        endpoint: str,
        status: int,
        response_time: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """
        Log API call to database for monitoring

        Args:
            api_name: Name of the API
            endpoint: API endpoint URL
            status: HTTP status code
            response_time: Response time in milliseconds
            error_message: Error message if call failed
        """
        try:
            db = get_db_connection()

            query = """
                INSERT INTO api_logs (
                    api_name, endpoint, request_method,
                    response_status, response_time_ms, error_message
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            params = (
                api_name,
                endpoint,
                'GET',
                status,
                int(response_time) if response_time else None,
                error_message
            )

            db.execute_query(query, params, fetch=False)
            logger.debug(f"Logged API call: {api_name} - Status: {status}")

        except Exception as e:
            logger.error(f"Failed to log API call: {e}")


class PAGASAIngestionService:
    """
    Service for ingesting PAGASA weather data into database
    """

    def __init__(self):
        """Initialize ingestion service"""
        self.connector = PAGASAConnector()
        self.db = get_db_connection()
        logger.info("PAGASA Ingestion Service initialized")

    def get_regions(self) -> List[Dict[str, Any]]:
        """
        Get all regions from database

        Returns:
            List of region dictionaries
        """
        query = "SELECT id, region_name, province, municipality FROM regions ORDER BY id"
        regions = self.db.execute_query(query)
        logger.info(f"Retrieved {len(regions)} regions from database")
        return regions

    def save_weather_forecasts(self, forecasts: List[WeatherForecast]) -> int:
        """
        Save weather forecasts to database

        Args:
            forecasts: List of WeatherForecast objects

        Returns:
            Number of forecasts saved
        """
        if not forecasts:
            logger.warning("No forecasts to save")
            return 0

        query = """
            INSERT INTO weather_forecasts (
                region_id, forecast_date, forecast_created_at,
                temperature_min, temperature_max, temperature_avg,
                humidity_percent, rainfall_mm, wind_speed_kph,
                weather_condition, weather_description,
                data_source, raw_data
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (region_id, forecast_date, forecast_created_at)
            DO UPDATE SET
                temperature_min = EXCLUDED.temperature_min,
                temperature_max = EXCLUDED.temperature_max,
                temperature_avg = EXCLUDED.temperature_avg,
                humidity_percent = EXCLUDED.humidity_percent,
                rainfall_mm = EXCLUDED.rainfall_mm,
                wind_speed_kph = EXCLUDED.wind_speed_kph,
                weather_condition = EXCLUDED.weather_condition,
                weather_description = EXCLUDED.weather_description,
                data_source = EXCLUDED.data_source,
                raw_data = EXCLUDED.raw_data
        """

        params_list = []
        for forecast in forecasts:
            params = (
                forecast.region_id,
                forecast.forecast_date,
                forecast.forecast_created_at,
                forecast.temperature_min,
                forecast.temperature_max,
                forecast.temperature_avg,
                forecast.humidity_percent,
                forecast.rainfall_mm,
                forecast.wind_speed_kph,
                forecast.weather_condition,
                forecast.weather_description,
                forecast.data_source,
                json.dumps(forecast.raw_data) if forecast.raw_data else None
            )
            params_list.append(params)

        try:
            self.db.execute_many(query, params_list)
            logger.info(f"Successfully saved {len(forecasts)} weather forecasts")
            return len(forecasts)

        except Exception as e:
            logger.error(f"Failed to save weather forecasts: {e}")
            raise

    def run_ingestion(self) -> Dict[str, Any]:
        """
        Run the full ingestion pipeline

        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        logger.info("Starting PAGASA weather data ingestion...")

        try:
            # Step 1: Fetch data from PAGASA
            raw_data = self.connector.fetch_forecast_data()
            if not raw_data:
                logger.error("Failed to fetch weather data")
                return {
                    'success': False,
                    'error': 'Failed to fetch data from PAGASA',
                    'duration_seconds': time.time() - start_time
                }

            # Step 2: Parse response
            pagasa_response = self.connector.parse_forecast_response(raw_data)

            # Step 3: Get regions from database
            regions = self.get_regions()
            if not regions:
                logger.error("No regions found in database")
                return {
                    'success': False,
                    'error': 'No regions found in database',
                    'duration_seconds': time.time() - start_time
                }

            # Step 4: Extract regional forecasts
            forecasts = self.connector.extract_regional_forecasts(
                pagasa_response,
                regions
            )

            # Step 5: Save to database
            saved_count = self.save_weather_forecasts(forecasts)

            duration = time.time() - start_time

            result = {
                'success': True,
                'forecasts_saved': saved_count,
                'regions_processed': len(regions),
                'issued_at': pagasa_response.issued_at.isoformat() if pagasa_response.issued_at else None,
                'duration_seconds': duration
            }

            logger.info(f"Ingestion completed successfully in {duration:.2f}s")
            logger.info(f"Results: {result}")

            return result

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }


def main():
    """Main function for running ingestion standalone"""
    service = PAGASAIngestionService()
    result = service.run_ingestion()

    if result['success']:
        print(f"✓ Ingestion successful: {result['forecasts_saved']} forecasts saved")
    else:
        print(f"✗ Ingestion failed: {result['error']}")

    return result


if __name__ == "__main__":
    main()
