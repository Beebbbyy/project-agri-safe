"""
Unit tests for Weather ETL Pipeline (PySpark)

Tests the complete ETL pipeline including:
- Data extraction from PostgreSQL
- Daily statistics computation
- Rolling window feature calculation
- Data validation
- Loading to PostgreSQL and Redis
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

from src.processing.spark_jobs.weather_etl import WeatherETL


@pytest.fixture(scope="module")
def spark_session():
    """Create a test Spark session"""
    spark = SparkSession.builder \
        .appName("TestWeatherETL") \
        .master("local[2]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

    yield spark

    spark.stop()


@pytest.fixture
def sample_weather_data(spark_session):
    """Create sample weather forecast data"""
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("region_id", StringType(), True),
        StructField("region_name", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("elevation", DoubleType(), True),
        StructField("forecast_date", DateType(), True),
        StructField("temperature_high", DoubleType(), True),
        StructField("temperature_low", DoubleType(), True),
        StructField("rainfall_mm", DoubleType(), True),
        StructField("wind_speed", DoubleType(), True),
        StructField("weather_condition", StringType(), True),
    ])

    data = [
        ("1", "REG001", "Region A", 14.5, 121.0, 50.0, date(2025, 1, 1), 32.0, 24.0, 10.0, 25.0, "Partly Cloudy"),
        ("2", "REG001", "Region A", 14.5, 121.0, 50.0, date(2025, 1, 2), 33.0, 25.0, 15.0, 30.0, "Cloudy"),
        ("3", "REG001", "Region A", 14.5, 121.0, 50.0, date(2025, 1, 3), 31.0, 23.0, 50.0, 35.0, "Rainy"),
        ("4", "REG002", "Region B", 15.0, 120.5, 100.0, date(2025, 1, 1), 30.0, 22.0, 5.0, 20.0, "Sunny"),
        ("5", "REG002", "Region B", 15.0, 120.5, 100.0, date(2025, 1, 2), 31.0, 23.0, 8.0, 22.0, "Partly Cloudy"),
        ("6", "REG002", "Region B", 15.0, 120.5, 100.0, date(2025, 1, 3), 32.0, 24.0, 100.0, 45.0, "Heavy Rain"),
    ]

    return spark_session.createDataFrame(data, schema)


@pytest.fixture
def etl_instance(spark_session):
    """Create WeatherETL instance with mocked database connection"""
    with patch('src.processing.spark_jobs.weather_etl.os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            'POSTGRES_HOST': 'test_host',
            'POSTGRES_DB': 'test_db',
            'POSTGRES_USER': 'test_user',
            'POSTGRES_PASSWORD': 'test_password',
            'REDIS_HOST': 'test_redis',
            'REDIS_PORT': '6379'
        }.get(key, default)

        etl = WeatherETL(spark=spark_session)
        return etl


class TestWeatherETLInitialization:
    """Test ETL initialization and configuration"""

    def test_init_with_provided_spark_session(self, spark_session):
        """Test initialization with provided Spark session"""
        etl = WeatherETL(spark=spark_session)
        assert etl.spark is not None
        assert etl.spark == spark_session

    def test_init_creates_spark_session_if_none(self):
        """Test that Spark session is created if not provided"""
        with patch('src.processing.spark_jobs.weather_etl.SparkSession') as mock_spark:
            mock_builder = MagicMock()
            mock_spark.builder = mock_builder

            etl = WeatherETL()

            mock_builder.appName.assert_called()
            mock_builder.config.assert_called()

    def test_database_configuration(self, etl_instance):
        """Test database connection parameters"""
        assert etl_instance.db_host == 'test_host'
        assert etl_instance.db_name == 'test_db'
        assert etl_instance.db_user == 'test_user'
        assert etl_instance.db_password == 'test_password'
        assert 'postgresql://test_host:5432/test_db' in etl_instance.jdbc_url

    def test_jdbc_properties(self, etl_instance):
        """Test JDBC connection properties"""
        assert etl_instance.jdbc_properties['user'] == 'test_user'
        assert etl_instance.jdbc_properties['password'] == 'test_password'
        assert etl_instance.jdbc_properties['driver'] == 'org.postgresql.Driver'


class TestExtractWeatherData:
    """Test data extraction from PostgreSQL"""

    def test_extract_weather_data_builds_correct_query(self, etl_instance, sample_weather_data):
        """Test that extraction builds correct SQL query"""
        with patch.object(etl_instance.spark.read, 'jdbc', return_value=sample_weather_data) as mock_jdbc:
            result = etl_instance.extract_weather_data('2025-01-01', '2025-01-03')

            mock_jdbc.assert_called_once()
            call_args = mock_jdbc.call_args

            # Verify table query contains date filter
            table_query = call_args[1]['table']
            assert '2025-01-01' in table_query
            assert '2025-01-03' in table_query

    def test_extract_with_region_filter(self, etl_instance, sample_weather_data):
        """Test extraction with region ID filter"""
        with patch.object(etl_instance.spark.read, 'jdbc', return_value=sample_weather_data) as mock_jdbc:
            result = etl_instance.extract_weather_data('2025-01-01', '2025-01-03', region_ids=['REG001'])

            table_query = mock_jdbc.call_args[1]['table']
            assert 'REG001' in table_query

    def test_extract_returns_dataframe(self, etl_instance, sample_weather_data):
        """Test that extraction returns a DataFrame"""
        with patch.object(etl_instance.spark.read, 'jdbc', return_value=sample_weather_data):
            result = etl_instance.extract_weather_data('2025-01-01', '2025-01-03')

            assert isinstance(result, DataFrame)
            assert result.count() > 0

    def test_extract_handles_database_error(self, etl_instance):
        """Test error handling for database connection issues"""
        with patch.object(etl_instance.spark.read, 'jdbc', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception) as exc_info:
                etl_instance.extract_weather_data('2025-01-01', '2025-01-03')

            assert "Connection failed" in str(exc_info.value)


class TestComputeDailyStats:
    """Test daily statistics computation"""

    def test_compute_daily_stats_aggregates_correctly(self, etl_instance, sample_weather_data):
        """Test that daily stats are computed correctly"""
        result = etl_instance.compute_daily_stats(sample_weather_data)

        # Check that we get one row per region-date combination
        result_count = result.count()
        assert result_count == 6  # 2 regions × 3 dates

        # Check required columns exist
        required_cols = ['region_id', 'forecast_date', 'temp_high_avg', 'temp_low_avg',
                        'rainfall_total', 'wind_speed_max', 'dominant_condition']
        for col in required_cols:
            assert col in result.columns

    def test_daily_stats_rainfall_totals(self, etl_instance, sample_weather_data):
        """Test rainfall is summed correctly"""
        result = etl_instance.compute_daily_stats(sample_weather_data)

        # Get rainfall for REG001 on 2025-01-01 (should be 10.0)
        result_pd = result.filter("region_id = 'REG001' AND forecast_date = '2025-01-01'").toPandas()
        assert len(result_pd) == 1
        assert result_pd.iloc[0]['rainfall_total'] == 10.0

    def test_daily_stats_temperature_averages(self, etl_instance, sample_weather_data):
        """Test temperature averages are calculated correctly"""
        result = etl_instance.compute_daily_stats(sample_weather_data)

        result_pd = result.filter("region_id = 'REG001' AND forecast_date = '2025-01-01'").toPandas()
        assert result_pd.iloc[0]['temp_high_avg'] == 32.0
        assert result_pd.iloc[0]['temp_low_avg'] == 24.0

    def test_daily_stats_wind_speed_max(self, etl_instance, sample_weather_data):
        """Test maximum wind speed is captured"""
        result = etl_instance.compute_daily_stats(sample_weather_data)

        result_pd = result.filter("region_id = 'REG001' AND forecast_date = '2025-01-03'").toPandas()
        assert result_pd.iloc[0]['wind_speed_max'] == 35.0

    def test_daily_stats_includes_created_at(self, etl_instance, sample_weather_data):
        """Test that created_at timestamp is added"""
        result = etl_instance.compute_daily_stats(sample_weather_data)

        assert 'created_at' in result.columns


class TestComputeRollingFeatures:
    """Test rolling window feature computation"""

    def test_rolling_features_3day_rainfall(self, etl_instance, spark_session):
        """Test 3-day rolling rainfall calculation"""
        # Create sequential data
        data = [
            ("REG001", "Region A", date(2025, 1, 1), 30.0, 20.0, 10.0, 25.0, 50.0, 1),
            ("REG001", "Region A", date(2025, 1, 2), 32.0, 22.0, 20.0, 30.0, 50.0, 1),
            ("REG001", "Region A", date(2025, 1, 3), 31.0, 21.0, 30.0, 28.0, 50.0, 1),
            ("REG001", "Region A", date(2025, 1, 4), 33.0, 23.0, 15.0, 32.0, 50.0, 1),
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'region_name', 'forecast_date', 'temp_high_avg', 'temp_low_avg',
             'rainfall_total', 'temp_avg', 'elevation', 'forecast_count'])

        result = etl_instance.compute_rolling_features(df)
        result_pd = result.toPandas().sort_values('forecast_date')

        # Day 3 should have 3-day sum of 10+20+30 = 60
        day3 = result_pd[result_pd['forecast_date'] == date(2025, 1, 3)]
        assert day3.iloc[0]['rainfall_3d'] == 60.0

    def test_rolling_features_7day_rainfall(self, etl_instance, spark_session):
        """Test 7-day rolling rainfall calculation"""
        data = []
        for i in range(10):
            data.append(("REG001", "Region A", date(2025, 1, 1) + timedelta(days=i),
                        30.0, 20.0, 10.0, 25.0, 50.0, 1))

        df = spark_session.createDataFrame(data,
            ['region_id', 'region_name', 'forecast_date', 'temp_high_avg', 'temp_low_avg',
             'rainfall_total', 'temp_avg', 'elevation', 'forecast_count'])

        result = etl_instance.compute_rolling_features(df)
        result_pd = result.toPandas().sort_values('forecast_date')

        # Day 7 should have 7-day sum of 7*10 = 70
        day7 = result_pd[result_pd['forecast_date'] == date(2025, 1, 7)]
        assert day7.iloc[0]['rainfall_7d'] == 70.0

    def test_rolling_features_temperature_average(self, etl_instance, spark_session):
        """Test 7-day temperature average"""
        data = []
        for i in range(10):
            data.append(("REG001", "Region A", date(2025, 1, 1) + timedelta(days=i),
                        30.0, 20.0, 10.0, 25.0, 50.0, 1))

        df = spark_session.createDataFrame(data,
            ['region_id', 'region_name', 'forecast_date', 'temp_high_avg', 'temp_low_avg',
             'rainfall_total', 'temp_avg', 'elevation', 'forecast_count'])

        result = etl_instance.compute_rolling_features(df)
        result_pd = result.toPandas()

        # Check temp_avg_7d exists and is reasonable
        assert 'temp_avg_7d' in result_pd.columns
        assert result_pd['temp_avg_7d'].max() == 25.0

    def test_rolling_features_rainy_days_count(self, etl_instance, spark_session):
        """Test rainy days counter (rainfall > 5mm)"""
        data = [
            ("REG001", "Region A", date(2025, 1, 1), 30.0, 20.0, 2.0, 25.0, 50.0, 1),   # Not rainy
            ("REG001", "Region A", date(2025, 1, 2), 30.0, 20.0, 10.0, 25.0, 50.0, 1),  # Rainy
            ("REG001", "Region A", date(2025, 1, 3), 30.0, 20.0, 15.0, 25.0, 50.0, 1),  # Rainy
            ("REG001", "Region A", date(2025, 1, 4), 30.0, 20.0, 3.0, 25.0, 50.0, 1),   # Not rainy
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'region_name', 'forecast_date', 'temp_high_avg', 'temp_low_avg',
             'rainfall_total', 'temp_avg', 'elevation', 'forecast_count'])

        result = etl_instance.compute_rolling_features(df)
        result_pd = result.toPandas().sort_values('forecast_date')

        # Should count days with rainfall > 5.0
        assert 'rainy_days_7d' in result_pd.columns

    def test_rolling_features_handles_nulls(self, etl_instance, spark_session):
        """Test that null values are handled properly"""
        data = [
            ("REG001", "Region A", date(2025, 1, 1), 30.0, 20.0, 10.0, 25.0, 50.0, 1),
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'region_name', 'forecast_date', 'temp_high_avg', 'temp_low_avg',
             'rainfall_total', 'temp_avg', 'elevation', 'forecast_count'])

        result = etl_instance.compute_rolling_features(df)
        result_pd = result.toPandas()

        # Check that nulls are filled with 0
        assert result_pd['rainfall_3d'].isna().sum() == 0
        assert result_pd['rainfall_7d'].isna().sum() == 0


class TestLoadToPostgres:
    """Test loading data to PostgreSQL"""

    def test_load_to_postgres_calls_jdbc_write(self, etl_instance, sample_weather_data):
        """Test that load calls JDBC write with correct parameters"""
        with patch.object(sample_weather_data.write, 'jdbc') as mock_write:
            etl_instance.load_to_postgres(sample_weather_data, 'test_table', mode='append')

            mock_write.assert_called_once()
            call_args = mock_write.call_args[1]

            assert call_args['table'] == 'test_table'
            assert call_args['mode'] == 'append'

    def test_load_handles_write_error(self, etl_instance, sample_weather_data):
        """Test error handling during database write"""
        with patch.object(sample_weather_data.write, 'jdbc', side_effect=Exception("Write failed")):
            with pytest.raises(Exception) as exc_info:
                etl_instance.load_to_postgres(sample_weather_data, 'test_table')

            assert "Write failed" in str(exc_info.value)


class TestCacheFeaturesToRedis:
    """Test Redis caching functionality"""

    @patch('src.processing.spark_jobs.weather_etl.redis.Redis')
    def test_cache_features_to_redis(self, mock_redis_class, etl_instance, spark_session):
        """Test caching features to Redis"""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance

        data = [
            ("REG001", date(2025, 1, 1), 10.0, 30.0, 60.0, 25.0, 2.0, 28.0, 3),
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'forecast_date', 'rainfall_3d', 'rainfall_7d', 'rainfall_14d',
             'temp_avg_7d', 'temp_variance_30d', 'wind_speed_avg_7d', 'rainy_days_7d'])

        etl_instance.cache_features_to_redis(df)

        # Verify Redis client was created
        mock_redis_class.assert_called_once()

        # Verify setex was called
        assert mock_redis_instance.setex.called

        # Verify close was called
        mock_redis_instance.close.assert_called_once()

    @patch('src.processing.spark_jobs.weather_etl.redis.Redis')
    def test_cache_handles_redis_unavailable(self, mock_redis_class, etl_instance, spark_session):
        """Test graceful handling when Redis is unavailable"""
        mock_redis_class.side_effect = ImportError("Redis not available")

        data = [("REG001", date(2025, 1, 1), 10.0, 30.0, 60.0, 25.0, 2.0, 28.0, 3)]
        df = spark_session.createDataFrame(data,
            ['region_id', 'forecast_date', 'rainfall_3d', 'rainfall_7d', 'rainfall_14d',
             'temp_avg_7d', 'temp_variance_30d', 'wind_speed_avg_7d', 'rainy_days_7d'])

        # Should not raise exception
        etl_instance.cache_features_to_redis(df)


class TestValidateData:
    """Test data validation"""

    def test_validate_data_passes_for_valid_data(self, etl_instance, sample_weather_data):
        """Test validation passes for valid DataFrame"""
        result = etl_instance.validate_data(sample_weather_data, "test_stage")
        assert result is True

    def test_validate_data_fails_for_empty_dataframe(self, etl_instance, spark_session):
        """Test validation fails for empty DataFrame"""
        empty_df = spark_session.createDataFrame([],
            StructType([StructField("region_id", StringType(), True)]))

        result = etl_instance.validate_data(empty_df, "test_stage")
        assert result is False

    def test_validate_data_checks_required_columns(self, etl_instance, spark_session):
        """Test validation checks for required columns"""
        # DataFrame missing forecast_date
        df = spark_session.createDataFrame([("REG001",)], ["region_id"])

        result = etl_instance.validate_data(df, "test_stage")
        assert result is False

    def test_validate_data_checks_null_values(self, etl_instance, sample_weather_data):
        """Test validation detects null values in critical columns"""
        result = etl_instance.validate_data(sample_weather_data, "test_stage")

        # Should pass if no critical nulls
        assert result is True


class TestRunPipeline:
    """Test complete ETL pipeline execution"""

    @patch.object(WeatherETL, 'cache_features_to_redis')
    @patch.object(WeatherETL, 'load_to_postgres')
    @patch.object(WeatherETL, 'extract_weather_data')
    def test_run_pipeline_executes_all_steps(self, mock_extract, mock_load, mock_cache,
                                            etl_instance, sample_weather_data):
        """Test that run() executes all pipeline steps"""
        mock_extract.return_value = sample_weather_data

        result = etl_instance.run('2025-01-01', '2025-01-03', save_to_db=True, cache_to_redis=True)

        # Verify all steps were called
        mock_extract.assert_called_once_with('2025-01-01', '2025-01-03', None)
        assert mock_load.called
        assert mock_cache.called

        # Verify result contains statistics
        assert 'status' in result
        assert result['status'] == 'success'
        assert 'duration_seconds' in result
        assert 'records_extracted' in result

    @patch.object(WeatherETL, 'extract_weather_data')
    def test_run_pipeline_skips_save_when_disabled(self, mock_extract, etl_instance, sample_weather_data):
        """Test that saving can be disabled"""
        mock_extract.return_value = sample_weather_data

        with patch.object(etl_instance, 'load_to_postgres') as mock_load:
            etl_instance.run('2025-01-01', '2025-01-03', save_to_db=False, cache_to_redis=False)

            mock_load.assert_not_called()

    @patch.object(WeatherETL, 'extract_weather_data')
    def test_run_pipeline_handles_errors(self, mock_extract, etl_instance):
        """Test error handling in pipeline"""
        mock_extract.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            etl_instance.run('2025-01-01', '2025-01-03')

        assert "Database error" in str(exc_info.value)


class TestSparkSessionManagement:
    """Test Spark session lifecycle"""

    def test_stop_closes_spark_session(self, etl_instance):
        """Test that stop() closes the Spark session"""
        with patch.object(etl_instance.spark, 'stop') as mock_stop:
            etl_instance.stop()
            mock_stop.assert_called_once()

    def test_create_spark_session_with_correct_config(self):
        """Test Spark session creation with optimized config"""
        with patch('src.processing.spark_jobs.weather_etl.SparkSession') as mock_spark:
            mock_builder = MagicMock()
            mock_spark.builder = mock_builder

            etl = WeatherETL()

            # Verify configurations were set
            assert mock_builder.config.called


@pytest.mark.parametrize("start_date,end_date,expected_valid", [
    ("2025-01-01", "2025-01-31", True),
    ("2025-01-01", "2025-01-01", True),
    ("2024-12-01", "2024-12-31", True),
])
def test_date_range_validation(etl_instance, start_date, end_date, expected_valid, sample_weather_data):
    """Parametrized test for different date ranges"""
    with patch.object(etl_instance, 'extract_weather_data', return_value=sample_weather_data):
        try:
            result = etl_instance.run(start_date, end_date, save_to_db=False, cache_to_redis=False)
            assert result['status'] == 'success'
        except Exception:
            assert not expected_valid
