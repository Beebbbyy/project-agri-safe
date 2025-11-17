"""
Unit tests for Rolling Feature Engineering

Tests all feature computation functions:
- Rainfall rolling features
- Temperature rolling features
- Wind rolling features
- Seasonal features
- Derived features
- Historical flood risk features
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType

from src.processing.spark_jobs.rolling_features import RollingFeatureEngine


@pytest.fixture(scope="module")
def spark_session():
    """Create a test Spark session"""
    spark = SparkSession.builder \
        .appName("TestRollingFeatures") \
        .master("local[2]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

    yield spark

    spark.stop()


@pytest.fixture
def sample_daily_stats(spark_session):
    """Create sample daily statistics data"""
    data = []
    for i in range(40):  # Create 40 days of data
        for region_id in ['REG001', 'REG002']:
            data.append((
                region_id,
                date(2025, 1, 1) + timedelta(days=i),
                30.0 + (i % 5),  # temp_high_avg
                22.0 + (i % 3),  # temp_low_avg
                10.0 + (i % 10) * 5,  # rainfall_total
                25.0 + (i % 8) * 5,  # wind_speed_max
                "Cloudy",
                50.0 if region_id == 'REG001' else 100.0,  # elevation
                14.5 if region_id == 'REG001' else 15.0,  # latitude
                121.0 if region_id == 'REG001' else 120.5,  # longitude
            ))

    schema = StructType([
        StructField("region_id", StringType(), True),
        StructField("stat_date", DateType(), True),
        StructField("temp_high_avg", DoubleType(), True),
        StructField("temp_low_avg", DoubleType(), True),
        StructField("rainfall_total", DoubleType(), True),
        StructField("wind_speed_max", DoubleType(), True),
        StructField("dominant_condition", StringType(), True),
        StructField("elevation", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
    ])

    return spark_session.createDataFrame(data, schema)


@pytest.fixture
def engine_instance(spark_session):
    """Create RollingFeatureEngine instance with mocked database"""
    with patch('src.processing.spark_jobs.rolling_features.os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            'POSTGRES_HOST': 'test_host',
            'POSTGRES_DB': 'test_db',
            'POSTGRES_USER': 'test_user',
            'POSTGRES_PASSWORD': 'test_password'
        }.get(key, default)

        engine = RollingFeatureEngine(spark=spark_session)
        return engine


class TestRollingFeatureEngineInitialization:
    """Test initialization and configuration"""

    def test_init_with_provided_spark_session(self, spark_session):
        """Test initialization with provided Spark session"""
        engine = RollingFeatureEngine(spark=spark_session)
        assert engine.spark is not None
        assert engine.spark == spark_session

    def test_database_configuration(self, engine_instance):
        """Test database connection parameters"""
        assert engine_instance.db_host == 'test_host'
        assert engine_instance.db_name == 'test_db'
        assert engine_instance.jdbc_url is not None


class TestLoadDailyStats:
    """Test loading daily statistics from database"""

    def test_load_daily_stats_builds_query(self, engine_instance, sample_daily_stats):
        """Test that load builds correct SQL query"""
        with patch.object(engine_instance.spark.read, 'jdbc', return_value=sample_daily_stats) as mock_jdbc:
            result = engine_instance.load_daily_stats('2025-01-01', '2025-01-31')

            mock_jdbc.assert_called_once()
            table_query = mock_jdbc.call_args[1]['table']
            assert '2025-01-01' in table_query
            assert '2025-01-31' in table_query

    def test_load_with_region_filter(self, engine_instance, sample_daily_stats):
        """Test loading with specific regions"""
        with patch.object(engine_instance.spark.read, 'jdbc', return_value=sample_daily_stats) as mock_jdbc:
            result = engine_instance.load_daily_stats('2025-01-01', '2025-01-31', region_ids=['REG001'])

            table_query = mock_jdbc.call_args[1]['table']
            assert 'REG001' in table_query

    def test_load_returns_dataframe(self, engine_instance, sample_daily_stats):
        """Test that load returns a DataFrame"""
        with patch.object(engine_instance.spark.read, 'jdbc', return_value=sample_daily_stats):
            result = engine_instance.load_daily_stats('2025-01-01', '2025-01-31')

            assert isinstance(result, DataFrame)
            assert result.count() > 0


class TestComputeRainfallFeatures:
    """Test rainfall feature computation"""

    def test_rainfall_features_columns_created(self, engine_instance, sample_daily_stats):
        """Test that all rainfall feature columns are created"""
        result = engine_instance.compute_rainfall_features(sample_daily_stats)

        expected_cols = ['rainfall_1d', 'rainfall_3d', 'rainfall_7d', 'rainfall_14d', 'rainfall_30d',
                        'rainy_days_7d', 'heavy_rain_days_7d', 'extreme_rain_days_7d',
                        'max_daily_rainfall_7d', 'rainfall_intensity_ratio']

        for col in expected_cols:
            assert col in result.columns

    def test_rainfall_1d_equals_total(self, engine_instance, sample_daily_stats):
        """Test rainfall_1d equals rainfall_total"""
        result = engine_instance.compute_rainfall_features(sample_daily_stats)
        result_pd = result.toPandas()

        assert (result_pd['rainfall_1d'] == result_pd['rainfall_total']).all()

    def test_rainfall_3d_accumulation(self, engine_instance, spark_session):
        """Test 3-day rainfall accumulation"""
        data = [
            ('REG001', date(2025, 1, 1), 30.0, 22.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),
            ('REG001', date(2025, 1, 2), 30.0, 22.0, 20.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),
            ('REG001', date(2025, 1, 3), 30.0, 22.0, 30.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_rainfall_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        # Day 3 should have sum of 10+20+30 = 60
        assert result_pd.iloc[2]['rainfall_3d'] == 60.0

    def test_rainy_days_counter(self, engine_instance, spark_session):
        """Test rainy days counting (> 2.5mm threshold)"""
        data = [
            ('REG001', date(2025, 1, 1), 30.0, 22.0, 1.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Not rainy
            ('REG001', date(2025, 1, 2), 30.0, 22.0, 5.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Rainy
            ('REG001', date(2025, 1, 3), 30.0, 22.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0), # Rainy
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_rainfall_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        # Last day should count 2 rainy days (days 2 and 3)
        assert result_pd.iloc[2]['rainy_days_7d'] == 2

    def test_heavy_rain_days_counter(self, engine_instance, spark_session):
        """Test heavy rain days counting (> 50mm threshold)"""
        data = [
            ('REG001', date(2025, 1, 1), 30.0, 22.0, 60.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Heavy
            ('REG001', date(2025, 1, 2), 30.0, 22.0, 30.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Not heavy
            ('REG001', date(2025, 1, 3), 30.0, 22.0, 70.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Heavy
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_rainfall_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        # Last day should count 2 heavy rain days
        assert result_pd.iloc[2]['heavy_rain_days_7d'] == 2


class TestComputeTemperatureFeatures:
    """Test temperature feature computation"""

    def test_temperature_features_columns_created(self, engine_instance, sample_daily_stats):
        """Test that temperature feature columns are created"""
        result = engine_instance.compute_temperature_features(sample_daily_stats)

        expected_cols = ['temp_avg', 'temp_range', 'temp_avg_7d', 'temp_avg_14d', 'temp_avg_30d',
                        'temp_variance_7d', 'temp_variance_30d', 'temp_max_7d', 'temp_min_7d',
                        'heat_stress_days_7d']

        for col in expected_cols:
            assert col in result.columns

    def test_temp_avg_calculation(self, engine_instance, spark_session):
        """Test average temperature calculation"""
        data = [
            ('REG001', date(2025, 1, 1), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_temperature_features(df)
        result_pd = result.toPandas()

        # Average of 30 and 20 should be 25
        assert result_pd.iloc[0]['temp_avg'] == 25.0

    def test_temp_range_calculation(self, engine_instance, spark_session):
        """Test temperature range calculation"""
        data = [
            ('REG001', date(2025, 1, 1), 35.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_temperature_features(df)
        result_pd = result.toPandas()

        # Range should be 35 - 20 = 15
        assert result_pd.iloc[0]['temp_range'] == 15.0

    def test_heat_stress_days_counter(self, engine_instance, spark_session):
        """Test heat stress days counting (> 35°C)"""
        data = [
            ('REG001', date(2025, 1, 1), 36.0, 25.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Heat stress
            ('REG001', date(2025, 1, 2), 32.0, 24.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Normal
            ('REG001', date(2025, 1, 3), 37.0, 26.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Heat stress
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_temperature_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        # Last day should count 2 heat stress days
        assert result_pd.iloc[2]['heat_stress_days_7d'] == 2


class TestComputeWindFeatures:
    """Test wind feature computation"""

    def test_wind_features_columns_created(self, engine_instance, sample_daily_stats):
        """Test that wind feature columns are created"""
        result = engine_instance.compute_wind_features(sample_daily_stats)

        expected_cols = ['wind_speed_avg_7d', 'wind_speed_max_7d', 'wind_speed_avg_14d',
                        'high_wind_days_7d', 'storm_wind_days_7d']

        for col in expected_cols:
            assert col in result.columns

    def test_high_wind_days_counter(self, engine_instance, spark_session):
        """Test high wind days counting (> 60 km/h)"""
        data = [
            ('REG001', date(2025, 1, 1), 30.0, 20.0, 10.0, 70.0, 'Cloudy', 50.0, 14.5, 121.0),  # High wind
            ('REG001', date(2025, 1, 2), 30.0, 20.0, 10.0, 40.0, 'Cloudy', 50.0, 14.5, 121.0),  # Normal
            ('REG001', date(2025, 1, 3), 30.0, 20.0, 10.0, 80.0, 'Cloudy', 50.0, 14.5, 121.0),  # High wind
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_wind_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        # Should count 2 high wind days
        assert result_pd.iloc[2]['high_wind_days_7d'] == 2

    def test_storm_wind_days_counter(self, engine_instance, spark_session):
        """Test storm wind days counting (> 100 km/h)"""
        data = [
            ('REG001', date(2025, 1, 1), 30.0, 20.0, 10.0, 110.0, 'Cloudy', 50.0, 14.5, 121.0),  # Storm
            ('REG001', date(2025, 1, 2), 30.0, 20.0, 10.0, 80.0, 'Cloudy', 50.0, 14.5, 121.0),   # Not storm
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_wind_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        # Should count 1 storm day
        assert result_pd.iloc[1]['storm_wind_days_7d'] == 1


class TestComputeSeasonalFeatures:
    """Test seasonal feature computation"""

    def test_seasonal_features_columns_created(self, engine_instance, sample_daily_stats):
        """Test that seasonal feature columns are created"""
        result = engine_instance.compute_seasonal_features(sample_daily_stats)

        expected_cols = ['month', 'day_of_week', 'is_typhoon_season',
                        'is_wet_season', 'is_dry_season', 'season_category']

        for col in expected_cols:
            assert col in result.columns

    def test_typhoon_season_flag(self, engine_instance, spark_session):
        """Test typhoon season flag (June-November)"""
        data = [
            ('REG001', date(2025, 7, 15), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # July - typhoon
            ('REG001', date(2025, 1, 15), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Jan - not typhoon
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_seasonal_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        # January should not be typhoon season
        assert result_pd.iloc[0]['is_typhoon_season'] == 0
        # July should be typhoon season
        assert result_pd.iloc[1]['is_typhoon_season'] == 1

    def test_season_categories(self, engine_instance, spark_session):
        """Test season category assignment"""
        data = [
            ('REG001', date(2025, 1, 15), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Winter
            ('REG001', date(2025, 4, 15), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Summer
            ('REG001', date(2025, 7, 15), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0),  # Monsoon
            ('REG001', date(2025, 10, 15), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0), # Post-monsoon
        ]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        result = engine_instance.compute_seasonal_features(df)
        result_pd = result.toPandas().sort_values('stat_date')

        assert result_pd.iloc[0]['season_category'] == 'winter'
        assert result_pd.iloc[1]['season_category'] == 'summer'
        assert result_pd.iloc[2]['season_category'] == 'monsoon'
        assert result_pd.iloc[3]['season_category'] == 'post_monsoon'


class TestComputeDerivedFeatures:
    """Test derived feature computation"""

    def test_derived_features_columns_created(self, engine_instance, sample_daily_stats):
        """Test that derived feature columns are created"""
        # First need to add required columns
        df = engine_instance.compute_rainfall_features(sample_daily_stats)
        df = engine_instance.compute_temperature_features(df)

        result = engine_instance.compute_derived_features(df)

        expected_cols = ['soil_moisture_proxy', 'evapotranspiration_estimate',
                        'flood_risk_indicator', 'growth_condition_index']

        for col in expected_cols:
            assert col in result.columns

    def test_soil_moisture_proxy_calculation(self, engine_instance, spark_session):
        """Test soil moisture proxy formula"""
        data = [('REG001', date(2025, 1, 1), 30.0, 20.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0)]

        df = spark_session.createDataFrame(data,
            ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
             'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

        # Add required features
        df = engine_instance.compute_rainfall_features(df)
        df = engine_instance.compute_temperature_features(df)

        result = engine_instance.compute_derived_features(df)

        assert 'soil_moisture_proxy' in result.columns

    def test_growth_condition_index_ranges(self, engine_instance, spark_session):
        """Test growth condition index categorization"""
        # Optimal conditions
        data_optimal = [('REG001', date(2025, 1, 1), 28.0, 22.0, 50.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0)]
        # Sub-optimal conditions
        data_suboptimal = [('REG001', date(2025, 1, 1), 32.0, 22.0, 180.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0)]

        for data, expected_high in [(data_optimal, True), (data_suboptimal, False)]:
            df = spark_session.createDataFrame(data,
                ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
                 'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

            df = engine_instance.compute_rainfall_features(df)
            df = engine_instance.compute_temperature_features(df)
            result = engine_instance.compute_derived_features(df)
            result_pd = result.toPandas()

            assert 'growth_condition_index' in result_pd.columns
            if expected_high:
                assert result_pd.iloc[0]['growth_condition_index'] >= 70


class TestComputeHistoricalFloodRisk:
    """Test historical flood risk feature addition"""

    def test_historical_flood_risk_with_data(self, engine_instance, sample_daily_stats):
        """Test adding historical flood data when available"""
        with patch.object(engine_instance.spark.read, 'jdbc') as mock_jdbc:
            # Mock historical flood data
            flood_data = [
                ('REG001', 5, 2, 10, 50.0),
                ('REG002', 2, 1, 8, 25.0),
            ]

            flood_df = engine_instance.spark.createDataFrame(flood_data,
                ['region_id', 'historical_high_risk_count', 'historical_critical_count',
                 'total_assessments', 'region_vulnerability_score'])

            mock_jdbc.return_value = flood_df

            result = engine_instance.compute_historical_flood_risk(sample_daily_stats)

            expected_cols = ['historical_high_risk_count', 'region_vulnerability_score']
            for col in expected_cols:
                assert col in result.columns

    def test_historical_flood_risk_handles_missing_table(self, engine_instance, sample_daily_stats):
        """Test graceful handling when flood table doesn't exist"""
        with patch.object(engine_instance.spark.read, 'jdbc', side_effect=Exception("Table not found")):
            result = engine_instance.compute_historical_flood_risk(sample_daily_stats)

            # Should add dummy columns
            assert 'historical_high_risk_count' in result.columns
            assert 'region_vulnerability_score' in result.columns

            result_pd = result.toPandas()
            assert result_pd['historical_high_risk_count'].iloc[0] == 0


class TestComputeAllFeatures:
    """Test complete feature computation pipeline"""

    def test_compute_all_features_includes_all_columns(self, engine_instance, sample_daily_stats):
        """Test that all feature categories are computed"""
        result = engine_instance.compute_all_features(sample_daily_stats)

        # Check for features from each category
        rainfall_cols = ['rainfall_3d', 'rainfall_7d']
        temp_cols = ['temp_avg', 'temp_avg_7d']
        wind_cols = ['wind_speed_avg_7d']
        seasonal_cols = ['month', 'is_typhoon_season']
        derived_cols = ['soil_moisture_proxy', 'flood_risk_indicator']

        all_expected = rainfall_cols + temp_cols + wind_cols + seasonal_cols + derived_cols

        for col in all_expected:
            assert col in result.columns

    def test_compute_all_features_handles_nulls(self, engine_instance, sample_daily_stats):
        """Test that nulls are properly handled"""
        result = engine_instance.compute_all_features(sample_daily_stats)
        result_pd = result.toPandas()

        # Check that numeric columns don't have nulls
        numeric_cols = result_pd.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            assert result_pd[col].isna().sum() == 0


class TestSaveFeatures:
    """Test saving features to database"""

    def test_save_features_calls_jdbc_write(self, engine_instance, sample_daily_stats):
        """Test that save calls JDBC write"""
        with patch.object(sample_daily_stats.write, 'jdbc') as mock_write:
            engine_instance.save_features(sample_daily_stats, 'test_table', mode='append')

            mock_write.assert_called_once()
            call_args = mock_write.call_args[1]
            assert call_args['table'] == 'test_table'
            assert call_args['mode'] == 'append'

    def test_save_handles_write_error(self, engine_instance, sample_daily_stats):
        """Test error handling during save"""
        with patch.object(sample_daily_stats.write, 'jdbc', side_effect=Exception("Write failed")):
            with pytest.raises(Exception) as exc_info:
                engine_instance.save_features(sample_daily_stats, 'test_table')

            assert "Write failed" in str(exc_info.value)


class TestRunPipeline:
    """Test complete feature engineering pipeline"""

    @patch.object(RollingFeatureEngine, 'save_features')
    @patch.object(RollingFeatureEngine, 'load_daily_stats')
    def test_run_pipeline_executes_all_steps(self, mock_load, mock_save, engine_instance, sample_daily_stats):
        """Test that run() executes all steps"""
        mock_load.return_value = sample_daily_stats

        result = engine_instance.run('2025-01-01', '2025-01-31', save_to_db=True)

        # Verify steps were called
        mock_load.assert_called_once()
        mock_save.assert_called_once()

        # Verify result contains statistics
        assert 'status' in result
        assert result['status'] == 'success'
        assert 'duration_seconds' in result
        assert 'feature_columns' in result

    @patch.object(RollingFeatureEngine, 'load_daily_stats')
    def test_run_pipeline_skips_save_when_disabled(self, mock_load, engine_instance, sample_daily_stats):
        """Test that saving can be disabled"""
        mock_load.return_value = sample_daily_stats

        with patch.object(engine_instance, 'save_features') as mock_save:
            engine_instance.run('2025-01-01', '2025-01-31', save_to_db=False)

            mock_save.assert_not_called()


@pytest.mark.parametrize("days,expected_features", [
    (7, True),   # 7 days should compute all features
    (30, True),  # 30 days should compute all features
    (60, True),  # 60 days should compute all features
])
def test_different_time_windows(engine_instance, spark_session, days, expected_features):
    """Parametrized test for different time windows"""
    data = []
    for i in range(days):
        data.append(('REG001', date(2025, 1, 1) + timedelta(days=i),
                    30.0, 22.0, 10.0, 25.0, 'Cloudy', 50.0, 14.5, 121.0))

    df = spark_session.createDataFrame(data,
        ['region_id', 'stat_date', 'temp_high_avg', 'temp_low_avg', 'rainfall_total',
         'wind_speed_max', 'dominant_condition', 'elevation', 'latitude', 'longitude'])

    result = engine_instance.compute_all_features(df)

    if expected_features:
        assert 'rainfall_7d' in result.columns
        assert result.count() == days
