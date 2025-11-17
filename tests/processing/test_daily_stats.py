"""
Unit tests for Daily Weather Statistics Job
"""

import pytest
from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DateType, DecimalType, TimestampType
)


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    spark = SparkSession.builder \
        .appName("AgriSafe-Test-DailyStats") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    yield spark

    spark.stop()


@pytest.fixture
def sample_weather_data(spark):
    """Create sample weather forecast data."""
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("region_id", IntegerType(), False),
        StructField("forecast_date", DateType(), False),
        StructField("temperature_min", DecimalType(5, 2), True),
        StructField("temperature_max", DecimalType(5, 2), True),
        StructField("temperature_avg", DecimalType(5, 2), True),
        StructField("humidity_percent", DecimalType(5, 2), True),
        StructField("rainfall_mm", DecimalType(8, 2), True),
        StructField("wind_speed_kph", DecimalType(6, 2), True),
    ])

    data = [
        Row("1", 1, date(2025, 1, 1), 22.0, 32.0, 27.0, 75.0, 10.5, 15.0),
        Row("2", 1, date(2025, 1, 1), 21.0, 31.0, 26.0, 76.0, 12.0, 16.0),
        Row("3", 1, date(2025, 1, 2), 23.0, 33.0, 28.0, 74.0, 5.5, 14.0),
        Row("4", 2, date(2025, 1, 1), 20.0, 30.0, 25.0, 80.0, 25.0, 20.0),
        Row("5", 2, date(2025, 1, 2), 19.0, 29.0, 24.0, 82.0, 30.0, 22.0),
    ]

    return spark.createDataFrame(data, schema)


def test_compute_daily_stats(spark, sample_weather_data):
    """Test daily statistics computation."""
    from src.processing.jobs.daily_weather_stats import DailyWeatherStatsJob

    # Create job instance (without cache for testing)
    job = DailyWeatherStatsJob(spark, use_cache=False)

    # Compute daily stats
    result = job.compute_daily_stats(sample_weather_data)

    # Collect results
    results = result.collect()

    # Assertions
    assert len(results) == 3  # 2 regions * 2 days, but region 1 has data for both days

    # Check region 1, date 2025-01-01
    region1_day1 = [r for r in results if r.region_id == 1 and r.stat_date == date(2025, 1, 1)][0]

    assert region1_day1.temp_min == 21.0  # min of 22.0, 21.0
    assert region1_day1.temp_max == 32.0  # max of 32.0, 31.0
    assert abs(region1_day1.temp_avg - 26.5) < 0.1  # avg of 27.0, 26.0
    assert abs(region1_day1.rainfall_total - 22.5) < 0.1  # sum of 10.5, 12.0
    assert region1_day1.forecast_count == 2


def test_empty_dataframe(spark):
    """Test handling of empty input data."""
    from src.processing.jobs.daily_weather_stats import DailyWeatherStatsJob

    # Create empty DataFrame with correct schema
    schema = StructType([
        StructField("region_id", IntegerType(), False),
        StructField("forecast_date", DateType(), False),
        StructField("temperature_avg", DecimalType(5, 2), True),
        StructField("rainfall_mm", DecimalType(8, 2), True),
    ])

    empty_df = spark.createDataFrame([], schema)

    job = DailyWeatherStatsJob(spark, use_cache=False)
    result = job.compute_daily_stats(empty_df)

    assert result.count() == 0


def test_null_handling(spark):
    """Test handling of null values in weather data."""
    from src.processing.jobs.daily_weather_stats import DailyWeatherStatsJob

    schema = StructType([
        StructField("region_id", IntegerType(), False),
        StructField("forecast_date", DateType(), False),
        StructField("temperature_min", DecimalType(5, 2), True),
        StructField("temperature_max", DecimalType(5, 2), True),
        StructField("temperature_avg", DecimalType(5, 2), True),
        StructField("rainfall_mm", DecimalType(8, 2), True),
    ])

    data = [
        Row(1, date(2025, 1, 1), 22.0, 32.0, 27.0, None),  # Null rainfall
        Row(1, date(2025, 1, 1), None, None, 26.0, 10.0),  # Null temps
    ]

    df = spark.createDataFrame(data, schema)

    job = DailyWeatherStatsJob(spark, use_cache=False)
    result = job.compute_daily_stats(df)

    # Should handle nulls gracefully
    assert result.count() == 1


def test_data_completeness_calculation(spark, sample_weather_data):
    """Test data completeness percentage calculation."""
    from src.processing.jobs.daily_weather_stats import DailyWeatherStatsJob

    job = DailyWeatherStatsJob(spark, use_cache=False)
    result = job.compute_daily_stats(sample_weather_data)

    results = result.collect()

    # With 2 forecasts for region 1 on 2025-01-01
    region1_day1 = [r for r in results if r.region_id == 1 and r.stat_date == date(2025, 1, 1)][0]

    # 2 forecasts out of expected 24 = 8.33%
    expected_completeness = (2 / 24.0) * 100
    assert abs(region1_day1.data_completeness - expected_completeness) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
