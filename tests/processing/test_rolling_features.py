"""
Unit tests for Rolling Features Job
"""

import pytest
from datetime import datetime, date, timedelta
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, IntegerType,
    DateType, DecimalType
)


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    spark = SparkSession.builder \
        .appName("AgriSafe-Test-RollingFeatures") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    yield spark

    spark.stop()


@pytest.fixture
def sample_daily_stats(spark):
    """Create sample daily statistics data for 14 days."""
    schema = StructType([
        StructField("region_id", IntegerType(), False),
        StructField("stat_date", DateType(), False),
        StructField("temp_avg", DecimalType(5, 2), True),
        StructField("temp_min", DecimalType(5, 2), True),
        StructField("temp_max", DecimalType(5, 2), True),
        StructField("rainfall_total", DecimalType(8, 2), True),
        StructField("rainfall_max", DecimalType(8, 2), True),
        StructField("wind_speed_avg", DecimalType(6, 2), True),
        StructField("wind_speed_max", DecimalType(6, 2), True),
        StructField("humidity_avg", DecimalType(5, 2), True),
    ])

    # Generate 14 days of data
    base_date = date(2025, 1, 1)
    data = []

    for i in range(14):
        current_date = base_date + timedelta(days=i)

        # Region 1: Increasing rainfall pattern
        data.append(Row(
            1, current_date,
            27.0 + i * 0.1,  # temp_avg
            22.0, 32.0,      # temp_min, temp_max
            float(5 + i * 5),  # rainfall increases
            float(10 + i * 2),  # rainfall_max
            15.0, 20.0,      # wind
            75.0             # humidity
        ))

        # Region 2: Stable weather
        data.append(Row(
            2, current_date,
            25.0, 20.0, 30.0,
            2.0, 5.0,
            12.0, 18.0,
            70.0
        ))

    return spark.createDataFrame(data, schema)


def test_compute_7day_rolling_features(spark, sample_daily_stats):
    """Test 7-day rolling window features."""
    from src.processing.jobs.rolling_features import RollingFeaturesJob

    job = RollingFeaturesJob(spark, use_cache=False)
    result = job.compute_rolling_features(sample_daily_stats, window_days=[7])

    # Collect results
    results = result.collect()

    # Should have features for both regions across all dates
    assert len(results) > 0

    # Check region 1, last date (should have full 7-day window)
    region1_last = [r for r in results
                    if r.region_id == 1 and r.feature_date == date(2025, 1, 14)][0]

    # Rainfall should be sum of last 7 days (days 7-13: indices 7-13)
    # Days 7-13: rainfall = 40, 45, 50, 55, 60, 65, 70
    expected_sum = sum([40 + i * 5 for i in range(7)])
    assert abs(region1_last.rainfall_rolling_sum - expected_sum) < 1.0

    # Window days should be 7
    assert region1_last.window_days == 7


def test_multiple_windows(spark, sample_daily_stats):
    """Test computation of multiple window sizes."""
    from src.processing.jobs.rolling_features import RollingFeaturesJob

    job = RollingFeaturesJob(spark, use_cache=False)
    result = job.compute_rolling_features(sample_daily_stats, window_days=[7, 14])

    results = result.collect()

    # Should have results for both 7-day and 14-day windows
    window_7 = [r for r in results if r.window_days == 7]
    window_14 = [r for r in results if r.window_days == 14]

    assert len(window_7) > 0
    assert len(window_14) > 0


def test_rainy_days_count(spark):
    """Test counting of rainy days."""
    from src.processing.jobs.rolling_features import RollingFeaturesJob

    schema = StructType([
        StructField("region_id", IntegerType(), False),
        StructField("stat_date", DateType(), False),
        StructField("temp_avg", DecimalType(5, 2), True),
        StructField("temp_min", DecimalType(5, 2), True),
        StructField("temp_max", DecimalType(5, 2), True),
        StructField("rainfall_total", DecimalType(8, 2), True),
        StructField("rainfall_max", DecimalType(8, 2), True),
        StructField("wind_speed_avg", DecimalType(6, 2), True),
        StructField("wind_speed_max", DecimalType(6, 2), True),
        StructField("humidity_avg", DecimalType(5, 2), True),
    ])

    # 7 days: 5 rainy (rainfall > 0), 2 dry (rainfall = 0)
    base_date = date(2025, 1, 1)
    data = [
        Row(1, base_date, 27.0, 22.0, 32.0, 10.0, 15.0, 15.0, 20.0, 75.0),  # rainy
        Row(1, base_date + timedelta(1), 27.0, 22.0, 32.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # dry
        Row(1, base_date + timedelta(2), 27.0, 22.0, 32.0, 20.0, 25.0, 15.0, 20.0, 75.0),  # rainy
        Row(1, base_date + timedelta(3), 27.0, 22.0, 32.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # dry
        Row(1, base_date + timedelta(4), 27.0, 22.0, 32.0, 5.0, 10.0, 15.0, 20.0, 75.0),  # rainy
        Row(1, base_date + timedelta(5), 27.0, 22.0, 32.0, 30.0, 35.0, 15.0, 20.0, 75.0),  # rainy
        Row(1, base_date + timedelta(6), 27.0, 22.0, 32.0, 15.0, 20.0, 15.0, 20.0, 75.0),  # rainy
    ]

    df = spark.createDataFrame(data, schema)

    job = RollingFeaturesJob(spark, use_cache=False)
    result = job.compute_rolling_features(df, window_days=[7])

    # Get last day result
    last_result = result.filter(result.feature_date == base_date + timedelta(6)).collect()[0]

    # Should count 5 rainy days
    assert last_result.rainfall_days_count == 5


def test_heavy_rainfall_days(spark):
    """Test counting of heavy rainfall days (> 50mm)."""
    from src.processing.jobs.rolling_features import RollingFeaturesJob

    schema = StructType([
        StructField("region_id", IntegerType(), False),
        StructField("stat_date", DateType(), False),
        StructField("temp_avg", DecimalType(5, 2), True),
        StructField("temp_min", DecimalType(5, 2), True),
        StructField("temp_max", DecimalType(5, 2), True),
        StructField("rainfall_total", DecimalType(8, 2), True),
        StructField("rainfall_max", DecimalType(8, 2), True),
        StructField("wind_speed_avg", DecimalType(6, 2), True),
        StructField("wind_speed_max", DecimalType(6, 2), True),
        StructField("humidity_avg", DecimalType(5, 2), True),
    ])

    # 7 days: 2 heavy rainfall days, 3 moderate, 2 dry
    base_date = date(2025, 1, 1)
    data = [
        Row(1, base_date, 27.0, 22.0, 32.0, 60.0, 65.0, 15.0, 20.0, 75.0),  # heavy
        Row(1, base_date + timedelta(1), 27.0, 22.0, 32.0, 20.0, 25.0, 15.0, 20.0, 75.0),  # moderate
        Row(1, base_date + timedelta(2), 27.0, 22.0, 32.0, 75.0, 80.0, 15.0, 20.0, 75.0),  # heavy
        Row(1, base_date + timedelta(3), 27.0, 22.0, 32.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # dry
        Row(1, base_date + timedelta(4), 27.0, 22.0, 32.0, 10.0, 15.0, 15.0, 20.0, 75.0),  # moderate
        Row(1, base_date + timedelta(5), 27.0, 22.0, 32.0, 30.0, 35.0, 15.0, 20.0, 75.0),  # moderate
        Row(1, base_date + timedelta(6), 27.0, 22.0, 32.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # dry
    ]

    df = spark.createDataFrame(data, schema)

    job = RollingFeaturesJob(spark, use_cache=False)
    result = job.compute_rolling_features(df, window_days=[7])

    # Get last day result
    last_result = result.filter(result.feature_date == base_date + timedelta(6)).collect()[0]

    # Should count 2 heavy rainfall days
    assert last_result.rainfall_heavy_days == 2


def test_extreme_temp_days(spark):
    """Test counting of extreme temperature days."""
    from src.processing.jobs.rolling_features import RollingFeaturesJob

    schema = StructType([
        StructField("region_id", IntegerType(), False),
        StructField("stat_date", DateType(), False),
        StructField("temp_avg", DecimalType(5, 2), True),
        StructField("temp_min", DecimalType(5, 2), True),
        StructField("temp_max", DecimalType(5, 2), True),
        StructField("rainfall_total", DecimalType(8, 2), True),
        StructField("rainfall_max", DecimalType(8, 2), True),
        StructField("wind_speed_avg", DecimalType(6, 2), True),
        StructField("wind_speed_max", DecimalType(6, 2), True),
        StructField("humidity_avg", DecimalType(5, 2), True),
    ])

    # 7 days: 2 extreme (very hot), 1 extreme (very cold), 4 normal
    base_date = date(2025, 1, 1)
    data = [
        Row(1, base_date, 27.0, 22.0, 36.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # extreme hot
        Row(1, base_date + timedelta(1), 27.0, 22.0, 30.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # normal
        Row(1, base_date + timedelta(2), 27.0, 22.0, 32.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # normal
        Row(1, base_date + timedelta(3), 20.0, 8.0, 25.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # extreme cold
        Row(1, base_date + timedelta(4), 27.0, 22.0, 31.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # normal
        Row(1, base_date + timedelta(5), 28.0, 23.0, 37.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # extreme hot
        Row(1, base_date + timedelta(6), 27.0, 22.0, 32.0, 0.0, 0.0, 15.0, 20.0, 75.0),  # normal
    ]

    df = spark.createDataFrame(data, schema)

    job = RollingFeaturesJob(spark, use_cache=False)
    result = job.compute_rolling_features(df, window_days=[7])

    # Get last day result
    last_result = result.filter(result.feature_date == base_date + timedelta(6)).collect()[0]

    # Should count 3 extreme temperature days (2 hot + 1 cold)
    assert last_result.extreme_temp_days == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
