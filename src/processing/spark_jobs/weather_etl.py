"""
PySpark ETL Pipeline for Weather Data Processing

This module implements a distributed ETL pipeline using PySpark to:
- Extract weather forecasts from PostgreSQL
- Aggregate daily statistics per region
- Compute rolling window features for ML
- Load processed data back to PostgreSQL and Redis cache

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, avg, sum as spark_sum, max as spark_max, min as spark_min,
    count, stddev, first, last, lit, to_date, current_timestamp,
    window, lag, lead, when, expr, coalesce, date_format, round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    DoubleType, IntegerType, TimestampType
)
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherETL:
    """
    PySpark ETL pipeline for weather data processing and aggregation

    This class handles:
    - Extraction of raw weather forecasts from PostgreSQL
    - Daily statistical aggregations per region
    - Rolling window feature computations
    - Data quality checks and validation
    - Loading results to PostgreSQL and caching in Redis
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        db_host: str = None,
        db_port: int = 5432,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None
    ):
        """
        Initialize the Weather ETL pipeline

        Args:
            spark: Existing SparkSession or None to create new one
            db_host: PostgreSQL host (defaults to env var)
            db_port: PostgreSQL port
            db_name: Database name (defaults to env var)
            db_user: Database user (defaults to env var)
            db_password: Database password (defaults to env var)
        """
        self.spark = spark or self._create_spark_session()

        # Database configuration
        self.db_host = db_host or os.getenv('POSTGRES_HOST', 'postgres')
        self.db_port = db_port
        self.db_name = db_name or os.getenv('POSTGRES_DB', 'agrisafe_db')
        self.db_user = db_user or os.getenv('POSTGRES_USER', 'agrisafe')
        self.db_password = db_password or os.getenv('POSTGRES_PASSWORD', 'agrisafe_password')

        self.jdbc_url = f"jdbc:postgresql://{self.db_host}:{self.db_port}/{self.db_name}"
        self.jdbc_properties = {
            "user": self.db_user,
            "password": self.db_password,
            "driver": "org.postgresql.Driver"
        }

        logger.info(f"Initialized WeatherETL with database: {self.jdbc_url}")

    def _create_spark_session(self) -> SparkSession:
        """
        Create a new SparkSession with optimized configurations

        Returns:
            Configured SparkSession
        """
        return SparkSession.builder \
            .appName("AgriSafe-WeatherETL") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()

    def extract_weather_data(
        self,
        start_date: str,
        end_date: str,
        region_ids: Optional[list] = None
    ) -> DataFrame:
        """
        Extract weather forecasts from PostgreSQL

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            region_ids: Optional list of region IDs to filter

        Returns:
            PySpark DataFrame with weather forecast data
        """
        logger.info(f"Extracting weather data from {start_date} to {end_date}")

        # Build SQL query
        region_filter = ""
        if region_ids:
            region_list = ','.join([f"'{rid}'" for rid in region_ids])
            region_filter = f"AND wf.region_id IN ({region_list})"

        query = f"""
            (SELECT
                wf.id,
                wf.region_id,
                r.name as region_name,
                r.latitude,
                r.longitude,
                r.elevation,
                wf.forecast_date,
                wf.temperature_high,
                wf.temperature_low,
                wf.rainfall_mm,
                wf.wind_speed,
                wf.weather_condition,
                wf.created_at
            FROM weather_forecasts wf
            JOIN regions r ON wf.region_id = r.id
            WHERE wf.forecast_date BETWEEN '{start_date}' AND '{end_date}'
            {region_filter}
            ORDER BY wf.region_id, wf.forecast_date
            ) as weather_data
        """

        try:
            df = self.spark.read.jdbc(
                url=self.jdbc_url,
                table=query,
                properties=self.jdbc_properties
            )

            record_count = df.count()
            logger.info(f"Extracted {record_count} weather forecast records")

            return df

        except Exception as e:
            logger.error(f"Failed to extract weather data: {str(e)}")
            raise

    def compute_daily_stats(self, df: DataFrame) -> DataFrame:
        """
        Aggregate weather data to daily statistics per region

        Computes:
        - Average high/low temperatures
        - Total rainfall
        - Maximum wind speed
        - Dominant weather condition
        - Forecast count per day

        Args:
            df: Raw weather forecast DataFrame

        Returns:
            DataFrame with daily aggregated statistics
        """
        logger.info("Computing daily statistics per region")

        daily_stats = df.groupBy("region_id", "region_name", "forecast_date") \
            .agg(
                spark_round(avg("temperature_high"), 2).alias("temp_high_avg"),
                spark_round(avg("temperature_low"), 2).alias("temp_low_avg"),
                spark_round(spark_sum("rainfall_mm"), 2).alias("rainfall_total"),
                spark_round(spark_max("wind_speed"), 2).alias("wind_speed_max"),
                spark_round(spark_min("wind_speed"), 2).alias("wind_speed_min"),
                spark_round(avg(col("temperature_high") + col("temperature_low")) / 2, 2).alias("temp_avg"),
                first("weather_condition").alias("dominant_condition"),
                first("elevation").alias("elevation"),
                count("*").alias("forecast_count")
            ) \
            .withColumn("created_at", current_timestamp()) \
            .orderBy("region_id", "forecast_date")

        stats_count = daily_stats.count()
        logger.info(f"Computed daily stats for {stats_count} region-date combinations")

        return daily_stats

    def compute_rolling_features(self, df: DataFrame) -> DataFrame:
        """
        Calculate rolling window features for ML model

        Features computed:
        - 3-day rainfall accumulation
        - 7-day rainfall accumulation
        - 14-day rainfall accumulation
        - 7-day average temperature
        - 30-day temperature variance
        - Number of rainy days in past 7 days

        Args:
            df: Daily statistics DataFrame

        Returns:
            DataFrame with rolling window features
        """
        logger.info("Computing rolling window features")

        # Define window specifications
        window_3d = Window.partitionBy("region_id") \
            .orderBy("forecast_date") \
            .rowsBetween(-2, 0)

        window_7d = Window.partitionBy("region_id") \
            .orderBy("forecast_date") \
            .rowsBetween(-6, 0)

        window_14d = Window.partitionBy("region_id") \
            .orderBy("forecast_date") \
            .rowsBetween(-13, 0)

        window_30d = Window.partitionBy("region_id") \
            .orderBy("forecast_date") \
            .rowsBetween(-29, 0)

        # Compute rolling features
        features = df \
            .withColumn("rainfall_3d", spark_round(spark_sum("rainfall_total").over(window_3d), 2)) \
            .withColumn("rainfall_7d", spark_round(spark_sum("rainfall_total").over(window_7d), 2)) \
            .withColumn("rainfall_14d", spark_round(spark_sum("rainfall_total").over(window_14d), 2)) \
            .withColumn("temp_avg_7d", spark_round(avg("temp_avg").over(window_7d), 2)) \
            .withColumn("temp_variance_30d", spark_round(stddev("temp_avg").over(window_30d), 2)) \
            .withColumn("wind_speed_avg_7d", spark_round(avg("wind_speed_max").over(window_7d), 2)) \
            .withColumn(
                "rainy_days_7d",
                spark_sum(when(col("rainfall_total") > 5.0, 1).otherwise(0)).over(window_7d)
            ) \
            .withColumn(
                "heavy_rain_days_7d",
                spark_sum(when(col("rainfall_total") > 50.0, 1).otherwise(0)).over(window_7d)
            ) \
            .withColumn(
                "rainfall_intensity",
                spark_round(
                    col("rainfall_total") / (col("rainfall_7d") + lit(0.01)),
                    4
                )
            ) \
            .withColumn(
                "soil_moisture_proxy",
                spark_round(
                    col("rainfall_7d") / (col("temp_avg") + lit(1)),
                    2
                )
            )

        # Handle nulls for initial windows
        features = features.fillna({
            "rainfall_3d": 0.0,
            "rainfall_7d": 0.0,
            "rainfall_14d": 0.0,
            "temp_variance_30d": 0.0,
            "rainy_days_7d": 0,
            "heavy_rain_days_7d": 0
        })

        feature_count = features.count()
        logger.info(f"Computed rolling features for {feature_count} records")

        return features

    def load_to_postgres(
        self,
        df: DataFrame,
        table_name: str,
        mode: str = "append"
    ):
        """
        Load processed data to PostgreSQL

        Args:
            df: DataFrame to save
            table_name: Target table name
            mode: Write mode ('append', 'overwrite', 'ignore', 'error')
        """
        logger.info(f"Loading data to PostgreSQL table: {table_name}")

        try:
            df.write.jdbc(
                url=self.jdbc_url,
                table=table_name,
                mode=mode,
                properties=self.jdbc_properties
            )

            logger.info(f"Successfully loaded data to {table_name}")

        except Exception as e:
            logger.error(f"Failed to load data to {table_name}: {str(e)}")
            raise

    def cache_features_to_redis(self, df: DataFrame, ttl: int = 604800):
        """
        Cache rolling features to Redis for fast access

        Args:
            df: DataFrame with features to cache
            ttl: Time to live in seconds (default: 7 days)
        """
        logger.info("Caching features to Redis")

        try:
            import redis
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True
            )

            # Convert to list of dicts for caching
            features_list = df.select(
                "region_id", "forecast_date",
                "rainfall_3d", "rainfall_7d", "rainfall_14d",
                "temp_avg_7d", "temp_variance_30d",
                "wind_speed_avg_7d", "rainy_days_7d"
            ).collect()

            cached_count = 0
            for row in features_list:
                key = f"weather:features:{row['region_id']}:{row['forecast_date']}"
                value = {
                    "rainfall_3d": float(row['rainfall_3d']) if row['rainfall_3d'] else 0.0,
                    "rainfall_7d": float(row['rainfall_7d']) if row['rainfall_7d'] else 0.0,
                    "rainfall_14d": float(row['rainfall_14d']) if row['rainfall_14d'] else 0.0,
                    "temp_avg_7d": float(row['temp_avg_7d']) if row['temp_avg_7d'] else 0.0,
                    "temp_variance_30d": float(row['temp_variance_30d']) if row['temp_variance_30d'] else 0.0,
                    "wind_speed_avg_7d": float(row['wind_speed_avg_7d']) if row['wind_speed_avg_7d'] else 0.0,
                    "rainy_days_7d": int(row['rainy_days_7d']) if row['rainy_days_7d'] else 0
                }

                redis_client.setex(key, ttl, json.dumps(value))
                cached_count += 1

            logger.info(f"Cached {cached_count} feature records to Redis")
            redis_client.close()

        except ImportError:
            logger.warning("Redis library not available, skipping cache")
        except Exception as e:
            logger.error(f"Failed to cache features to Redis: {str(e)}")

    def validate_data(self, df: DataFrame, stage: str) -> bool:
        """
        Perform data quality checks

        Args:
            df: DataFrame to validate
            stage: Name of the processing stage

        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating data at stage: {stage}")

        # Check for empty DataFrame
        if df.count() == 0:
            logger.error(f"Validation failed: DataFrame is empty at {stage}")
            return False

        # Check for required columns
        required_columns = ["region_id", "forecast_date"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Validation failed: Missing columns {missing_columns} at {stage}")
            return False

        # Check for null values in critical columns
        null_counts = df.select([
            spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in ["region_id", "forecast_date"]
        ]).collect()[0].asDict()

        if any(count > 0 for count in null_counts.values()):
            logger.warning(f"Found null values at {stage}: {null_counts}")

        logger.info(f"Validation passed at {stage}")
        return True

    def run(
        self,
        start_date: str,
        end_date: str,
        region_ids: Optional[list] = None,
        save_to_db: bool = True,
        cache_to_redis: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete ETL pipeline

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            region_ids: Optional list of region IDs to process
            save_to_db: Whether to save results to PostgreSQL
            cache_to_redis: Whether to cache features to Redis

        Returns:
            Dictionary with pipeline execution statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting Weather ETL pipeline for {start_date} to {end_date}")

        try:
            # Extract
            raw_data = self.extract_weather_data(start_date, end_date, region_ids)
            self.validate_data(raw_data, "extraction")

            # Transform - Daily Stats
            daily_stats = self.compute_daily_stats(raw_data)
            self.validate_data(daily_stats, "daily_stats")

            # Transform - Rolling Features
            features = self.compute_rolling_features(daily_stats)
            self.validate_data(features, "rolling_features")

            # Load to PostgreSQL
            if save_to_db:
                self.load_to_postgres(daily_stats, "weather_daily_stats", mode="append")

            # Cache to Redis
            if cache_to_redis:
                self.cache_features_to_redis(features)

            # Compute statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            stats = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "records_extracted": raw_data.count(),
                "daily_stats_computed": daily_stats.count(),
                "features_computed": features.count(),
                "status": "success"
            }

            logger.info(f"ETL pipeline completed successfully in {duration:.2f} seconds")
            logger.info(f"Statistics: {stats}")

            return stats

        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise

    def stop(self):
        """Stop the Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """
    Main entry point for running the ETL pipeline as a standalone script
    """
    import argparse

    parser = argparse.ArgumentParser(description='Weather ETL Pipeline')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--regions', nargs='*', help='Optional list of region IDs')
    parser.add_argument('--no-save', action='store_true', help='Skip saving to database')
    parser.add_argument('--no-cache', action='store_true', help='Skip caching to Redis')

    args = parser.parse_args()

    # Create ETL instance and run
    etl = WeatherETL()

    try:
        stats = etl.run(
            start_date=args.start_date,
            end_date=args.end_date,
            region_ids=args.regions,
            save_to_db=not args.no_save,
            cache_to_redis=not args.no_cache
        )

        print(f"\n{'='*60}")
        print("ETL Pipeline Completed Successfully")
        print(f"{'='*60}")
        print(f"Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"Records processed: {stats['records_extracted']}")
        print(f"Daily stats computed: {stats['daily_stats_computed']}")
        print(f"Features computed: {stats['features_computed']}")
        print(f"{'='*60}\n")

        sys.exit(0)

    except Exception as e:
        print(f"\nETL Pipeline Failed: {str(e)}")
        sys.exit(1)

    finally:
        etl.stop()


if __name__ == "__main__":
    main()
