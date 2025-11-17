"""
Daily Weather Statistics Aggregation Job

Aggregates raw weather forecasts into daily statistics per region.
Calculates min, max, avg, and stddev for temperature, rainfall, humidity, and wind.
"""

import sys
from datetime import datetime, timedelta
from typing import Optional
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from loguru import logger

from src.processing.utils.spark_session import (
    get_spark_session,
    read_postgres_table,
    write_postgres_table,
    stop_spark_session
)
from src.cache.redis_cache import RedisFeatureCache


class DailyWeatherStatsJob:
    """
    PySpark job to compute daily weather statistics.
    """

    def __init__(self, spark: SparkSession, use_cache: bool = True):
        """
        Initialize the job.

        Args:
            spark: SparkSession instance
            use_cache: Whether to cache results in Redis
        """
        self.spark = spark
        self.use_cache = use_cache
        self.cache = RedisFeatureCache() if use_cache else None
        self.job_name = "daily_weather_stats"

    def load_weather_forecasts(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Load weather forecasts from PostgreSQL.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame: Raw weather forecast data
        """
        logger.info("Loading weather forecasts...")

        # Build query with date filters
        query = "(SELECT * FROM weather_forecasts WHERE 1=1"

        if start_date:
            query += f" AND forecast_date >= '{start_date}'"
        if end_date:
            query += f" AND forecast_date <= '{end_date}'"

        query += ") AS forecasts"

        df = read_postgres_table(self.spark, query)

        logger.info(f"Loaded {df.count()} weather forecast records")
        return df

    def compute_daily_stats(self, df):
        """
        Compute daily statistics from raw forecasts.

        Args:
            df: DataFrame with raw weather forecasts

        Returns:
            DataFrame: Aggregated daily statistics
        """
        logger.info("Computing daily statistics...")

        # Ensure forecast_date is date type
        df = df.withColumn("forecast_date", F.to_date("forecast_date"))

        # Group by region and date, compute aggregations
        daily_stats = df.groupBy("region_id", "forecast_date") \
            .agg(
                # Temperature statistics
                F.min("temperature_min").alias("temp_min"),
                F.max("temperature_max").alias("temp_max"),
                F.avg("temperature_avg").alias("temp_avg"),
                F.stddev("temperature_avg").alias("temp_stddev"),

                # Rainfall statistics
                F.sum("rainfall_mm").alias("rainfall_total"),
                F.max("rainfall_mm").alias("rainfall_max"),
                F.avg("rainfall_mm").alias("rainfall_avg"),

                # Wind statistics
                F.avg("wind_speed_kph").alias("wind_speed_avg"),
                F.max("wind_speed_kph").alias("wind_speed_max"),

                # Humidity statistics
                F.avg("humidity_percent").alias("humidity_avg"),
                F.min("humidity_percent").alias("humidity_min"),
                F.max("humidity_percent").alias("humidity_max"),

                # Data quality metrics
                F.count("*").alias("forecast_count"),
            )

        # Calculate data completeness (assuming 24 hourly forecasts expected)
        daily_stats = daily_stats.withColumn(
            "data_completeness",
            (F.col("forecast_count") / 24.0) * 100
        )

        # Rename forecast_date to stat_date for consistency
        daily_stats = daily_stats.withColumnRenamed("forecast_date", "stat_date")

        # Generate UUID for primary key
        daily_stats = daily_stats.withColumn(
            "id",
            F.expr("uuid()")
        )

        # Add timestamps
        current_timestamp = F.current_timestamp()
        daily_stats = daily_stats \
            .withColumn("created_at", current_timestamp) \
            .withColumn("updated_at", current_timestamp)

        # Round numeric values
        numeric_cols = [
            "temp_min", "temp_max", "temp_avg", "temp_stddev",
            "rainfall_total", "rainfall_max", "rainfall_avg",
            "wind_speed_avg", "wind_speed_max",
            "humidity_avg", "humidity_min", "humidity_max",
            "data_completeness"
        ]

        for col in numeric_cols:
            if col in daily_stats.columns:
                daily_stats = daily_stats.withColumn(
                    col,
                    F.round(F.col(col), 2)
                )

        logger.info(f"Computed statistics for {daily_stats.count()} region-date combinations")

        return daily_stats

    def save_to_postgres(self, df, mode: str = "append"):
        """
        Save daily statistics to PostgreSQL.

        Args:
            df: DataFrame with daily statistics
            mode: Write mode (append, overwrite)
        """
        logger.info(f"Saving daily statistics to PostgreSQL (mode: {mode})...")

        # Select only columns that match the table schema
        columns = [
            "id", "region_id", "stat_date",
            "temp_min", "temp_max", "temp_avg", "temp_stddev",
            "rainfall_total", "rainfall_max", "rainfall_avg",
            "wind_speed_avg", "wind_speed_max",
            "humidity_avg", "humidity_min", "humidity_max",
            "forecast_count", "data_completeness",
            "created_at", "updated_at"
        ]

        df_to_save = df.select(*columns)

        write_postgres_table(df_to_save, "weather_daily_stats", mode=mode)

        logger.info(f"Successfully saved {df_to_save.count()} records")

    def cache_results(self, df):
        """
        Cache computed statistics in Redis.

        Args:
            df: DataFrame with daily statistics
        """
        if not self.cache:
            logger.info("Caching disabled, skipping...")
            return

        logger.info("Caching results in Redis...")

        # Use Spark collect() instead of toPandas() to avoid distutils dependency
        rows = df.select(
            "region_id", "stat_date",
            "temp_avg", "rainfall_total", "wind_speed_avg", "humidity_avg"
        ).collect()

        cached_count = 0
        for row in rows:
            stats = {
                "temp_avg": float(row["temp_avg"]) if row["temp_avg"] else None,
                "rainfall_total": float(row["rainfall_total"]) if row["rainfall_total"] else None,
                "wind_speed_avg": float(row["wind_speed_avg"]) if row["wind_speed_avg"] else None,
                "humidity_avg": float(row["humidity_avg"]) if row["humidity_avg"] else None,
                "date": str(row["stat_date"]),
            }

            self.cache.cache_weather_stats(
                region_id=int(row["region_id"]),
                stats=stats,
                ttl=3600  # 1 hour
            )
            cached_count += 1

        logger.info(f"Cached {cached_count} records in Redis")

    def log_job_metadata(
        self,
        start_time: datetime,
        end_time: datetime,
        status: str,
        records_processed: int,
        records_created: int,
        error_message: Optional[str] = None
    ):
        """
        Log job execution metadata to database.

        Args:
            start_time: Job start timestamp
            end_time: Job end timestamp
            status: Job status (success/failed)
            records_processed: Number of records processed
            records_created: Number of records created
            error_message: Error message if failed
        """
        duration_seconds = int((end_time - start_time).total_seconds())

        # Create metadata as a list of tuples with proper types
        metadata_data = [(
            self.job_name,
            "daily_stats",
            start_time,
            end_time,
            duration_seconds,
            status,
            records_processed,
            records_created,
            0,  # records_updated
            0,  # records_failed
            None,  # date_from
            None,  # date_to
            error_message,
            None,  # error_traceback
            self.spark.sparkContext.applicationId,
            "2g",  # executor_memory
            "2g",  # driver_memory
            datetime.now()
        )]

        # Define schema explicitly to avoid type inference issues
        from pyspark.sql.types import (
            StructType, StructField, StringType, TimestampType,
            IntegerType
        )

        schema = StructType([
            StructField("job_name", StringType(), False),
            StructField("job_type", StringType(), False),
            StructField("start_time", TimestampType(), False),
            StructField("end_time", TimestampType(), True),
            StructField("duration_seconds", IntegerType(), True),
            StructField("status", StringType(), True),
            StructField("records_processed", IntegerType(), True),
            StructField("records_created", IntegerType(), True),
            StructField("records_updated", IntegerType(), True),
            StructField("records_failed", IntegerType(), True),
            StructField("date_from", StringType(), True),
            StructField("date_to", StringType(), True),
            StructField("error_message", StringType(), True),
            StructField("error_traceback", StringType(), True),
            StructField("spark_app_id", StringType(), True),
            StructField("executor_memory", StringType(), True),
            StructField("driver_memory", StringType(), True),
            StructField("created_at", TimestampType(), True),
        ])

        # Create DataFrame with explicit schema
        metadata_df = self.spark.createDataFrame(metadata_data, schema)

        # Save to database
        write_postgres_table(metadata_df, "feature_metadata", mode="append")

        logger.info("Job metadata logged")

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mode: str = "append"
    ):
        """
        Execute the complete job.

        Args:
            start_date: Start date for processing (YYYY-MM-DD)
            end_date: End date for processing (YYYY-MM-DD)
            mode: Write mode (append, overwrite)

        Returns:
            dict: Job execution summary
        """
        job_start = datetime.now()
        logger.info(f"Starting {self.job_name} job...")
        logger.info(f"Date range: {start_date} to {end_date}")

        try:
            # Load data
            forecasts_df = self.load_weather_forecasts(start_date, end_date)
            records_processed = forecasts_df.count()

            # Compute statistics
            stats_df = self.compute_daily_stats(forecasts_df)
            records_created = stats_df.count()

            # Save to PostgreSQL
            self.save_to_postgres(stats_df, mode=mode)

            # Cache results
            if self.use_cache:
                self.cache_results(stats_df)

            # Log metadata
            job_end = datetime.now()
            self.log_job_metadata(
                start_time=job_start,
                end_time=job_end,
                status="success",
                records_processed=records_processed,
                records_created=records_created
            )

            duration = (job_end - job_start).total_seconds()
            logger.info(f"Job completed successfully in {duration:.2f} seconds")

            return {
                "status": "success",
                "records_processed": records_processed,
                "records_created": records_created,
                "duration_seconds": duration
            }

        except Exception as e:
            job_end = datetime.now()
            logger.error(f"Job failed: {e}")

            # Log failure
            try:
                self.log_job_metadata(
                    start_time=job_start,
                    end_time=job_end,
                    status="failed",
                    records_processed=0,
                    records_created=0,
                    error_message=str(e)
                )
            except Exception as log_error:
                logger.error(f"Failed to log metadata: {log_error}")

            raise


def main():
    """
    Main entry point for running the job standalone.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Daily Weather Statistics Aggregation")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--mode", default="append", choices=["append", "overwrite"],
                        help="Write mode")
    parser.add_argument("--no-cache", action="store_true", help="Disable Redis caching")

    args = parser.parse_args()

    # Create Spark session
    spark = get_spark_session(app_name="AgriSafe-DailyStats")

    try:
        # Run job
        job = DailyWeatherStatsJob(spark, use_cache=not args.no_cache)
        result = job.run(
            start_date=args.start_date,
            end_date=args.end_date,
            mode=args.mode
        )

        logger.info(f"Job result: {result}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Job failed with error: {e}")
        sys.exit(1)

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
