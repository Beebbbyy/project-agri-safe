"""
Rolling Window Features Job

Computes rolling window features (7-day, 14-day, 30-day) for weather data.
Includes temperature trends, rainfall patterns, and extreme event indicators.
"""

import sys
from datetime import datetime
from typing import Optional, List
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


class RollingFeaturesJob:
    """
    PySpark job to compute rolling window features.
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
        self.job_name = "rolling_features"

    def load_daily_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Load daily weather statistics from PostgreSQL.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame: Daily statistics
        """
        logger.info("Loading daily weather statistics...")

        query = "(SELECT * FROM weather_daily_stats WHERE 1=1"

        if start_date:
            query += f" AND stat_date >= '{start_date}'"
        if end_date:
            query += f" AND stat_date <= '{end_date}'"

        query += " ORDER BY region_id, stat_date) AS daily_stats"

        df = read_postgres_table(self.spark, query)

        logger.info(f"Loaded {df.count()} daily statistics records")
        return df

    def compute_rolling_features(
        self,
        df,
        window_days: List[int] = [7, 14, 30]
    ):
        """
        Compute rolling window features for specified windows.

        Args:
            df: DataFrame with daily statistics
            window_days: List of window sizes in days

        Returns:
            DataFrame: Rolling features for all windows
        """
        logger.info(f"Computing rolling features for windows: {window_days}")

        # Ensure stat_date is date type
        df = df.withColumn("stat_date", F.to_date("stat_date"))

        all_features = []

        for window in window_days:
            logger.info(f"Computing {window}-day rolling features...")

            # Define window specification
            # Partition by region, order by date, rows from (window-1) preceding to current
            window_spec = Window \
                .partitionBy("region_id") \
                .orderBy("stat_date") \
                .rowsBetween(-(window - 1), 0)

            # Compute rolling features
            features = df.select(
                "region_id",
                F.col("stat_date").alias("feature_date"),

                # Temperature features
                F.avg("temp_avg").over(window_spec).alias("temp_rolling_avg"),
                F.min("temp_min").over(window_spec).alias("temp_rolling_min"),
                F.max("temp_max").over(window_spec).alias("temp_rolling_max"),
                F.stddev("temp_avg").over(window_spec).alias("temp_rolling_stddev"),

                # Rainfall features
                F.sum("rainfall_total").over(window_spec).alias("rainfall_rolling_sum"),
                F.avg("rainfall_total").over(window_spec).alias("rainfall_rolling_avg"),
                F.max("rainfall_max").over(window_spec).alias("rainfall_rolling_max"),
                F.stddev("rainfall_total").over(window_spec).alias("rainfall_rolling_stddev"),

                # Count rainy days (rainfall > 0)
                F.sum(
                    F.when(F.col("rainfall_total") > 0, 1).otherwise(0)
                ).over(window_spec).alias("rainfall_days_count"),

                # Count heavy rainfall days (> 50mm)
                F.sum(
                    F.when(F.col("rainfall_total") > 50, 1).otherwise(0)
                ).over(window_spec).alias("rainfall_heavy_days"),

                # Wind features
                F.avg("wind_speed_avg").over(window_spec).alias("wind_rolling_avg"),
                F.max("wind_speed_max").over(window_spec).alias("wind_rolling_max"),

                # Humidity features
                F.avg("humidity_avg").over(window_spec).alias("humidity_rolling_avg"),
                F.stddev("humidity_avg").over(window_spec).alias("humidity_rolling_stddev"),

                # Extreme temperature days (< 10°C or > 35°C)
                F.sum(
                    F.when(
                        (F.col("temp_min") < 10) | (F.col("temp_max") > 35),
                        1
                    ).otherwise(0)
                ).over(window_spec).alias("extreme_temp_days")
            )

            # Compute temperature trend (simple linear regression coefficient)
            features = self._compute_temp_trend(features, window, window_spec)

            # Compute consecutive rainy days
            features = self._compute_consecutive_rain_days(features, df, window)

            # Add window size
            features = features.withColumn("window_days", F.lit(window))

            all_features.append(features)

        # Combine all windows
        combined_features = all_features[0]
        for feat_df in all_features[1:]:
            combined_features = combined_features.union(feat_df)

        # Generate UUID and timestamps
        combined_features = combined_features \
            .withColumn("id", F.expr("uuid()")) \
            .withColumn("created_at", F.current_timestamp()) \
            .withColumn("updated_at", F.current_timestamp())

        # Round numeric values
        numeric_cols = [
            "temp_rolling_avg", "temp_rolling_min", "temp_rolling_max", "temp_rolling_stddev",
            "temp_trend", "rainfall_rolling_sum", "rainfall_rolling_avg",
            "rainfall_rolling_max", "rainfall_rolling_stddev",
            "wind_rolling_avg", "wind_rolling_max",
            "humidity_rolling_avg", "humidity_rolling_stddev"
        ]

        for col in numeric_cols:
            if col in combined_features.columns:
                combined_features = combined_features.withColumn(
                    col,
                    F.round(F.col(col), 2)
                )

        logger.info(f"Computed {combined_features.count()} rolling feature records")

        return combined_features

    def _compute_temp_trend(self, df, window_days: int, window_spec):
        """
        Compute simple temperature trend coefficient.

        Args:
            df: DataFrame with features
            window_days: Window size
            window_spec: Window specification

        Returns:
            DataFrame: With temp_trend column added
        """
        # Simple trend: (latest temp - earliest temp) / window_days
        # More sophisticated: use linear regression (requires collect_list + UDF)

        # For simplicity, use difference between recent avg and older avg
        half_window = window_days // 2

        recent_window = Window \
            .partitionBy("region_id") \
            .orderBy("feature_date") \
            .rowsBetween(-half_window, 0)

        older_window = Window \
            .partitionBy("region_id") \
            .orderBy("feature_date") \
            .rowsBetween(-(window_days - 1), -half_window)

        # This requires temp_avg from original data
        # For now, use a placeholder calculation
        df = df.withColumn(
            "temp_trend",
            F.lit(0.0)  # Placeholder - can be enhanced with UDF
        )

        return df

    def _compute_consecutive_rain_days(self, features_df, daily_df, window_days: int):
        """
        Compute maximum consecutive rainy days within the window.

        Args:
            features_df: DataFrame with rolling features
            daily_df: Original daily statistics
            window_days: Window size

        Returns:
            DataFrame: With consecutive_rain_days column added
        """
        # Create a binary indicator for rainy days
        daily_with_rain = daily_df.select(
            "region_id",
            "stat_date",
            F.when(F.col("rainfall_total") > 0, 1).otherwise(0).alias("is_rainy")
        )

        # Define window for consecutive calculation
        window_spec = Window \
            .partitionBy("region_id") \
            .orderBy("stat_date") \
            .rowsBetween(-(window_days - 1), 0)

        # Collect rainy day indicators as array
        daily_with_rain = daily_with_rain.withColumn(
            "rainy_days_array",
            F.collect_list("is_rainy").over(window_spec)
        )

        # UDF to calculate max consecutive rainy days
        @F.udf("int")
        def max_consecutive(arr):
            if not arr:
                return 0

            max_consecutive = 0
            current_consecutive = 0

            for is_rainy in arr:
                if is_rainy == 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            return max_consecutive

        consecutive_df = daily_with_rain.select(
            "region_id",
            F.col("stat_date").alias("feature_date"),
            max_consecutive("rainy_days_array").alias("consecutive_rain_days")
        )

        # Join with features
        features_df = features_df.join(
            consecutive_df,
            on=["region_id", "feature_date"],
            how="left"
        )

        # Fill nulls with 0
        features_df = features_df.fillna({"consecutive_rain_days": 0})

        return features_df

    def save_to_postgres(self, df, mode: str = "append"):
        """
        Save rolling features to PostgreSQL.

        Args:
            df: DataFrame with rolling features
            mode: Write mode
        """
        logger.info(f"Saving rolling features to PostgreSQL (mode: {mode})...")

        columns = [
            "id", "region_id", "feature_date", "window_days",
            "temp_rolling_avg", "temp_rolling_min", "temp_rolling_max",
            "temp_rolling_stddev", "temp_trend",
            "rainfall_rolling_sum", "rainfall_rolling_avg", "rainfall_rolling_max",
            "rainfall_rolling_stddev", "rainfall_days_count", "rainfall_heavy_days",
            "wind_rolling_avg", "wind_rolling_max",
            "humidity_rolling_avg", "humidity_rolling_stddev",
            "extreme_temp_days", "consecutive_rain_days",
            "created_at", "updated_at"
        ]

        df_to_save = df.select(*columns)

        write_postgres_table(df_to_save, "weather_rolling_features", mode=mode)

        logger.info(f"Successfully saved {df_to_save.count()} records")

    def cache_results(self, df):
        """
        Cache rolling features in Redis.

        Args:
            df: DataFrame with rolling features
        """
        if not self.cache:
            logger.info("Caching disabled, skipping...")
            return

        logger.info("Caching rolling features in Redis...")

        # Cache only latest features for each region and window
        # Use Spark collect() instead of toPandas() to avoid distutils dependency
        rows = df.select(
            "region_id", "window_days", "feature_date",
            "rainfall_rolling_sum", "rainfall_heavy_days",
            "temp_rolling_avg", "consecutive_rain_days"
        ).collect()

        cached_count = 0
        for row in rows:
            features = {
                "rainfall_sum": float(row["rainfall_rolling_sum"]) if row["rainfall_rolling_sum"] else None,
                "heavy_rain_days": int(row["rainfall_heavy_days"]) if row["rainfall_heavy_days"] else 0,
                "temp_avg": float(row["temp_rolling_avg"]) if row["temp_rolling_avg"] else None,
                "consecutive_rain_days": int(row["consecutive_rain_days"]) if row["consecutive_rain_days"] else 0,
                "date": str(row["feature_date"]),
            }

            self.cache.cache_rolling_features(
                region_id=int(row["region_id"]),
                window_days=int(row["window_days"]),
                features=features,
                ttl=3600  # 1 hour
            )
            cached_count += 1

        logger.info(f"Cached {cached_count} feature sets in Redis")

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        window_days: List[int] = [7, 14, 30],
        mode: str = "append"
    ):
        """
        Execute the complete job.

        Args:
            start_date: Start date for processing
            end_date: End date for processing
            window_days: List of window sizes
            mode: Write mode

        Returns:
            dict: Job execution summary
        """
        job_start = datetime.now()
        logger.info(f"Starting {self.job_name} job...")
        logger.info(f"Windows: {window_days}")

        try:
            # Load data
            daily_stats = self.load_daily_stats(start_date, end_date)
            records_processed = daily_stats.count()

            # Compute rolling features
            features_df = self.compute_rolling_features(daily_stats, window_days)
            records_created = features_df.count()

            # Save to PostgreSQL
            self.save_to_postgres(features_df, mode=mode)

            # Cache results
            if self.use_cache:
                self.cache_results(features_df)

            duration = (datetime.now() - job_start).total_seconds()
            logger.info(f"Job completed successfully in {duration:.2f} seconds")

            return {
                "status": "success",
                "records_processed": records_processed,
                "records_created": records_created,
                "duration_seconds": duration
            }

        except Exception as e:
            logger.error(f"Job failed: {e}")
            raise


def main():
    """
    Main entry point for running the job standalone.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Rolling Window Features")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--windows", default="7,14,30",
                        help="Comma-separated window sizes (default: 7,14,30)")
    parser.add_argument("--mode", default="append", choices=["append", "overwrite"])
    parser.add_argument("--no-cache", action="store_true", help="Disable Redis caching")

    args = parser.parse_args()

    # Parse window sizes
    window_days = [int(w) for w in args.windows.split(",")]

    # Create Spark session
    spark = get_spark_session(app_name="AgriSafe-RollingFeatures")

    try:
        # Run job
        job = RollingFeaturesJob(spark, use_cache=not args.no_cache)
        result = job.run(
            start_date=args.start_date,
            end_date=args.end_date,
            window_days=window_days,
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
