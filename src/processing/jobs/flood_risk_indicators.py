"""
Flood Risk Indicators Job

Calculates flood risk indicators based on rainfall patterns and historical data.
Uses rule-based and statistical approaches to assess flood risk levels.
"""

import sys
from datetime import datetime
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


class FloodRiskIndicatorsJob:
    """
    PySpark job to calculate flood risk indicators.
    """

    # Risk thresholds
    RAINFALL_HEAVY_THRESHOLD = 50.0  # mm per day
    RAINFALL_EXTREME_THRESHOLD = 100.0  # mm per day
    CUMULATIVE_7D_HIGH_RISK = 150.0  # mm in 7 days
    CUMULATIVE_7D_CRITICAL_RISK = 250.0  # mm in 7 days
    CONSECUTIVE_RAIN_HIGH_RISK = 5  # days

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
        self.job_name = "flood_risk_indicators"

    def load_rolling_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        window_days: int = 7
    ):
        """
        Load rolling window features from PostgreSQL.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            window_days: Window size to use (default: 7)

        Returns:
            DataFrame: Rolling features
        """
        logger.info(f"Loading {window_days}-day rolling features...")

        query = f"(SELECT * FROM weather_rolling_features WHERE window_days = {window_days}"

        if start_date:
            query += f" AND feature_date >= '{start_date}'"
        if end_date:
            query += f" AND feature_date <= '{end_date}'"

        query += ") AS features"

        df = read_postgres_table(self.spark, query)

        logger.info(f"Loaded {df.count()} rolling feature records")
        return df

    def calculate_risk_scores(self, df):
        """
        Calculate flood risk scores and indicators.

        Args:
            df: DataFrame with rolling features

        Returns:
            DataFrame: With risk scores and indicators
        """
        logger.info("Calculating flood risk scores...")

        # Ensure feature_date is date type
        df = df.withColumn("feature_date", F.to_date("feature_date"))

        # Calculate rainfall intensity score (0-100)
        df = df.withColumn(
            "rainfall_intensity_score",
            F.least(
                (F.col("rainfall_rolling_avg") / self.RAINFALL_HEAVY_THRESHOLD) * 100,
                F.lit(100.0)
            )
        )

        # Calculate rainfall duration score based on consecutive rainy days (0-100)
        df = df.withColumn(
            "rainfall_duration_score",
            F.least(
                (F.col("consecutive_rain_days") / 10.0) * 100,
                F.lit(100.0)
            )
        )

        # Calculate 7-day and 14-day cumulative rainfall
        # Since we're working with 7-day features, rainfall_rolling_sum is 7-day cumulative
        df = df.withColumn("cumulative_rainfall_7d", F.col("rainfall_rolling_sum"))

        # Load 14-day features for 14-day cumulative
        df_14d = self.load_rolling_features(window_days=14)
        df_14d_cum = df_14d.select(
            "region_id",
            F.col("feature_date"),
            F.col("rainfall_rolling_sum").alias("cumulative_rainfall_14d")
        )

        # Join with main dataframe
        df = df.join(
            df_14d_cum,
            on=["region_id", "feature_date"],
            how="left"
        )

        # Calculate risk factors (0-1 scale)
        # Heavy rainfall factor
        df = df.withColumn(
            "heavy_rainfall_factor",
            F.when(
                F.col("cumulative_rainfall_7d") >= self.CUMULATIVE_7D_CRITICAL_RISK,
                F.lit(1.0)
            ).when(
                F.col("cumulative_rainfall_7d") >= self.CUMULATIVE_7D_HIGH_RISK,
                F.col("cumulative_rainfall_7d") / self.CUMULATIVE_7D_CRITICAL_RISK
            ).otherwise(
                F.col("cumulative_rainfall_7d") / self.CUMULATIVE_7D_HIGH_RISK * 0.5
            )
        )

        # Prolonged rain factor
        df = df.withColumn(
            "prolonged_rain_factor",
            F.least(
                F.col("consecutive_rain_days") / self.CONSECUTIVE_RAIN_HIGH_RISK,
                F.lit(1.0)
            )
        )

        # Soil saturation proxy (based on recent rainfall)
        df = df.withColumn(
            "soil_saturation_proxy",
            F.least(
                (F.col("rainfall_heavy_days") / 7.0) +
                (F.col("consecutive_rain_days") / 10.0),
                F.lit(1.0)
            )
        )

        # Calculate composite flood risk score (0-100)
        # Weighted average of factors
        df = df.withColumn(
            "flood_risk_score",
            (
                F.col("heavy_rainfall_factor") * 0.5 +
                F.col("prolonged_rain_factor") * 0.3 +
                F.col("soil_saturation_proxy") * 0.2
            ) * 100
        )

        # Determine flood risk level
        df = df.withColumn(
            "flood_risk_level",
            F.when(F.col("flood_risk_score") >= 75, "Critical")
            .when(F.col("flood_risk_score") >= 50, "High")
            .when(F.col("flood_risk_score") >= 25, "Moderate")
            .otherwise("Low")
        )

        # Set alert flags
        df = df.withColumn(
            "is_high_risk",
            F.col("flood_risk_score") >= 50
        )

        df = df.withColumn(
            "is_critical_risk",
            F.col("flood_risk_score") >= 75
        )

        # Generate alert messages
        df = df.withColumn(
            "alert_message",
            F.when(
                F.col("is_critical_risk"),
                F.concat(
                    F.lit("CRITICAL FLOOD RISK: "),
                    F.round(F.col("cumulative_rainfall_7d"), 1),
                    F.lit("mm rainfall in 7 days. "),
                    F.col("consecutive_rain_days"),
                    F.lit(" consecutive rainy days. Immediate action recommended.")
                )
            ).when(
                F.col("is_high_risk"),
                F.concat(
                    F.lit("HIGH FLOOD RISK: "),
                    F.round(F.col("cumulative_rainfall_7d"), 1),
                    F.lit("mm rainfall in 7 days. Monitor conditions closely.")
                )
            ).otherwise(
                F.lit(None)
            )
        )

        # Calculate historical percentile (placeholder - would need historical data)
        df = df.withColumn("percentile_vs_historical", F.lit(None))

        # Add model metadata
        df = df.withColumn("model_version", F.lit("rule_based_v1.0"))
        df = df.withColumn(
            "confidence_score",
            F.when(F.col("rainfall_days_count") >= 5, F.lit(90.0))
            .when(F.col("rainfall_days_count") >= 3, F.lit(75.0))
            .otherwise(F.lit(60.0))
        )

        # Rename feature_date to indicator_date
        df = df.withColumnRenamed("feature_date", "indicator_date")

        # Generate UUID and timestamps
        df = df \
            .withColumn("id", F.expr("uuid()")) \
            .withColumn("created_at", F.current_timestamp()) \
            .withColumn("updated_at", F.current_timestamp())

        # Round numeric values
        numeric_cols = [
            "rainfall_intensity_score", "rainfall_duration_score",
            "cumulative_rainfall_7d", "cumulative_rainfall_14d",
            "flood_risk_score", "heavy_rainfall_factor",
            "prolonged_rain_factor", "soil_saturation_proxy",
            "percentile_vs_historical", "confidence_score"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df = df.withColumn(col, F.round(F.col(col), 2))

        logger.info(f"Calculated risk indicators for {df.count()} records")

        return df

    def save_to_postgres(self, df, mode: str = "append"):
        """
        Save flood risk indicators to PostgreSQL.

        Args:
            df: DataFrame with risk indicators
            mode: Write mode
        """
        logger.info(f"Saving flood risk indicators to PostgreSQL (mode: {mode})...")

        columns = [
            "id", "region_id", "indicator_date",
            "rainfall_intensity_score", "rainfall_duration_score",
            "cumulative_rainfall_7d", "cumulative_rainfall_14d",
            "flood_risk_score", "flood_risk_level",
            "heavy_rainfall_factor", "prolonged_rain_factor", "soil_saturation_proxy",
            "percentile_vs_historical",
            "is_high_risk", "is_critical_risk", "alert_message",
            "model_version", "confidence_score",
            "created_at", "updated_at"
        ]

        df_to_save = df.select(*columns)

        write_postgres_table(df_to_save, "flood_risk_indicators", mode=mode)

        logger.info(f"Successfully saved {df_to_save.count()} records")

    def cache_results(self, df):
        """
        Cache risk indicators in Redis.

        Args:
            df: DataFrame with risk indicators
        """
        if not self.cache:
            logger.info("Caching disabled, skipping...")
            return

        logger.info("Caching risk indicators in Redis...")

        # Cache latest indicators for each region
        # Use Spark collect() instead of toPandas() to avoid distutils dependency
        rows = df.select(
            "region_id", "indicator_date",
            "flood_risk_score", "flood_risk_level",
            "cumulative_rainfall_7d", "is_high_risk", "is_critical_risk",
            "alert_message"
        ).collect()

        cached_count = 0
        for row in rows:
            indicators = {
                "risk_score": float(row["flood_risk_score"]) if row["flood_risk_score"] else None,
                "risk_level": str(row["flood_risk_level"]),
                "cumulative_rainfall_7d": float(row["cumulative_rainfall_7d"]) if row["cumulative_rainfall_7d"] else None,
                "is_high_risk": bool(row["is_high_risk"]),
                "is_critical_risk": bool(row["is_critical_risk"]),
                "alert_message": str(row["alert_message"]) if row["alert_message"] else None,
                "date": str(row["indicator_date"]),
            }

            self.cache.cache_risk_indicators(
                region_id=int(row["region_id"]),
                indicators=indicators,
                ttl=1800  # 30 minutes
            )
            cached_count += 1

        logger.info(f"Cached {cached_count} risk indicators in Redis")

    def generate_summary_stats(self, df):
        """
        Generate summary statistics for monitoring.

        Args:
            df: DataFrame with risk indicators

        Returns:
            dict: Summary statistics
        """
        # Count by risk level
        risk_counts = df.groupBy("flood_risk_level").count().collect()
        risk_summary = {row["flood_risk_level"]: row["count"] for row in risk_counts}

        # Count high and critical risk regions
        high_risk_count = df.filter(F.col("is_high_risk")).count()
        critical_risk_count = df.filter(F.col("is_critical_risk")).count()

        # Average risk score
        avg_risk = df.agg(F.avg("flood_risk_score")).collect()[0][0]

        summary = {
            "total_regions": df.count(),
            "risk_level_distribution": risk_summary,
            "high_risk_count": high_risk_count,
            "critical_risk_count": critical_risk_count,
            "average_risk_score": round(float(avg_risk), 2) if avg_risk else 0.0
        }

        logger.info(f"Summary: {summary}")

        return summary

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mode: str = "append"
    ):
        """
        Execute the complete job.

        Args:
            start_date: Start date for processing
            end_date: End date for processing
            mode: Write mode

        Returns:
            dict: Job execution summary
        """
        job_start = datetime.now()
        logger.info(f"Starting {self.job_name} job...")

        try:
            # Load rolling features (7-day)
            features_df = self.load_rolling_features(start_date, end_date, window_days=7)
            records_processed = features_df.count()

            # Calculate risk indicators
            indicators_df = self.calculate_risk_scores(features_df)
            records_created = indicators_df.count()

            # Generate summary stats
            summary = self.generate_summary_stats(indicators_df)

            # Save to PostgreSQL
            self.save_to_postgres(indicators_df, mode=mode)

            # Cache results
            if self.use_cache:
                self.cache_results(indicators_df)

            duration = (datetime.now() - job_start).total_seconds()
            logger.info(f"Job completed successfully in {duration:.2f} seconds")

            return {
                "status": "success",
                "records_processed": records_processed,
                "records_created": records_created,
                "summary": summary,
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

    parser = argparse.ArgumentParser(description="Flood Risk Indicators Calculation")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--mode", default="append", choices=["append", "overwrite"])
    parser.add_argument("--no-cache", action="store_true", help="Disable Redis caching")

    args = parser.parse_args()

    # Create Spark session
    spark = get_spark_session(app_name="AgriSafe-FloodRisk")

    try:
        # Run job
        job = FloodRiskIndicatorsJob(spark, use_cache=not args.no_cache)
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
