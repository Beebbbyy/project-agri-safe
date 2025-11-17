"""
Rolling Window Feature Engineering for Weather Data

This module implements rolling window calculations for machine learning features:
- Multi-day rainfall accumulations
- Temperature trends and variances
- Weather pattern indicators
- Seasonal features

These features are used by the flood risk prediction models.

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, avg, sum as spark_sum, max as spark_max, min as spark_min,
    count, stddev, first, when, lit, current_timestamp,
    date_format, dayofweek, month, year, datediff,
    round as spark_round, coalesce
)
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RollingFeatureEngine:
    """
    Advanced feature engineering for weather data using rolling windows

    This class computes time-series features that are critical for
    flood risk prediction and harvest timing recommendations.
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
        Initialize the Rolling Feature Engine

        Args:
            spark: Existing SparkSession or None to create new one
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
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

        logger.info("Initialized RollingFeatureEngine")

    def _create_spark_session(self) -> SparkSession:
        """Create a new SparkSession"""
        return SparkSession.builder \
            .appName("AgriSafe-RollingFeatures") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()

    def load_daily_stats(
        self,
        start_date: str,
        end_date: str,
        region_ids: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Load daily weather statistics from PostgreSQL

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            region_ids: Optional list of region IDs

        Returns:
            DataFrame with daily statistics
        """
        logger.info(f"Loading daily stats from {start_date} to {end_date}")

        region_filter = ""
        if region_ids:
            region_list = ','.join([f"'{rid}'" for rid in region_ids])
            region_filter = f"AND wds.region_id IN ({region_list})"

        query = f"""
            (SELECT
                wds.region_id,
                wds.stat_date,
                wds.temp_high_avg,
                wds.temp_low_avg,
                wds.rainfall_total,
                wds.wind_speed_max,
                wds.dominant_condition,
                r.elevation,
                r.latitude,
                r.longitude
            FROM weather_daily_stats wds
            JOIN regions r ON wds.region_id = r.id
            WHERE wds.stat_date BETWEEN '{start_date}' AND '{end_date}'
            {region_filter}
            ORDER BY wds.region_id, wds.stat_date
            ) as daily_stats
        """

        df = self.spark.read.jdbc(
            url=self.jdbc_url,
            table=query,
            properties=self.jdbc_properties
        )

        logger.info(f"Loaded {df.count()} daily statistics records")
        return df

    def compute_rainfall_features(self, df: DataFrame) -> DataFrame:
        """
        Compute rainfall-related rolling features

        Features:
        - 1, 3, 7, 14, 30-day rainfall accumulations
        - Rainfall intensity indicators
        - Dry spell counters
        - Heavy rain day counts

        Args:
            df: Input DataFrame with daily stats

        Returns:
            DataFrame with rainfall features
        """
        logger.info("Computing rainfall features")

        # Define windows
        window_3d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-2, 0)
        window_7d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-6, 0)
        window_14d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-13, 0)
        window_30d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-29, 0)

        result = df \
            .withColumn("rainfall_1d", col("rainfall_total")) \
            .withColumn("rainfall_3d", spark_round(spark_sum("rainfall_total").over(window_3d), 2)) \
            .withColumn("rainfall_7d", spark_round(spark_sum("rainfall_total").over(window_7d), 2)) \
            .withColumn("rainfall_14d", spark_round(spark_sum("rainfall_total").over(window_14d), 2)) \
            .withColumn("rainfall_30d", spark_round(spark_sum("rainfall_total").over(window_30d), 2)) \
            .withColumn(
                "rainy_days_7d",
                spark_sum(when(col("rainfall_total") > 2.5, 1).otherwise(0)).over(window_7d)
            ) \
            .withColumn(
                "heavy_rain_days_7d",
                spark_sum(when(col("rainfall_total") > 50, 1).otherwise(0)).over(window_7d)
            ) \
            .withColumn(
                "extreme_rain_days_7d",
                spark_sum(when(col("rainfall_total") > 100, 1).otherwise(0)).over(window_7d)
            ) \
            .withColumn(
                "max_daily_rainfall_7d",
                spark_round(spark_max("rainfall_total").over(window_7d), 2)
            ) \
            .withColumn(
                "rainfall_intensity_ratio",
                spark_round(
                    col("rainfall_total") / (col("rainfall_7d") + lit(0.1)),
                    4
                )
            )

        return result

    def compute_temperature_features(self, df: DataFrame) -> DataFrame:
        """
        Compute temperature-related rolling features

        Features:
        - Average temperatures over multiple windows
        - Temperature variance and stability
        - Heat/cold stress indicators
        - Diurnal temperature range

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with temperature features
        """
        logger.info("Computing temperature features")

        window_7d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-6, 0)
        window_14d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-13, 0)
        window_30d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-29, 0)

        result = df \
            .withColumn("temp_avg", spark_round((col("temp_high_avg") + col("temp_low_avg")) / 2, 2)) \
            .withColumn("temp_range", spark_round(col("temp_high_avg") - col("temp_low_avg"), 2)) \
            .withColumn("temp_avg_7d", spark_round(avg("temp_avg").over(window_7d), 2)) \
            .withColumn("temp_avg_14d", spark_round(avg("temp_avg").over(window_14d), 2)) \
            .withColumn("temp_avg_30d", spark_round(avg("temp_avg").over(window_30d), 2)) \
            .withColumn("temp_variance_7d", spark_round(stddev("temp_avg").over(window_7d), 2)) \
            .withColumn("temp_variance_30d", spark_round(stddev("temp_avg").over(window_30d), 2)) \
            .withColumn("temp_max_7d", spark_round(spark_max("temp_high_avg").over(window_7d), 2)) \
            .withColumn("temp_min_7d", spark_round(spark_min("temp_low_avg").over(window_7d), 2)) \
            .withColumn(
                "heat_stress_days_7d",
                spark_sum(when(col("temp_high_avg") > 35, 1).otherwise(0)).over(window_7d)
            )

        return result

    def compute_wind_features(self, df: DataFrame) -> DataFrame:
        """
        Compute wind-related features

        Features:
        - Average and maximum wind speeds
        - High wind day counts
        - Wind speed trends

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with wind features
        """
        logger.info("Computing wind features")

        window_7d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-6, 0)
        window_14d = Window.partitionBy("region_id").orderBy("stat_date").rowsBetween(-13, 0)

        result = df \
            .withColumn("wind_speed_avg_7d", spark_round(avg("wind_speed_max").over(window_7d), 2)) \
            .withColumn("wind_speed_max_7d", spark_round(spark_max("wind_speed_max").over(window_7d), 2)) \
            .withColumn("wind_speed_avg_14d", spark_round(avg("wind_speed_max").over(window_14d), 2)) \
            .withColumn(
                "high_wind_days_7d",
                spark_sum(when(col("wind_speed_max") > 60, 1).otherwise(0)).over(window_7d)
            ) \
            .withColumn(
                "storm_wind_days_7d",
                spark_sum(when(col("wind_speed_max") > 100, 1).otherwise(0)).over(window_7d)
            )

        return result

    def compute_seasonal_features(self, df: DataFrame) -> DataFrame:
        """
        Compute seasonal and cyclical features

        Features:
        - Month indicators
        - Typhoon season flag
        - Wet/dry season indicators
        - Day of week

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with seasonal features
        """
        logger.info("Computing seasonal features")

        result = df \
            .withColumn("month", month("stat_date")) \
            .withColumn("day_of_week", dayofweek("stat_date")) \
            .withColumn(
                "is_typhoon_season",
                when(col("month").isin([6, 7, 8, 9, 10, 11]), 1).otherwise(0)
            ) \
            .withColumn(
                "is_wet_season",
                when(col("month").isin([6, 7, 8, 9, 10, 11]), 1).otherwise(0)
            ) \
            .withColumn(
                "is_dry_season",
                when(col("month").isin([12, 1, 2, 3, 4, 5]), 1).otherwise(0)
            ) \
            .withColumn(
                "season_category",
                when(col("month").isin([12, 1, 2]), "winter")
                .when(col("month").isin([3, 4, 5]), "summer")
                .when(col("month").isin([6, 7, 8]), "monsoon")
                .otherwise("post_monsoon")
            )

        return result

    def compute_derived_features(self, df: DataFrame) -> DataFrame:
        """
        Compute derived features from combinations

        Features:
        - Soil moisture proxy
        - Flood risk indicators
        - Evapotranspiration estimate
        - Growing condition index

        Args:
            df: Input DataFrame with basic features

        Returns:
            DataFrame with derived features
        """
        logger.info("Computing derived features")

        result = df \
            .withColumn(
                "soil_moisture_proxy",
                spark_round(
                    col("rainfall_7d") / (col("temp_avg_7d") + lit(1)),
                    2
                )
            ) \
            .withColumn(
                "evapotranspiration_estimate",
                spark_round(
                    col("temp_avg_7d") * 0.5 - col("rainfall_7d") * 0.1,
                    2
                )
            ) \
            .withColumn(
                "flood_risk_indicator",
                spark_round(
                    (col("rainfall_7d") * 0.4 + col("rainfall_1d") * 0.6) / (col("elevation") + lit(1)),
                    4
                )
            ) \
            .withColumn(
                "growth_condition_index",
                spark_round(
                    when(
                        (col("temp_avg_7d").between(20, 30)) &
                        (col("rainfall_7d").between(20, 150)),
                        100
                    ).when(
                        (col("temp_avg_7d").between(15, 35)) &
                        (col("rainfall_7d").between(10, 200)),
                        70
                    ).otherwise(40),
                    0
                ).cast(IntegerType())
            )

        return result

    def compute_historical_flood_risk(self, df: DataFrame) -> DataFrame:
        """
        Add historical flood risk data per region

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with historical flood counts
        """
        logger.info("Computing historical flood risk")

        try:
            # Load historical flood assessments
            flood_query = """
                (SELECT
                    region_id,
                    COUNT(*) FILTER (WHERE risk_level IN ('high', 'critical')) as historical_high_risk_count,
                    COUNT(*) FILTER (WHERE risk_level = 'critical') as historical_critical_count,
                    COUNT(*) as total_assessments,
                    ROUND(
                        100.0 * COUNT(*) FILTER (WHERE risk_level IN ('high', 'critical')) / NULLIF(COUNT(*), 0),
                        2
                    ) as region_vulnerability_score
                FROM flood_risk_assessments
                GROUP BY region_id
                ) as historical_risk
            """

            flood_df = self.spark.read.jdbc(
                url=self.jdbc_url,
                table=flood_query,
                properties=self.jdbc_properties
            )

            # Join with main DataFrame
            result = df.join(flood_df, on="region_id", how="left")

            # Fill nulls for regions with no history
            result = result.fillna({
                "historical_high_risk_count": 0,
                "historical_critical_count": 0,
                "total_assessments": 0,
                "region_vulnerability_score": 0.0
            })

            return result

        except Exception as e:
            logger.warning(f"Could not load historical flood data: {str(e)}")
            # Add dummy columns if table doesn't exist
            return df \
                .withColumn("historical_high_risk_count", lit(0)) \
                .withColumn("historical_critical_count", lit(0)) \
                .withColumn("total_assessments", lit(0)) \
                .withColumn("region_vulnerability_score", lit(0.0))

    def compute_all_features(self, df: DataFrame) -> DataFrame:
        """
        Compute all rolling window features in sequence

        Args:
            df: Input DataFrame with daily statistics

        Returns:
            DataFrame with all features
        """
        logger.info("Computing all rolling features")

        # Apply all feature computations
        df = self.compute_rainfall_features(df)
        df = self.compute_temperature_features(df)
        df = self.compute_wind_features(df)
        df = self.compute_seasonal_features(df)
        df = self.compute_derived_features(df)
        df = self.compute_historical_flood_risk(df)

        # Add metadata
        df = df.withColumn("features_computed_at", current_timestamp())

        # Handle any remaining nulls
        numeric_columns = [
            field.name for field in df.schema.fields
            if isinstance(field.dataType, (DoubleType, IntegerType))
        ]

        for col_name in numeric_columns:
            df = df.withColumn(col_name, coalesce(col(col_name), lit(0)))

        logger.info("All features computed successfully")
        return df

    def save_features(
        self,
        df: DataFrame,
        table_name: str = "weather_features",
        mode: str = "append"
    ):
        """
        Save computed features to PostgreSQL

        Args:
            df: DataFrame with features
            table_name: Target table name
            mode: Write mode
        """
        logger.info(f"Saving features to {table_name}")

        try:
            df.write.jdbc(
                url=self.jdbc_url,
                table=table_name,
                mode=mode,
                properties=self.jdbc_properties
            )
            logger.info("Features saved successfully")

        except Exception as e:
            logger.error(f"Failed to save features: {str(e)}")
            raise

    def run(
        self,
        start_date: str,
        end_date: str,
        region_ids: Optional[List[str]] = None,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete feature engineering pipeline

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            region_ids: Optional region IDs
            save_to_db: Whether to save to database

        Returns:
            Execution statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting feature engineering for {start_date} to {end_date}")

        try:
            # Load daily statistics
            df = self.load_daily_stats(start_date, end_date, region_ids)

            # Compute all features
            features = self.compute_all_features(df)

            # Save to database
            if save_to_db:
                self.save_features(features)

            # Statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            stats = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "records_processed": df.count(),
                "features_computed": features.count(),
                "feature_columns": len(features.columns),
                "status": "success"
            }

            logger.info(f"Feature engineering completed in {duration:.2f} seconds")
            return stats

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def stop(self):
        """Stop the Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Rolling Feature Engineering')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--regions', nargs='*', help='Optional region IDs')
    parser.add_argument('--no-save', action='store_true', help='Skip saving to database')

    args = parser.parse_args()

    engine = RollingFeatureEngine()

    try:
        stats = engine.run(
            start_date=args.start_date,
            end_date=args.end_date,
            region_ids=args.regions,
            save_to_db=not args.no_save
        )

        print(f"\n{'='*60}")
        print("Feature Engineering Completed")
        print(f"{'='*60}")
        print(f"Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"Records: {stats['records_processed']}")
        print(f"Features: {stats['feature_columns']} columns")
        print(f"{'='*60}\n")

        sys.exit(0)

    except Exception as e:
        print(f"\nFeature Engineering Failed: {str(e)}")
        sys.exit(1)

    finally:
        engine.stop()


if __name__ == "__main__":
    main()
