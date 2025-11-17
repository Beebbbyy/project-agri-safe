"""
Weather Data Processing DAG - Phase 3

This DAG orchestrates daily weather data processing:
- Runs Spark ETL job for weather aggregations
- Generates rolling window features
- Caches features to Redis
- Performs data quality checks

Schedule: Daily at 8:00 AM UTC (4:00 PM PHT)
Author: AgriSafe Development Team
Date: 2025-01-17
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.processing.spark_jobs.weather_etl import WeatherETL
from src.processing.spark_jobs.rolling_features import RollingFeatureEngine
from src.quality.validators import WeatherDataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


# DAG default arguments
default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 17),
}


def run_spark_etl(**context):
    """
    Execute Spark ETL pipeline for weather data aggregation

    This task:
    - Extracts weather forecasts from PostgreSQL
    - Computes daily statistics per region
    - Calculates basic rolling features
    - Saves results to weather_daily_stats table
    - Optionally caches to Redis

    Args:
        **context: Airflow context with execution date, etc.
    """
    logger.info("Starting Spark ETL pipeline for weather aggregations")

    execution_date = context['execution_date']
    logger.info(f"Execution date: {execution_date}")

    # Define date range for processing
    # Process data from 7 days ago to yesterday to ensure completeness
    end_date = execution_date.date()
    start_date = end_date - timedelta(days=7)

    logger.info(f"Processing date range: {start_date} to {end_date}")

    try:
        # Initialize ETL pipeline
        etl = WeatherETL()

        # Run the ETL pipeline
        stats = etl.run(
            start_date=str(start_date),
            end_date=str(end_date),
            save_to_db=True,
            cache_to_redis=False  # Caching handled by rolling_features task
        )

        # Stop Spark session
        etl.stop()

        # Push results to XCom for downstream tasks
        context['task_instance'].xcom_push(key='etl_stats', value=stats)

        logger.info(f"Spark ETL completed successfully: {stats}")

        return stats

    except Exception as e:
        logger.error(f"Spark ETL failed: {str(e)}")
        raise


def generate_rolling_features(**context):
    """
    Generate advanced rolling window features

    This task:
    - Loads daily statistics from database
    - Computes rolling features (3d, 7d, 14d, 30d windows)
    - Calculates derived features (soil moisture, flood indicators, etc.)
    - Adds historical flood risk data
    - Saves features to weather_features table
    - Caches features to Redis

    Args:
        **context: Airflow context
    """
    logger.info("Starting rolling feature generation")

    execution_date = context['execution_date']

    # Process last 60 days to ensure sufficient window for 30-day features
    end_date = execution_date.date()
    start_date = end_date - timedelta(days=60)

    logger.info(f"Generating features for: {start_date} to {end_date}")

    try:
        # Initialize feature engine
        engine = RollingFeatureEngine()

        # Run feature engineering
        stats = engine.run(
            start_date=str(start_date),
            end_date=str(end_date),
            save_to_db=True
        )

        # Stop Spark session
        engine.stop()

        # Push results to XCom
        context['task_instance'].xcom_push(key='feature_stats', value=stats)

        logger.info(f"Rolling features generated successfully: {stats}")

        return stats

    except Exception as e:
        logger.error(f"Feature generation failed: {str(e)}")
        raise


def cache_features_to_redis(**context):
    """
    Cache latest features to Redis for fast access

    This task:
    - Loads latest weather features from database
    - Caches them to Redis with 7-day TTL
    - Provides fast access for prediction services

    Args:
        **context: Airflow context
    """
    logger.info("Caching features to Redis")

    try:
        import redis
        import pandas as pd
        import json
        import os
        from src.utils.database import get_db_connection

        # Connect to Redis
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True
        )

        # Fetch latest features from database
        query = """
            SELECT
                region_id,
                stat_date,
                rainfall_1d,
                rainfall_3d,
                rainfall_7d,
                rainfall_14d,
                temp_avg_7d,
                temp_variance_30d,
                wind_speed_avg_7d,
                rainy_days_7d,
                soil_moisture_proxy,
                flood_risk_indicator
            FROM weather_features
            WHERE stat_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY region_id, stat_date DESC
        """

        db = get_db_connection()
        with db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        # Cache each record
        cached_count = 0
        ttl = 7 * 24 * 3600  # 7 days

        for _, row in df.iterrows():
            key = f"weather:features:{row['region_id']}:{row['stat_date']}"
            value = {
                'rainfall_1d': float(row['rainfall_1d']) if row['rainfall_1d'] else 0.0,
                'rainfall_3d': float(row['rainfall_3d']) if row['rainfall_3d'] else 0.0,
                'rainfall_7d': float(row['rainfall_7d']) if row['rainfall_7d'] else 0.0,
                'rainfall_14d': float(row['rainfall_14d']) if row['rainfall_14d'] else 0.0,
                'temp_avg_7d': float(row['temp_avg_7d']) if row['temp_avg_7d'] else 0.0,
                'temp_variance_30d': float(row['temp_variance_30d']) if row['temp_variance_30d'] else 0.0,
                'wind_speed_avg_7d': float(row['wind_speed_avg_7d']) if row['wind_speed_avg_7d'] else 0.0,
                'rainy_days_7d': int(row['rainy_days_7d']) if row['rainy_days_7d'] else 0,
                'soil_moisture_proxy': float(row['soil_moisture_proxy']) if row['soil_moisture_proxy'] else 0.0,
                'flood_risk_indicator': float(row['flood_risk_indicator']) if row['flood_risk_indicator'] else 0.0
            }

            redis_client.setex(key, ttl, json.dumps(value))
            cached_count += 1

        redis_client.close()

        logger.info(f"Cached {cached_count} feature records to Redis")

        return {'cached_records': cached_count}

    except ImportError:
        logger.warning("Redis library not available, skipping cache")
        return {'cached_records': 0}
    except Exception as e:
        logger.error(f"Redis caching failed: {str(e)}")
        # Don't fail the task if caching fails
        return {'cached_records': 0, 'error': str(e)}


def run_data_quality_checks(**context):
    """
    Run comprehensive data quality checks

    This task:
    - Validates null values
    - Checks value ranges
    - Verifies data freshness
    - Checks regional coverage
    - Detects statistical anomalies
    - Validates data consistency
    - Saves results to database
    - Raises error if critical checks fail

    Args:
        **context: Airflow context
    """
    logger.info("Running data quality checks")

    try:
        # Initialize validator
        validator = WeatherDataValidator()

        # Run all checks
        summary = validator.run_all_checks(save_results=True)

        # Push results to XCom
        context['task_instance'].xcom_push(key='quality_summary', value=summary)

        # Log summary
        logger.info(f"Quality checks complete: {summary['passed']}/{summary['total_checks']} passed")

        # Print detailed report
        report = validator.get_summary_report()
        logger.info(report)

        # Fail task if critical checks failed
        if not summary['all_passed']:
            critical_failures = [
                check['check'] for check in summary['checks']
                if not check['passed'] and check['severity'] == 'critical'
            ]

            if critical_failures:
                error_msg = f"Critical quality checks failed: {critical_failures}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        return summary

    except Exception as e:
        logger.error(f"Data quality checks failed: {str(e)}")
        raise


def send_pipeline_notification(**context):
    """
    Send notification about pipeline execution status

    This task:
    - Collects results from all previous tasks
    - Generates summary report
    - Logs summary (in production: sends email/Slack)

    Args:
        **context: Airflow context
    """
    logger.info("Sending pipeline notification")

    task_instance = context['task_instance']

    # Get results from previous tasks
    etl_stats = task_instance.xcom_pull(task_ids='run_spark_etl', key='etl_stats')
    feature_stats = task_instance.xcom_pull(task_ids='generate_rolling_features', key='feature_stats')
    quality_summary = task_instance.xcom_pull(task_ids='data_quality_checks', key='quality_summary')

    # Generate summary report
    logger.info("="*70)
    logger.info("WEATHER PROCESSING PIPELINE SUMMARY")
    logger.info("="*70)
    logger.info(f"Execution Date: {context['execution_date']}")
    logger.info("")

    if etl_stats:
        logger.info("SPARK ETL:")
        logger.info(f"  Duration: {etl_stats.get('duration_seconds', 0):.2f}s")
        logger.info(f"  Records Extracted: {etl_stats.get('records_extracted', 0)}")
        logger.info(f"  Daily Stats Computed: {etl_stats.get('daily_stats_computed', 0)}")
        logger.info(f"  Status: {etl_stats.get('status', 'unknown')}")
        logger.info("")

    if feature_stats:
        logger.info("ROLLING FEATURES:")
        logger.info(f"  Duration: {feature_stats.get('duration_seconds', 0):.2f}s")
        logger.info(f"  Records Processed: {feature_stats.get('records_processed', 0)}")
        logger.info(f"  Features Computed: {feature_stats.get('features_computed', 0)}")
        logger.info(f"  Feature Columns: {feature_stats.get('feature_columns', 0)}")
        logger.info(f"  Status: {feature_stats.get('status', 'unknown')}")
        logger.info("")

    if quality_summary:
        logger.info("DATA QUALITY:")
        logger.info(f"  Checks Run: {quality_summary.get('total_checks', 0)}")
        logger.info(f"  Passed: {quality_summary.get('passed', 0)}")
        logger.info(f"  Failed: {quality_summary.get('failed', 0)}")
        logger.info(f"  Success Rate: {quality_summary.get('success_rate', 0):.1f}%")
        logger.info(f"  Critical Failures: {quality_summary.get('critical_failures', 0)}")
        logger.info(f"  Warnings: {quality_summary.get('warnings', 0)}")
        logger.info("")

    logger.info("="*70)

    # In production, this would send email or Slack notification
    # For now, logging is sufficient

    return True


# Define the DAG
with DAG(
    'weather_data_processing',
    default_args=default_args,
    description='Daily weather data ETL and feature engineering pipeline',
    schedule_interval='0 8 * * *',  # 8:00 AM UTC daily (4:00 PM PHT)
    catchup=False,
    max_active_runs=1,
    tags=['phase3', 'etl', 'weather', 'spark', 'features'],
) as dag:

    # Task 1: Run Spark ETL for weather aggregations
    spark_etl_task = PythonOperator(
        task_id='run_spark_etl',
        python_callable=run_spark_etl,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
        doc_md="""
        ### Spark ETL for Weather Aggregations

        **Purpose:** Process raw weather forecasts into daily statistics

        **Operations:**
        - Extracts weather forecasts from PostgreSQL
        - Aggregates to daily statistics per region
        - Computes averages, totals, and maximums
        - Saves to `weather_daily_stats` table

        **Output:** Daily weather statistics for ML feature engineering

        **Timeout:** 15 minutes
        """
    )

    # Task 2: Generate rolling window features
    rolling_features_task = PythonOperator(
        task_id='generate_rolling_features',
        python_callable=generate_rolling_features,
        provide_context=True,
        execution_timeout=timedelta(minutes=20),
        doc_md="""
        ### Rolling Window Feature Engineering

        **Purpose:** Generate advanced time-series features for ML models

        **Features Computed:**
        - Rainfall accumulations (3d, 7d, 14d, 30d)
        - Temperature averages and variances
        - Wind speed metrics
        - Rainy day counts
        - Derived features (soil moisture proxy, flood indicators)
        - Historical flood risk data

        **Output:** `weather_features` table for model training/prediction

        **Timeout:** 20 minutes
        """
    )

    # Task 3: Cache features to Redis
    cache_redis_task = PythonOperator(
        task_id='cache_features_to_redis',
        python_callable=cache_features_to_redis,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Cache Features to Redis

        **Purpose:** Cache latest features for fast access by prediction services

        **Operations:**
        - Fetches latest 7 days of features
        - Stores in Redis with 7-day TTL
        - Key pattern: `weather:features:{region_id}:{date}`

        **Timeout:** 5 minutes
        """
    )

    # Task 4: Run data quality checks
    quality_checks_task = PythonOperator(
        task_id='data_quality_checks',
        python_callable=run_data_quality_checks,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        ### Data Quality Validation

        **Purpose:** Ensure data quality and reliability

        **Checks Performed:**
        - Null value validation
        - Value range checks
        - Data freshness verification
        - Regional coverage validation
        - Statistical anomaly detection
        - Data consistency checks

        **Action:** Fails task if critical checks fail

        **Timeout:** 10 minutes
        """
    )

    # Task 5: Send notification
    notify_task = PythonOperator(
        task_id='send_notification',
        python_callable=send_pipeline_notification,
        provide_context=True,
        trigger_rule='all_done',  # Run even if previous tasks fail
        doc_md="""
        ### Pipeline Notification

        **Purpose:** Send summary of pipeline execution

        **Operations:**
        - Collects results from all tasks
        - Generates summary report
        - Logs summary (production: sends email/Slack)

        **Trigger:** Always runs (even if previous tasks fail)
        """
    )

    # Task 6: Cleanup old data (optional maintenance)
    cleanup_task = BashOperator(
        task_id='cleanup_old_features',
        bash_command="""
        echo "Checking for old feature data to cleanup..."
        # In production, this would run SQL to delete features older than 90 days
        # For now, just a placeholder
        echo "Cleanup check completed"
        """,
        doc_md="""
        ### Cleanup Old Features

        **Purpose:** Maintain database size by removing old features

        **Operations:**
        - Identifies features older than 90 days
        - Removes obsolete data
        - Maintains database performance

        **Note:** Currently a placeholder - implement SQL cleanup in production
        """
    )

    # Define task dependencies
    # Spark ETL must complete first
    spark_etl_task >> rolling_features_task

    # After rolling features, cache to Redis and run quality checks in parallel
    rolling_features_task >> [cache_redis_task, quality_checks_task]

    # Both Redis caching and quality checks must complete before notification
    [cache_redis_task, quality_checks_task] >> notify_task

    # Cleanup runs after quality checks
    quality_checks_task >> cleanup_task
