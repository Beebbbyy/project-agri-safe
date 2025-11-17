"""
Spark ETL Pipeline DAG

Orchestrates the complete Spark ETL workflow:
1. Daily weather statistics aggregation
2. Rolling window features computation
3. Flood risk indicators calculation

Runs daily after PAGASA data ingestion completes.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from loguru import logger
import sys
import os

# Add project root to Python path
project_root = "/opt/airflow"
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Default arguments
default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}

# DAG definition
dag = DAG(
    'spark_etl_pipeline',
    default_args=default_args,
    description='Spark ETL pipeline for weather feature engineering',
    schedule_interval='0 23 * * *',  # Daily at 11 PM UTC (7 AM PHT) - after PAGASA ingestion
    start_date=days_ago(1),
    catchup=False,
    tags=['spark', 'etl', 'features', 'ml'],
    max_active_runs=1,
)


def run_daily_stats_job(**context):
    """
    Run daily weather statistics aggregation.
    """
    from src.processing.jobs.daily_weather_stats import DailyWeatherStatsJob
    from src.processing.utils.spark_session import get_spark_session, stop_spark_session

    logger.info("Starting daily stats job...")

    # Get execution date
    execution_date = context['execution_date']
    start_date = (execution_date - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = execution_date.strftime('%Y-%m-%d')

    spark = None
    try:
        # Create Spark session
        spark = get_spark_session(app_name="AgriSafe-DailyStats-Airflow")

        # Run job
        job = DailyWeatherStatsJob(spark, use_cache=True)
        result = job.run(
            start_date=start_date,
            end_date=end_date,
            mode="append"
        )

        logger.info(f"Daily stats job completed: {result}")

        # Push result to XCom for downstream tasks
        context['task_instance'].xcom_push(key='stats_result', value=result)

        return result

    except Exception as e:
        logger.error(f"Daily stats job failed: {e}")
        raise

    finally:
        if spark:
            stop_spark_session(spark)


def run_rolling_features_job(**context):
    """
    Run rolling window features computation.
    """
    from src.processing.jobs.rolling_features import RollingFeaturesJob
    from src.processing.utils.spark_session import get_spark_session, stop_spark_session

    logger.info("Starting rolling features job...")

    # Get execution date
    execution_date = context['execution_date']
    # Process last 45 days to ensure we have enough data for 30-day windows
    start_date = (execution_date - timedelta(days=45)).strftime('%Y-%m-%d')
    end_date = execution_date.strftime('%Y-%m-%d')

    spark = None
    try:
        # Create Spark session
        spark = get_spark_session(app_name="AgriSafe-RollingFeatures-Airflow")

        # Run job with all window sizes
        job = RollingFeaturesJob(spark, use_cache=True)
        result = job.run(
            start_date=start_date,
            end_date=end_date,
            window_days=[7, 14, 30],
            mode="append"
        )

        logger.info(f"Rolling features job completed: {result}")

        # Push result to XCom
        context['task_instance'].xcom_push(key='features_result', value=result)

        return result

    except Exception as e:
        logger.error(f"Rolling features job failed: {e}")
        raise

    finally:
        if spark:
            stop_spark_session(spark)


def run_flood_risk_job(**context):
    """
    Run flood risk indicators calculation.
    """
    from src.processing.jobs.flood_risk_indicators import FloodRiskIndicatorsJob
    from src.processing.utils.spark_session import get_spark_session, stop_spark_session

    logger.info("Starting flood risk indicators job...")

    # Get execution date
    execution_date = context['execution_date']
    # Process last 14 days
    start_date = (execution_date - timedelta(days=14)).strftime('%Y-%m-%d')
    end_date = execution_date.strftime('%Y-%m-%d')

    spark = None
    try:
        # Create Spark session
        spark = get_spark_session(app_name="AgriSafe-FloodRisk-Airflow")

        # Run job
        job = FloodRiskIndicatorsJob(spark, use_cache=True)
        result = job.run(
            start_date=start_date,
            end_date=end_date,
            mode="append"
        )

        logger.info(f"Flood risk job completed: {result}")

        # Push result to XCom
        context['task_instance'].xcom_push(key='risk_result', value=result)

        return result

    except Exception as e:
        logger.error(f"Flood risk job failed: {e}")
        raise

    finally:
        if spark:
            stop_spark_session(spark)


def validate_pipeline_results(**context):
    """
    Validate that all jobs completed successfully and check data quality.
    """
    from src.utils.database import DatabaseConnection

    logger.info("Validating pipeline results...")

    # Get results from previous tasks
    ti = context['task_instance']
    stats_result = ti.xcom_pull(task_ids='run_daily_stats', key='stats_result')
    features_result = ti.xcom_pull(task_ids='run_rolling_features', key='features_result')
    risk_result = ti.xcom_pull(task_ids='run_flood_risk', key='risk_result')

    logger.info(f"Stats: {stats_result}")
    logger.info(f"Features: {features_result}")
    logger.info(f"Risk: {risk_result}")

    # Validate all jobs succeeded
    validation_results = {
        'daily_stats': stats_result.get('status') == 'success' if stats_result else False,
        'rolling_features': features_result.get('status') == 'success' if features_result else False,
        'flood_risk': risk_result.get('status') == 'success' if risk_result else False,
    }

    # Check database record counts
    db = DatabaseConnection()
    try:
        with db.get_cursor() as cur:
            # Check recent records in each table
            execution_date = context['execution_date'].strftime('%Y-%m-%d')

            # Daily stats
            cur.execute(
                "SELECT COUNT(*) FROM weather_daily_stats WHERE stat_date >= %s",
                (execution_date,)
            )
            stats_count = cur.fetchone()[0]

            # Rolling features
            cur.execute(
                "SELECT COUNT(*) FROM weather_rolling_features WHERE feature_date >= %s",
                (execution_date,)
            )
            features_count = cur.fetchone()[0]

            # Flood risk
            cur.execute(
                "SELECT COUNT(*) FROM flood_risk_indicators WHERE indicator_date >= %s",
                (execution_date,)
            )
            risk_count = cur.fetchone()[0]

            validation_results['record_counts'] = {
                'daily_stats': stats_count,
                'rolling_features': features_count,
                'flood_risk': risk_count
            }

            logger.info(f"Validation results: {validation_results}")

            # Raise error if critical validations fail
            if not all([
                validation_results['daily_stats'],
                validation_results['rolling_features'],
                validation_results['flood_risk']
            ]):
                raise ValueError("One or more ETL jobs failed")

            return validation_results

    finally:
        db.close()


def send_completion_notification(**context):
    """
    Send notification about pipeline completion.
    """
    ti = context['task_instance']
    validation = ti.xcom_pull(task_ids='validate_results')

    logger.info("=" * 60)
    logger.info("SPARK ETL PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Execution Date: {context['execution_date']}")
    logger.info(f"Validation Results: {validation}")
    logger.info("=" * 60)

    # In production, send email/Slack notification here
    return {
        "status": "completed",
        "validation": validation
    }


# Define tasks
task_daily_stats = PythonOperator(
    task_id='run_daily_stats',
    python_callable=run_daily_stats_job,
    provide_context=True,
    dag=dag,
)

task_rolling_features = PythonOperator(
    task_id='run_rolling_features',
    python_callable=run_rolling_features_job,
    provide_context=True,
    dag=dag,
)

task_flood_risk = PythonOperator(
    task_id='run_flood_risk',
    python_callable=run_flood_risk_job,
    provide_context=True,
    dag=dag,
)

task_validate = PythonOperator(
    task_id='validate_results',
    python_callable=validate_pipeline_results,
    provide_context=True,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_completion_notification,
    provide_context=True,
    dag=dag,
)

# Optional: Cleanup old data
task_cleanup = BashOperator(
    task_id='cleanup_old_data',
    bash_command="""
    echo "Cleaning up old processed data..."
    # Add cleanup logic here if needed
    # e.g., delete data older than 1 year
    echo "Cleanup completed"
    """,
    dag=dag,
)

# Define task dependencies
# Linear pipeline: stats -> features -> risk -> validate -> notify
#                                              \\-> cleanup
task_daily_stats >> task_rolling_features >> task_flood_risk >> task_validate
task_validate >> task_notify
task_validate >> task_cleanup

# Documentation
dag.doc_md = """
# Spark ETL Pipeline

This DAG orchestrates the complete ETL workflow for weather feature engineering.

## Workflow

1. **Daily Stats** - Aggregate raw weather forecasts into daily statistics
2. **Rolling Features** - Compute 7-day, 14-day, and 30-day rolling window features
3. **Flood Risk** - Calculate flood risk indicators based on rainfall patterns
4. **Validation** - Verify all jobs completed successfully
5. **Notification** - Send completion notification
6. **Cleanup** - Remove old processed data (optional)

## Schedule

Runs daily at 11 PM UTC (7 AM Philippine Time), one hour after PAGASA ingestion.

## Dependencies

- PAGASA ingestion must complete before this pipeline runs
- PostgreSQL database with schema from Phase 1
- Redis for feature caching
- PySpark 3.5+

## Monitoring

Check Airflow UI for task status and logs. Review XCom data for detailed results.
"""
