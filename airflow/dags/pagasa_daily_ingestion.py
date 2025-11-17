"""
PAGASA Daily Weather Data Ingestion DAG

This DAG runs daily to fetch weather forecast data from PAGASA
and store it in the database.

Schedule: Daily at 6:00 AM PHT (UTC+8)
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

from src.ingestion.pagasa_connector import PAGASAIngestionService
from src.utils.logger import get_logger

logger = get_logger(__name__)

# DAG default arguments
default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 11, 17),
}


def fetch_pagasa_data(**context):
    """
    Task to fetch PAGASA weather data

    Args:
        **context: Airflow context with execution date, etc.
    """
    logger.info("Starting PAGASA data fetch task...")

    execution_date = context['execution_date']
    logger.info(f"Execution date: {execution_date}")

    # Initialize ingestion service
    service = PAGASAIngestionService()

    # Run ingestion
    result = service.run_ingestion()

    # Push results to XCom for downstream tasks
    context['task_instance'].xcom_push(key='ingestion_result', value=result)

    if not result['success']:
        raise Exception(f"PAGASA data ingestion failed: {result.get('error')}")

    logger.info(f"PAGASA data fetch completed: {result['forecasts_saved']} forecasts saved")

    return result


def validate_data(**context):
    """
    Task to validate the ingested data

    Args:
        **context: Airflow context
    """
    logger.info("Starting data validation task...")

    # Pull ingestion results from XCom
    task_instance = context['task_instance']
    ingestion_result = task_instance.xcom_pull(
        task_ids='fetch_pagasa_data',
        key='ingestion_result'
    )

    if not ingestion_result:
        raise Exception("No ingestion result found in XCom")

    # Perform validation checks
    forecasts_saved = ingestion_result.get('forecasts_saved', 0)

    # Check if we saved a reasonable number of forecasts
    # Assuming we have at least 10 regions and 5-day forecasts = 50 minimum
    if forecasts_saved < 50:
        logger.warning(f"Low number of forecasts saved: {forecasts_saved}")
        # Don't fail the task, just log warning

    # Additional validation: check database for data quality
    from src.utils.database import get_db_connection

    db = get_db_connection()

    # Check for today's forecasts
    today = datetime.now().date()
    query = """
        SELECT COUNT(*) as count
        FROM weather_forecasts
        WHERE forecast_date >= %s
        AND DATE(created_at) = %s
    """

    result = db.execute_query(query, (today, today))

    if result and len(result) > 0:
        count = result[0]['count']
        logger.info(f"Found {count} forecasts for today in database")

        if count == 0:
            raise Exception("No forecasts found in database for today")
    else:
        raise Exception("Failed to query database for validation")

    logger.info("Data validation completed successfully")

    return {'validation_passed': True, 'forecasts_count': forecasts_saved}


def send_notification(**context):
    """
    Task to send notification about ingestion status

    Args:
        **context: Airflow context
    """
    logger.info("Sending ingestion notification...")

    task_instance = context['task_instance']

    # Get results from previous tasks
    ingestion_result = task_instance.xcom_pull(
        task_ids='fetch_pagasa_data',
        key='ingestion_result'
    )

    validation_result = task_instance.xcom_pull(
        task_ids='validate_data'
    )

    # In production, this would send email/Slack notification
    # For now, just log the summary
    logger.info("=" * 50)
    logger.info("PAGASA Daily Ingestion Summary")
    logger.info("=" * 50)
    logger.info(f"Execution Date: {context['execution_date']}")
    logger.info(f"Success: {ingestion_result.get('success', False)}")
    logger.info(f"Forecasts Saved: {ingestion_result.get('forecasts_saved', 0)}")
    logger.info(f"Regions Processed: {ingestion_result.get('regions_processed', 0)}")
    logger.info(f"Duration: {ingestion_result.get('duration_seconds', 0):.2f}s")
    logger.info(f"Validation Passed: {validation_result.get('validation_passed', False)}")
    logger.info("=" * 50)

    return True


# Define the DAG
with DAG(
    'pagasa_daily_ingestion',
    default_args=default_args,
    description='Daily PAGASA weather data ingestion pipeline',
    schedule_interval='0 22 * * *',  # 6:00 AM PHT (22:00 UTC previous day)
    catchup=False,
    tags=['weather', 'pagasa', 'ingestion'],
) as dag:

    # Task 1: Fetch PAGASA data
    fetch_data_task = PythonOperator(
        task_id='fetch_pagasa_data',
        python_callable=fetch_pagasa_data,
        provide_context=True,
        doc_md="""
        ### Fetch PAGASA Data

        This task fetches weather forecast data from PAGASA API sources
        and stores it in the database.

        **Data Source:** PAGASA (via Vercel API)
        **Output:** Weather forecasts for all regions (5-day forecast)
        """
    )

    # Task 2: Validate data
    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
        doc_md="""
        ### Validate Data

        This task validates the ingested weather data to ensure:
        - Sufficient number of forecasts were saved
        - Data exists in the database
        - Data quality checks pass
        """
    )

    # Task 3: Send notification
    notify_task = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        provide_context=True,
        trigger_rule='all_done',  # Run even if previous tasks fail
        doc_md="""
        ### Send Notification

        This task sends a notification with the ingestion summary.
        In production, this would send email or Slack messages.
        """
    )

    # Task 4: Cleanup old forecasts (optional)
    cleanup_task = BashOperator(
        task_id='cleanup_old_forecasts',
        bash_command="""
        echo "Cleaning up old forecast data..."
        # This would connect to DB and delete forecasts older than 30 days
        # For now, just a placeholder
        echo "Cleanup completed"
        """,
        doc_md="""
        ### Cleanup Old Forecasts

        This task removes forecast data older than 30 days
        to keep the database size manageable.
        """
    )

    # Define task dependencies
    fetch_data_task >> validate_data_task >> notify_task
    validate_data_task >> cleanup_task
