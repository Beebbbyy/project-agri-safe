"""
Data Quality Monitoring DAG - Phase 3

This DAG orchestrates comprehensive data quality monitoring:
- Runs all quality validators (null checks, ranges, freshness, etc.)
- Generates quality reports and dashboards
- Sends alerts for critical failures
- Tracks quality trends over time

Schedule: Every 6 hours
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

from src.quality.validators import WeatherDataValidator
from src.quality.monitoring import QualityMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)


# DAG default arguments
default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 17),
}


def run_all_quality_checks(**context):
    """
    Execute comprehensive data quality validation

    This task:
    - Checks for null values
    - Validates value ranges
    - Verifies data freshness
    - Checks regional coverage
    - Detects statistical anomalies
    - Validates data consistency
    - Saves all results to database

    Args:
        **context: Airflow context
    """
    logger.info("Starting comprehensive quality checks")

    execution_date = context['execution_date']
    logger.info(f"Execution date: {execution_date}")

    try:
        # Initialize validator
        validator = WeatherDataValidator()

        # Run all checks and save to database
        summary = validator.run_all_checks(save_results=True)

        # Push results to XCom
        context['task_instance'].xcom_push(key='quality_summary', value=summary)

        # Log detailed report
        report = validator.get_summary_report()
        logger.info(report)

        # Return summary
        return summary

    except Exception as e:
        logger.error(f"Quality checks failed: {str(e)}")
        raise


def generate_quality_reports(**context):
    """
    Generate comprehensive quality reports

    This task:
    - Creates text report
    - Generates HTML dashboard
    - Analyzes quality trends
    - Identifies patterns and issues

    Args:
        **context: Airflow context
    """
    logger.info("Generating quality reports")

    try:
        # Initialize monitor
        monitor = QualityMonitor()

        # Generate text report
        text_report = monitor.generate_report(
            days=1,
            include_trends=True,
            include_failures=True
        )

        logger.info(text_report)

        # Generate HTML dashboard
        html_dashboard = monitor.generate_html_dashboard(days=7)

        # Save HTML to file (in production: upload to S3 or serve via web)
        report_dir = Path('/tmp/quality_reports')
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = report_dir / f'quality_dashboard_{timestamp}.html'

        with open(html_path, 'w') as f:
            f.write(html_dashboard)

        logger.info(f"HTML dashboard saved to: {html_path}")

        # Get quality trends
        trends = monitor.get_quality_trends(days=7)

        # Push to XCom
        reports = {
            'text_report': text_report,
            'html_path': str(html_path),
            'trends_count': len(trends)
        }

        context['task_instance'].xcom_push(key='reports', value=reports)

        return reports

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise


def analyze_quality_trends(**context):
    """
    Analyze data quality trends over time

    This task:
    - Examines quality metrics over past 30 days
    - Identifies deteriorating trends
    - Compares current vs historical performance
    - Highlights areas needing attention

    Args:
        **context: Airflow context
    """
    logger.info("Analyzing quality trends")

    try:
        # Initialize monitor
        monitor = QualityMonitor()

        # Get comprehensive statistics
        stats = monitor.get_check_statistics(days=30)

        # Analyze trends
        logger.info("="*70)
        logger.info("QUALITY TRENDS ANALYSIS (Past 30 Days)")
        logger.info("="*70)

        for check_name, check_stats in stats.items():
            success_rate = check_stats['success_rate']
            total_runs = check_stats['total_runs']
            critical_count = check_stats['critical_count']

            # Determine trend status
            if success_rate >= 95:
                status = "✓ EXCELLENT"
            elif success_rate >= 90:
                status = "✓ GOOD"
            elif success_rate >= 80:
                status = "⚠ NEEDS ATTENTION"
            else:
                status = "✗ CRITICAL"

            logger.info(f"\n{check_name.upper()}:")
            logger.info(f"  Status: {status}")
            logger.info(f"  Success Rate: {success_rate:.1f}%")
            logger.info(f"  Total Runs: {total_runs}")
            logger.info(f"  Critical Failures: {critical_count}")

            if check_stats['avg_actual_value'] is not None:
                logger.info(f"  Avg Value: {check_stats['avg_actual_value']:.2f}")
                logger.info(f"  Max Value: {check_stats['max_actual_value']:.2f}")
                logger.info(f"  Min Value: {check_stats['min_actual_value']:.2f}")

        logger.info("")
        logger.info("="*70)

        # Push to XCom
        trend_analysis = {
            'total_checks': len(stats),
            'stats': stats
        }

        context['task_instance'].xcom_push(key='trend_analysis', value=trend_analysis)

        return trend_analysis

    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        # Don't fail the task - analysis is informational
        return {'error': str(e)}


def generate_alerts(**context):
    """
    Generate alerts for critical quality issues

    This task:
    - Identifies critical failures
    - Generates actionable alerts
    - Provides recommendations
    - Prepares notifications

    Args:
        **context: Airflow context
    """
    logger.info("Generating quality alerts")

    task_instance = context['task_instance']

    # Get quality summary from validation task
    quality_summary = task_instance.xcom_pull(
        task_ids='run_quality_checks',
        key='quality_summary'
    )

    if not quality_summary:
        logger.warning("No quality summary available for alert generation")
        return {'alerts': []}

    try:
        # Initialize monitor
        monitor = QualityMonitor()

        # Get alert recommendations
        alerts = monitor.get_alert_recommendations()

        # Log alerts
        if alerts:
            logger.info("="*70)
            logger.info("DATA QUALITY ALERTS")
            logger.info("="*70)

            for alert in alerts:
                level_icon = "🔴" if alert['level'] == 'critical' else "🟡"

                logger.info(f"\n{level_icon} [{alert['level'].upper()}] {alert['check']}")
                logger.info(f"  Message: {alert['message']}")
                logger.info(f"  Recommendation: {alert['recommendation']}")
                logger.info(f"  Timestamp: {alert['timestamp']}")

            logger.info("")
            logger.info(f"Total Alerts: {len(alerts)}")
            logger.info("="*70)

        else:
            logger.info("✓ No critical quality issues detected")

        # Add summary information to alerts
        alert_summary = {
            'total_alerts': len(alerts),
            'critical_count': len([a for a in alerts if a['level'] == 'critical']),
            'warning_count': len([a for a in alerts if a['level'] == 'warning']),
            'alerts': alerts,
            'quality_summary': quality_summary
        }

        # Push to XCom
        context['task_instance'].xcom_push(key='alerts', value=alert_summary)

        # Return alerts for downstream tasks
        return alert_summary

    except Exception as e:
        logger.error(f"Alert generation failed: {str(e)}")
        return {'error': str(e), 'alerts': []}


def send_quality_notifications(**context):
    """
    Send notifications about data quality status

    This task:
    - Sends email alerts for critical issues (production)
    - Posts to Slack channel (production)
    - Logs comprehensive summary (development)

    Args:
        **context: Airflow context
    """
    logger.info("Sending quality notifications")

    task_instance = context['task_instance']

    # Get results from all tasks
    quality_summary = task_instance.xcom_pull(
        task_ids='run_quality_checks',
        key='quality_summary'
    )

    reports = task_instance.xcom_pull(
        task_ids='generate_reports',
        key='reports'
    )

    alerts = task_instance.xcom_pull(
        task_ids='generate_alerts',
        key='alerts'
    )

    # Generate comprehensive notification
    logger.info("="*70)
    logger.info("DATA QUALITY MONITORING SUMMARY")
    logger.info("="*70)
    logger.info(f"Execution Date: {context['execution_date']}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Overall Status
    if quality_summary:
        all_passed = quality_summary.get('all_passed', False)
        status_icon = "✓" if all_passed else "✗"
        status_text = "HEALTHY" if all_passed else "ISSUES DETECTED"

        logger.info(f"OVERALL STATUS: {status_icon} {status_text}")
        logger.info("")

        logger.info("CHECK SUMMARY:")
        logger.info(f"  Total Checks: {quality_summary.get('total_checks', 0)}")
        logger.info(f"  Passed: {quality_summary.get('passed', 0)}")
        logger.info(f"  Failed: {quality_summary.get('failed', 0)}")
        logger.info(f"  Success Rate: {quality_summary.get('success_rate', 0):.1f}%")
        logger.info(f"  Critical Failures: {quality_summary.get('critical_failures', 0)}")
        logger.info(f"  Warnings: {quality_summary.get('warnings', 0)}")
        logger.info("")

    # Alert Summary
    if alerts and alerts.get('total_alerts', 0) > 0:
        logger.info("ACTIVE ALERTS:")
        logger.info(f"  Total: {alerts.get('total_alerts', 0)}")
        logger.info(f"  Critical: {alerts.get('critical_count', 0)}")
        logger.info(f"  Warnings: {alerts.get('warning_count', 0)}")
        logger.info("")

        # List top alerts
        for alert in alerts.get('alerts', [])[:5]:  # Top 5 alerts
            logger.info(f"  • [{alert['level'].upper()}] {alert['check']}")
            logger.info(f"    {alert['message']}")

        if len(alerts.get('alerts', [])) > 5:
            logger.info(f"  ... and {len(alerts['alerts']) - 5} more")

        logger.info("")

    # Report Information
    if reports:
        logger.info("REPORTS GENERATED:")
        logger.info(f"  HTML Dashboard: {reports.get('html_path', 'N/A')}")
        logger.info(f"  Trends Analyzed: {reports.get('trends_count', 0)} records")
        logger.info("")

    logger.info("="*70)

    # In production, this would:
    # 1. Send email if critical failures
    # 2. Post to Slack channel
    # 3. Update monitoring dashboard
    # 4. Trigger PagerDuty if necessary

    notification_result = {
        'notifications_sent': True,
        'timestamp': datetime.now().isoformat(),
        'status': 'healthy' if quality_summary.get('all_passed', False) else 'issues_detected',
        'alert_count': alerts.get('total_alerts', 0) if alerts else 0
    }

    return notification_result


def cleanup_old_quality_checks(**context):
    """
    Clean up old quality check records

    This task:
    - Removes quality checks older than 90 days
    - Maintains database performance
    - Preserves summary statistics

    Args:
        **context: Airflow context
    """
    logger.info("Cleaning up old quality check records")

    try:
        from src.utils.database import get_db_connection

        db = get_db_connection()

        # Delete quality checks older than 90 days
        query = """
            DELETE FROM data_quality_checks
            WHERE checked_at < CURRENT_DATE - INTERVAL '90 days'
        """

        with db.get_cursor() as cursor:
            cursor.execute(query)
            deleted_count = cursor.rowcount

        logger.info(f"Deleted {deleted_count} old quality check records")

        return {'deleted_records': deleted_count}

    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        # Don't fail the task - cleanup is maintenance
        return {'error': str(e), 'deleted_records': 0}


def check_critical_failures(**context):
    """
    Check if critical failures exist and fail task if needed

    This task:
    - Reviews quality check results
    - Fails the DAG if critical issues exist
    - Ensures visibility of critical problems

    Args:
        **context: Airflow context
    """
    logger.info("Checking for critical failures")

    task_instance = context['task_instance']

    quality_summary = task_instance.xcom_pull(
        task_ids='run_quality_checks',
        key='quality_summary'
    )

    if not quality_summary:
        logger.warning("No quality summary available")
        return

    critical_failures = quality_summary.get('critical_failures', 0)

    if critical_failures > 0:
        # Get details of critical failures
        critical_checks = [
            check for check in quality_summary.get('checks', [])
            if not check['passed'] and check['severity'] == 'critical'
        ]

        error_msg = f"CRITICAL: {critical_failures} critical quality check(s) failed:\n"

        for check in critical_checks:
            error_msg += f"\n  • {check['check']}: {check['message']}"
            if check.get('threshold') and check.get('actual_value'):
                error_msg += f"\n    Threshold: {check['threshold']}, Actual: {check['actual_value']:.2f}"

        logger.error(error_msg)

        # Fail the task to alert team
        raise ValueError(error_msg)

    else:
        logger.info("✓ No critical failures detected")


# Define the DAG
with DAG(
    'data_quality_monitoring',
    default_args=default_args,
    description='Comprehensive data quality validation and monitoring',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    max_active_runs=1,
    tags=['phase3', 'quality', 'monitoring', 'validation'],
) as dag:

    # Task 1: Run all quality checks
    quality_checks_task = PythonOperator(
        task_id='run_quality_checks',
        python_callable=run_all_quality_checks,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        ### Run All Quality Checks

        **Purpose:** Execute comprehensive data quality validation

        **Checks Performed:**
        - Null value detection
        - Value range validation
        - Data freshness verification
        - Regional coverage assessment
        - Statistical anomaly detection
        - Data consistency validation

        **Output:** Quality summary with pass/fail status for each check

        **Timeout:** 10 minutes
        """
    )

    # Task 2: Generate reports
    generate_reports_task = PythonOperator(
        task_id='generate_reports',
        python_callable=generate_quality_reports,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Generate Quality Reports

        **Purpose:** Create comprehensive quality reports

        **Outputs:**
        - Text report with current status
        - HTML dashboard for visualization
        - Trend analysis

        **Timeout:** 5 minutes
        """
    )

    # Task 3: Analyze trends
    analyze_trends_task = PythonOperator(
        task_id='analyze_trends',
        python_callable=analyze_quality_trends,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Analyze Quality Trends

        **Purpose:** Examine quality metrics over time

        **Analysis:**
        - 30-day trend analysis
        - Success rate trends
        - Deteriorating patterns
        - Performance comparisons

        **Timeout:** 5 minutes
        """
    )

    # Task 4: Generate alerts
    alerts_task = PythonOperator(
        task_id='generate_alerts',
        python_callable=generate_alerts,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Generate Alerts

        **Purpose:** Create actionable alerts for quality issues

        **Outputs:**
        - Critical failure alerts
        - Warning notifications
        - Actionable recommendations

        **Timeout:** 5 minutes
        """
    )

    # Task 5: Send notifications
    notify_task = PythonOperator(
        task_id='send_notifications',
        python_callable=send_quality_notifications,
        provide_context=True,
        trigger_rule='all_done',  # Run even if previous tasks fail
        doc_md="""
        ### Send Quality Notifications

        **Purpose:** Notify team about quality status

        **Channels:**
        - Log summary (always)
        - Email alerts (critical issues)
        - Slack notifications (production)

        **Trigger:** Always runs (even if checks fail)
        """
    )

    # Task 6: Check for critical failures (fails DAG if critical)
    check_critical_task = PythonOperator(
        task_id='check_critical_failures',
        python_callable=check_critical_failures,
        provide_context=True,
        doc_md="""
        ### Check Critical Failures

        **Purpose:** Fail DAG if critical quality issues exist

        **Action:** Raises error if critical checks failed

        **Result:** Ensures visibility of critical problems
        """
    )

    # Task 7: Cleanup old records
    cleanup_task = PythonOperator(
        task_id='cleanup_old_checks',
        python_callable=cleanup_old_quality_checks,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Cleanup Old Quality Checks

        **Purpose:** Maintain database performance

        **Operation:** Deletes quality check records older than 90 days

        **Note:** Does not fail on errors (maintenance task)

        **Timeout:** 5 minutes
        """
    )

    # Task 8: Export quality metrics (for external monitoring)
    export_metrics_task = BashOperator(
        task_id='export_quality_metrics',
        bash_command="""
        echo "Exporting quality metrics to monitoring system..."
        # In production, this would export metrics to Prometheus, Datadog, etc.
        # For now, just a placeholder
        echo "Metrics export completed (placeholder)"
        """,
        doc_md="""
        ### Export Quality Metrics

        **Purpose:** Export metrics to external monitoring systems

        **Targets:**
        - Prometheus (time-series metrics)
        - Datadog (dashboards)
        - CloudWatch (AWS monitoring)

        **Note:** Currently a placeholder for future implementation
        """
    )

    # Define task dependencies
    # Run quality checks first
    quality_checks_task >> [generate_reports_task, analyze_trends_task, alerts_task]

    # All analysis tasks must complete before notifications
    [generate_reports_task, analyze_trends_task, alerts_task] >> notify_task

    # After notifications, check for critical failures
    notify_task >> check_critical_task

    # Cleanup and export run in parallel after critical check
    check_critical_task >> [cleanup_task, export_metrics_task]
