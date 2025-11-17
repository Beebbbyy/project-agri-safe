"""
Data Quality Monitoring and Reporting

This module provides monitoring dashboards and reports for data quality:
- Quality trends over time
- Critical failure tracking
- Automated report generation
- Alert generation for quality issues

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

from src.utils.database import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityMonitor:
    """
    Monitor and report on data quality metrics

    Provides:
    - Historical quality trends
    - Critical failure tracking
    - Automated report generation
    - Quality dashboards
    """

    def __init__(self):
        """Initialize quality monitor"""
        self.db = get_db_connection()
        logger.info("Initialized QualityMonitor")

    def get_quality_trends(
        self,
        days: int = 7,
        check_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get quality check trends over time

        Args:
            days: Number of days to analyze
            check_names: Optional filter for specific checks

        Returns:
            DataFrame with quality trends
        """
        logger.info(f"Fetching quality trends for past {days} days")

        check_filter = ""
        if check_names:
            check_list = ','.join([f"'{name}'" for name in check_names])
            check_filter = f"AND check_name IN ({check_list})"

        query = f"""
            SELECT
                DATE(checked_at) as check_date,
                check_name,
                COUNT(*) as total_checks,
                SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_checks,
                SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed_checks,
                ROUND(
                    100.0 * SUM(CASE WHEN passed THEN 1 ELSE 0 END) / COUNT(*),
                    2
                ) as pass_rate,
                AVG(actual_value) as avg_actual_value,
                MAX(threshold) as threshold_value
            FROM data_quality_checks
            WHERE checked_at >= CURRENT_DATE - INTERVAL '{days} days'
            {check_filter}
            GROUP BY DATE(checked_at), check_name
            ORDER BY check_date DESC, check_name
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Retrieved {len(df)} trend records")
        return df

    def get_critical_failures(
        self,
        days: int = 7,
        severity: str = 'critical'
    ) -> pd.DataFrame:
        """
        Get recent critical failures

        Args:
            days: Number of days to look back
            severity: Severity level to filter ('critical', 'warning', 'info')

        Returns:
            DataFrame with failure details
        """
        logger.info(f"Fetching {severity} failures for past {days} days")

        query = f"""
            SELECT
                check_name,
                severity,
                message,
                details,
                threshold,
                actual_value,
                checked_at
            FROM data_quality_checks
            WHERE severity = '{severity}'
              AND NOT passed
              AND checked_at >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY checked_at DESC
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Found {len(df)} {severity} failures")
        return df

    def get_check_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get overall statistics for all checks

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        logger.info(f"Computing check statistics for past {days} days")

        query = f"""
            SELECT
                check_name,
                COUNT(*) as total_runs,
                SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_runs,
                ROUND(
                    100.0 * SUM(CASE WHEN passed THEN 1 ELSE 0 END) / COUNT(*),
                    2
                ) as success_rate,
                COUNT(*) FILTER (WHERE severity = 'critical') as critical_count,
                COUNT(*) FILTER (WHERE severity = 'warning') as warning_count,
                MIN(checked_at) as first_check,
                MAX(checked_at) as last_check,
                AVG(actual_value) as avg_actual_value,
                MAX(actual_value) as max_actual_value,
                MIN(actual_value) as min_actual_value
            FROM data_quality_checks
            WHERE checked_at >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY check_name
            ORDER BY check_name
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        stats = {}
        for _, row in df.iterrows():
            stats[row['check_name']] = {
                'total_runs': int(row['total_runs']),
                'passed_runs': int(row['passed_runs']),
                'success_rate': float(row['success_rate']),
                'critical_count': int(row['critical_count']),
                'warning_count': int(row['warning_count']),
                'first_check': row['first_check'].isoformat(),
                'last_check': row['last_check'].isoformat(),
                'avg_actual_value': float(row['avg_actual_value']) if row['avg_actual_value'] else None,
                'max_actual_value': float(row['max_actual_value']) if row['max_actual_value'] else None,
                'min_actual_value': float(row['min_actual_value']) if row['min_actual_value'] else None
            }

        return stats

    def get_recent_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of most recent quality checks

        Args:
            hours: Number of hours to look back

        Returns:
            Summary dictionary
        """
        logger.info(f"Getting quality summary for past {hours} hours")

        query = f"""
            WITH latest_checks AS (
                SELECT
                    check_name,
                    passed,
                    severity,
                    message,
                    checked_at,
                    ROW_NUMBER() OVER (PARTITION BY check_name ORDER BY checked_at DESC) as rn
                FROM data_quality_checks
                WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '{hours} hours'
            )
            SELECT
                check_name,
                passed,
                severity,
                message,
                checked_at
            FROM latest_checks
            WHERE rn = 1
            ORDER BY check_name
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        if len(df) == 0:
            return {
                'timestamp': datetime.now().isoformat(),
                'checks_found': 0,
                'all_passed': False,
                'critical_failures': 0,
                'warnings': 0,
                'checks': []
            }

        summary = {
            'timestamp': datetime.now().isoformat(),
            'checks_found': len(df),
            'all_passed': df['passed'].all(),
            'critical_failures': len(df[(~df['passed']) & (df['severity'] == 'critical')]),
            'warnings': len(df[(~df['passed']) & (df['severity'] == 'warning')]),
            'checks': [
                {
                    'check_name': row['check_name'],
                    'passed': bool(row['passed']),
                    'severity': row['severity'],
                    'message': row['message'],
                    'checked_at': row['checked_at'].isoformat()
                }
                for _, row in df.iterrows()
            ]
        }

        return summary

    def generate_report(
        self,
        days: int = 1,
        include_trends: bool = True,
        include_failures: bool = True
    ) -> str:
        """
        Generate comprehensive text report

        Args:
            days: Number of days to cover
            include_trends: Include trend analysis
            include_failures: Include failure details

        Returns:
            Formatted report string
        """
        logger.info(f"Generating quality report for past {days} day(s)")

        report_lines = [
            "",
            "="*70,
            "DATA QUALITY REPORT",
            "="*70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Period: Past {days} day(s)",
            ""
        ]

        # Recent summary
        summary = self.get_recent_summary(hours=days*24)

        report_lines.append("CURRENT STATUS")
        report_lines.append("-"*70)
        report_lines.append(f"Checks Monitored: {summary['checks_found']}")
        report_lines.append(f"All Passed: {'YES ✓' if summary['all_passed'] else 'NO ✗'}")
        report_lines.append(f"Critical Failures: {summary['critical_failures']}")
        report_lines.append(f"Warnings: {summary['warnings']}")
        report_lines.append("")

        # Check details
        report_lines.append("CHECK DETAILS")
        report_lines.append("-"*70)

        for check in summary['checks']:
            status = "✓ PASS" if check['passed'] else "✗ FAIL"
            severity_icon = {
                'critical': '🔴',
                'warning': '🟡',
                'info': '🔵'
            }.get(check['severity'], '')

            report_lines.append(f"{status} {severity_icon} {check['check_name'].upper()}")
            report_lines.append(f"  {check['message']}")
            report_lines.append(f"  Last checked: {check['checked_at']}")
            report_lines.append("")

        # Critical failures detail
        if include_failures:
            failures = self.get_critical_failures(days=days)

            if len(failures) > 0:
                report_lines.append("")
                report_lines.append("CRITICAL FAILURES")
                report_lines.append("-"*70)

                for _, failure in failures.head(10).iterrows():
                    report_lines.append(f"• {failure['check_name']}")
                    report_lines.append(f"  {failure['message']}")
                    report_lines.append(f"  Time: {failure['checked_at']}")

                    if failure['threshold'] and failure['actual_value']:
                        report_lines.append(
                            f"  Threshold: {failure['threshold']} | "
                            f"Actual: {failure['actual_value']:.2f}"
                        )

                    report_lines.append("")

        # Trends
        if include_trends and days > 1:
            trends = self.get_quality_trends(days=days)

            if len(trends) > 0:
                report_lines.append("")
                report_lines.append("QUALITY TRENDS")
                report_lines.append("-"*70)

                # Group by check name and show trend
                for check_name in trends['check_name'].unique():
                    check_trends = trends[trends['check_name'] == check_name].sort_values('check_date')

                    if len(check_trends) > 0:
                        latest_pass_rate = check_trends.iloc[-1]['pass_rate']
                        avg_pass_rate = check_trends['pass_rate'].mean()

                        trend_symbol = "→"
                        if len(check_trends) > 1:
                            if latest_pass_rate > check_trends.iloc[0]['pass_rate']:
                                trend_symbol = "↑"
                            elif latest_pass_rate < check_trends.iloc[0]['pass_rate']:
                                trend_symbol = "↓"

                        report_lines.append(
                            f"{check_name}: {latest_pass_rate:.1f}% pass rate "
                            f"(avg: {avg_pass_rate:.1f}%) {trend_symbol}"
                        )

        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("")

        return "\n".join(report_lines)

    def generate_html_dashboard(self, days: int = 7) -> str:
        """
        Generate HTML dashboard for quality metrics

        Args:
            days: Number of days to visualize

        Returns:
            HTML string
        """
        logger.info("Generating HTML dashboard")

        summary = self.get_recent_summary(hours=days*24)
        trends = self.get_quality_trends(days=days)
        failures = self.get_critical_failures(days=days)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Dashboard - AgriSafe</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                }}
                .metric {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    flex: 1;
                    margin: 0 10px;
                }}
                .metric-value {{
                    font-size: 36px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 14px;
                }}
                .checks {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .check-item {{
                    padding: 10px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                .pass {{ color: #27ae60; }}
                .fail {{ color: #e74c3c; }}
                .critical {{ color: #c0392b; font-weight: bold; }}
                .warning {{ color: #f39c12; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Dashboard</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Period: Past {days} day(s)</p>
            </div>

            <div class="summary">
                <div class="metric">
                    <div class="metric-label">Total Checks</div>
                    <div class="metric-value">{summary['checks_found']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Critical Failures</div>
                    <div class="metric-value" style="color: #e74c3c;">
                        {summary['critical_failures']}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Warnings</div>
                    <div class="metric-value" style="color: #f39c12;">
                        {summary['warnings']}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div class="metric-value" style="color: {'#27ae60' if summary['all_passed'] else '#e74c3c'};">
                        {'PASS' if summary['all_passed'] else 'FAIL'}
                    </div>
                </div>
            </div>

            <div class="checks">
                <h2>Recent Checks</h2>
        """

        for check in summary['checks']:
            status_class = 'pass' if check['passed'] else 'fail'
            severity_class = check['severity']

            html += f"""
                <div class="check-item">
                    <span class="{status_class}">
                        {'✓' if check['passed'] else '✗'}
                    </span>
                    <strong class="{severity_class}">{check['check_name'].upper()}</strong>
                    <br/>
                    <small>{check['message']}</small>
                    <br/>
                    <small style="color: #95a5a6;">
                        Last checked: {check['checked_at']}
                    </small>
                </div>
            """

        html += """
            </div>
        """

        if len(failures) > 0:
            html += """
            <div class="checks">
                <h2>Recent Critical Failures</h2>
            """

            for _, failure in failures.head(10).iterrows():
                html += f"""
                <div class="check-item critical">
                    <strong>{failure['check_name']}</strong>
                    <br/>
                    <small>{failure['message']}</small>
                    <br/>
                    <small style="color: #95a5a6;">
                        {failure['checked_at']}
                    </small>
                </div>
                """

            html += """
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def get_alert_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate recommendations based on recent failures

        Returns:
            List of alert/recommendation dictionaries
        """
        logger.info("Generating alert recommendations")

        failures = self.get_critical_failures(days=1, severity='critical')
        warnings = self.get_critical_failures(days=1, severity='warning')

        alerts = []

        # Critical alerts
        for _, failure in failures.iterrows():
            alerts.append({
                'level': 'critical',
                'check': failure['check_name'],
                'message': failure['message'],
                'recommendation': self._get_recommendation(failure['check_name']),
                'timestamp': failure['checked_at'].isoformat()
            })

        # Warnings
        for _, warning in warnings.iterrows():
            alerts.append({
                'level': 'warning',
                'check': warning['check_name'],
                'message': warning['message'],
                'recommendation': self._get_recommendation(warning['check_name']),
                'timestamp': warning['checked_at'].isoformat()
            })

        return alerts

    def _get_recommendation(self, check_name: str) -> str:
        """Get recommendation for a specific check failure"""
        recommendations = {
            'null_values': "Check data ingestion pipeline. Verify PAGASA API connection.",
            'value_ranges': "Investigate data source. Review data transformation logic.",
            'data_freshness': "Check Airflow DAG status. Verify weather data ingestion job.",
            'regional_coverage': "Review region-specific data sources. Check for API rate limits.",
            'anomaly_detection': "Investigate statistical anomalies. May indicate extreme weather events.",
            'data_consistency': "Review data validation rules. Check for data type issues."
        }

        return recommendations.get(check_name, "Review data pipeline and data sources.")


if __name__ == "__main__":
    # Generate report
    monitor = QualityMonitor()

    # Text report
    report = monitor.generate_report(days=7)
    print(report)

    # Statistics
    stats = monitor.get_check_statistics(days=30)
    print("\nCheck Statistics (30 days):")
    for check_name, check_stats in stats.items():
        print(f"\n{check_name}:")
        print(f"  Success Rate: {check_stats['success_rate']:.1f}%")
        print(f"  Total Runs: {check_stats['total_runs']}")
        print(f"  Critical Count: {check_stats['critical_count']}")

    # Alerts
    alerts = monitor.get_alert_recommendations()
    if alerts:
        print("\n\nActive Alerts:")
        for alert in alerts:
            print(f"\n[{alert['level'].upper()}] {alert['check']}")
            print(f"  {alert['message']}")
            print(f"  Recommendation: {alert['recommendation']}")
