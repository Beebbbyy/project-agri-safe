"""
Data Quality Validation Framework

This module provides comprehensive data quality checks for weather data:
- Null value validation
- Range validation
- Data freshness checks
- Regional coverage validation
- Statistical anomaly detection
- Consistency checks

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
from enum import Enum

import pandas as pd
import numpy as np
from psycopg2.extras import Json

from src.utils.database import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation results"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """
    Result of a single validation check

    Attributes:
        check_name: Name of the validation check
        passed: Whether the check passed
        severity: Severity level
        message: Human-readable message
        details: Additional details about the check
        timestamp: When the check was performed
        threshold: The threshold value used (if applicable)
        actual_value: The actual value measured
    """
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    threshold: Optional[float] = None
    actual_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'check_name': self.check_name,
            'passed': self.passed,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'threshold': self.threshold,
            'actual_value': self.actual_value
        }


class WeatherDataValidator:
    """
    Comprehensive data quality validation for weather data

    Performs various checks to ensure data quality and reliability
    """

    # Thresholds for validation checks
    NULL_PERCENTAGE_THRESHOLD = 5.0  # Max % of null values
    INVALID_PERCENTAGE_THRESHOLD = 1.0  # Max % of out-of-range values
    DATA_FRESHNESS_HOURS = 48  # Max hours since last update
    REGIONAL_COVERAGE_THRESHOLD = 90.0  # Min % of regions with data
    ANOMALY_COUNT_THRESHOLD = 10  # Max number of statistical anomalies

    # Valid ranges for weather data
    VALID_RANGES = {
        'temperature_high': (15, 45),
        'temperature_low': (10, 40),
        'rainfall_mm': (0, 500),
        'wind_speed': (0, 250)
    }

    def __init__(self):
        """Initialize validator"""
        self.db = get_db_connection()
        self.results: List[ValidationResult] = []
        logger.info("Initialized WeatherDataValidator")

    def check_null_values(self, days_back: int = 7) -> ValidationResult:
        """
        Check for unexpected NULL values in weather data

        Args:
            days_back: Number of days to check

        Returns:
            ValidationResult with null value statistics
        """
        logger.info("Running null value check")

        query = f"""
            SELECT
                COUNT(*) as total_records,
                COUNT(*) FILTER (WHERE temperature_high IS NULL) as null_temp_high,
                COUNT(*) FILTER (WHERE temperature_low IS NULL) as null_temp_low,
                COUNT(*) FILTER (WHERE rainfall_mm IS NULL) as null_rainfall,
                COUNT(*) FILTER (WHERE wind_speed IS NULL) as null_wind,
                COUNT(*) FILTER (WHERE weather_condition IS NULL) as null_condition
            FROM weather_forecasts
            WHERE created_at >= CURRENT_DATE - INTERVAL '{days_back} days'
        """

        with self.db.get_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        total_records = result['total_records']
        total_fields = total_records * 5  # 5 key fields checked

        if total_records == 0:
            return ValidationResult(
                check_name='null_values',
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="No records found in the specified time period",
                details={'days_checked': days_back}
            )

        null_count = (
            result['null_temp_high'] +
            result['null_temp_low'] +
            result['null_rainfall'] +
            result['null_wind'] +
            result['null_condition']
        )

        null_percentage = (null_count / total_fields) * 100

        passed = null_percentage < self.NULL_PERCENTAGE_THRESHOLD
        severity = ValidationSeverity.CRITICAL if null_percentage > 10 else ValidationSeverity.WARNING

        return ValidationResult(
            check_name='null_values',
            passed=passed,
            severity=severity,
            message=f"Found {null_percentage:.2f}% NULL values in weather data",
            details={
                'null_temp_high': int(result['null_temp_high']),
                'null_temp_low': int(result['null_temp_low']),
                'null_rainfall': int(result['null_rainfall']),
                'null_wind': int(result['null_wind']),
                'null_condition': int(result['null_condition']),
                'total_records': int(total_records),
                'days_checked': days_back
            },
            threshold=self.NULL_PERCENTAGE_THRESHOLD,
            actual_value=null_percentage
        )

    def check_value_ranges(self, days_back: int = 7) -> ValidationResult:
        """
        Check if values are within expected ranges

        Args:
            days_back: Number of days to check

        Returns:
            ValidationResult with range violation statistics
        """
        logger.info("Running value range check")

        temp_high_min, temp_high_max = self.VALID_RANGES['temperature_high']
        temp_low_min, temp_low_max = self.VALID_RANGES['temperature_low']
        rain_min, rain_max = self.VALID_RANGES['rainfall_mm']
        wind_min, wind_max = self.VALID_RANGES['wind_speed']

        query = f"""
            SELECT
                COUNT(*) as total_records,
                COUNT(*) FILTER (
                    WHERE temperature_high < {temp_high_min} OR temperature_high > {temp_high_max}
                ) as invalid_temp_high,
                COUNT(*) FILTER (
                    WHERE temperature_low < {temp_low_min} OR temperature_low > {temp_low_max}
                ) as invalid_temp_low,
                COUNT(*) FILTER (
                    WHERE rainfall_mm < {rain_min} OR rainfall_mm > {rain_max}
                ) as invalid_rainfall,
                COUNT(*) FILTER (
                    WHERE wind_speed < {wind_min} OR wind_speed > {wind_max}
                ) as invalid_wind
            FROM weather_forecasts
            WHERE created_at >= CURRENT_DATE - INTERVAL '{days_back} days'
        """

        with self.db.get_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        total_records = result['total_records']
        total_fields = total_records * 4

        if total_records == 0:
            return ValidationResult(
                check_name='value_ranges',
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="No records found for range validation",
                details={'days_checked': days_back}
            )

        invalid_count = (
            result['invalid_temp_high'] +
            result['invalid_temp_low'] +
            result['invalid_rainfall'] +
            result['invalid_wind']
        )

        invalid_percentage = (invalid_count / total_fields) * 100

        passed = invalid_percentage < self.INVALID_PERCENTAGE_THRESHOLD
        severity = ValidationSeverity.CRITICAL if invalid_percentage > 5 else ValidationSeverity.WARNING

        return ValidationResult(
            check_name='value_ranges',
            passed=passed,
            severity=severity,
            message=f"Found {invalid_percentage:.2f}% out-of-range values",
            details={
                'invalid_temp_high': int(result['invalid_temp_high']),
                'invalid_temp_low': int(result['invalid_temp_low']),
                'invalid_rainfall': int(result['invalid_rainfall']),
                'invalid_wind': int(result['invalid_wind']),
                'total_records': int(total_records),
                'valid_ranges': self.VALID_RANGES
            },
            threshold=self.INVALID_PERCENTAGE_THRESHOLD,
            actual_value=invalid_percentage
        )

    def check_data_freshness(self) -> ValidationResult:
        """
        Check if data is being updated regularly

        Returns:
            ValidationResult with data freshness information
        """
        logger.info("Running data freshness check")

        query = """
            SELECT
                MAX(created_at) as last_update,
                EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(created_at))) / 3600 as hours_since_update
            FROM weather_forecasts
        """

        with self.db.get_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        hours_old = result['hours_since_update']

        if pd.isna(hours_old):
            return ValidationResult(
                check_name='data_freshness',
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="No weather data found in database",
                details={}
            )

        passed = hours_old < self.DATA_FRESHNESS_HOURS
        severity = ValidationSeverity.CRITICAL if hours_old > 72 else ValidationSeverity.WARNING

        return ValidationResult(
            check_name='data_freshness',
            passed=passed,
            severity=severity,
            message=f"Last update: {hours_old:.1f} hours ago",
            details={
                'last_update': result['last_update'].isoformat(),
                'hours_since_update': float(hours_old)
            },
            threshold=self.DATA_FRESHNESS_HOURS,
            actual_value=hours_old
        )

    def check_regional_coverage(self, hours_back: int = 24) -> ValidationResult:
        """
        Ensure all regions have recent data

        Args:
            hours_back: Number of hours to check coverage

        Returns:
            ValidationResult with coverage statistics
        """
        logger.info("Running regional coverage check")

        query = f"""
            SELECT
                COUNT(DISTINCT wf.region_id) as regions_with_data,
                (SELECT COUNT(*) FROM regions) as total_regions
            FROM weather_forecasts wf
            WHERE wf.created_at >= CURRENT_TIMESTAMP - INTERVAL '{hours_back} hours'
        """

        with self.db.get_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        regions_with_data = result['regions_with_data']
        total_regions = result['total_regions']

        if total_regions == 0:
            return ValidationResult(
                check_name='regional_coverage',
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="No regions found in database",
                details={}
            )

        coverage = (regions_with_data / total_regions) * 100

        passed = coverage >= self.REGIONAL_COVERAGE_THRESHOLD
        severity = ValidationSeverity.WARNING if coverage < 90 else ValidationSeverity.INFO

        return ValidationResult(
            check_name='regional_coverage',
            passed=passed,
            severity=severity,
            message=f"{coverage:.0f}% regional coverage ({regions_with_data}/{total_regions} regions)",
            details={
                'regions_with_data': int(regions_with_data),
                'total_regions': int(total_regions),
                'hours_checked': hours_back
            },
            threshold=self.REGIONAL_COVERAGE_THRESHOLD,
            actual_value=coverage
        )

    def check_anomalies(self, days_back: int = 7, std_threshold: float = 3.0) -> ValidationResult:
        """
        Detect statistical anomalies in weather data

        Args:
            days_back: Days to check for context
            std_threshold: Standard deviations for anomaly detection

        Returns:
            ValidationResult with anomaly statistics
        """
        logger.info("Running statistical anomaly detection")

        query = f"""
            WITH stats AS (
                SELECT
                    region_id,
                    AVG(rainfall_mm) as avg_rainfall,
                    STDDEV(rainfall_mm) as stddev_rainfall,
                    AVG(temperature_high) as avg_temp_high,
                    STDDEV(temperature_high) as stddev_temp_high
                FROM weather_forecasts
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY region_id
            ),
            recent_data AS (
                SELECT
                    wf.region_id,
                    wf.rainfall_mm,
                    wf.temperature_high,
                    wf.created_at
                FROM weather_forecasts wf
                WHERE wf.created_at >= CURRENT_DATE - INTERVAL '{days_back} days'
            )
            SELECT
                COUNT(*) as anomaly_count,
                COUNT(*) FILTER (
                    WHERE ABS(rd.rainfall_mm - s.avg_rainfall) > {std_threshold} * s.stddev_rainfall
                ) as rainfall_anomalies,
                COUNT(*) FILTER (
                    WHERE ABS(rd.temperature_high - s.avg_temp_high) > {std_threshold} * s.stddev_temp_high
                ) as temperature_anomalies
            FROM recent_data rd
            JOIN stats s ON rd.region_id = s.region_id
            WHERE ABS(rd.rainfall_mm - s.avg_rainfall) > {std_threshold} * s.stddev_rainfall
               OR ABS(rd.temperature_high - s.avg_temp_high) > {std_threshold} * s.stddev_temp_high
        """

        with self.db.get_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        anomaly_count = result['anomaly_count']
        rainfall_anomalies = result['rainfall_anomalies']
        temperature_anomalies = result['temperature_anomalies']

        passed = anomaly_count < self.ANOMALY_COUNT_THRESHOLD
        severity = ValidationSeverity.WARNING if anomaly_count < 20 else ValidationSeverity.INFO

        return ValidationResult(
            check_name='anomaly_detection',
            passed=passed,
            severity=severity,
            message=f"Found {anomaly_count} statistical anomalies ({std_threshold}σ threshold)",
            details={
                'anomaly_count': int(anomaly_count),
                'rainfall_anomalies': int(rainfall_anomalies),
                'temperature_anomalies': int(temperature_anomalies),
                'std_threshold': std_threshold,
                'days_checked': days_back
            },
            threshold=self.ANOMALY_COUNT_THRESHOLD,
            actual_value=anomaly_count
        )

    def check_data_consistency(self, days_back: int = 7) -> ValidationResult:
        """
        Check for data consistency issues

        Args:
            days_back: Days to check

        Returns:
            ValidationResult with consistency check results
        """
        logger.info("Running data consistency check")

        query = f"""
            SELECT
                COUNT(*) as total_records,
                COUNT(*) FILTER (WHERE temperature_low > temperature_high) as inverted_temps,
                COUNT(*) FILTER (WHERE wind_speed < 0) as negative_wind,
                COUNT(*) FILTER (WHERE rainfall_mm < 0) as negative_rainfall
            FROM weather_forecasts
            WHERE created_at >= CURRENT_DATE - INTERVAL '{days_back} days'
        """

        with self.db.get_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        total_records = result['total_records']
        inconsistencies = (
            result['inverted_temps'] +
            result['negative_wind'] +
            result['negative_rainfall']
        )

        if total_records == 0:
            return ValidationResult(
                check_name='data_consistency',
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="No records for consistency check",
                details={}
            )

        inconsistency_rate = (inconsistencies / total_records) * 100

        passed = inconsistency_rate < 0.5  # Less than 0.5%
        severity = ValidationSeverity.CRITICAL if inconsistency_rate > 1 else ValidationSeverity.WARNING

        return ValidationResult(
            check_name='data_consistency',
            passed=passed,
            severity=severity,
            message=f"Found {inconsistency_rate:.2f}% inconsistent records",
            details={
                'inverted_temps': int(result['inverted_temps']),
                'negative_wind': int(result['negative_wind']),
                'negative_rainfall': int(result['negative_rainfall']),
                'total_records': int(total_records)
            },
            threshold=0.5,
            actual_value=inconsistency_rate
        )

    def run_all_checks(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Execute all validation checks

        Args:
            save_results: Whether to save results to database

        Returns:
            Summary of all validation results
        """
        logger.info("Running all validation checks")

        checks = [
            self.check_null_values(),
            self.check_value_ranges(),
            self.check_data_freshness(),
            self.check_regional_coverage(),
            self.check_anomalies(),
            self.check_data_consistency()
        ]

        self.results = checks

        # Save to database
        if save_results:
            self.save_results(checks)

        # Generate summary
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.passed)
        critical_failures = [c for c in checks if not c.passed and c.severity == ValidationSeverity.CRITICAL]
        warnings = [c for c in checks if not c.passed and c.severity == ValidationSeverity.WARNING]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': total_checks - passed_checks,
            'success_rate': (passed_checks / total_checks) * 100,
            'critical_failures': len(critical_failures),
            'warnings': len(warnings),
            'all_passed': len(critical_failures) == 0,
            'checks': [
                {
                    'check': c.check_name,
                    'passed': c.passed,
                    'severity': c.severity.value,
                    'message': c.message,
                    'threshold': c.threshold,
                    'actual_value': c.actual_value
                }
                for c in checks
            ]
        }

        logger.info(f"Validation complete: {passed_checks}/{total_checks} passed")
        if critical_failures:
            logger.error(f"Critical failures: {[c.check_name for c in critical_failures]}")

        return summary

    def save_results(self, results: List[ValidationResult]):
        """
        Save validation results to database

        Args:
            results: List of ValidationResult objects
        """
        logger.info(f"Saving {len(results)} validation results to database")

        # Create table if not exists
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS data_quality_checks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                check_name VARCHAR(100) NOT NULL,
                passed BOOLEAN NOT NULL,
                severity VARCHAR(20) NOT NULL,
                message TEXT,
                details JSONB,
                threshold NUMERIC,
                actual_value NUMERIC,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_quality_checks_time (checked_at DESC),
                INDEX idx_quality_checks_name (check_name)
            );
        """

        with self.db.get_cursor() as cursor:
            # Create table
            cursor.execute(create_table_sql.replace('INDEX', ''))

            # Insert results
            for result in results:
                cursor.execute("""
                    INSERT INTO data_quality_checks
                    (check_name, passed, severity, message, details, threshold, actual_value, checked_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                """, (
                    result.check_name,
                    result.passed,
                    result.severity.value,
                    result.message,
                    Json(result.details),
                    result.threshold,
                    result.actual_value,
                    result.timestamp
                ))

        logger.info("Validation results saved successfully")

    def get_summary_report(self) -> str:
        """
        Generate a text summary report of validation results

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available. Run checks first."

        report = [
            "\n" + "="*60,
            "Data Quality Validation Report",
            "="*60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Checks: {len(self.results)}",
            ""
        ]

        for result in self.results:
            status_icon = "✓" if result.passed else "✗"
            severity_icon = {
                ValidationSeverity.CRITICAL: "🔴",
                ValidationSeverity.WARNING: "🟡",
                ValidationSeverity.INFO: "🔵"
            }.get(result.severity, "")

            report.append(f"{status_icon} {result.check_name.upper()} {severity_icon}")
            report.append(f"   {result.message}")

            if result.threshold and result.actual_value:
                report.append(f"   Threshold: {result.threshold} | Actual: {result.actual_value:.2f}")

            report.append("")

        report.append("="*60 + "\n")

        return "\n".join(report)


if __name__ == "__main__":
    # Run validation checks
    validator = WeatherDataValidator()
    summary = validator.run_all_checks()

    # Print report
    print(validator.get_summary_report())

    # Print summary
    import json
    print("\nSummary:")
    print(json.dumps(summary, indent=2, default=str))
