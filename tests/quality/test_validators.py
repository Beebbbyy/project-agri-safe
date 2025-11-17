"""
Unit tests for Data Quality Validators

Tests comprehensive data quality checks including:
- Null value validation
- Range validation
- Data freshness checks
- Regional coverage validation
- Anomaly detection
- Data consistency checks
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from src.quality.validators import (
    WeatherDataValidator,
    ValidationResult,
    ValidationSeverity
)


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value.__enter__.return_value = mock_conn
    mock_db.get_cursor.return_value.__enter__.return_value = MagicMock()
    return mock_db


@pytest.fixture
def validator_instance(mock_db_connection):
    """Create validator instance with mocked database"""
    with patch('src.quality.validators.get_db_connection', return_value=mock_db_connection):
        validator = WeatherDataValidator()
        return validator


class TestWeatherDataValidatorInitialization:
    """Test validator initialization"""

    def test_initialization_sets_thresholds(self, validator_instance):
        """Test that thresholds are set correctly"""
        assert validator_instance.NULL_PERCENTAGE_THRESHOLD == 5.0
        assert validator_instance.INVALID_PERCENTAGE_THRESHOLD == 1.0
        assert validator_instance.DATA_FRESHNESS_HOURS == 48
        assert validator_instance.REGIONAL_COVERAGE_THRESHOLD == 90.0

    def test_initialization_sets_valid_ranges(self, validator_instance):
        """Test that valid ranges are defined"""
        assert 'temperature_high' in validator_instance.VALID_RANGES
        assert 'rainfall_mm' in validator_instance.VALID_RANGES
        assert 'wind_speed' in validator_instance.VALID_RANGES


class TestCheckNullValues:
    """Test null value checking"""

    def test_check_null_values_passes_for_clean_data(self, validator_instance):
        """Test that check passes when no nulls"""
        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'null_temp_high': 0,
            'null_temp_low': 0,
            'null_rainfall': 0,
            'null_wind': 0,
            'null_condition': 0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_null_values()

            assert result.passed is True
            assert result.check_name == 'null_values'
            assert result.actual_value < validator_instance.NULL_PERCENTAGE_THRESHOLD

    def test_check_null_values_fails_for_high_nulls(self, validator_instance):
        """Test that check fails when too many nulls"""
        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'null_temp_high': 100,
            'null_temp_low': 100,
            'null_rainfall': 100,
            'null_wind': 100,
            'null_condition': 100
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_null_values()

            assert result.passed is False
            assert result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.WARNING]

    def test_check_null_values_fails_for_no_records(self, validator_instance):
        """Test handling when no records found"""
        mock_result = pd.DataFrame([{
            'total_records': 0,
            'null_temp_high': 0,
            'null_temp_low': 0,
            'null_rainfall': 0,
            'null_wind': 0,
            'null_condition': 0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_null_values()

            assert result.passed is False
            assert result.severity == ValidationSeverity.CRITICAL


class TestCheckValueRanges:
    """Test value range validation"""

    def test_check_value_ranges_passes_for_valid_data(self, validator_instance):
        """Test that check passes when values in range"""
        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'invalid_temp_high': 0,
            'invalid_temp_low': 0,
            'invalid_rainfall': 0,
            'invalid_wind': 0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_value_ranges()

            assert result.passed is True

    def test_check_value_ranges_fails_for_invalid_data(self, validator_instance):
        """Test that check fails when values out of range"""
        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'invalid_temp_high': 50,
            'invalid_temp_low': 40,
            'invalid_rainfall': 30,
            'invalid_wind': 20
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_value_ranges()

            assert result.passed is False
            assert result.actual_value > validator_instance.INVALID_PERCENTAGE_THRESHOLD


class TestCheckDataFreshness:
    """Test data freshness validation"""

    def test_check_data_freshness_passes_for_recent_data(self, validator_instance):
        """Test that check passes when data is recent"""
        recent_time = datetime.now() - timedelta(hours=12)
        mock_result = pd.DataFrame([{
            'last_update': recent_time,
            'hours_since_update': 12.0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_data_freshness()

            assert result.passed is True
            assert result.actual_value < validator_instance.DATA_FRESHNESS_HOURS

    def test_check_data_freshness_fails_for_old_data(self, validator_instance):
        """Test that check fails when data is too old"""
        old_time = datetime.now() - timedelta(hours=72)
        mock_result = pd.DataFrame([{
            'last_update': old_time,
            'hours_since_update': 72.0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_data_freshness()

            assert result.passed is False
            assert result.severity == ValidationSeverity.CRITICAL

    def test_check_data_freshness_handles_no_data(self, validator_instance):
        """Test handling when no data exists"""
        mock_result = pd.DataFrame([{
            'last_update': None,
            'hours_since_update': None
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_data_freshness()

            assert result.passed is False
            assert result.severity == ValidationSeverity.CRITICAL


class TestCheckRegionalCoverage:
    """Test regional coverage validation"""

    def test_check_regional_coverage_passes_for_full_coverage(self, validator_instance):
        """Test that check passes when all regions have data"""
        mock_result = pd.DataFrame([{
            'regions_with_data': 45,
            'total_regions': 50
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_regional_coverage()

            assert result.passed is True
            assert result.actual_value >= validator_instance.REGIONAL_COVERAGE_THRESHOLD

    def test_check_regional_coverage_fails_for_low_coverage(self, validator_instance):
        """Test that check fails when coverage is low"""
        mock_result = pd.DataFrame([{
            'regions_with_data': 30,
            'total_regions': 50
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_regional_coverage()

            assert result.passed is False


class TestCheckAnomalies:
    """Test statistical anomaly detection"""

    def test_check_anomalies_passes_for_few_anomalies(self, validator_instance):
        """Test that check passes when few anomalies"""
        mock_result = pd.DataFrame([{
            'anomaly_count': 5,
            'rainfall_anomalies': 3,
            'temperature_anomalies': 2
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_anomalies()

            assert result.passed is True
            assert result.actual_value < validator_instance.ANOMALY_COUNT_THRESHOLD

    def test_check_anomalies_fails_for_many_anomalies(self, validator_instance):
        """Test that check fails when too many anomalies"""
        mock_result = pd.DataFrame([{
            'anomaly_count': 25,
            'rainfall_anomalies': 15,
            'temperature_anomalies': 10
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_anomalies()

            assert result.passed is False


class TestCheckDataConsistency:
    """Test data consistency validation"""

    def test_check_data_consistency_passes_for_consistent_data(self, validator_instance):
        """Test that check passes when data is consistent"""
        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'inverted_temps': 0,
            'negative_wind': 0,
            'negative_rainfall': 0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_data_consistency()

            assert result.passed is True

    def test_check_data_consistency_fails_for_inconsistent_data(self, validator_instance):
        """Test that check fails for inconsistent data"""
        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'inverted_temps': 50,
            'negative_wind': 30,
            'negative_rainfall': 20
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            result = validator_instance.check_data_consistency()

            assert result.passed is False
            assert result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.WARNING]


class TestRunAllChecks:
    """Test running all validation checks"""

    @patch.object(WeatherDataValidator, 'check_null_values')
    @patch.object(WeatherDataValidator, 'check_value_ranges')
    @patch.object(WeatherDataValidator, 'check_data_freshness')
    @patch.object(WeatherDataValidator, 'check_regional_coverage')
    @patch.object(WeatherDataValidator, 'check_anomalies')
    @patch.object(WeatherDataValidator, 'check_data_consistency')
    def test_run_all_checks_executes_all(self, mock_consistency, mock_anomalies, mock_coverage,
                                        mock_freshness, mock_ranges, mock_nulls, validator_instance):
        """Test that all checks are executed"""
        # Mock all checks to pass
        for mock_check in [mock_nulls, mock_ranges, mock_freshness, mock_coverage, mock_anomalies, mock_consistency]:
            mock_check.return_value = ValidationResult(
                check_name='test',
                passed=True,
                severity=ValidationSeverity.INFO,
                message='Test passed'
            )

        summary = validator_instance.run_all_checks(save_results=False)

        assert summary['total_checks'] == 6
        assert summary['passed'] == 6
        assert summary['success_rate'] == 100.0

    def test_run_all_checks_calculates_summary(self, validator_instance):
        """Test that summary is calculated correctly"""
        with patch('pandas.read_sql') as mock_read_sql:
            # Mock successful checks
            mock_read_sql.return_value = pd.DataFrame([{
                'total_records': 1000,
                'null_temp_high': 0, 'null_temp_low': 0,
                'null_rainfall': 0, 'null_wind': 0, 'null_condition': 0,
                'invalid_temp_high': 0, 'invalid_temp_low': 0,
                'invalid_rainfall': 0, 'invalid_wind': 0,
                'last_update': datetime.now(), 'hours_since_update': 12.0,
                'regions_with_data': 45, 'total_regions': 50,
                'anomaly_count': 5, 'rainfall_anomalies': 3, 'temperature_anomalies': 2,
                'inverted_temps': 0, 'negative_wind': 0, 'negative_rainfall': 0
            }])

            summary = validator_instance.run_all_checks(save_results=False)

            assert 'timestamp' in summary
            assert 'total_checks' in summary
            assert 'passed' in summary
            assert 'failed' in summary
            assert 'success_rate' in summary


class TestSaveResults:
    """Test saving validation results to database"""

    def test_save_results_inserts_to_database(self, validator_instance):
        """Test that results are saved to database"""
        results = [
            ValidationResult(
                check_name='test_check',
                passed=True,
                severity=ValidationSeverity.INFO,
                message='Test passed',
                threshold=5.0,
                actual_value=2.0
            )
        ]

        mock_cursor = MagicMock()
        with patch.object(validator_instance.db, 'get_cursor') as mock_get_cursor:
            mock_get_cursor.return_value.__enter__.return_value = mock_cursor

            validator_instance.save_results(results)

            assert mock_cursor.execute.call_count >= 1


class TestGetSummaryReport:
    """Test generating summary report"""

    def test_get_summary_report_without_results(self, validator_instance):
        """Test report generation with no results"""
        report = validator_instance.get_summary_report()

        assert "No validation results available" in report

    def test_get_summary_report_with_results(self, validator_instance):
        """Test report generation with results"""
        validator_instance.results = [
            ValidationResult(
                check_name='test_check',
                passed=True,
                severity=ValidationSeverity.INFO,
                message='Test passed'
            )
        ]

        report = validator_instance.get_summary_report()

        assert "Data Quality Validation Report" in report
        assert "test_check" in report.upper()


@pytest.mark.parametrize("null_percentage,expected_pass", [
    (0.0, True),
    (3.0, True),
    (5.0, True),
    (8.0, False),
])
def test_null_threshold_validation(validator_instance, null_percentage, expected_pass):
    """Parametrized test for null value thresholds"""
    total_fields = 1000
    null_count = int(total_fields * null_percentage / 100)

    mock_result = pd.DataFrame([{
        'total_records': 1000,
        'null_temp_high': null_count // 5,
        'null_temp_low': null_count // 5,
        'null_rainfall': null_count // 5,
        'null_wind': null_count // 5,
        'null_condition': null_count // 5
    }])

    with patch('pandas.read_sql', return_value=mock_result):
        result = validator_instance.check_null_values()
        assert result.passed == expected_pass
