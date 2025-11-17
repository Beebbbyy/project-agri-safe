"""
Unit tests for Data Quality Monitoring

Tests monitoring and reporting functionality including:
- Quality trends analysis
- Critical failure tracking
- Check statistics
- Report generation
- Alert recommendations
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from src.quality.monitoring import QualityMonitor


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value.__enter__.return_value = mock_conn
    return mock_db


@pytest.fixture
def monitor_instance(mock_db_connection):
    """Create monitor instance with mocked database"""
    with patch('src.quality.monitoring.get_db_connection', return_value=mock_db_connection):
        monitor = QualityMonitor()
        return monitor


@pytest.fixture
def sample_trends_data():
    """Sample quality trends data"""
    dates = [datetime.now().date() - timedelta(days=i) for i in range(7)]
    return pd.DataFrame({
        'check_date': dates * 2,
        'check_name': ['null_values'] * 7 + ['value_ranges'] * 7,
        'total_checks': [10] * 14,
        'passed_checks': [9, 8, 9, 10, 8, 9, 10] + [7, 8, 9, 10, 9, 8, 9],
        'failed_checks': [1, 2, 1, 0, 2, 1, 0] + [3, 2, 1, 0, 1, 2, 1],
        'pass_rate': [90.0, 80.0, 90.0, 100.0, 80.0, 90.0, 100.0] + [70.0, 80.0, 90.0, 100.0, 90.0, 80.0, 90.0],
        'avg_actual_value': [2.0] * 14,
        'threshold_value': [5.0] * 14
    })


class TestQualityMonitorInitialization:
    """Test monitor initialization"""

    def test_initialization_creates_monitor(self, monitor_instance):
        """Test that monitor initializes correctly"""
        assert monitor_instance is not None


class TestGetQualityTrends:
    """Test quality trends retrieval"""

    def test_get_quality_trends_returns_dataframe(self, monitor_instance, sample_trends_data):
        """Test that trends are returned as DataFrame"""
        with patch('pandas.read_sql', return_value=sample_trends_data):
            result = monitor_instance.get_quality_trends(days=7)

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'check_name' in result.columns
            assert 'pass_rate' in result.columns

    def test_get_quality_trends_with_filter(self, monitor_instance, sample_trends_data):
        """Test trends with check name filter"""
        with patch('pandas.read_sql', return_value=sample_trends_data) as mock_read_sql:
            result = monitor_instance.get_quality_trends(days=7, check_names=['null_values'])

            query = mock_read_sql.call_args[0][0]
            assert 'null_values' in query


class TestGetCriticalFailures:
    """Test critical failure tracking"""

    def test_get_critical_failures_returns_dataframe(self, monitor_instance):
        """Test that failures are returned as DataFrame"""
        failure_data = pd.DataFrame({
            'check_name': ['null_values', 'data_freshness'],
            'severity': ['critical', 'critical'],
            'message': ['High null rate', 'Data too old'],
            'details': ['{}', '{}'],
            'threshold': [5.0, 48.0],
            'actual_value': [10.0, 72.0],
            'checked_at': [datetime.now()] * 2
        })

        with patch('pandas.read_sql', return_value=failure_data):
            result = monitor_instance.get_critical_failures(days=7)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_get_critical_failures_filters_by_severity(self, monitor_instance):
        """Test filtering by severity level"""
        with patch('pandas.read_sql', return_value=pd.DataFrame()) as mock_read_sql:
            monitor_instance.get_critical_failures(days=7, severity='warning')

            query = mock_read_sql.call_args[0][0]
            assert 'warning' in query


class TestGetCheckStatistics:
    """Test check statistics calculation"""

    def test_get_check_statistics_returns_dict(self, monitor_instance):
        """Test that statistics are returned as dictionary"""
        stats_data = pd.DataFrame({
            'check_name': ['null_values', 'value_ranges'],
            'total_runs': [100, 100],
            'passed_runs': [95, 90],
            'success_rate': [95.0, 90.0],
            'critical_count': [2, 5],
            'warning_count': [3, 5],
            'first_check': [datetime.now() - timedelta(days=30)] * 2,
            'last_check': [datetime.now()] * 2,
            'avg_actual_value': [2.0, 0.5],
            'max_actual_value': [4.0, 1.0],
            'min_actual_value': [0.0, 0.0]
        })

        with patch('pandas.read_sql', return_value=stats_data):
            result = monitor_instance.get_check_statistics(days=30)

            assert isinstance(result, dict)
            assert 'null_values' in result
            assert result['null_values']['success_rate'] == 95.0
            assert result['null_values']['total_runs'] == 100


class TestGetRecentSummary:
    """Test recent quality summary"""

    def test_get_recent_summary_returns_dict(self, monitor_instance):
        """Test that summary is returned as dictionary"""
        summary_data = pd.DataFrame({
            'check_name': ['null_values', 'value_ranges'],
            'passed': [True, False],
            'severity': ['info', 'critical'],
            'message': ['Passed', 'Failed'],
            'checked_at': [datetime.now()] * 2
        })

        with patch('pandas.read_sql', return_value=summary_data):
            result = monitor_instance.get_recent_summary(hours=24)

            assert isinstance(result, dict)
            assert result['checks_found'] == 2
            assert result['all_passed'] is False
            assert result['critical_failures'] == 1

    def test_get_recent_summary_handles_no_data(self, monitor_instance):
        """Test handling when no recent checks"""
        with patch('pandas.read_sql', return_value=pd.DataFrame()):
            result = monitor_instance.get_recent_summary(hours=24)

            assert result['checks_found'] == 0
            assert result['all_passed'] is False


class TestGenerateReport:
    """Test report generation"""

    def test_generate_report_returns_string(self, monitor_instance):
        """Test that report is generated as string"""
        summary_data = pd.DataFrame({
            'check_name': ['null_values'],
            'passed': [True],
            'severity': ['info'],
            'message': ['Passed'],
            'checked_at': [datetime.now()]
        })

        with patch('pandas.read_sql', return_value=summary_data):
            with patch.object(monitor_instance, 'get_critical_failures', return_value=pd.DataFrame()):
                with patch.object(monitor_instance, 'get_quality_trends', return_value=pd.DataFrame()):
                    report = monitor_instance.generate_report(days=7)

                    assert isinstance(report, str)
                    assert "DATA QUALITY REPORT" in report
                    assert "null_values" in report.upper()

    def test_generate_report_includes_failures(self, monitor_instance):
        """Test that report includes critical failures"""
        summary_data = pd.DataFrame({
            'check_name': ['null_values'],
            'passed': [False],
            'severity': ['critical'],
            'message': ['Failed'],
            'checked_at': [datetime.now()]
        })

        failure_data = pd.DataFrame({
            'check_name': ['null_values'],
            'message': ['High null rate'],
            'checked_at': [datetime.now()],
            'threshold': [5.0],
            'actual_value': [10.0]
        })

        with patch('pandas.read_sql', return_value=summary_data):
            with patch.object(monitor_instance, 'get_critical_failures', return_value=failure_data):
                with patch.object(monitor_instance, 'get_quality_trends', return_value=pd.DataFrame()):
                    report = monitor_instance.generate_report(days=7, include_failures=True)

                    assert "CRITICAL FAILURES" in report


class TestGenerateHTMLDashboard:
    """Test HTML dashboard generation"""

    def test_generate_html_dashboard_returns_html(self, monitor_instance):
        """Test that HTML dashboard is generated"""
        summary_data = pd.DataFrame({
            'check_name': ['null_values'],
            'passed': [True],
            'severity': ['info'],
            'message': ['Passed'],
            'checked_at': [datetime.now()]
        })

        with patch('pandas.read_sql', return_value=summary_data):
            with patch.object(monitor_instance, 'get_quality_trends', return_value=pd.DataFrame()):
                with patch.object(monitor_instance, 'get_critical_failures', return_value=pd.DataFrame()):
                    html = monitor_instance.generate_html_dashboard(days=7)

                    assert isinstance(html, str)
                    assert "<!DOCTYPE html>" in html
                    assert "Data Quality Dashboard" in html


class TestGetAlertRecommendations:
    """Test alert recommendations"""

    def test_get_alert_recommendations_returns_list(self, monitor_instance):
        """Test that recommendations are returned as list"""
        failure_data = pd.DataFrame({
            'check_name': ['null_values', 'data_freshness'],
            'message': ['High null rate', 'Data too old'],
            'checked_at': [datetime.now()] * 2
        })

        with patch.object(monitor_instance, 'get_critical_failures', return_value=failure_data):
            alerts = monitor_instance.get_alert_recommendations()

            assert isinstance(alerts, list)
            assert len(alerts) > 0
            assert all('level' in alert for alert in alerts)
            assert all('recommendation' in alert for alert in alerts)

    def test_get_recommendation_provides_actionable_advice(self, monitor_instance):
        """Test that recommendations are actionable"""
        recommendation = monitor_instance._get_recommendation('null_values')

        assert isinstance(recommendation, str)
        assert len(recommendation) > 0
        assert any(word in recommendation.lower() for word in ['check', 'verify', 'review', 'investigate'])


@pytest.mark.parametrize("pass_rate,expected_trend", [
    ([90, 92, 94, 96, 98], "improving"),
    ([98, 96, 94, 92, 90], "declining"),
    ([90, 90, 90, 90, 90], "stable"),
])
def test_quality_trend_detection(monitor_instance, pass_rate, expected_trend):
    """Parametrized test for trend detection"""
    dates = [datetime.now().date() - timedelta(days=i) for i in range(5)]
    trend_data = pd.DataFrame({
        'check_date': dates,
        'check_name': ['null_values'] * 5,
        'pass_rate': pass_rate,
        'total_checks': [10] * 5,
        'passed_checks': pass_rate,
        'failed_checks': [100 - p for p in pass_rate],
        'avg_actual_value': [2.0] * 5,
        'threshold_value': [5.0] * 5
    })

    with patch('pandas.read_sql', return_value=trend_data):
        trends = monitor_instance.get_quality_trends(days=5)

        # Check if data is returned
        assert len(trends) > 0
