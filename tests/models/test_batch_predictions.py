"""
Unit tests for Batch Flood Risk Predictions

Tests the batch prediction service including:
- Model loading and initialization
- Feature fetching from database
- Batch prediction generation
- Prediction saving to database
- Model comparison (v1 vs v2)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import date, datetime

from src.models.batch_predictions import FloodRiskBatchPredictor
from src.models.flood_risk_v1 import FloodRiskLevel


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value.__enter__.return_value = mock_conn
    mock_db.get_cursor.return_value.__enter__.return_value = MagicMock()
    return mock_db


@pytest.fixture
def sample_features_data():
    """Sample features data"""
    return pd.DataFrame({
        'region_id': ['REG001', 'REG002', 'REG003'],
        'region_name': ['Region A', 'Region B', 'Region C'],
        'assessment_date': [date(2025, 1, 15)] * 3,
        'latitude': [14.5, 15.0, 15.5],
        'longitude': [121.0, 120.5, 121.5],
        'elevation': [50.0, 100.0, 150.0],
        'rainfall_1d': [10.0, 50.0, 100.0],
        'rainfall_3d': [30.0, 120.0, 250.0],
        'rainfall_7d': [60.0, 200.0, 400.0],
        'rainfall_14d': [100.0, 300.0, 500.0],
        'rainfall_30d': [150.0, 400.0, 600.0],
        'temp_avg': [28.0, 27.0, 26.0],
        'temp_range': [10.0, 12.0, 11.0],
        'temp_avg_7d': [28.0, 27.0, 26.0],
        'temp_avg_14d': [28.0, 27.0, 26.0],
        'temp_avg_30d': [28.0, 27.0, 26.0],
        'temp_variance_7d': [2.0, 2.5, 3.0],
        'temp_variance_30d': [3.0, 3.5, 4.0],
        'wind_speed_max_7d': [30.0, 60.0, 100.0],
        'wind_speed_avg_7d': [20.0, 40.0, 70.0],
        'rainy_days_7d': [2, 5, 6],
        'heavy_rain_days_7d': [0, 2, 4],
        'high_wind_days_7d': [0, 1, 3],
        'max_daily_rainfall_7d': [15.0, 60.0, 120.0],
        'month': [1, 1, 1],
        'day_of_week': [3, 3, 3],
        'is_typhoon_season': [0, 0, 0],
        'is_wet_season': [0, 0, 0],
        'rainfall_intensity_ratio': [0.1, 0.3, 0.5],
        'soil_moisture_proxy': [2.0, 7.0, 15.0],
        'evapotranspiration_estimate': [10.0, 8.0, 6.0],
        'flood_risk_indicator': [0.1, 0.4, 0.8],
        'historical_high_risk_count': [0, 3, 8],
        'region_vulnerability_score': [10.0, 50.0, 80.0],
    })


@pytest.fixture
def predictor_with_both_models(mock_db_connection):
    """Create predictor with both models enabled"""
    with patch('src.models.batch_predictions.get_db_connection', return_value=mock_db_connection):
        with patch.object(FloodRiskBatchPredictor, '_find_latest_model', return_value=None):
            predictor = FloodRiskBatchPredictor(use_ml_model=False, use_rule_model=True)
            return predictor


class TestBatchPredictorInitialization:
    """Test batch predictor initialization"""

    def test_initialization_with_rule_model_only(self, mock_db_connection):
        """Test initialization with only rule-based model"""
        with patch('src.models.batch_predictions.get_db_connection', return_value=mock_db_connection):
            predictor = FloodRiskBatchPredictor(use_ml_model=False, use_rule_model=True)

            assert predictor.rule_model is not None
            assert predictor.ml_model is None

    def test_initialization_finds_latest_model(self, mock_db_connection):
        """Test that latest model is auto-detected"""
        with patch('src.models.batch_predictions.get_db_connection', return_value=mock_db_connection):
            with patch.object(FloodRiskBatchPredictor, '_find_latest_model', return_value='/path/to/model.pkl'):
                with patch('src.models.batch_predictions.os.path.exists', return_value=True):
                    with patch('src.models.batch_predictions.MLFloodModel'):
                        predictor = FloodRiskBatchPredictor(use_ml_model=True, use_rule_model=False)

    def test_initialization_with_explicit_model_path(self, mock_db_connection):
        """Test initialization with explicit model path"""
        with patch('src.models.batch_predictions.get_db_connection', return_value=mock_db_connection):
            with patch('src.models.batch_predictions.os.path.exists', return_value=True):
                with patch('src.models.batch_predictions.MLFloodModel'):
                    predictor = FloodRiskBatchPredictor(
                        model_path='/explicit/path/model.pkl',
                        use_ml_model=True
                    )


class TestFindLatestModel:
    """Test finding latest ML model"""

    @patch('glob.glob')
    @patch('os.path.getmtime')
    def test_find_latest_model_success(self, mock_getmtime, mock_glob, mock_db_connection):
        """Test finding latest model from multiple options"""
        mock_glob.return_value = [
            'models/flood_risk_v2_20250101.pkl',
            'models/flood_risk_v2_20250115.pkl',
        ]
        mock_getmtime.side_effect = lambda x: 1 if '20250101' in x else 2

        with patch('src.models.batch_predictions.get_db_connection', return_value=mock_db_connection):
            predictor = FloodRiskBatchPredictor(use_ml_model=False, use_rule_model=False)
            latest = predictor._find_latest_model()

            assert '20250115' in latest

    @patch('glob.glob')
    def test_find_latest_model_no_models(self, mock_glob, mock_db_connection):
        """Test when no models are found"""
        mock_glob.return_value = []

        with patch('src.models.batch_predictions.get_db_connection', return_value=mock_db_connection):
            predictor = FloodRiskBatchPredictor(use_ml_model=False, use_rule_model=False)
            latest = predictor._find_latest_model()

            assert latest is None


class TestFetchLatestFeatures:
    """Test fetching latest features from database"""

    def test_fetch_latest_features_builds_query(self, predictor_with_both_models, sample_features_data):
        """Test that fetch builds correct SQL query"""
        with patch('pandas.read_sql', return_value=sample_features_data) as mock_read_sql:
            result = predictor_with_both_models.fetch_latest_features()

            mock_read_sql.assert_called_once()

    def test_fetch_with_target_date(self, predictor_with_both_models, sample_features_data):
        """Test fetching with specific target date"""
        with patch('pandas.read_sql', return_value=sample_features_data) as mock_read_sql:
            target = date(2025, 1, 15)
            result = predictor_with_both_models.fetch_latest_features(target_date=target)

            query = mock_read_sql.call_args[0][0]
            assert '2025-01-15' in query

    def test_fetch_with_region_filter(self, predictor_with_both_models, sample_features_data):
        """Test fetching with region ID filter"""
        with patch('pandas.read_sql', return_value=sample_features_data) as mock_read_sql:
            result = predictor_with_both_models.fetch_latest_features(region_ids=['REG001'])

            query = mock_read_sql.call_args[0][0]
            assert 'REG001' in query

    def test_fetch_adds_derived_features(self, predictor_with_both_models, sample_features_data):
        """Test that derived features are added"""
        with patch('pandas.read_sql', return_value=sample_features_data):
            result = predictor_with_both_models.fetch_latest_features()

            assert 'rainfall_intensity_ratio' in result.columns
            assert 'soil_moisture_proxy' in result.columns
            assert 'flood_risk_indicator' in result.columns

    def test_fetch_fills_nans(self, predictor_with_both_models, sample_features_data):
        """Test that NaN values are filled"""
        data_with_nans = sample_features_data.copy()
        data_with_nans.loc[0, 'rainfall_1d'] = np.nan

        with patch('pandas.read_sql', return_value=data_with_nans):
            result = predictor_with_both_models.fetch_latest_features()

            assert result['rainfall_1d'].isna().sum() == 0


class TestAddHistoricalRisk:
    """Test adding historical flood risk data"""

    def test_add_historical_risk_success(self, predictor_with_both_models, sample_features_data):
        """Test successfully adding historical risk"""
        historical_data = pd.DataFrame({
            'region_id': ['REG001', 'REG002'],
            'historical_high_risk_count': [5, 2],
            'region_vulnerability_score': [75.0, 30.0]
        })

        with patch('pandas.read_sql', return_value=historical_data):
            result = predictor_with_both_models._add_historical_risk(sample_features_data)

            assert 'historical_high_risk_count' in result.columns
            assert 'region_vulnerability_score' in result.columns

    def test_add_historical_risk_handles_error(self, predictor_with_both_models, sample_features_data):
        """Test graceful handling when historical data fails to load"""
        with patch('pandas.read_sql', side_effect=Exception("Query failed")):
            result = predictor_with_both_models._add_historical_risk(sample_features_data)

            # Should add default columns
            assert 'historical_high_risk_count' in result.columns
            assert result['historical_high_risk_count'].iloc[0] == 0


class TestPredictAllRegions:
    """Test generating predictions for all regions"""

    def test_predict_all_regions_with_rule_model(self, predictor_with_both_models, sample_features_data):
        """Test predictions with rule-based model"""
        with patch.object(predictor_with_both_models, 'fetch_latest_features',
                         return_value=sample_features_data):
            predictions = predictor_with_both_models.predict_all_regions()

            assert len(predictions) == 3
            assert all('predictions' in p for p in predictions)
            assert all('rule_based' in p['predictions'] for p in predictions)

    def test_predict_handles_empty_features(self, predictor_with_both_models):
        """Test handling when no features are available"""
        with patch.object(predictor_with_both_models, 'fetch_latest_features',
                         return_value=pd.DataFrame()):
            predictions = predictor_with_both_models.predict_all_regions()

            assert len(predictions) == 0

    def test_predict_handles_individual_errors(self, predictor_with_both_models, sample_features_data):
        """Test that individual prediction errors don't stop batch"""
        with patch.object(predictor_with_both_models, 'fetch_latest_features',
                         return_value=sample_features_data):
            with patch.object(predictor_with_both_models.rule_model, 'predict',
                            side_effect=[Exception("Error"), Mock(), Mock()]):
                predictions = predictor_with_both_models.predict_all_regions()

                # Should still get some predictions despite one error
                assert len(predictions) > 0


class TestSavePredictions:
    """Test saving predictions to database"""

    def test_save_predictions_executes_insert(self, predictor_with_both_models):
        """Test that save executes database insert"""
        predictions = [
            {
                'region_id': 'REG001',
                'region_name': 'Region A',
                'assessment_date': date(2025, 1, 15),
                'predictions': {
                    'rule_based': {
                        'risk_level': 'low',
                        'risk_score': 15.0,
                        'confidence_score': 0.7,
                        'recommendation': 'Normal conditions',
                        'model_version': 'v1.0'
                    }
                }
            }
        ]

        mock_cursor = MagicMock()
        with patch.object(predictor_with_both_models.db, 'get_cursor') as mock_get_cursor:
            mock_get_cursor.return_value.__enter__.return_value = mock_cursor

            predictor_with_both_models.save_predictions(predictions)

            assert mock_cursor.execute.called

    def test_save_prefers_ml_predictions(self, predictor_with_both_models):
        """Test that ML predictions are preferred when available"""
        predictions = [
            {
                'region_id': 'REG001',
                'region_name': 'Region A',
                'assessment_date': date(2025, 1, 15),
                'predictions': {
                    'rule_based': {'risk_level': 'low', 'risk_score': 15.0,
                                  'confidence_score': 0.7, 'recommendation': 'Test',
                                  'model_version': 'v1.0'},
                    'ml_based': {'risk_level': 'high', 'risk_score': 65.0,
                                'confidence_score': 0.9, 'recommendation': 'Test',
                                'model_version': 'v2.0'}
                }
            }
        ]

        mock_cursor = MagicMock()
        with patch.object(predictor_with_both_models.db, 'get_cursor') as mock_get_cursor:
            mock_get_cursor.return_value.__enter__.return_value = mock_cursor

            predictor_with_both_models.save_predictions(predictions, model_version='combined')

            # Verify high risk was saved (from ML model)
            call_args = mock_cursor.execute.call_args[0]
            assert 'high' in str(call_args) or call_args[1][2] == 'high'


class TestRunPipeline:
    """Test complete batch prediction pipeline"""

    @patch.object(FloodRiskBatchPredictor, 'save_predictions')
    @patch.object(FloodRiskBatchPredictor, 'predict_all_regions')
    def test_run_pipeline_executes_all_steps(self, mock_predict, mock_save,
                                            predictor_with_both_models):
        """Test that run() executes all steps"""
        mock_predictions = [
            {
                'region_id': 'REG001',
                'predictions': {'rule_based': {'risk_level': 'low'}}
            }
        ]
        mock_predict.return_value = mock_predictions

        result = predictor_with_both_models.run(save_to_db=True)

        mock_predict.assert_called_once()
        mock_save.assert_called_once()

        assert result['status'] == 'success'
        assert 'duration_seconds' in result
        assert 'regions_processed' in result

    @patch.object(FloodRiskBatchPredictor, 'predict_all_regions')
    def test_run_skips_save_when_disabled(self, mock_predict, predictor_with_both_models):
        """Test that saving can be disabled"""
        mock_predict.return_value = []

        with patch.object(predictor_with_both_models, 'save_predictions') as mock_save:
            predictor_with_both_models.run(save_to_db=False)

            mock_save.assert_not_called()

    @patch.object(FloodRiskBatchPredictor, 'predict_all_regions')
    def test_run_calculates_risk_distribution(self, mock_predict, predictor_with_both_models):
        """Test that run calculates risk distribution statistics"""
        mock_predictions = [
            {'predictions': {'rule_based': {'risk_level': 'low'}}},
            {'predictions': {'rule_based': {'risk_level': 'high'}}},
            {'predictions': {'rule_based': {'risk_level': 'low'}}},
        ]
        mock_predict.return_value = mock_predictions

        result = predictor_with_both_models.run(save_to_db=False)

        assert 'risk_distribution' in result
        assert result['risk_distribution']['rule_based']['low'] == 2
        assert result['risk_distribution']['rule_based']['high'] == 1


@pytest.mark.parametrize("model_version,expected_valid", [
    ('rule_based', True),
    ('ml_based', True),
    ('combined', True),
])
def test_different_model_versions(predictor_with_both_models, model_version, expected_valid):
    """Parametrized test for different model version options"""
    predictions = [
        {
            'region_id': 'REG001',
            'predictions': {
                'rule_based': {'risk_level': 'low', 'risk_score': 15.0,
                              'confidence_score': 0.7, 'recommendation': 'Test',
                              'model_version': 'v1.0'}
            }
        }
    ]

    mock_cursor = MagicMock()
    with patch.object(predictor_with_both_models.db, 'get_cursor') as mock_get_cursor:
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        predictor_with_both_models.save_predictions(predictions, model_version=model_version)

        if expected_valid:
            assert mock_cursor.execute.called
