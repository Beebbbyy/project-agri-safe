"""
Unit tests for Model Training Pipeline

Tests the end-to-end training pipeline including:
- Data fetching from database
- Feature engineering
- Model training
- Model evaluation
- Model persistence
- Metadata tracking
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime, date
import tempfile
import os

from src.models.training_pipeline import FloodModelTrainingPipeline


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value.__enter__.return_value = mock_conn
    mock_db.get_cursor.return_value.__enter__.return_value = MagicMock()
    return mock_db


@pytest.fixture
def sample_db_data():
    """Sample training data from database"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'region_id': [f'REG{i:03d}' for i in range(n_samples)],
        'assessment_date': [date(2025, 1, 1) for _ in range(n_samples)],
        'region_name': [f'Region {i}' for i in range(n_samples)],
        'latitude': np.random.uniform(5, 20, n_samples),
        'longitude': np.random.uniform(120, 125, n_samples),
        'elevation': np.random.uniform(0, 500, n_samples),
        'rainfall_1d': np.random.exponential(30, n_samples),
        'rainfall_3d': np.random.exponential(80, n_samples),
        'rainfall_7d': np.random.exponential(150, n_samples),
        'rainfall_14d': np.random.exponential(250, n_samples),
        'rainfall_30d': np.random.exponential(400, n_samples),
        'temp_avg': np.random.normal(28, 3, n_samples),
        'temp_range': np.random.uniform(5, 15, n_samples),
        'temp_avg_7d': np.random.normal(28, 2, n_samples),
        'temp_avg_14d': np.random.normal(28, 2, n_samples),
        'temp_avg_30d': np.random.normal(28, 2, n_samples),
        'temp_variance_7d': np.random.uniform(0, 5, n_samples),
        'temp_variance_30d': np.random.uniform(0, 5, n_samples),
        'wind_speed_max_7d': np.random.exponential(40, n_samples),
        'wind_speed_avg_7d': np.random.exponential(25, n_samples),
        'rainy_days_7d': np.random.poisson(3, n_samples),
        'heavy_rain_days_7d': np.random.poisson(0.5, n_samples),
        'high_wind_days_7d': np.random.poisson(1, n_samples),
        'max_daily_rainfall_7d': np.random.exponential(50, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day_of_week': np.random.randint(1, 8, n_samples),
        'is_typhoon_season': np.random.binomial(1, 0.5, n_samples),
        'is_wet_season': np.random.binomial(1, 0.5, n_samples),
        'rainfall_intensity_ratio': np.random.uniform(0, 1, n_samples),
        'soil_moisture_proxy': np.random.uniform(0, 20, n_samples),
        'evapotranspiration_estimate': np.random.normal(10, 3, n_samples),
        'flood_risk_indicator': np.random.uniform(0, 1, n_samples),
        'historical_high_risk_count': np.random.poisson(2, n_samples),
        'region_vulnerability_score': np.random.uniform(0, 100, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def pipeline_instance(mock_db_connection):
    """Create pipeline instance with mocked database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.models.training_pipeline.get_db_connection', return_value=mock_db_connection):
            pipeline = FloodModelTrainingPipeline(model_dir=tmpdir)
            yield pipeline


class TestTrainingPipelineInitialization:
    """Test pipeline initialization"""

    def test_initialization_creates_model_directory(self):
        """Test that model directory is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, 'models')

            with patch('src.models.training_pipeline.get_db_connection'):
                pipeline = FloodModelTrainingPipeline(model_dir=model_dir)

                assert os.path.exists(model_dir)

    def test_initialization_sets_min_training_days(self):
        """Test min_training_days configuration"""
        with patch('src.models.training_pipeline.get_db_connection'):
            pipeline = FloodModelTrainingPipeline(min_training_days=100)

            assert pipeline.min_training_days == 100


class TestFetchTrainingData:
    """Test fetching training data from database"""

    def test_fetch_training_data_builds_query(self, pipeline_instance, sample_db_data, mock_db_connection):
        """Test that fetch builds correct SQL query"""
        with patch('pandas.read_sql', return_value=sample_db_data) as mock_read_sql:
            result = pipeline_instance.fetch_training_data(days_back=180)

            mock_read_sql.assert_called_once()
            query = mock_read_sql.call_args[0][0]
            assert '180 days' in query

    def test_fetch_with_region_filter(self, pipeline_instance, sample_db_data):
        """Test fetching with region filter"""
        with patch('pandas.read_sql', return_value=sample_db_data) as mock_read_sql:
            result = pipeline_instance.fetch_training_data(days_back=180, region_ids=['REG001'])

            query = mock_read_sql.call_args[0][0]
            assert 'REG001' in query

    def test_fetch_adds_derived_features(self, pipeline_instance, sample_db_data):
        """Test that derived features are added"""
        with patch('pandas.read_sql', return_value=sample_db_data):
            result = pipeline_instance.fetch_training_data(days_back=180)

            assert 'rainfall_intensity_ratio' in result.columns
            assert 'soil_moisture_proxy' in result.columns
            assert 'flood_risk_indicator' in result.columns

    def test_fetch_fills_nan_values(self, pipeline_instance, sample_db_data):
        """Test that NaN values are filled"""
        data_with_nans = sample_db_data.copy()
        data_with_nans.loc[0, 'rainfall_1d'] = np.nan

        with patch('pandas.read_sql', return_value=data_with_nans):
            result = pipeline_instance.fetch_training_data(days_back=180)

            assert result['rainfall_1d'].isna().sum() == 0


class TestAddHistoricalRiskData:
    """Test adding historical flood risk data"""

    def test_add_historical_risk_data_success(self, pipeline_instance, sample_db_data):
        """Test successfully adding historical risk data"""
        historical_data = pd.DataFrame({
            'region_id': ['REG001', 'REG002'],
            'historical_high_risk_count': [5, 2],
            'region_vulnerability_score': [75.0, 30.0]
        })

        with patch('pandas.read_sql', return_value=historical_data):
            result = pipeline_instance._add_historical_risk_data(sample_db_data)

            assert 'historical_high_risk_count' in result.columns
            assert 'region_vulnerability_score' in result.columns

    def test_add_historical_risk_data_handles_missing_table(self, pipeline_instance, sample_db_data):
        """Test graceful handling when historical table doesn't exist"""
        with patch('pandas.read_sql', side_effect=Exception("Table not found")):
            result = pipeline_instance._add_historical_risk_data(sample_db_data)

            # Should add default columns
            assert 'historical_high_risk_count' in result.columns
            assert result['historical_high_risk_count'].iloc[0] == 0


class TestTrainMLModel:
    """Test ML model training"""

    def test_train_ml_model_success(self, pipeline_instance, sample_db_data):
        """Test successful model training"""
        model, metrics = pipeline_instance.train_ml_model(sample_db_data, test_size=0.2)

        assert model is not None
        assert model.model is not None
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics

    def test_train_with_custom_xgb_params(self, pipeline_instance, sample_db_data):
        """Test training with custom XGBoost parameters"""
        model, metrics = pipeline_instance.train_ml_model(
            sample_db_data,
            test_size=0.2,
            n_estimators=50,
            max_depth=3
        )

        assert model.model.n_estimators == 50
        assert model.model.max_depth == 3


class TestSaveModelArtifacts:
    """Test saving model artifacts"""

    def test_save_model_artifacts(self, pipeline_instance, sample_db_data):
        """Test saving model and metrics"""
        model, metrics = pipeline_instance.train_ml_model(sample_db_data, test_size=0.2)

        model_path = pipeline_instance.save_model_artifacts(model, metrics, version_tag='test')

        assert os.path.exists(model_path)
        assert 'test' in model_path

    def test_save_creates_metrics_file(self, pipeline_instance, sample_db_data):
        """Test that metrics JSON file is created"""
        model, metrics = pipeline_instance.train_ml_model(sample_db_data, test_size=0.2)

        pipeline_instance.save_model_artifacts(model, metrics, version_tag='test')

        metrics_path = os.path.join(pipeline_instance.model_dir, 'metrics_v2_test.json')
        assert os.path.exists(metrics_path)


class TestMakeJSONSerializable:
    """Test JSON serialization helper"""

    def test_numpy_array_serialization(self, pipeline_instance):
        """Test converting numpy arrays to lists"""
        obj = {'array': np.array([1, 2, 3])}

        result = pipeline_instance._make_json_serializable(obj)

        assert isinstance(result['array'], list)
        assert result['array'] == [1, 2, 3]

    def test_numpy_scalar_serialization(self, pipeline_instance):
        """Test converting numpy scalars"""
        obj = {
            'int_val': np.int64(42),
            'float_val': np.float64(3.14)
        }

        result = pipeline_instance._make_json_serializable(obj)

        assert isinstance(result['int_val'], int)
        assert isinstance(result['float_val'], float)

    def test_nested_structure_serialization(self, pipeline_instance):
        """Test serializing nested structures"""
        obj = {
            'nested': {
                'array': np.array([1, 2]),
                'scalar': np.int32(10)
            }
        }

        result = pipeline_instance._make_json_serializable(obj)

        assert isinstance(result['nested']['array'], list)
        assert isinstance(result['nested']['scalar'], int)


class TestSaveTrainingRunMetadata:
    """Test saving training metadata to database"""

    def test_save_training_run_metadata(self, pipeline_instance):
        """Test saving metadata to database"""
        metrics = {'test_accuracy': 0.85}

        # Mock cursor execute
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {'id': 1}

        with patch.object(pipeline_instance.db, 'get_cursor') as mock_get_cursor:
            mock_get_cursor.return_value.__enter__.return_value = mock_cursor

            pipeline_instance.save_training_run_metadata(
                model_name='test_model',
                model_version='v1.0',
                model_path='/path/to/model.pkl',
                metrics=metrics
            )

            mock_cursor.execute.assert_called_once()


class TestRunPipeline:
    """Test complete pipeline execution"""

    @patch.object(FloodModelTrainingPipeline, 'save_training_run_metadata')
    @patch.object(FloodModelTrainingPipeline, 'save_model_artifacts')
    @patch.object(FloodModelTrainingPipeline, 'fetch_training_data')
    def test_run_pipeline_executes_all_steps(self, mock_fetch, mock_save_artifacts, mock_save_metadata,
                                            pipeline_instance, sample_db_data):
        """Test that run() executes all pipeline steps"""
        mock_fetch.return_value = sample_db_data
        mock_save_artifacts.return_value = '/path/to/model.pkl'

        result = pipeline_instance.run(days_back=180, save_to_db=True)

        mock_fetch.assert_called_once()
        mock_save_artifacts.assert_called_once()
        mock_save_metadata.assert_called_once()

        assert result['status'] == 'success'
        assert 'duration_seconds' in result
        assert 'training_samples' in result

    @patch.object(FloodModelTrainingPipeline, 'fetch_training_data')
    def test_run_raises_error_for_insufficient_data(self, mock_fetch, pipeline_instance):
        """Test error handling for insufficient training data"""
        # Return less data than minimum required
        small_data = pd.DataFrame({'region_id': ['REG001']})
        mock_fetch.return_value = small_data

        with pytest.raises(ValueError, match="Insufficient training data"):
            pipeline_instance.run(days_back=30)

    @patch.object(FloodModelTrainingPipeline, 'save_training_run_metadata')
    @patch.object(FloodModelTrainingPipeline, 'save_model_artifacts')
    @patch.object(FloodModelTrainingPipeline, 'fetch_training_data')
    def test_run_skips_db_save_when_disabled(self, mock_fetch, mock_save_artifacts, mock_save_metadata,
                                             pipeline_instance, sample_db_data):
        """Test that database save can be disabled"""
        mock_fetch.return_value = sample_db_data
        mock_save_artifacts.return_value = '/path/to/model.pkl'

        result = pipeline_instance.run(days_back=180, save_to_db=False)

        mock_save_metadata.assert_not_called()


@pytest.mark.parametrize("days_back,expected_valid", [
    (90, True),
    (180, True),
    (365, True),
])
def test_different_training_periods(pipeline_instance, sample_db_data, days_back, expected_valid):
    """Parametrized test for different training periods"""
    with patch.object(pipeline_instance, 'fetch_training_data', return_value=sample_db_data):
        with patch.object(pipeline_instance, 'save_model_artifacts', return_value='/tmp/model.pkl'):
            with patch.object(pipeline_instance, 'save_training_run_metadata'):
                result = pipeline_instance.run(days_back=days_back, save_to_db=False)

                if expected_valid:
                    assert result['status'] == 'success'
