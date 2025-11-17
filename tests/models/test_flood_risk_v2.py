"""
Unit tests for ML-Based Flood Risk Model (v2)

Tests the XGBoost-based prediction model including:
- Model initialization and loading
- Feature preparation
- Label creation
- Training process
- Prediction (single and batch)
- Model persistence
- Feature importance
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import date
import tempfile
import os

from src.models.flood_risk_v2 import MLFloodModel, FloodRiskLevel


@pytest.fixture
def sample_training_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame({
        'region_id': [f'REG{i:03d}' for i in range(n_samples)],
        'rainfall_1d': np.random.exponential(30, n_samples),
        'rainfall_3d': np.random.exponential(80, n_samples),
        'rainfall_7d': np.random.exponential(150, n_samples),
        'rainfall_14d': np.random.exponential(250, n_samples),
        'rainfall_30d': np.random.exponential(400, n_samples),
        'elevation': np.random.uniform(0, 500, n_samples),
        'temp_avg': np.random.normal(28, 3, n_samples),
        'temp_avg_7d': np.random.normal(28, 2, n_samples),
        'temp_avg_14d': np.random.normal(28, 2, n_samples),
        'temp_avg_30d': np.random.normal(28, 2, n_samples),
        'temp_variance_7d': np.random.uniform(0, 5, n_samples),
        'temp_variance_30d': np.random.uniform(0, 5, n_samples),
        'temp_range': np.random.uniform(5, 15, n_samples),
        'wind_speed_max_7d': np.random.exponential(40, n_samples),
        'wind_speed_avg_7d': np.random.exponential(25, n_samples),
        'high_wind_days_7d': np.random.poisson(1, n_samples),
        'rainy_days_7d': np.random.poisson(3, n_samples),
        'heavy_rain_days_7d': np.random.poisson(0.5, n_samples),
        'max_daily_rainfall_7d': np.random.exponential(50, n_samples),
        'rainfall_intensity_ratio': np.random.uniform(0, 1, n_samples),
        'soil_moisture_proxy': np.random.uniform(0, 20, n_samples),
        'evapotranspiration_estimate': np.random.normal(10, 3, n_samples),
        'flood_risk_indicator': np.random.uniform(0, 1, n_samples),
        'historical_high_risk_count': np.random.poisson(2, n_samples),
        'region_vulnerability_score': np.random.uniform(0, 100, n_samples),
        'latitude': np.random.uniform(5, 20, n_samples),
        'longitude': np.random.uniform(120, 125, n_samples),
        'is_typhoon_season': np.random.binomial(1, 0.5, n_samples),
        'is_wet_season': np.random.binomial(1, 0.5, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day_of_week': np.random.randint(1, 8, n_samples)
    })

    return data


@pytest.fixture
def model_instance():
    """Create a fresh ML model instance"""
    return MLFloodModel()


@pytest.fixture
def trained_model(model_instance, sample_training_data):
    """Create a trained model for testing"""
    y = model_instance.create_labels_from_rules(sample_training_data)
    model_instance.train(sample_training_data, y, test_size=0.2, random_state=42)
    return model_instance


class TestMLFloodModelInitialization:
    """Test ML model initialization"""

    def test_initialization_without_model_path(self):
        """Test initialization creates new model"""
        model = MLFloodModel()
        assert model.model is None
        assert model.version == "v2.0"
        assert model.feature_names == model.FEATURE_NAMES

    def test_initialization_with_invalid_path(self):
        """Test initialization with non-existent path"""
        model = MLFloodModel(model_path="/nonexistent/path/model.pkl")
        assert model.model is None

    def test_feature_names_defined(self, model_instance):
        """Test that feature names are properly defined"""
        assert len(model_instance.FEATURE_NAMES) > 0
        assert 'rainfall_1d' in model_instance.FEATURE_NAMES
        assert 'elevation' in model_instance.FEATURE_NAMES
        assert 'month' in model_instance.FEATURE_NAMES

    def test_label_mapping_correct(self, model_instance):
        """Test label mapping is configured correctly"""
        assert model_instance.LABEL_MAPPING[0] == FloodRiskLevel.LOW
        assert model_instance.LABEL_MAPPING[1] == FloodRiskLevel.MEDIUM
        assert model_instance.LABEL_MAPPING[2] == FloodRiskLevel.HIGH
        assert model_instance.LABEL_MAPPING[3] == FloodRiskLevel.CRITICAL


class TestFeaturePreparation:
    """Test feature preparation"""

    def test_prepare_features_selects_correct_columns(self, model_instance, sample_training_data):
        """Test feature preparation selects and orders columns"""
        prepared = model_instance.prepare_features(sample_training_data)

        assert list(prepared.columns) == model_instance.FEATURE_NAMES
        assert len(prepared) == len(sample_training_data)

    def test_prepare_features_handles_missing_features(self, model_instance):
        """Test preparation handles missing features"""
        incomplete_data = pd.DataFrame({
            'rainfall_1d': [10.0, 20.0],
            'elevation': [100.0, 200.0]
        })

        prepared = model_instance.prepare_features(incomplete_data)

        # Should fill missing features with defaults
        assert len(prepared.columns) == len(model_instance.FEATURE_NAMES)
        assert prepared['rainfall_1d'].tolist() == [10.0, 20.0]

    def test_prepare_features_handles_nulls(self, model_instance, sample_training_data):
        """Test preparation handles null values"""
        data_with_nulls = sample_training_data.copy()
        data_with_nulls.loc[0, 'rainfall_1d'] = np.nan

        prepared = model_instance.prepare_features(data_with_nulls)

        assert prepared['rainfall_1d'].isna().sum() == 0

    def test_prepare_features_handles_infinite_values(self, model_instance, sample_training_data):
        """Test preparation handles infinite values"""
        data_with_inf = sample_training_data.copy()
        data_with_inf.loc[0, 'rainfall_1d'] = np.inf

        prepared = model_instance.prepare_features(data_with_inf)

        assert not np.isinf(prepared['rainfall_1d']).any()

    def test_prepare_features_correct_dtypes(self, model_instance, sample_training_data):
        """Test that dtypes are correct after preparation"""
        prepared = model_instance.prepare_features(sample_training_data)

        # Check boolean/int columns
        assert prepared['is_typhoon_season'].dtype == int
        assert prepared['month'].dtype == int

        # Check float columns
        assert prepared['rainfall_1d'].dtype == float


class TestLabelCreation:
    """Test label creation methods"""

    def test_create_labels_from_rules(self, model_instance, sample_training_data):
        """Test rule-based label creation"""
        labels = model_instance.create_labels_from_rules(sample_training_data)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_training_data)
        assert labels.min() >= 0
        assert labels.max() <= 3

    def test_create_labels_from_rules_distribution(self, model_instance, sample_training_data):
        """Test that labels have reasonable distribution"""
        labels = model_instance.create_labels_from_rules(sample_training_data)

        # Should have multiple classes
        unique_labels = np.unique(labels)
        assert len(unique_labels) >= 2

    def test_create_labels_from_data_with_risk_level(self, model_instance):
        """Test creating labels from existing risk_level column"""
        data = pd.DataFrame({
            'risk_level': [FloodRiskLevel.LOW, FloodRiskLevel.HIGH, FloodRiskLevel.CRITICAL]
        })

        labels = model_instance.create_labels_from_data(data)

        assert labels is not None
        assert list(labels) == [0, 2, 3]

    def test_create_labels_from_data_with_flood_occurred(self, model_instance):
        """Test creating labels from flood_occurred column"""
        data = pd.DataFrame({
            'flood_occurred': [False, True, True],
            'flood_severity': [0, 2, 3]
        })

        labels = model_instance.create_labels_from_data(data)

        assert labels is not None
        assert labels[0] == 0  # No flood

    def test_create_labels_from_data_returns_none_when_no_labels(self, model_instance):
        """Test that None is returned when no label columns exist"""
        data = pd.DataFrame({'rainfall_1d': [10.0, 20.0]})

        labels = model_instance.create_labels_from_data(data)

        assert labels is None


class TestModelTraining:
    """Test model training"""

    def test_train_model_successfully(self, model_instance, sample_training_data):
        """Test successful model training"""
        y = model_instance.create_labels_from_rules(sample_training_data)

        metrics = model_instance.train(sample_training_data, y, test_size=0.2, random_state=42)

        assert model_instance.model is not None
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert 'cv_mean_accuracy' in metrics
        assert metrics['train_accuracy'] > 0

    def test_train_sets_feature_importance(self, trained_model):
        """Test that training sets feature importance"""
        assert trained_model.feature_importance_ is not None
        assert len(trained_model.feature_importance_) > 0

    def test_train_sets_training_metrics(self, trained_model):
        """Test that training metrics are stored"""
        assert trained_model.training_metrics_ is not None
        assert 'confusion_matrix' in trained_model.training_metrics_
        assert 'classification_report' in trained_model.training_metrics_

    def test_train_sets_trained_at_timestamp(self, trained_model):
        """Test that trained_at timestamp is set"""
        assert trained_model.trained_at is not None

    def test_train_with_custom_xgb_params(self, model_instance, sample_training_data):
        """Test training with custom XGBoost parameters"""
        y = model_instance.create_labels_from_rules(sample_training_data)

        metrics = model_instance.train(
            sample_training_data, y,
            test_size=0.2,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05
        )

        assert model_instance.model is not None
        assert model_instance.model.n_estimators == 50


class TestPrediction:
    """Test model prediction"""

    def test_predict_single_sample(self, trained_model):
        """Test prediction for a single sample"""
        features = {
            'region_id': 'TEST001',
            'assessment_date': date.today(),
            'rainfall_1d': 50.0,
            'rainfall_7d': 150.0,
            'elevation': 100.0
        }

        assessment = trained_model.predict(features)

        assert assessment.region_id == 'TEST001'
        assert assessment.risk_level in FloodRiskLevel
        assert 0 <= assessment.confidence_score <= 1
        assert 0 <= assessment.risk_score <= 100

    def test_predict_without_trained_model_raises_error(self, model_instance):
        """Test that prediction fails without trained model"""
        features = {'region_id': 'TEST001', 'rainfall_1d': 10.0}

        with pytest.raises(ValueError, match="Model not trained"):
            model_instance.predict(features)

    def test_predict_returns_correct_risk_levels(self, trained_model):
        """Test that predictions return valid risk levels"""
        high_risk_features = {
            'region_id': 'TEST001',
            'rainfall_1d': 150.0,
            'rainfall_7d': 400.0,
            'elevation': 30.0,
            'region_vulnerability_score': 80.0
        }

        assessment = trained_model.predict(high_risk_features)

        assert assessment.risk_level in [FloodRiskLevel.MEDIUM, FloodRiskLevel.HIGH, FloodRiskLevel.CRITICAL]

    def test_predict_with_probabilities(self, trained_model):
        """Test prediction includes probabilities"""
        features = {'region_id': 'TEST001', 'rainfall_1d': 10.0}

        assessment = trained_model.predict(features, return_probabilities=True)

        assert 'probabilities' in assessment.metadata
        probs = assessment.metadata['probabilities']
        assert len(probs) == 4  # 4 risk levels
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Should sum to ~1

    def test_predict_includes_model_metadata(self, trained_model):
        """Test prediction includes model metadata"""
        features = {'region_id': 'TEST001', 'rainfall_1d': 10.0}

        assessment = trained_model.predict(features)

        assert 'model_trained_at' in assessment.metadata
        assert 'predicted_class' in assessment.metadata


class TestBatchPrediction:
    """Test batch prediction functionality"""

    def test_batch_predict_multiple_samples(self, trained_model, sample_training_data):
        """Test batch prediction for multiple samples"""
        features_df = sample_training_data.head(10)

        assessments = trained_model.batch_predict(features_df)

        assert len(assessments) > 0
        assert all(a.risk_level in FloodRiskLevel for a in assessments)

    def test_batch_predict_without_trained_model_raises_error(self, model_instance, sample_training_data):
        """Test batch prediction fails without trained model"""
        with pytest.raises(ValueError, match="Model not trained"):
            model_instance.batch_predict(sample_training_data.head(5))


class TestModelPersistence:
    """Test model saving and loading"""

    def test_save_model(self, trained_model):
        """Test model saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')

            trained_model.save_model(model_path)

            assert os.path.exists(model_path)

    def test_load_model(self, trained_model):
        """Test model loading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')

            # Save model
            trained_model.save_model(model_path)

            # Load model
            new_model = MLFloodModel(model_path=model_path)

            assert new_model.model is not None
            assert new_model.feature_importance_ is not None

    def test_load_nonexistent_model_raises_error(self):
        """Test loading non-existent model raises error"""
        model = MLFloodModel()

        with pytest.raises(FileNotFoundError):
            model.load_model('/nonexistent/model.pkl')

    def test_save_model_without_training_raises_error(self, model_instance):
        """Test saving untrained model raises error"""
        with pytest.raises(ValueError, match="No model to save"):
            model_instance.save_model('/tmp/model.pkl')


class TestFeatureImportance:
    """Test feature importance reporting"""

    def test_get_feature_importance_report(self, trained_model):
        """Test feature importance report generation"""
        report = trained_model.get_feature_importance_report(top_n=10)

        assert isinstance(report, str)
        assert "Feature Importance Report" in report
        assert "Top 10 Features" in report

    def test_feature_importance_report_without_training(self, model_instance):
        """Test feature importance report without training"""
        report = model_instance.get_feature_importance_report()

        assert "No feature importance data available" in report

    def test_feature_importance_shows_top_features(self, trained_model):
        """Test that top features are shown in importance report"""
        report = trained_model.get_feature_importance_report(top_n=5)

        # Should show at least some feature names
        assert any(fname in report for fname in trained_model.FEATURE_NAMES[:10])


@pytest.mark.parametrize("test_size,expected_valid", [
    (0.1, True),
    (0.2, True),
    (0.3, True),
    (0.5, True),
])
def test_different_test_sizes(model_instance, sample_training_data, test_size, expected_valid):
    """Parametrized test for different test set sizes"""
    y = model_instance.create_labels_from_rules(sample_training_data)

    metrics = model_instance.train(sample_training_data, y, test_size=test_size, random_state=42)

    if expected_valid:
        assert metrics['test_accuracy'] > 0
        assert 'cv_mean_accuracy' in metrics
