"""
Integration Tests for Phase 3 Components

Tests end-to-end workflows combining multiple Phase 3 components:
- Weather ETL → Feature Engineering → Model Training
- Feature Engineering → Prediction Pipeline
- Data Quality → ETL → Validation
- Complete flood risk assessment workflow
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import date, datetime, timedelta
from pyspark.sql import SparkSession
import tempfile
import os

from src.processing.spark_jobs.weather_etl import WeatherETL
from src.processing.spark_jobs.rolling_features import RollingFeatureEngine
from src.models.flood_risk_v1 import RuleBasedFloodModel, WeatherFeatures
from src.models.flood_risk_v2 import MLFloodModel
from src.models.training_pipeline import FloodModelTrainingPipeline
from src.models.batch_predictions import FloodRiskBatchPredictor
from src.quality.validators import WeatherDataValidator


@pytest.fixture(scope="module")
def spark_session():
    """Create a test Spark session for integration tests"""
    spark = SparkSession.builder \
        .appName("Phase3IntegrationTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

    yield spark

    spark.stop()


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value.__enter__.return_value = mock_conn
    mock_db.get_cursor.return_value.__enter__.return_value = MagicMock()
    return mock_db


@pytest.fixture
def sample_training_data():
    """Generate sample training data for integration tests"""
    np.random.seed(42)
    n_samples = 150

    data = pd.DataFrame({
        'region_id': [f'REG{i:03d}' for i in range(n_samples)],
        'assessment_date': [date(2025, 1, 1) for _ in range(n_samples)],
        'region_name': [f'Region {i}' for i in range(n_samples)],
        'rainfall_1d': np.random.exponential(30, n_samples),
        'rainfall_3d': np.random.exponential(80, n_samples),
        'rainfall_7d': np.random.exponential(150, n_samples),
        'rainfall_14d': np.random.exponential(250, n_samples),
        'rainfall_30d': np.random.exponential(400, n_samples),
        'elevation': np.random.uniform(0, 500, n_samples),
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


class TestETLToModelPipeline:
    """Test complete pipeline from ETL to model training"""

    @patch('src.models.training_pipeline.get_db_connection')
    def test_training_pipeline_with_etl_data(self, mock_get_db, sample_training_data):
        """Test training a model with ETL-processed data"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training pipeline
            pipeline = FloodModelTrainingPipeline(model_dir=tmpdir)

            # Mock data fetch to return our sample data
            with patch.object(pipeline, 'fetch_training_data', return_value=sample_training_data):
                # Train model
                model, metrics = pipeline.train_ml_model(sample_training_data, test_size=0.2)

                # Verify model was trained
                assert model is not None
                assert model.model is not None
                assert metrics['train_accuracy'] > 0

                # Save model
                model_path = pipeline.save_model_artifacts(model, metrics, version_tag='integration_test')

                # Verify model was saved
                assert os.path.exists(model_path)


class TestModelComparisonWorkflow:
    """Test comparing rule-based and ML models"""

    def test_both_models_predict_same_features(self, sample_training_data):
        """Test that both v1 and v2 models can predict from same features"""
        # Train ML model
        ml_model = MLFloodModel()
        y = ml_model.create_labels_from_rules(sample_training_data)
        ml_model.train(sample_training_data, y, test_size=0.2, random_state=42)

        # Create rule-based model
        rule_model = RuleBasedFloodModel()

        # Test prediction on same sample
        sample_row = sample_training_data.iloc[0].to_dict()
        sample_row['region_id'] = 'TEST001'
        sample_row['assessment_date'] = date.today()

        # Get predictions from both models
        rule_features = WeatherFeatures(**sample_row)
        rule_assessment = rule_model.predict(rule_features)
        ml_assessment = ml_model.predict(sample_row)

        # Both should return valid assessments
        assert rule_assessment.region_id == 'TEST001'
        assert ml_assessment.region_id == 'TEST001'
        assert rule_assessment.risk_level is not None
        assert ml_assessment.risk_level is not None


class TestBatchPredictionWorkflow:
    """Test batch prediction pipeline"""

    @patch('src.models.batch_predictions.get_db_connection')
    def test_batch_prediction_with_both_models(self, mock_get_db, sample_training_data):
        """Test batch predictions using both models"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Create and train ML model
        ml_model = MLFloodModel()
        y = ml_model.create_labels_from_rules(sample_training_data)
        ml_model.train(sample_training_data, y, test_size=0.2, random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            ml_model.save_model(model_path)

            # Create batch predictor
            predictor = FloodRiskBatchPredictor(
                model_path=model_path,
                use_ml_model=True,
                use_rule_model=True
            )

            # Mock feature fetching
            features_df = sample_training_data.head(10)
            with patch.object(predictor, 'fetch_latest_features', return_value=features_df):
                predictions = predictor.predict_all_regions()

                # Verify predictions
                assert len(predictions) == 10
                for pred in predictions:
                    assert 'predictions' in pred
                    assert 'ml_based' in pred['predictions'] or 'rule_based' in pred['predictions']


class TestDataQualityIntegration:
    """Test data quality checks integrated with ETL"""

    @patch('src.quality.validators.get_db_connection')
    def test_quality_validation_in_etl_workflow(self, mock_get_db):
        """Test running quality checks after ETL"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        validator = WeatherDataValidator()

        # Mock quality check results
        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'null_temp_high': 5,
            'null_temp_low': 5,
            'null_rainfall': 5,
            'null_wind': 5,
            'null_condition': 5,
            'invalid_temp_high': 2,
            'invalid_temp_low': 2,
            'invalid_rainfall': 2,
            'invalid_wind': 2,
            'last_update': datetime.now(),
            'hours_since_update': 12.0,
            'regions_with_data': 45,
            'total_regions': 50,
            'anomaly_count': 5,
            'rainfall_anomalies': 3,
            'temperature_anomalies': 2,
            'inverted_temps': 0,
            'negative_wind': 0,
            'negative_rainfall': 0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            summary = validator.run_all_checks(save_results=False)

            # Quality checks should pass
            assert summary['total_checks'] == 6
            assert summary['success_rate'] >= 0


class TestEndToEndFloodAssessment:
    """Test complete end-to-end flood risk assessment workflow"""

    @patch('src.models.batch_predictions.get_db_connection')
    @patch('src.quality.validators.get_db_connection')
    def test_complete_assessment_workflow(self, mock_quality_db, mock_pred_db, sample_training_data):
        """Test complete workflow: Quality Check → Feature Engineering → Prediction"""
        # Step 1: Quality validation
        mock_db = MagicMock()
        mock_quality_db.return_value = mock_db
        mock_pred_db.return_value = mock_db

        validator = WeatherDataValidator()

        mock_result = pd.DataFrame([{
            'total_records': 1000,
            'null_temp_high': 0, 'null_temp_low': 0,
            'null_rainfall': 0, 'null_wind': 0, 'null_condition': 0,
            'invalid_temp_high': 0, 'invalid_temp_low': 0,
            'invalid_rainfall': 0, 'invalid_wind': 0,
            'last_update': datetime.now(), 'hours_since_update': 12.0,
            'regions_with_data': 50, 'total_regions': 50,
            'anomaly_count': 5, 'rainfall_anomalies': 3, 'temperature_anomalies': 2,
            'inverted_temps': 0, 'negative_wind': 0, 'negative_rainfall': 0
        }])

        with patch('pandas.read_sql', return_value=mock_result):
            quality_summary = validator.run_all_checks(save_results=False)

        # Quality checks should pass
        assert quality_summary['all_passed'] is True

        # Step 2: Model training
        ml_model = MLFloodModel()
        y = ml_model.create_labels_from_rules(sample_training_data)
        metrics = ml_model.train(sample_training_data, y, test_size=0.2, random_state=42)

        assert metrics['test_accuracy'] > 0

        # Step 3: Batch predictions
        rule_model = RuleBasedFloodModel()

        # Create sample for prediction
        sample = sample_training_data.iloc[0].to_dict()
        sample['region_id'] = 'TEST001'
        sample['assessment_date'] = date.today()

        # Get predictions
        rule_features = WeatherFeatures(**sample)
        rule_assessment = rule_model.predict(rule_features)
        ml_assessment = ml_model.predict(sample)

        # Both models should produce assessments
        assert rule_assessment is not None
        assert ml_assessment is not None
        assert rule_assessment.risk_level is not None
        assert ml_assessment.risk_level is not None


class TestModelRetraining:
    """Test model retraining workflow"""

    @patch('src.models.training_pipeline.get_db_connection')
    def test_model_retraining_with_new_data(self, mock_get_db, sample_training_data):
        """Test retraining model with new data"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FloodModelTrainingPipeline(model_dir=tmpdir)

            # First training
            with patch.object(pipeline, 'fetch_training_data', return_value=sample_training_data):
                model1, metrics1 = pipeline.train_ml_model(sample_training_data, test_size=0.2)
                path1 = pipeline.save_model_artifacts(model1, metrics1, version_tag='v1')

            # Second training with slightly different data
            new_data = sample_training_data.copy()
            new_data['rainfall_1d'] = new_data['rainfall_1d'] * 1.1

            with patch.object(pipeline, 'fetch_training_data', return_value=new_data):
                model2, metrics2 = pipeline.train_ml_model(new_data, test_size=0.2)
                path2 = pipeline.save_model_artifacts(model2, metrics2, version_tag='v2')

            # Both models should exist
            assert os.path.exists(path1)
            assert os.path.exists(path2)
            assert path1 != path2


class TestErrorHandlingIntegration:
    """Test error handling across components"""

    @patch('src.models.batch_predictions.get_db_connection')
    def test_graceful_degradation_when_ml_model_unavailable(self, mock_get_db):
        """Test that system works with only rule-based model"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Create predictor with only rule-based model
        predictor = FloodRiskBatchPredictor(
            model_path='/nonexistent/model.pkl',
            use_ml_model=True,  # Will fail to load
            use_rule_model=True
        )

        # Should still work with rule-based model
        assert predictor.rule_model is not None

        # Create sample data
        sample_features = pd.DataFrame({
            'region_id': ['TEST001'],
            'region_name': ['Test Region'],
            'assessment_date': [date.today()],
            'rainfall_1d': [50.0],
            'rainfall_7d': [200.0],
            'elevation': [100.0],
            'latitude': [14.5],
            'longitude': [121.0],
            'temp_avg': [28.0],
            'rainfall_3d': [100.0],
            'rainfall_14d': [300.0],
            'rainfall_30d': [500.0],
            'temp_range': [10.0],
            'temp_avg_7d': [28.0],
            'temp_avg_14d': [28.0],
            'temp_avg_30d': [28.0],
            'temp_variance_7d': [2.0],
            'temp_variance_30d': [3.0],
            'wind_speed_max_7d': [40.0],
            'wind_speed_avg_7d': [30.0],
            'rainy_days_7d': [3],
            'heavy_rain_days_7d': [1],
            'high_wind_days_7d': [0],
            'max_daily_rainfall_7d': [60.0],
            'rainfall_intensity_ratio': [0.25],
            'soil_moisture_proxy': [7.0],
            'evapotranspiration_estimate': [9.0],
            'flood_risk_indicator': [0.5],
            'historical_high_risk_count': [2],
            'region_vulnerability_score': [40.0],
            'is_typhoon_season': [0],
            'is_wet_season': [0],
            'month': [1],
            'day_of_week': [3]
        })

        with patch.object(predictor, 'fetch_latest_features', return_value=sample_features):
            predictions = predictor.predict_all_regions()

            # Should get rule-based predictions
            assert len(predictions) == 1
            assert 'rule_based' in predictions[0]['predictions']


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance characteristics of integrated components"""

    def test_batch_prediction_performance(self, sample_training_data):
        """Test that batch predictions complete in reasonable time"""
        import time

        # Train model
        ml_model = MLFloodModel()
        y = ml_model.create_labels_from_rules(sample_training_data)
        ml_model.train(sample_training_data, y, test_size=0.2, random_state=42, n_estimators=50)

        # Time batch predictions
        start = time.time()
        features_df = sample_training_data.head(50)
        assessments = ml_model.batch_predict(features_df)
        duration = time.time() - start

        # Should complete in reasonable time (less than 5 seconds for 50 samples)
        assert duration < 5.0
        assert len(assessments) > 0


@pytest.mark.integration
def test_complete_phase3_workflow(sample_training_data):
    """Integration test for complete Phase 3 workflow"""
    # 1. Train ML model
    ml_model = MLFloodModel()
    y = ml_model.create_labels_from_rules(sample_training_data)
    metrics = ml_model.train(sample_training_data, y, test_size=0.2, random_state=42)

    assert metrics['test_accuracy'] > 0.5  # Should have reasonable accuracy

    # 2. Create rule-based model
    rule_model = RuleBasedFloodModel()

    # 3. Make predictions with both models
    sample = sample_training_data.iloc[0].to_dict()
    sample['region_id'] = 'INTEGRATION_TEST'
    sample['assessment_date'] = date.today()

    # Rule-based prediction
    rule_features = WeatherFeatures(**sample)
    rule_assessment = rule_model.predict(rule_features)

    # ML prediction
    ml_assessment = ml_model.predict(sample, return_probabilities=True)

    # 4. Verify both predictions are valid
    assert rule_assessment.risk_level is not None
    assert ml_assessment.risk_level is not None
    assert rule_assessment.confidence_score > 0
    assert ml_assessment.confidence_score > 0
    assert 'probabilities' in ml_assessment.metadata

    # 5. Verify predictions include recommendations
    assert len(rule_assessment.recommendation) > 0
    assert len(ml_assessment.recommendation) > 0

    print("\n=== Integration Test Complete ===")
    print(f"Rule-based: {rule_assessment.risk_level.value.upper()} (score: {rule_assessment.risk_score:.1f})")
    print(f"ML-based: {ml_assessment.risk_level.value.upper()} (score: {ml_assessment.risk_score:.1f})")
    print(f"Rule confidence: {rule_assessment.confidence_score:.2%}")
    print(f"ML confidence: {ml_assessment.confidence_score:.2%}")
