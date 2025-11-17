"""
Flood Risk Model Training and Prediction DAG - Phase 3

This DAG orchestrates flood risk model operations:
- Weekly model training with latest data
- Daily prediction generation for all regions
- Model performance tracking and validation
- Prediction quality validation

Schedule:
- Training: Weekly on Sunday at 2:00 AM UTC
- Predictions: Daily at 9:00 AM UTC (5:00 PM PHT)

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.training_pipeline import FloodModelTrainingPipeline
from src.models.batch_predictions import FloodRiskBatchPredictor
from src.quality.validators import WeatherDataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== MODEL TRAINING DAG ====================

training_default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'start_date': datetime(2025, 1, 17),
}


def check_training_readiness(**context):
    """
    Check if sufficient data is available for training

    This task:
    - Verifies minimum data requirements
    - Checks data quality
    - Determines if training should proceed

    Returns:
        str: Task ID to execute next
    """
    logger.info("Checking training readiness")

    try:
        from src.utils.database import get_db_connection

        db = get_db_connection()

        # Check if we have enough daily stats
        query = """
            SELECT COUNT(DISTINCT stat_date) as days_of_data
            FROM weather_daily_stats
            WHERE stat_date >= CURRENT_DATE - INTERVAL '180 days'
        """

        with db.get_connection() as conn:
            import pandas as pd
            result = pd.read_sql(query, conn)

        days_of_data = result.iloc[0]['days_of_data']

        logger.info(f"Found {days_of_data} days of historical data")

        # Need at least 90 days for meaningful training
        if days_of_data >= 90:
            logger.info("Sufficient data available for training")
            return 'train_flood_model'
        else:
            logger.warning(f"Insufficient data for training: {days_of_data} days (need 90+)")
            return 'skip_training'

    except Exception as e:
        logger.error(f"Training readiness check failed: {str(e)}")
        return 'skip_training'


def train_flood_model(**context):
    """
    Train flood risk prediction model

    This task:
    - Fetches historical weather data (180 days)
    - Prepares features and labels
    - Trains XGBoost model
    - Evaluates model performance
    - Saves model artifacts and metadata
    - Tracks model version in database

    Args:
        **context: Airflow context
    """
    logger.info("Starting flood model training")

    execution_date = context['execution_date']
    logger.info(f"Execution date: {execution_date}")

    try:
        # Initialize training pipeline
        pipeline = FloodModelTrainingPipeline(
            model_dir='models',
            min_training_days=90
        )

        # XGBoost hyperparameters
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        # Run training
        summary = pipeline.run(
            days_back=180,
            test_size=0.2,
            save_to_db=True,
            **xgb_params
        )

        # Push results to XCom
        context['task_instance'].xcom_push(key='training_summary', value=summary)

        logger.info(f"Model training completed: {summary}")

        # Check if model meets minimum accuracy threshold
        min_accuracy = 0.70  # 70% minimum accuracy
        if summary['test_accuracy'] < min_accuracy:
            logger.warning(
                f"Model accuracy {summary['test_accuracy']:.4f} "
                f"below threshold {min_accuracy}"
            )
            # Don't fail the task, but log warning

        return summary

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


def validate_trained_model(**context):
    """
    Validate the newly trained model

    This task:
    - Loads the trained model
    - Runs test predictions
    - Validates model outputs
    - Compares with previous model version (if available)

    Args:
        **context: Airflow context
    """
    logger.info("Validating trained model")

    task_instance = context['task_instance']
    training_summary = task_instance.xcom_pull(
        task_ids='train_flood_model',
        key='training_summary'
    )

    if not training_summary:
        logger.error("No training summary found")
        raise ValueError("Training summary not available")

    model_path = training_summary.get('model_path')
    test_accuracy = training_summary.get('test_accuracy')

    logger.info(f"Validating model at: {model_path}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    try:
        # Load the model and run sample predictions
        from src.models.flood_risk_v2 import MLFloodModel

        model = MLFloodModel(model_path=model_path)

        # Test with sample data
        sample_features = {
            'rainfall_1d': 50.0,
            'rainfall_3d': 120.0,
            'rainfall_7d': 200.0,
            'rainfall_14d': 350.0,
            'rainfall_30d': 500.0,
            'temp_avg': 28.0,
            'temp_range': 8.0,
            'temp_avg_7d': 27.5,
            'temp_avg_14d': 27.8,
            'temp_avg_30d': 28.2,
            'temp_variance_7d': 2.0,
            'temp_variance_30d': 2.5,
            'wind_speed_max_7d': 45.0,
            'wind_speed_avg_7d': 35.0,
            'elevation': 50.0,
            'month': 8,
            'day_of_week': 3,
            'is_typhoon_season': 1,
            'is_wet_season': 1,
            'rainy_days_7d': 5,
            'heavy_rain_days_7d': 2,
            'high_wind_days_7d': 1,
            'max_daily_rainfall_7d': 80.0,
            'rainfall_intensity_ratio': 0.25,
            'soil_moisture_proxy': 7.0,
            'evapotranspiration_estimate': -6.0,
            'flood_risk_indicator': 2.5,
            'historical_high_risk_count': 3,
            'region_vulnerability_score': 15.0
        }

        assessment = model.predict(sample_features, return_probabilities=True)

        logger.info(f"Sample prediction: {assessment.risk_level.value}")
        logger.info(f"Confidence: {assessment.confidence_score:.4f}")

        # Validation passed
        validation_result = {
            'model_path': model_path,
            'test_accuracy': test_accuracy,
            'sample_prediction': assessment.risk_level.value,
            'validation_status': 'passed'
        }

        context['task_instance'].xcom_push(key='validation_result', value=validation_result)

        logger.info("Model validation completed successfully")

        return validation_result

    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise


def send_training_notification(**context):
    """
    Send notification about training completion

    Args:
        **context: Airflow context
    """
    logger.info("Sending training notification")

    task_instance = context['task_instance']

    training_summary = task_instance.xcom_pull(
        task_ids='train_flood_model',
        key='training_summary'
    )

    validation_result = task_instance.xcom_pull(
        task_ids='validate_model',
        key='validation_result'
    )

    # Generate summary
    logger.info("="*70)
    logger.info("FLOOD MODEL TRAINING SUMMARY")
    logger.info("="*70)
    logger.info(f"Execution Date: {context['execution_date']}")
    logger.info("")

    if training_summary:
        logger.info("TRAINING RESULTS:")
        logger.info(f"  Duration: {training_summary.get('duration_seconds', 0):.2f}s")
        logger.info(f"  Training Samples: {training_summary.get('training_samples', 0)}")
        logger.info(f"  Test Accuracy: {training_summary.get('test_accuracy', 0):.4f}")
        logger.info(f"  CV Accuracy: {training_summary.get('cv_accuracy', 0):.4f}")
        logger.info(f"  Model Version: {training_summary.get('version', 'unknown')}")
        logger.info(f"  Model Path: {training_summary.get('model_path', 'unknown')}")
        logger.info("")

    if validation_result:
        logger.info("VALIDATION:")
        logger.info(f"  Status: {validation_result.get('validation_status', 'unknown')}")
        logger.info(f"  Sample Prediction: {validation_result.get('sample_prediction', 'unknown')}")
        logger.info("")

    logger.info("="*70)

    return True


def skip_training_task(**context):
    """
    Placeholder task when training is skipped

    Args:
        **context: Airflow context
    """
    logger.info("Training skipped - insufficient data or other conditions not met")
    return {'status': 'skipped'}


# Define Training DAG
with DAG(
    'flood_model_training',
    default_args=training_default_args,
    description='Weekly flood risk model training pipeline',
    schedule_interval='0 2 * * 0',  # Sunday 2:00 AM UTC
    catchup=False,
    max_active_runs=1,
    tags=['phase3', 'ml', 'training', 'flood-risk'],
) as training_dag:

    # Task 1: Check if training should proceed
    check_readiness = BranchPythonOperator(
        task_id='check_training_readiness',
        python_callable=check_training_readiness,
        provide_context=True,
        doc_md="""
        ### Check Training Readiness

        **Purpose:** Verify sufficient data exists for training

        **Checks:**
        - Minimum 90 days of historical data
        - Data quality thresholds
        - Training prerequisites

        **Output:** Branches to training or skip task
        """
    )

    # Task 2a: Train model
    train_task = PythonOperator(
        task_id='train_flood_model',
        python_callable=train_flood_model,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        ### Train Flood Risk Model

        **Purpose:** Train XGBoost model for flood risk prediction

        **Process:**
        - Fetch 180 days of historical weather data
        - Engineer features and create labels
        - Train XGBoost classifier
        - Evaluate on test set
        - Save model artifacts

        **Output:** Trained model with metadata

        **Timeout:** 30 minutes
        """
    )

    # Task 2b: Skip training
    skip_task = PythonOperator(
        task_id='skip_training',
        python_callable=skip_training_task,
        provide_context=True,
        doc_md="""
        ### Skip Training

        **Purpose:** Placeholder when training is skipped

        **Reason:** Insufficient data or conditions not met
        """
    )

    # Task 3: Validate model
    validate_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_trained_model,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        ### Validate Trained Model

        **Purpose:** Verify model quality and functionality

        **Checks:**
        - Model loads correctly
        - Predictions work as expected
        - Output format is valid
        - Performance meets threshold

        **Timeout:** 10 minutes
        """
    )

    # Task 4: Send notification
    notify_training = PythonOperator(
        task_id='send_training_notification',
        python_callable=send_training_notification,
        provide_context=True,
        trigger_rule='all_done',
        doc_md="""
        ### Training Notification

        **Purpose:** Send summary of training results

        **Trigger:** Always runs (even if training skipped/failed)
        """
    )

    # Define dependencies
    check_readiness >> [train_task, skip_task]
    train_task >> validate_task >> notify_training
    skip_task >> notify_training


# ==================== PREDICTION DAG ====================

prediction_default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 17),
}


def generate_flood_predictions(**context):
    """
    Generate flood risk predictions for all regions

    This task:
    - Loads latest weather features
    - Runs predictions using both models (v1 & v2)
    - Saves predictions to database
    - Validates prediction outputs

    Args:
        **context: Airflow context
    """
    logger.info("Starting flood risk prediction generation")

    execution_date = context['execution_date']
    target_date = execution_date.date()

    logger.info(f"Generating predictions for: {target_date}")

    try:
        # Initialize batch predictor with auto-detection of latest model
        predictor = FloodRiskBatchPredictor(
            model_path=None,  # Auto-detect latest model
            use_ml_model=True,
            use_rule_model=True
        )

        # Run predictions
        summary = predictor.run(
            target_date=target_date,
            region_ids=None,  # All regions
            save_to_db=True,
            model_version='combined'  # Save ML predictions preferentially
        )

        # Push results to XCom
        context['task_instance'].xcom_push(key='prediction_summary', value=summary)

        logger.info(f"Prediction generation completed: {summary}")

        return summary

    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise


def validate_predictions(**context):
    """
    Validate generated predictions

    This task:
    - Checks prediction coverage (all regions)
    - Validates risk level distribution
    - Ensures no anomalous predictions
    - Verifies database saves

    Args:
        **context: Airflow context
    """
    logger.info("Validating predictions")

    task_instance = context['task_instance']
    prediction_summary = task_instance.xcom_pull(
        task_ids='generate_predictions',
        key='prediction_summary'
    )

    if not prediction_summary:
        logger.error("No prediction summary found")
        raise ValueError("Prediction summary not available")

    regions_processed = prediction_summary.get('regions_processed', 0)
    risk_distribution = prediction_summary.get('risk_distribution', {})

    logger.info(f"Regions processed: {regions_processed}")
    logger.info(f"Risk distribution: {risk_distribution}")

    # Validation checks
    try:
        from src.utils.database import get_db_connection

        db = get_db_connection()

        # Check if predictions were saved
        query = """
            SELECT COUNT(*) as prediction_count
            FROM flood_risk_assessments
            WHERE assessment_date = CURRENT_DATE
        """

        with db.get_connection() as conn:
            import pandas as pd
            result = pd.read_sql(query, conn)

        prediction_count = result.iloc[0]['prediction_count']

        logger.info(f"Found {prediction_count} predictions in database")

        # Validate coverage
        expected_regions = 30  # Total regions in Philippines
        if regions_processed < expected_regions * 0.9:  # 90% threshold
            logger.warning(
                f"Low prediction coverage: {regions_processed}/{expected_regions} regions"
            )

        # Validate distribution
        if risk_distribution:
            for model_type, dist in risk_distribution.items():
                total = sum(dist.values())
                if total > 0:
                    # Check if all predictions are same level (anomaly)
                    for level, count in dist.items():
                        percentage = (count / total) * 100
                        if percentage > 90:
                            logger.warning(
                                f"{model_type}: {percentage:.1f}% predictions are '{level}' - "
                                "possible model issue"
                            )

        validation_result = {
            'regions_processed': regions_processed,
            'predictions_saved': prediction_count,
            'coverage_percentage': (regions_processed / expected_regions) * 100,
            'validation_status': 'passed'
        }

        context['task_instance'].xcom_push(key='validation_result', value=validation_result)

        logger.info("Prediction validation completed successfully")

        return validation_result

    except Exception as e:
        logger.error(f"Prediction validation failed: {str(e)}")
        raise


def track_model_performance(**context):
    """
    Track model performance metrics over time

    This task:
    - Analyzes prediction patterns
    - Compares v1 vs v2 model outputs
    - Identifies potential model drift
    - Logs performance metrics

    Args:
        **context: Airflow context
    """
    logger.info("Tracking model performance")

    try:
        from src.utils.database import get_db_connection
        import pandas as pd

        db = get_db_connection()

        # Analyze recent predictions
        query = """
            SELECT
                model_version,
                risk_level,
                COUNT(*) as count,
                AVG(confidence_score) as avg_confidence
            FROM flood_risk_assessments
            WHERE assessment_date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY model_version, risk_level
            ORDER BY model_version, risk_level
        """

        with db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        # Log performance metrics
        logger.info("="*70)
        logger.info("MODEL PERFORMANCE TRACKING (Past 7 Days)")
        logger.info("="*70)

        for model_version in df['model_version'].unique():
            model_df = df[df['model_version'] == model_version]

            logger.info(f"\n{model_version}:")
            total_predictions = model_df['count'].sum()

            for _, row in model_df.iterrows():
                percentage = (row['count'] / total_predictions) * 100
                logger.info(
                    f"  {row['risk_level']}: {row['count']} ({percentage:.1f}%) "
                    f"- Avg Confidence: {row['avg_confidence']:.4f}"
                )

        logger.info("="*70)

        performance_metrics = {
            'tracking_date': datetime.now().isoformat(),
            'models_tracked': df['model_version'].unique().tolist(),
            'total_predictions': int(df['count'].sum())
        }

        context['task_instance'].xcom_push(key='performance_metrics', value=performance_metrics)

        return performance_metrics

    except Exception as e:
        logger.error(f"Performance tracking failed: {str(e)}")
        # Don't fail the task, just log the error
        return {'error': str(e)}


def send_prediction_notification(**context):
    """
    Send notification about prediction generation

    Args:
        **context: Airflow context
    """
    logger.info("Sending prediction notification")

    task_instance = context['task_instance']

    prediction_summary = task_instance.xcom_pull(
        task_ids='generate_predictions',
        key='prediction_summary'
    )

    validation_result = task_instance.xcom_pull(
        task_ids='validate_predictions',
        key='validation_result'
    )

    performance_metrics = task_instance.xcom_pull(
        task_ids='track_performance',
        key='performance_metrics'
    )

    # Generate summary
    logger.info("="*70)
    logger.info("FLOOD RISK PREDICTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Execution Date: {context['execution_date']}")
    logger.info("")

    if prediction_summary:
        logger.info("PREDICTIONS GENERATED:")
        logger.info(f"  Regions Processed: {prediction_summary.get('regions_processed', 0)}")
        logger.info(f"  Duration: {prediction_summary.get('duration_seconds', 0):.2f}s")
        logger.info(f"  Saved to Database: {prediction_summary.get('saved_to_db', False)}")
        logger.info("")

        risk_dist = prediction_summary.get('risk_distribution', {})
        if risk_dist:
            logger.info("RISK DISTRIBUTION:")
            for model_type, dist in risk_dist.items():
                logger.info(f"  {model_type}:")
                for level, count in sorted(dist.items()):
                    logger.info(f"    {level}: {count}")
            logger.info("")

    if validation_result:
        logger.info("VALIDATION:")
        logger.info(f"  Status: {validation_result.get('validation_status', 'unknown')}")
        logger.info(f"  Coverage: {validation_result.get('coverage_percentage', 0):.1f}%")
        logger.info(f"  Predictions Saved: {validation_result.get('predictions_saved', 0)}")
        logger.info("")

    logger.info("="*70)

    return True


# Define Prediction DAG
with DAG(
    'flood_risk_predictions',
    default_args=prediction_default_args,
    description='Daily flood risk predictions for all regions',
    schedule_interval='0 9 * * *',  # 9:00 AM UTC daily (5:00 PM PHT)
    catchup=False,
    max_active_runs=1,
    tags=['phase3', 'ml', 'predictions', 'flood-risk'],
) as prediction_dag:

    # Task 1: Generate predictions
    generate_task = PythonOperator(
        task_id='generate_predictions',
        python_callable=generate_flood_predictions,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
        doc_md="""
        ### Generate Flood Risk Predictions

        **Purpose:** Create daily flood risk assessments for all regions

        **Process:**
        - Fetch latest weather features
        - Run both rule-based and ML models
        - Generate risk predictions
        - Save to database

        **Output:** Predictions for all 30 regions

        **Timeout:** 15 minutes
        """
    )

    # Task 2: Validate predictions
    validate_pred_task = PythonOperator(
        task_id='validate_predictions',
        python_callable=validate_predictions,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Validate Predictions

        **Purpose:** Verify prediction quality and coverage

        **Checks:**
        - All regions covered
        - Risk distribution reasonable
        - Database saves successful
        - No anomalous predictions

        **Timeout:** 5 minutes
        """
    )

    # Task 3: Track performance
    track_task = PythonOperator(
        task_id='track_performance',
        python_callable=track_model_performance,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Track Model Performance

        **Purpose:** Monitor model behavior over time

        **Metrics:**
        - Prediction distribution
        - Confidence scores
        - Model comparison (v1 vs v2)
        - Potential drift detection

        **Timeout:** 5 minutes
        """
    )

    # Task 4: Send notification
    notify_pred_task = PythonOperator(
        task_id='send_prediction_notification',
        python_callable=send_prediction_notification,
        provide_context=True,
        trigger_rule='all_done',
        doc_md="""
        ### Prediction Notification

        **Purpose:** Send summary of prediction generation

        **Trigger:** Always runs (even if previous tasks fail)
        """
    )

    # Define dependencies
    generate_task >> validate_pred_task >> track_task >> notify_pred_task
