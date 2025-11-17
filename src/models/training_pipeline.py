"""
Model Training Pipeline for Flood Risk Prediction

This module orchestrates the complete training workflow:
- Data extraction from database
- Feature engineering and preparation
- Model training and validation
- Model evaluation and comparison
- Model persistence and versioning

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import os
import sys
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

import pandas as pd
import numpy as np
from psycopg2.extras import Json

from src.utils.database import get_db_connection
from src.models.flood_risk_v2 import MLFloodModel
from src.models.flood_risk_v1 import RuleBasedFloodModel, WeatherFeatures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FloodModelTrainingPipeline:
    """
    Complete training pipeline for flood risk models

    Handles:
    - Historical data extraction
    - Feature engineering
    - Model training (both v1 and v2)
    - Model evaluation
    - Model persistence
    - Metadata tracking
    """

    def __init__(
        self,
        model_dir: str = "models",
        min_training_days: int = 90
    ):
        """
        Initialize training pipeline

        Args:
            model_dir: Directory to save trained models
            min_training_days: Minimum days of historical data required
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.min_training_days = min_training_days
        self.db = get_db_connection()

        logger.info(f"Initialized training pipeline. Models will be saved to: {self.model_dir}")

    def fetch_training_data(
        self,
        days_back: int = 180,
        region_ids: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Fetch historical weather data for training

        Args:
            days_back: Number of days of history to fetch
            region_ids: Optional list of specific regions

        Returns:
            DataFrame with weather features
        """
        logger.info(f"Fetching training data for past {days_back} days")

        region_filter = ""
        if region_ids:
            region_list = ','.join([f"'{rid}'" for rid in region_ids])
            region_filter = f"AND wds.region_id IN ({region_list})"

        query = f"""
            SELECT
                wds.region_id::text,
                wds.stat_date as assessment_date,
                r.name as region_name,
                r.latitude,
                r.longitude,
                r.elevation,

                -- Rainfall features
                wds.rainfall_total as rainfall_1d,

                -- 3-day rolling
                SUM(wds.rainfall_total) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as rainfall_3d,

                -- 7-day rolling
                SUM(wds.rainfall_total) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rainfall_7d,

                -- 14-day rolling
                SUM(wds.rainfall_total) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) as rainfall_14d,

                -- 30-day rolling
                SUM(wds.rainfall_total) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as rainfall_30d,

                -- Temperature features
                (wds.temp_high_avg + wds.temp_low_avg) / 2 as temp_avg,
                wds.temp_high_avg - wds.temp_low_avg as temp_range,

                AVG((wds.temp_high_avg + wds.temp_low_avg) / 2) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as temp_avg_7d,

                AVG((wds.temp_high_avg + wds.temp_low_avg) / 2) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) as temp_avg_14d,

                AVG((wds.temp_high_avg + wds.temp_low_avg) / 2) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as temp_avg_30d,

                STDDEV((wds.temp_high_avg + wds.temp_low_avg) / 2) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as temp_variance_7d,

                STDDEV((wds.temp_high_avg + wds.temp_low_avg) / 2) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as temp_variance_30d,

                -- Wind features
                wds.wind_speed_max as wind_speed_max_7d,
                AVG(wds.wind_speed_max) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as wind_speed_avg_7d,

                -- Rainy day counts
                COUNT(*) FILTER (WHERE wds.rainfall_total > 2.5) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rainy_days_7d,

                COUNT(*) FILTER (WHERE wds.rainfall_total > 50) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as heavy_rain_days_7d,

                COUNT(*) FILTER (WHERE wds.wind_speed_max > 60) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as high_wind_days_7d,

                MAX(wds.rainfall_total) OVER (
                    PARTITION BY wds.region_id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as max_daily_rainfall_7d,

                -- Seasonal features
                EXTRACT(MONTH FROM wds.stat_date)::int as month,
                EXTRACT(DOW FROM wds.stat_date)::int as day_of_week,
                CASE WHEN EXTRACT(MONTH FROM wds.stat_date) IN (6,7,8,9,10,11) THEN 1 ELSE 0 END as is_typhoon_season,
                CASE WHEN EXTRACT(MONTH FROM wds.stat_date) IN (6,7,8,9,10,11) THEN 1 ELSE 0 END as is_wet_season

            FROM weather_daily_stats wds
            JOIN regions r ON wds.region_id = r.id
            WHERE wds.stat_date >= CURRENT_DATE - INTERVAL '{days_back} days'
            {region_filter}
            ORDER BY wds.region_id, wds.stat_date
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        # Add derived features
        df['rainfall_intensity_ratio'] = df['rainfall_1d'] / (df['rainfall_7d'] + 0.1)
        df['soil_moisture_proxy'] = df['rainfall_7d'] / (df['temp_avg'] + 1)
        df['evapotranspiration_estimate'] = df['temp_avg'] * 0.5 - df['rainfall_7d'] * 0.1
        df['flood_risk_indicator'] = (df['rainfall_7d'] * 0.4 + df['rainfall_1d'] * 0.6) / (df['elevation'] + 1)

        # Add historical flood risk data
        df = self._add_historical_risk_data(df)

        # Fill NaN values
        df = df.fillna(0)

        logger.info(f"Fetched {len(df)} training samples across {df['region_id'].nunique()} regions")

        return df

    def _add_historical_risk_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add historical flood risk counts per region

        Args:
            df: DataFrame with training data

        Returns:
            DataFrame with historical risk columns added
        """
        try:
            query = """
                SELECT
                    region_id::text,
                    COUNT(*) FILTER (WHERE risk_level IN ('high', 'critical')) as historical_high_risk_count,
                    ROUND(
                        100.0 * COUNT(*) FILTER (WHERE risk_level IN ('high', 'critical')) /
                        NULLIF(COUNT(*), 0),
                        2
                    ) as region_vulnerability_score
                FROM flood_risk_assessments
                GROUP BY region_id
            """

            with self.db.get_connection() as conn:
                risk_df = pd.read_sql(query, conn)

            # Merge with main data
            df = df.merge(risk_df, on='region_id', how='left')

        except Exception as e:
            logger.warning(f"Could not load historical risk data: {str(e)}")
            df['historical_high_risk_count'] = 0
            df['region_vulnerability_score'] = 0.0

        # Fill missing values
        df['historical_high_risk_count'] = df['historical_high_risk_count'].fillna(0)
        df['region_vulnerability_score'] = df['region_vulnerability_score'].fillna(0.0)

        return df

    def train_ml_model(
        self,
        training_data: pd.DataFrame,
        test_size: float = 0.2,
        **xgb_params
    ) -> Tuple[MLFloodModel, Dict[str, Any]]:
        """
        Train the ML-based flood model

        Args:
            training_data: DataFrame with features
            test_size: Test set proportion
            **xgb_params: XGBoost hyperparameters

        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info("Training ML flood risk model (v2)")

        # Initialize model
        model = MLFloodModel()

        # Prepare features and labels
        X = training_data.copy()
        y = model.create_labels_from_rules(X)

        # Check for actual labels if available
        actual_labels = model.create_labels_from_data(X)
        if actual_labels is not None:
            logger.info("Using actual flood occurrence labels for training")
            y = actual_labels

        logger.info(f"Label distribution: {np.bincount(y)}")

        # Train model
        metrics = model.train(X, y, test_size=test_size, **xgb_params)

        return model, metrics

    def save_model_artifacts(
        self,
        model: MLFloodModel,
        metrics: Dict[str, Any],
        version_tag: Optional[str] = None
    ) -> str:
        """
        Save model and metadata

        Args:
            model: Trained model
            metrics: Training metrics
            version_tag: Optional version tag (defaults to timestamp)

        Returns:
            Path to saved model
        """
        if version_tag is None:
            version_tag = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_filename = f"flood_risk_v2_{version_tag}.pkl"
        model_path = self.model_dir / model_filename

        # Save model
        model.save_model(str(model_path))

        # Save metrics separately
        metrics_path = self.model_dir / f"metrics_v2_{version_tag}.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = self._make_json_serializable(metrics)
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metrics saved to: {metrics_path}")

        return str(model_path)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def save_training_run_metadata(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        metrics: Dict[str, Any]
    ):
        """
        Save training run metadata to database

        Args:
            model_name: Name of the model
            model_version: Version identifier
            model_path: Path to saved model
            metrics: Training metrics
        """
        try:
            query = """
                INSERT INTO model_training_runs (
                    model_name, model_version, training_date,
                    accuracy, metrics, model_path, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
            """

            with self.db.get_cursor() as cursor:
                cursor.execute(query, (
                    model_name,
                    model_version,
                    datetime.now(),
                    metrics.get('test_accuracy', 0.0),
                    Json(self._make_json_serializable(metrics)),
                    model_path,
                    'completed'
                ))

                run_id = cursor.fetchone()['id']
                logger.info(f"Training run metadata saved with ID: {run_id}")

        except Exception as e:
            logger.error(f"Failed to save training metadata: {str(e)}")

    def run(
        self,
        days_back: int = 180,
        region_ids: Optional[list] = None,
        test_size: float = 0.2,
        save_to_db: bool = True,
        **xgb_params
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline

        Args:
            days_back: Days of historical data to use
            region_ids: Optional region filter
            test_size: Test set size
            save_to_db: Whether to save metadata to DB
            **xgb_params: XGBoost parameters

        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("Starting Flood Risk Model Training Pipeline")
        logger.info("="*60)

        try:
            # Step 1: Fetch training data
            logger.info("\n[Step 1/4] Fetching training data...")
            training_data = self.fetch_training_data(days_back, region_ids)

            if len(training_data) < self.min_training_days:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} samples "
                    f"(minimum: {self.min_training_days})"
                )

            # Step 2: Train ML model
            logger.info("\n[Step 2/4] Training ML model...")
            ml_model, ml_metrics = self.train_ml_model(training_data, test_size, **xgb_params)

            # Step 3: Save model
            logger.info("\n[Step 3/4] Saving model artifacts...")
            version_tag = datetime.now().strftime('%Y%m%d')
            model_path = self.save_model_artifacts(ml_model, ml_metrics, version_tag)

            # Step 4: Save metadata to database
            if save_to_db:
                logger.info("\n[Step 4/4] Saving training metadata to database...")
                self.save_training_run_metadata(
                    model_name="flood_risk_ml",
                    model_version=f"v2.0_{version_tag}",
                    model_path=model_path,
                    metrics=ml_metrics
                )

            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            summary = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'training_samples': len(training_data),
                'test_accuracy': ml_metrics['test_accuracy'],
                'cv_accuracy': ml_metrics['cv_mean_accuracy'],
                'model_path': model_path,
                'version': f"v2.0_{version_tag}"
            }

            logger.info("\n" + "="*60)
            logger.info("Training Pipeline Completed Successfully!")
            logger.info("="*60)
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Training samples: {len(training_data)}")
            logger.info(f"Test accuracy: {ml_metrics['test_accuracy']:.4f}")
            logger.info(f"CV accuracy: {ml_metrics['cv_mean_accuracy']:.4f} ± {ml_metrics['cv_std_accuracy']:.4f}")
            logger.info(f"Model saved to: {model_path}")
            logger.info("="*60 + "\n")

            # Print feature importance
            logger.info(ml_model.get_feature_importance_report())

            return summary

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Flood Model Training Pipeline')
    parser.add_argument('--days', type=int, default=180, help='Days of historical data')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--model-dir', default='models', help='Directory to save models')
    parser.add_argument('--no-save-db', action='store_true', help='Skip database metadata')

    # XGBoost parameters
    parser.add_argument('--n-estimators', type=int, default=200, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=6, help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')

    args = parser.parse_args()

    # Create pipeline
    pipeline = FloodModelTrainingPipeline(model_dir=args.model_dir)

    # XGBoost parameters
    xgb_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate
    }

    # Run pipeline
    try:
        summary = pipeline.run(
            days_back=args.days,
            test_size=args.test_size,
            save_to_db=not args.no_save_db,
            **xgb_params
        )

        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("="*60)

        sys.exit(0)

    except Exception as e:
        print(f"\nTraining Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
