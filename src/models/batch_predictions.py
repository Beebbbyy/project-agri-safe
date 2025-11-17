"""
Batch Flood Risk Prediction Service

This module provides batch prediction capabilities for flood risk assessment:
- Load latest weather features for all regions
- Generate predictions using trained ML model
- Save predictions to database
- Compare v1 (rule-based) vs v2 (ML) predictions

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import glob

import pandas as pd
import numpy as np
from psycopg2.extras import Json

from src.utils.database import get_db_connection
from src.models.flood_risk_v1 import RuleBasedFloodModel, WeatherFeatures
from src.models.flood_risk_v2 import MLFloodModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FloodRiskBatchPredictor:
    """
    Generate batch flood risk predictions for all regions

    This service:
    - Fetches latest weather features
    - Runs predictions using both v1 and v2 models
    - Saves results to database
    - Provides comparison between models
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_ml_model: bool = True,
        use_rule_model: bool = True
    ):
        """
        Initialize batch predictor

        Args:
            model_path: Path to ML model (auto-detects latest if None)
            use_ml_model: Whether to use ML model for predictions
            use_rule_model: Whether to use rule-based model
        """
        self.db = get_db_connection()
        self.use_ml_model = use_ml_model
        self.use_rule_model = use_rule_model

        # Initialize models
        self.ml_model: Optional[MLFloodModel] = None
        self.rule_model: Optional[RuleBasedFloodModel] = None

        if use_ml_model:
            if model_path is None:
                model_path = self._find_latest_model()

            if model_path and os.path.exists(model_path):
                self.ml_model = MLFloodModel(model_path=model_path)
                logger.info(f"Loaded ML model from: {model_path}")
            else:
                logger.warning("No ML model found. Only rule-based predictions will be available.")
                self.use_ml_model = False

        if use_rule_model:
            self.rule_model = RuleBasedFloodModel()
            logger.info("Initialized rule-based model")

    def _find_latest_model(self) -> Optional[str]:
        """
        Find the most recent ML model file

        Returns:
            Path to latest model or None
        """
        model_patterns = [
            'models/flood_risk_v2_*.pkl',
            '../models/flood_risk_v2_*.pkl',
            '/opt/airflow/models/flood_risk_v2_*.pkl'
        ]

        for pattern in model_patterns:
            models = glob.glob(pattern)
            if models:
                latest = max(models, key=os.path.getmtime)
                logger.info(f"Auto-detected latest model: {latest}")
                return latest

        logger.warning("No ML models found in standard locations")
        return None

    def fetch_latest_features(
        self,
        target_date: Optional[date] = None,
        region_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch latest weather features for all regions

        Args:
            target_date: Date to fetch features for (defaults to today)
            region_ids: Optional list of specific regions

        Returns:
            DataFrame with features for each region
        """
        if target_date is None:
            target_date = date.today()

        logger.info(f"Fetching features for {target_date}")

        region_filter = ""
        if region_ids:
            region_list = ','.join([f"'{rid}'" for rid in region_ids])
            region_filter = f"AND wds.region_id IN ({region_list})"

        query = f"""
            WITH latest_stats AS (
                SELECT
                    wds.region_id,
                    wds.stat_date,
                    wds.rainfall_total as rainfall_1d,
                    wds.temp_high_avg,
                    wds.temp_low_avg,
                    wds.wind_speed_max,
                    r.name as region_name,
                    r.latitude,
                    r.longitude,
                    r.elevation,
                    EXTRACT(MONTH FROM wds.stat_date)::int as month,
                    EXTRACT(DOW FROM wds.stat_date)::int as day_of_week,
                    CASE WHEN EXTRACT(MONTH FROM wds.stat_date) IN (6,7,8,9,10,11) THEN 1 ELSE 0 END as is_typhoon_season,
                    CASE WHEN EXTRACT(MONTH FROM wds.stat_date) IN (6,7,8,9,10,11) THEN 1 ELSE 0 END as is_wet_season,
                    ROW_NUMBER() OVER (PARTITION BY wds.region_id ORDER BY wds.stat_date DESC) as rn
                FROM weather_daily_stats wds
                JOIN regions r ON wds.region_id = r.id
                WHERE wds.stat_date <= '{target_date}'
                {region_filter}
            )
            SELECT
                ls.region_id::text,
                ls.stat_date as assessment_date,
                ls.region_name,
                ls.latitude,
                ls.longitude,
                ls.elevation,
                ls.rainfall_1d,
                (ls.temp_high_avg + ls.temp_low_avg) / 2 as temp_avg,
                ls.temp_high_avg - ls.temp_low_avg as temp_range,
                ls.wind_speed_max as wind_speed_max_7d,
                ls.month,
                ls.day_of_week,
                ls.is_typhoon_season,
                ls.is_wet_season,

                -- Rolling features (3-day)
                COALESCE((
                    SELECT SUM(rainfall_total)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '2 days' AND ls.stat_date
                ), 0) as rainfall_3d,

                -- Rolling features (7-day)
                COALESCE((
                    SELECT SUM(rainfall_total)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                ), 0) as rainfall_7d,

                COALESCE((
                    SELECT AVG((temp_high_avg + temp_low_avg) / 2)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                ), ls.temp_high_avg) as temp_avg_7d,

                COALESCE((
                    SELECT STDDEV((temp_high_avg + temp_low_avg) / 2)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                ), 0) as temp_variance_7d,

                COALESCE((
                    SELECT AVG(wind_speed_max)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                ), ls.wind_speed_max) as wind_speed_avg_7d,

                COALESCE((
                    SELECT COUNT(*)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                      AND wds2.rainfall_total > 2.5
                ), 0) as rainy_days_7d,

                COALESCE((
                    SELECT COUNT(*)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                      AND wds2.rainfall_total > 50
                ), 0) as heavy_rain_days_7d,

                COALESCE((
                    SELECT COUNT(*)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                      AND wds2.wind_speed_max > 60
                ), 0) as high_wind_days_7d,

                COALESCE((
                    SELECT MAX(rainfall_total)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '6 days' AND ls.stat_date
                ), ls.rainfall_1d) as max_daily_rainfall_7d,

                -- Rolling features (14-day)
                COALESCE((
                    SELECT SUM(rainfall_total)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '13 days' AND ls.stat_date
                ), 0) as rainfall_14d,

                COALESCE((
                    SELECT AVG((temp_high_avg + temp_low_avg) / 2)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '13 days' AND ls.stat_date
                ), ls.temp_high_avg) as temp_avg_14d,

                -- Rolling features (30-day)
                COALESCE((
                    SELECT SUM(rainfall_total)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '29 days' AND ls.stat_date
                ), 0) as rainfall_30d,

                COALESCE((
                    SELECT AVG((temp_high_avg + temp_low_avg) / 2)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '29 days' AND ls.stat_date
                ), ls.temp_high_avg) as temp_avg_30d,

                COALESCE((
                    SELECT STDDEV((temp_high_avg + temp_low_avg) / 2)
                    FROM weather_daily_stats wds2
                    WHERE wds2.region_id = ls.region_id
                      AND wds2.stat_date BETWEEN ls.stat_date - INTERVAL '29 days' AND ls.stat_date
                ), 0) as temp_variance_30d

            FROM latest_stats ls
            WHERE ls.rn = 1
            ORDER BY ls.region_id
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)

        # Add derived features
        df['rainfall_intensity_ratio'] = df['rainfall_1d'] / (df['rainfall_7d'] + 0.1)
        df['soil_moisture_proxy'] = df['rainfall_7d'] / (df['temp_avg'] + 1)
        df['evapotranspiration_estimate'] = df['temp_avg'] * 0.5 - df['rainfall_7d'] * 0.1
        df['flood_risk_indicator'] = (df['rainfall_7d'] * 0.4 + df['rainfall_1d'] * 0.6) / (df['elevation'] + 1)

        # Add historical flood risk
        df = self._add_historical_risk(df)

        # Fill NaN values
        df = df.fillna(0)

        logger.info(f"Fetched features for {len(df)} regions")

        return df

    def _add_historical_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical flood risk data"""
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

            df = df.merge(risk_df, on='region_id', how='left')

        except Exception as e:
            logger.warning(f"Could not load historical risk: {str(e)}")
            df['historical_high_risk_count'] = 0
            df['region_vulnerability_score'] = 0.0

        df['historical_high_risk_count'] = df['historical_high_risk_count'].fillna(0)
        df['region_vulnerability_score'] = df['region_vulnerability_score'].fillna(0.0)

        return df

    def predict_all_regions(
        self,
        target_date: Optional[date] = None,
        region_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for all regions

        Args:
            target_date: Date to predict for
            region_ids: Optional region filter

        Returns:
            List of prediction dictionaries
        """
        logger.info("Generating batch predictions for all regions")

        # Fetch features
        features_df = self.fetch_latest_features(target_date, region_ids)

        if len(features_df) == 0:
            logger.warning("No features available for prediction")
            return []

        predictions = []

        for _, row in features_df.iterrows():
            region_prediction = {
                'region_id': row['region_id'],
                'region_name': row['region_name'],
                'assessment_date': row['assessment_date'],
                'predictions': {}
            }

            # Rule-based prediction (v1)
            if self.use_rule_model:
                try:
                    weather_features = WeatherFeatures(**row.to_dict())
                    rule_assessment = self.rule_model.predict(weather_features)

                    region_prediction['predictions']['rule_based'] = {
                        'risk_level': rule_assessment.risk_level.value,
                        'risk_score': rule_assessment.risk_score,
                        'confidence_score': rule_assessment.confidence_score,
                        'recommendation': rule_assessment.recommendation,
                        'model_version': rule_assessment.model_version
                    }

                except Exception as e:
                    logger.error(f"Rule-based prediction failed for {row['region_id']}: {str(e)}")

            # ML-based prediction (v2)
            if self.use_ml_model and self.ml_model:
                try:
                    ml_assessment = self.ml_model.predict(
                        row.to_dict(),
                        return_probabilities=True
                    )

                    region_prediction['predictions']['ml_based'] = {
                        'risk_level': ml_assessment.risk_level.value,
                        'risk_score': ml_assessment.risk_score,
                        'confidence_score': ml_assessment.confidence_score,
                        'recommendation': ml_assessment.recommendation,
                        'model_version': ml_assessment.model_version,
                        'probabilities': ml_assessment.metadata.get('probabilities', {})
                    }

                except Exception as e:
                    logger.error(f"ML prediction failed for {row['region_id']}: {str(e)}")

            predictions.append(region_prediction)

        logger.info(f"Generated {len(predictions)} predictions")

        return predictions

    def save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        model_version: str = 'combined'
    ):
        """
        Save predictions to database

        Args:
            predictions: List of prediction dictionaries
            model_version: Which model predictions to save ('rule_based', 'ml_based', or 'combined')
        """
        logger.info(f"Saving {len(predictions)} predictions to database")

        saved_count = 0

        with self.db.get_cursor() as cursor:
            for pred in predictions:
                # Determine which prediction to save
                if model_version == 'ml_based' and 'ml_based' in pred['predictions']:
                    selected_pred = pred['predictions']['ml_based']
                elif model_version == 'rule_based' and 'rule_based' in pred['predictions']:
                    selected_pred = pred['predictions']['rule_based']
                elif 'ml_based' in pred['predictions']:  # Prefer ML
                    selected_pred = pred['predictions']['ml_based']
                elif 'rule_based' in pred['predictions']:
                    selected_pred = pred['predictions']['rule_based']
                else:
                    continue

                try:
                    cursor.execute("""
                        INSERT INTO flood_risk_assessments (
                            region_id, assessment_date, risk_level,
                            confidence_score, contributing_factors,
                            recommendation, model_version
                        ) VALUES (
                            %s::uuid, %s, %s, %s, %s::jsonb, %s, %s
                        )
                        ON CONFLICT (region_id, assessment_date)
                        DO UPDATE SET
                            risk_level = EXCLUDED.risk_level,
                            confidence_score = EXCLUDED.confidence_score,
                            contributing_factors = EXCLUDED.contributing_factors,
                            recommendation = EXCLUDED.recommendation,
                            model_version = EXCLUDED.model_version,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        pred['region_id'],
                        pred['assessment_date'],
                        selected_pred['risk_level'],
                        selected_pred['confidence_score'],
                        Json({}),  # Empty for now, could include factors
                        selected_pred['recommendation'],
                        selected_pred['model_version']
                    ))

                    saved_count += 1

                except Exception as e:
                    logger.error(f"Failed to save prediction for {pred['region_id']}: {str(e)}")

        logger.info(f"Saved {saved_count} predictions to database")

    def run(
        self,
        target_date: Optional[date] = None,
        region_ids: Optional[List[str]] = None,
        save_to_db: bool = True,
        model_version: str = 'combined'
    ) -> Dict[str, Any]:
        """
        Execute batch prediction pipeline

        Args:
            target_date: Date to predict for
            region_ids: Optional region filter
            save_to_db: Whether to save to database
            model_version: Which model to use for saving

        Returns:
            Summary statistics
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("Starting Batch Flood Risk Prediction")
        logger.info("="*60)

        # Generate predictions
        predictions = self.predict_all_regions(target_date, region_ids)

        # Save to database
        if save_to_db:
            self.save_predictions(predictions, model_version)

        # Calculate statistics
        risk_distribution = {}
        for pred in predictions:
            for model_type, model_pred in pred['predictions'].items():
                if model_type not in risk_distribution:
                    risk_distribution[model_type] = {}

                level = model_pred['risk_level']
                risk_distribution[model_type][level] = risk_distribution[model_type].get(level, 0) + 1

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = {
            'status': 'success',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'regions_processed': len(predictions),
            'risk_distribution': risk_distribution,
            'saved_to_db': save_to_db
        }

        logger.info("\n" + "="*60)
        logger.info("Batch Prediction Completed")
        logger.info("="*60)
        logger.info(f"Regions processed: {len(predictions)}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("\nRisk Distribution:")
        for model_type, dist in risk_distribution.items():
            logger.info(f"  {model_type}:")
            for level, count in sorted(dist.items()):
                logger.info(f"    {level}: {count}")
        logger.info("="*60 + "\n")

        return summary


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Batch Flood Risk Predictions')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD)')
    parser.add_argument('--regions', nargs='*', help='Specific region IDs')
    parser.add_argument('--model-path', help='Path to ML model')
    parser.add_argument('--model-version', default='combined',
                        choices=['rule_based', 'ml_based', 'combined'],
                        help='Which model to use for saving')
    parser.add_argument('--no-save', action='store_true', help='Skip saving to database')
    parser.add_argument('--no-ml', action='store_true', help='Skip ML predictions')
    parser.add_argument('--no-rule', action='store_true', help='Skip rule-based predictions')

    args = parser.parse_args()

    # Parse target date
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()

    # Create predictor
    predictor = FloodRiskBatchPredictor(
        model_path=args.model_path,
        use_ml_model=not args.no_ml,
        use_rule_model=not args.no_rule
    )

    try:
        # Run predictions
        summary = predictor.run(
            target_date=target_date,
            region_ids=args.regions,
            save_to_db=not args.no_save,
            model_version=args.model_version
        )

        print("\nBatch Prediction Summary:")
        print(json.dumps(summary, indent=2, default=str))

        sys.exit(0)

    except Exception as e:
        print(f"\nBatch Prediction Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import json
    main()
