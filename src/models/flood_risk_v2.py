"""
ML-Based Flood Risk Prediction Model (Version 2)

This module implements a machine learning approach to flood risk prediction using XGBoost.
It provides:
- Multi-class classification (low, medium, high, critical)
- Feature importance analysis
- Probability estimates for each risk level
- Model persistence and versioning

The model learns from historical weather patterns and flood occurrences.

Author: AgriSafe Development Team
Date: 2025-01-17
"""

import os
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

from src.models.flood_risk_v1 import FloodRiskLevel, FloodRiskAssessment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLFloodModel:
    """
    Machine Learning-based flood risk prediction using XGBoost

    This model:
    - Trains on historical weather and flood data
    - Predicts risk level with confidence scores
    - Provides feature importance insights
    - Supports model versioning and updates
    """

    # Feature names used by the model
    FEATURE_NAMES = [
        # Rainfall features
        'rainfall_1d',
        'rainfall_3d',
        'rainfall_7d',
        'rainfall_14d',
        'rainfall_30d',
        'rainy_days_7d',
        'heavy_rain_days_7d',
        'max_daily_rainfall_7d',
        'rainfall_intensity_ratio',

        # Temperature features
        'temp_avg',
        'temp_avg_7d',
        'temp_avg_14d',
        'temp_avg_30d',
        'temp_variance_7d',
        'temp_variance_30d',
        'temp_range',

        # Wind features
        'wind_speed_max_7d',
        'wind_speed_avg_7d',
        'high_wind_days_7d',

        # Geographic features
        'elevation',
        'latitude',
        'longitude',

        # Historical features
        'historical_high_risk_count',
        'region_vulnerability_score',

        # Derived features
        'soil_moisture_proxy',
        'evapotranspiration_estimate',
        'flood_risk_indicator',

        # Seasonal features
        'is_typhoon_season',
        'is_wet_season',
        'month',
        'day_of_week'
    ]

    # Risk level mapping
    LABEL_MAPPING = {
        0: FloodRiskLevel.LOW,
        1: FloodRiskLevel.MEDIUM,
        2: FloodRiskLevel.HIGH,
        3: FloodRiskLevel.CRITICAL
    }

    REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML flood model

        Args:
            model_path: Path to load existing model from
        """
        self.model: Optional[XGBClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_names = self.FEATURE_NAMES
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.training_metrics_: Optional[Dict[str, Any]] = None
        self.version = "v2.0"
        self.trained_at: Optional[datetime] = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded existing model from {model_path}")
        else:
            logger.info("Initialized new ML flood model")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate features for training or prediction

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with prepared features
        """
        logger.debug(f"Preparing features for {len(df)} samples")

        # Ensure all required features exist
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with defaults.")
            for feature in missing_features:
                if feature.startswith('is_'):
                    df[feature] = 0
                else:
                    df[feature] = 0.0

        # Select and order features
        X = df[self.feature_names].copy()

        # Handle missing values
        X = X.fillna(0)

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        # Ensure correct dtypes
        for col in X.columns:
            if col.startswith('is_') or col in ['rainy_days_7d', 'heavy_rain_days_7d', 'high_wind_days_7d', 'month', 'day_of_week', 'historical_high_risk_count']:
                X[col] = X[col].astype(int)
            else:
                X[col] = X[col].astype(float)

        return X

    def create_labels_from_rules(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create training labels using rule-based approach
        (Used when historical flood data is not available)

        Args:
            df: DataFrame with features

        Returns:
            Array of labels (0-3)
        """
        logger.info("Creating labels from rule-based approach")

        labels = []
        for _, row in df.iterrows():
            # Simplified rule-based labeling
            score = 0

            # Rainfall contribution
            if row.get('rainfall_1d', 0) > 150 or row.get('rainfall_7d', 0) > 400:
                score += 40
            elif row.get('rainfall_1d', 0) > 100 or row.get('rainfall_7d', 0) > 250:
                score += 30
            elif row.get('rainfall_1d', 0) > 50 or row.get('rainfall_7d', 0) > 150:
                score += 15

            # Elevation contribution
            if row.get('elevation', 100) < 50:
                score += 15
            elif row.get('elevation', 100) < 100:
                score += 7

            # Historical contribution
            if row.get('region_vulnerability_score', 0) > 70:
                score += 10

            # Determine label
            if score >= 65:
                labels.append(3)  # critical
            elif score >= 45:
                labels.append(2)  # high
            elif score >= 25:
                labels.append(1)  # medium
            else:
                labels.append(0)  # low

        return np.array(labels)

    def create_labels_from_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Create labels from actual flood occurrence data
        (Preferred when historical flood data is available)

        Args:
            df: DataFrame with 'flood_occurred' or 'risk_level' column

        Returns:
            Array of labels or None if no label data
        """
        if 'risk_level' in df.columns:
            logger.info("Creating labels from risk_level column")
            return df['risk_level'].map(self.REVERSE_LABEL_MAPPING).values

        elif 'flood_occurred' in df.columns:
            logger.info("Creating labels from flood_occurred column")
            # Simple binary to multi-class conversion
            labels = []
            for _, row in df.iterrows():
                if row['flood_occurred']:
                    # Use severity if available, otherwise mark as high
                    severity = row.get('flood_severity', 2)
                    labels.append(min(severity, 3))
                else:
                    labels.append(0)  # low risk
            return np.array(labels)

        return None

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        **xgb_params
    ) -> Dict[str, Any]:
        """
        Train the XGBoost classifier

        Args:
            X: Feature DataFrame
            y: Target labels
            test_size: Proportion of data for testing
            random_state: Random seed
            **xgb_params: Additional XGBoost parameters

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training ML flood model on {len(X)} samples")

        # Prepare features
        X_prepared = self.prepare_features(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        # Default XGBoost parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softmax',
            'num_class': 4,
            'random_state': random_state,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'enable_categorical': False
        }

        # Override with custom parameters
        default_params.update(xgb_params)

        # Initialize and train model
        self.model = XGBClassifier(**default_params)

        logger.info("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Classification report
        target_names = [level.value for level in self.LABEL_MAPPING.values()]
        class_report = classification_report(
            y_test, y_pred_test,
            target_names=target_names,
            output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test)

        # Feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        self.feature_importance_ = feature_importance

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_prepared, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='accuracy'
        )

        # Store metrics
        self.training_metrics_ = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': np.bincount(y).tolist()
        }

        self.trained_at = datetime.now()

        # Log results
        logger.info(f"Training completed!")
        logger.info(f"Train accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        logger.info("\nTop 10 Important Features:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {feature}: {importance:.4f}")

        return self.training_metrics_

    def predict(
        self,
        features: Dict[str, Any],
        return_probabilities: bool = False
    ) -> FloodRiskAssessment:
        """
        Predict flood risk for given features

        Args:
            features: Dictionary of feature values
            return_probabilities: Whether to include probabilities in metadata

        Returns:
            FloodRiskAssessment with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Prepare features
        X = self.prepare_features(df)

        # Get prediction and probabilities
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]

        # Map to risk level
        risk_level = self.LABEL_MAPPING[pred_class]
        confidence = float(pred_proba[pred_class])

        # Generate recommendation
        recommendations = {
            FloodRiskLevel.LOW: "Normal conditions. Continue regular farming schedule. Maintain awareness of weather forecasts.",
            FloodRiskLevel.MEDIUM: "CAUTION: Moderate flood risk. Monitor weather conditions closely. Plan contingency measures.",
            FloodRiskLevel.HIGH: "WARNING: Elevated flood risk. Prepare for early harvest within 24-48 hours. Secure equipment.",
            FloodRiskLevel.CRITICAL: "URGENT: High flood risk. Harvest crops immediately if possible. Implement emergency measures."
        }

        # Get top contributing features
        if self.feature_importance_:
            top_features = {
                k: float(v) for k, v in
                sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        else:
            top_features = {}

        # Create assessment
        assessment = FloodRiskAssessment(
            region_id=features.get('region_id', 'unknown'),
            assessment_date=features.get('assessment_date', date.today()),
            risk_level=risk_level,
            confidence_score=round(confidence, 3),
            risk_score=round(pred_class * 33.33, 2),  # Convert 0-3 to 0-100 scale
            contributing_factors=top_features,
            recommendation=recommendations[risk_level],
            triggered_rules=[],  # ML model doesn't use explicit rules
            model_version=f"{self.version}_ml_xgboost",
            metadata={
                'probabilities': {
                    level.value: round(float(prob), 4)
                    for level, prob in zip(self.LABEL_MAPPING.values(), pred_proba)
                } if return_probabilities else {},
                'predicted_class': int(pred_class),
                'model_trained_at': self.trained_at.isoformat() if self.trained_at else None
            }
        )

        return assessment

    def batch_predict(
        self,
        features_df: pd.DataFrame,
        return_probabilities: bool = False
    ) -> List[FloodRiskAssessment]:
        """
        Predict flood risk for multiple samples

        Args:
            features_df: DataFrame with features for multiple regions
            return_probabilities: Whether to include probabilities

        Returns:
            List of FloodRiskAssessment results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        logger.info(f"Running batch prediction for {len(features_df)} samples")

        # Prepare all features at once
        X = self.prepare_features(features_df)

        # Get predictions
        pred_classes = self.model.predict(X)
        pred_probas = self.model.predict_proba(X)

        # Create assessments
        assessments = []
        for idx, (_, row) in enumerate(features_df.iterrows()):
            features_dict = row.to_dict()
            features_dict['_pred_class'] = pred_classes[idx]
            features_dict['_pred_proba'] = pred_probas[idx]

            try:
                assessment = self.predict(
                    features_dict,
                    return_probabilities=return_probabilities
                )
                assessments.append(assessment)
            except Exception as e:
                logger.error(f"Failed to create assessment for row {idx}: {str(e)}")
                continue

        logger.info(f"Completed {len(assessments)} predictions")
        return assessments

    def save_model(self, path: str):
        """
        Save trained model to disk

        Args:
            path: File path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'training_metrics': self.training_metrics_,
            'version': self.version,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load trained model from disk

        Args:
            path: File path to load model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names', self.FEATURE_NAMES)
        self.feature_importance_ = model_data.get('feature_importance')
        self.training_metrics_ = model_data.get('training_metrics')
        self.version = model_data.get('version', 'v2.0')

        trained_at_str = model_data.get('trained_at')
        if trained_at_str:
            self.trained_at = datetime.fromisoformat(trained_at_str)

        logger.info(f"Model loaded from {path}")
        logger.info(f"Model version: {self.version}, Trained: {trained_at_str}")

    def get_feature_importance_report(self, top_n: int = 20) -> str:
        """
        Generate feature importance report

        Args:
            top_n: Number of top features to include

        Returns:
            Formatted report string
        """
        if not self.feature_importance_:
            return "No feature importance data available. Train the model first."

        report = [
            "\nFeature Importance Report",
            "=" * 60,
            f"Model Version: {self.version}",
            f"Trained At: {self.trained_at}",
            "\nTop {} Features:".format(top_n),
            "-" * 60
        ]

        sorted_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        for rank, (feature, importance) in enumerate(sorted_features, 1):
            bar = "█" * int(importance * 50)
            report.append(f"{rank:2d}. {feature:30s} {importance:.4f} {bar}")

        report.append("=" * 60)
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Create sample training data
    print("Creating sample training data...")

    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic features
    sample_data = pd.DataFrame({
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

    # Train model
    print("\nTraining ML Flood Model...")
    model = MLFloodModel()

    # Create labels
    y = model.create_labels_from_rules(sample_data)

    # Train
    metrics = model.train(sample_data, y)

    # Print feature importance
    print(model.get_feature_importance_report())

    # Test prediction
    print("\nTesting prediction...")
    test_features = sample_data.iloc[0].to_dict()
    test_features['region_id'] = 'TEST001'
    test_features['assessment_date'] = date.today()

    assessment = model.predict(test_features, return_probabilities=True)
    print(f"\nPrediction: {assessment.risk_level.value.upper()}")
    print(f"Confidence: {assessment.confidence_score:.2%}")
    print(f"Recommendation: {assessment.recommendation}")
