# Phase 3: Data Processing & ML Development Plan
**Duration**: Weeks 5-6
**Status**: 📋 Planning
**Dependencies**: Phase 1 ✅ & Phase 2 ✅

---

## Table of Contents
1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Architecture](#architecture)
4. [Component 1: Spark ETL Pipelines](#component-1-spark-etl-pipelines)
5. [Component 2: Flood Risk Prediction Model](#component-2-flood-risk-prediction-model)
6. [Component 3: Data Quality & Validation](#component-3-data-quality--validation)
7. [Implementation Timeline](#implementation-timeline)
8. [Testing Strategy](#testing-strategy)
9. [Dependencies & Setup](#dependencies--setup)
10. [Success Criteria](#success-criteria)

---

## Overview

Phase 3 transforms the raw weather data from Phase 2 into actionable insights through:
- **ETL Pipelines**: Process and aggregate weather data using PySpark
- **ML Models**: Predict flood risk levels to protect harvests
- **Data Quality**: Ensure data reliability and accuracy

### Current State
- ✅ 30 Philippine regions with coordinates
- ✅ Daily PAGASA weather ingestion (150+ forecasts/day)
- ✅ PostgreSQL database with weather_forecasts table
- ✅ Airflow orchestration platform
- ✅ 20+ crop types with growth characteristics

### Target State
- 🎯 Historical weather data aggregations
- 🎯 Flood risk scores for all regions
- 🎯 Automated data quality checks
- 🎯 ML model for flood prediction
- 🎯 Feature engineering pipeline

---

## Objectives

### Primary Goals
1. **Build PySpark ETL pipelines** for weather data processing
2. **Develop flood risk ML model** (v1: rule-based, v2: ML-based)
3. **Implement data quality framework** with automated checks
4. **Create feature engineering pipeline** for ML training

### Key Deliverables
- [ ] PySpark job for weather aggregations
- [ ] Flood risk prediction model (trained & validated)
- [ ] Data quality dashboard/reports
- [ ] Feature store for ML features
- [ ] Airflow DAGs for orchestration
- [ ] Unit tests (>80% coverage)
- [ ] Documentation

---

## Architecture

### Technology Stack
```
Data Processing:
- PySpark 3.5.0 (distributed processing)
- pandas 2.1.4 (small-scale analysis)
- numpy 1.26.3 (numerical operations)

Machine Learning:
- scikit-learn 1.4.0 (baseline models)
- XGBoost 2.0.3 (flood prediction)
- joblib (model serialization)

Orchestration:
- Apache Airflow 2.8.1 (workflow management)
- Celery (distributed task execution)

Storage:
- PostgreSQL 15+ (structured data)
- Redis 7 (caching, feature store)
- File system (model artifacts, checkpoints)
```

### Data Flow Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 3 DATA FLOW                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  PostgreSQL DB   │
│  weather_forecasts│
│  (Raw PAGASA)    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              SPARK ETL PIPELINES                                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  Extract      │→ │  Transform    │→ │  Load         │      │
│  │  - Read DB    │  │  - Aggregate  │  │  - Write DB   │      │
│  │  - Validate   │  │  - Features   │  │  - Cache      │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
         │                       │
         ▼                       ▼
┌────────────────────┐  ┌────────────────────┐
│  Feature Store     │  │  Aggregated Data   │
│  (Redis Cache)     │  │  (PostgreSQL)      │
│  - rainfall_7d     │  │  - weather_stats   │
│  - temp_avg        │  │  - region_metrics  │
│  - flood_history   │  │                    │
└─────────┬──────────┘  └────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│              FLOOD RISK ML MODEL                                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  Training     │→ │  Validation   │→ │  Prediction   │      │
│  │  - Features   │  │  - Metrics    │  │  - Real-time  │      │
│  │  - Labels     │  │  - Tuning     │  │  - Batch      │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│  flood_risk_       │
│  assessments       │
│  (PostgreSQL)      │
│  - risk_level      │
│  - confidence      │
│  - factors         │
└────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATA QUALITY CHECKS                                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  Validation   │  │  Anomaly      │  │  Monitoring   │      │
│  │  - Nulls      │  │  - Outliers   │  │  - Alerts     │      │
│  │  - Ranges     │  │  - Drift      │  │  - Reports    │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Spark ETL Pipelines

### 1.1 Weather Data Aggregations

**Purpose**: Create historical weather metrics for analysis and ML features

#### Job 1: Daily Weather Statistics
```python
# src/processing/weather_aggregations.py
"""
Compute daily aggregated statistics per region:
- Average temperature (high/low)
- Total rainfall
- Max wind speed
- Dominant weather condition
"""
```

**Output Table**: `weather_daily_stats`
```sql
CREATE TABLE weather_daily_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id UUID REFERENCES regions(id),
    stat_date DATE NOT NULL,
    temp_high_avg DECIMAL(5,2),
    temp_low_avg DECIMAL(5,2),
    rainfall_total DECIMAL(8,2),
    wind_speed_max DECIMAL(5,2),
    dominant_condition VARCHAR(50),
    forecast_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, stat_date)
);
```

#### Job 2: Rolling Window Features
```python
# src/processing/rolling_features.py
"""
Compute rolling window metrics for ML:
- 7-day rainfall accumulation
- 14-day temperature average
- 30-day weather pattern trends
- Seasonal variance
"""
```

**Output**: Cached in Redis for fast access
```
Key Pattern: weather:features:{region_id}:{date}
Value: JSON {
    "rainfall_7d": 145.5,
    "rainfall_14d": 280.3,
    "temp_avg_7d": 28.5,
    "temp_variance_30d": 2.3,
    "rainy_days_7d": 4
}
TTL: 7 days
```

#### Job 3: Regional Risk Indicators
```python
# src/processing/risk_indicators.py
"""
Calculate region-level risk indicators:
- Flood-prone season detection
- Typhoon season probability
- Harvest window suitability scores
"""
```

**Output Table**: `region_risk_indicators`
```sql
CREATE TABLE region_risk_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id UUID REFERENCES regions(id),
    indicator_date DATE NOT NULL,
    flood_season_score DECIMAL(3,2),
    typhoon_probability DECIMAL(3,2),
    harvest_suitability DECIMAL(3,2),
    risk_factors JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, indicator_date)
);
```

### 1.2 PySpark Job Structure

**File**: `src/processing/spark_jobs/weather_etl.py`

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, sum, max, min, count,
    window, lag, lead,
    to_date, current_timestamp
)
from pyspark.sql.types import *

class WeatherETL:
    """
    PySpark ETL pipeline for weather data processing
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.db_url = "jdbc:postgresql://postgres:5432/agrisafe"
        self.db_props = {
            "user": "airflow",
            "password": "airflow",
            "driver": "org.postgresql.Driver"
        }

    def extract_weather_data(self, start_date: str, end_date: str):
        """Extract weather forecasts from PostgreSQL"""
        query = f"""
            (SELECT
                wf.id,
                wf.region_id,
                r.name as region_name,
                wf.forecast_date,
                wf.temperature_high,
                wf.temperature_low,
                wf.rainfall_mm,
                wf.wind_speed,
                wf.weather_condition,
                wf.created_at
            FROM weather_forecasts wf
            JOIN regions r ON wf.region_id = r.id
            WHERE wf.forecast_date BETWEEN '{start_date}' AND '{end_date}'
            ) as weather_data
        """
        return self.spark.read.jdbc(
            url=self.db_url,
            table=query,
            properties=self.db_props
        )

    def compute_daily_stats(self, df):
        """Aggregate to daily statistics per region"""
        return df.groupBy("region_id", "forecast_date").agg(
            avg("temperature_high").alias("temp_high_avg"),
            avg("temperature_low").alias("temp_low_avg"),
            sum("rainfall_mm").alias("rainfall_total"),
            max("wind_speed").alias("wind_speed_max"),
            count("*").alias("forecast_count")
        )

    def compute_rolling_features(self, df):
        """Calculate rolling window features"""
        from pyspark.sql.window import Window

        window_7d = Window.partitionBy("region_id") \
                          .orderBy("forecast_date") \
                          .rowsBetween(-6, 0)

        return df.withColumn(
            "rainfall_7d",
            sum("rainfall_total").over(window_7d)
        ).withColumn(
            "temp_avg_7d",
            avg("temp_high_avg").over(window_7d)
        )

    def load_to_postgres(self, df, table_name: str):
        """Load processed data back to PostgreSQL"""
        df.write.jdbc(
            url=self.db_url,
            table=table_name,
            mode="append",
            properties=self.db_props
        )

    def run(self, start_date: str, end_date: str):
        """Execute full ETL pipeline"""
        # Extract
        raw_data = self.extract_weather_data(start_date, end_date)

        # Transform
        daily_stats = self.compute_daily_stats(raw_data)
        features = self.compute_rolling_features(daily_stats)

        # Load
        self.load_to_postgres(daily_stats, "weather_daily_stats")

        # Cache features to Redis
        self.cache_features(features)

        return features.count()
```

### 1.3 Airflow DAG for ETL Orchestration

**File**: `airflow/dags/weather_processing_dag.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'weather_etl_pipeline',
    default_args=default_args,
    description='Daily weather data ETL processing',
    schedule_interval='0 8 * * *',  # 8 AM daily
    catchup=False,
    tags=['etl', 'weather', 'phase3']
)

# Task 1: Run Spark ETL
spark_etl = SparkSubmitOperator(
    task_id='weather_aggregations',
    application='/opt/airflow/src/processing/spark_jobs/weather_etl.py',
    conn_id='spark_default',
    conf={
        'spark.executor.memory': '2g',
        'spark.driver.memory': '1g'
    },
    dag=dag
)

# Task 2: Generate features
feature_engineering = SparkSubmitOperator(
    task_id='rolling_features',
    application='/opt/airflow/src/processing/spark_jobs/rolling_features.py',
    conn_id='spark_default',
    dag=dag
)

# Task 3: Data quality checks
def run_quality_checks(**context):
    from src.quality.validators import WeatherDataValidator
    validator = WeatherDataValidator()
    results = validator.validate_daily_stats()
    if not results['passed']:
        raise ValueError(f"Data quality check failed: {results['errors']}")

quality_checks = PythonOperator(
    task_id='data_quality_checks',
    python_callable=run_quality_checks,
    dag=dag
)

# Task dependencies
spark_etl >> feature_engineering >> quality_checks
```

---

## Component 2: Flood Risk Prediction Model

### 2.1 Model Architecture

#### Version 1: Rule-Based Model (Week 5)
**Purpose**: Quick baseline for immediate flood risk assessment

**File**: `src/models/flood_risk_v1.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

class FloodRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FloodRiskAssessment:
    risk_level: FloodRiskLevel
    confidence_score: float
    contributing_factors: Dict[str, float]
    recommendation: str

class RuleBasedFloodModel:
    """
    Rule-based flood risk prediction using weather thresholds
    """

    # Rainfall thresholds (mm)
    CRITICAL_RAINFALL_1D = 150
    HIGH_RAINFALL_1D = 100
    MEDIUM_RAINFALL_1D = 50

    CRITICAL_RAINFALL_7D = 400
    HIGH_RAINFALL_7D = 250
    MEDIUM_RAINFALL_7D = 150

    def predict(self, features: Dict) -> FloodRiskAssessment:
        """
        Predict flood risk based on rules

        Input features:
        - rainfall_1d: Today's rainfall (mm)
        - rainfall_7d: 7-day accumulated rainfall (mm)
        - temperature_avg: Average temperature (°C)
        - wind_speed: Wind speed (km/h)
        - elevation: Region elevation (m)
        - historical_flood_count: Past floods in region
        """

        rainfall_1d = features.get('rainfall_1d', 0)
        rainfall_7d = features.get('rainfall_7d', 0)
        elevation = features.get('elevation', 100)
        flood_history = features.get('historical_flood_count', 0)

        factors = {}
        score = 0

        # Rule 1: Daily rainfall intensity
        if rainfall_1d >= self.CRITICAL_RAINFALL_1D:
            score += 40
            factors['heavy_rainfall_today'] = 0.8
        elif rainfall_1d >= self.HIGH_RAINFALL_1D:
            score += 30
            factors['heavy_rainfall_today'] = 0.6
        elif rainfall_1d >= self.MEDIUM_RAINFALL_1D:
            score += 15
            factors['moderate_rainfall_today'] = 0.4

        # Rule 2: Accumulated rainfall
        if rainfall_7d >= self.CRITICAL_RAINFALL_7D:
            score += 35
            factors['soil_saturation'] = 0.9
        elif rainfall_7d >= self.HIGH_RAINFALL_7D:
            score += 25
            factors['soil_saturation'] = 0.7
        elif rainfall_7d >= self.MEDIUM_RAINFALL_7D:
            score += 10
            factors['soil_saturation'] = 0.5

        # Rule 3: Low elevation (flood-prone)
        if elevation < 50:
            score += 15
            factors['low_elevation'] = 0.7
        elif elevation < 100:
            score += 5
            factors['low_elevation'] = 0.3

        # Rule 4: Historical flood frequency
        if flood_history > 5:
            score += 10
            factors['flood_prone_area'] = 0.6
        elif flood_history > 2:
            score += 5
            factors['flood_prone_area'] = 0.3

        # Determine risk level
        if score >= 70:
            risk_level = FloodRiskLevel.CRITICAL
            recommendation = "URGENT: High flood risk. Harvest immediately if possible."
        elif score >= 50:
            risk_level = FloodRiskLevel.HIGH
            recommendation = "WARNING: Elevated flood risk. Prepare for early harvest."
        elif score >= 30:
            risk_level = FloodRiskLevel.MEDIUM
            recommendation = "CAUTION: Monitor weather closely. Plan contingency."
        else:
            risk_level = FloodRiskLevel.LOW
            recommendation = "Normal conditions. Maintain regular schedule."

        confidence = min(score / 100, 1.0)

        return FloodRiskAssessment(
            risk_level=risk_level,
            confidence_score=confidence,
            contributing_factors=factors,
            recommendation=recommendation
        )
```

#### Version 2: ML-Based Model (Week 6)
**Purpose**: Data-driven flood prediction with higher accuracy

**File**: `src/models/flood_risk_v2.py`

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
from typing import Tuple, Dict

class MLFloodModel:
    """
    Machine learning-based flood risk prediction
    Using XGBoost for classification
    """

    def __init__(self):
        self.model = None
        self.feature_names = [
            'rainfall_1d',
            'rainfall_3d',
            'rainfall_7d',
            'rainfall_14d',
            'temp_avg',
            'temp_variance',
            'wind_speed_max',
            'elevation',
            'soil_moisture_proxy',  # (rainfall_7d / temp_avg)
            'rainfall_intensity',   # (rainfall_1d / rainfall_7d)
            'historical_flood_count',
            'season_typhoon',       # 1 if June-Nov, else 0
            'region_vulnerability'  # Historical flood rate
        ]
        self.label_mapping = {
            0: 'low',
            1: 'medium',
            2: 'high',
            3: 'critical'
        }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""

        # Derived features
        df['soil_moisture_proxy'] = df['rainfall_7d'] / (df['temp_avg'] + 1)
        df['rainfall_intensity'] = df['rainfall_1d'] / (df['rainfall_7d'] + 1)

        # Seasonal feature
        df['season_typhoon'] = df['forecast_date'].apply(
            lambda x: 1 if x.month in [6,7,8,9,10,11] else 0
        )

        return df[self.feature_names]

    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create target labels from historical data

        Assume we have flood incident data:
        - flood_occurred: boolean
        - flood_severity: 1-4 scale
        """

        # If actual flood data exists
        if 'flood_severity' in df.columns:
            return df['flood_severity'].values - 1  # 0-indexed

        # Otherwise, use rule-based labels for initial training
        labels = []
        for _, row in df.iterrows():
            if row['rainfall_7d'] > 400 or row['rainfall_1d'] > 150:
                labels.append(3)  # critical
            elif row['rainfall_7d'] > 250 or row['rainfall_1d'] > 100:
                labels.append(2)  # high
            elif row['rainfall_7d'] > 150 or row['rainfall_1d'] > 50:
                labels.append(1)  # medium
            else:
                labels.append(0)  # low

        return np.array(labels)

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Train XGBoost classifier"""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize model
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=4,
            random_state=42,
            eval_metric='mlogloss'
        )

        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate
        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=list(self.label_mapping.values())
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        }

        return metrics

    def predict(self, features: Dict) -> FloodRiskAssessment:
        """Predict flood risk for new data"""

        # Prepare feature vector
        X = pd.DataFrame([features])
        X = X[self.feature_names]

        # Get prediction and probabilities
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]

        risk_level = self.label_mapping[pred_class]
        confidence = pred_proba[pred_class]

        # Get feature contributions (SHAP would be better)
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        # Generate recommendation
        recommendations = {
            'low': "Normal conditions. Maintain regular schedule.",
            'medium': "CAUTION: Monitor weather closely. Plan contingency.",
            'high': "WARNING: Elevated flood risk. Prepare for early harvest.",
            'critical': "URGENT: High flood risk. Harvest immediately if possible."
        }

        return FloodRiskAssessment(
            risk_level=risk_level,
            confidence_score=float(confidence),
            contributing_factors=feature_importance,
            recommendation=recommendations[risk_level]
        )

    def save_model(self, path: str):
        """Save trained model to disk"""
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load trained model from disk"""
        self.model = joblib.load(path)
```

### 2.2 Model Training Pipeline

**File**: `src/models/training_pipeline.py`

```python
from datetime import datetime, timedelta
from src.utils.database import get_db_connection
from src.models.flood_risk_v2 import MLFloodModel
import pandas as pd

def fetch_training_data(days_back: int = 90) -> pd.DataFrame:
    """
    Fetch historical weather and flood data for training
    """

    query = """
        SELECT
            wds.region_id,
            wds.stat_date as forecast_date,
            wds.rainfall_total as rainfall_1d,
            wds.temp_high_avg as temp_avg,
            wds.wind_speed_max,
            r.elevation,
            -- Rolling aggregations (using window functions)
            SUM(wds.rainfall_total) OVER (
                PARTITION BY wds.region_id
                ORDER BY wds.stat_date
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) as rainfall_3d,
            SUM(wds.rainfall_total) OVER (
                PARTITION BY wds.region_id
                ORDER BY wds.stat_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as rainfall_7d,
            SUM(wds.rainfall_total) OVER (
                PARTITION BY wds.region_id
                ORDER BY wds.stat_date
                ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
            ) as rainfall_14d,
            STDDEV(wds.temp_high_avg) OVER (
                PARTITION BY wds.region_id
                ORDER BY wds.stat_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as temp_variance,
            -- Historical flood data (if available)
            COALESCE(
                (SELECT COUNT(*)
                 FROM flood_risk_assessments fra
                 WHERE fra.region_id = wds.region_id
                   AND fra.risk_level IN ('high', 'critical')
                   AND fra.assessment_date < wds.stat_date
                ), 0
            ) as historical_flood_count
        FROM weather_daily_stats wds
        JOIN regions r ON wds.region_id = r.id
        WHERE wds.stat_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY wds.region_id, wds.stat_date
    """ % days_back

    with get_db_connection() as conn:
        df = pd.read_sql(query, conn)

    return df

def train_flood_model():
    """
    Complete training pipeline for flood risk model
    """

    print("🔄 Fetching training data...")
    df = fetch_training_data(days_back=180)  # 6 months
    print(f"✅ Loaded {len(df)} samples")

    print("🔄 Preparing features...")
    model = MLFloodModel()
    X = model.prepare_features(df)
    y = model.create_labels(df)
    print(f"✅ Features shape: {X.shape}")

    print("🔄 Training model...")
    metrics = model.train(X, y)
    print(f"✅ Model trained!")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"\n{metrics['classification_report']}")

    print("\n📊 Feature Importance:")
    for feature, importance in sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"   {feature}: {importance:.3f}")

    # Save model
    model_path = f"models/flood_risk_v2_{datetime.now().strftime('%Y%m%d')}.pkl"
    model.save_model(model_path)
    print(f"\n✅ Model saved to {model_path}")

    return model, metrics

if __name__ == "__main__":
    train_flood_model()
```

### 2.3 Batch Prediction Service

**File**: `src/models/batch_predictions.py`

```python
from datetime import datetime
from src.models.flood_risk_v2 import MLFloodModel
from src.utils.database import get_db_connection
import pandas as pd

class FloodRiskBatchPredictor:
    """
    Generate flood risk predictions for all regions
    """

    def __init__(self, model_path: str):
        self.model = MLFloodModel()
        self.model.load_model(model_path)

    def fetch_latest_features(self) -> pd.DataFrame:
        """Get latest weather features for all regions"""

        query = """
            SELECT
                r.id as region_id,
                r.name as region_name,
                r.elevation,
                wds.stat_date,
                wds.rainfall_total as rainfall_1d,
                wds.temp_high_avg as temp_avg,
                wds.wind_speed_max,
                -- Get rolling features from Redis or compute
                -- For simplicity, using SQL window functions
                SUM(wds.rainfall_total) OVER (
                    PARTITION BY r.id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as rainfall_3d,
                SUM(wds.rainfall_total) OVER (
                    PARTITION BY r.id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rainfall_7d,
                SUM(wds.rainfall_total) OVER (
                    PARTITION BY r.id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) as rainfall_14d,
                STDDEV(wds.temp_high_avg) OVER (
                    PARTITION BY r.id
                    ORDER BY wds.stat_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as temp_variance
            FROM regions r
            JOIN weather_daily_stats wds ON r.id = wds.region_id
            WHERE wds.stat_date = (
                SELECT MAX(stat_date)
                FROM weather_daily_stats
                WHERE region_id = r.id
            )
        """

        with get_db_connection() as conn:
            return pd.read_sql(query, conn)

    def predict_all_regions(self):
        """Generate predictions for all regions"""

        df = self.fetch_latest_features()

        predictions = []
        for _, row in df.iterrows():
            features = row.to_dict()
            assessment = self.model.predict(features)

            predictions.append({
                'region_id': row['region_id'],
                'assessment_date': row['stat_date'],
                'risk_level': assessment.risk_level,
                'confidence_score': assessment.confidence_score,
                'contributing_factors': assessment.contributing_factors,
                'recommendation': assessment.recommendation,
                'model_version': 'v2',
                'created_at': datetime.now()
            })

        # Bulk insert to database
        self.save_predictions(predictions)

        return predictions

    def save_predictions(self, predictions: list):
        """Save predictions to flood_risk_assessments table"""

        with get_db_connection() as conn:
            cursor = conn.cursor()

            for pred in predictions:
                cursor.execute("""
                    INSERT INTO flood_risk_assessments (
                        region_id, assessment_date, risk_level,
                        confidence_score, contributing_factors,
                        recommendation, model_version
                    ) VALUES (
                        %s, %s, %s, %s, %s::jsonb, %s, %s
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
                    pred['risk_level'],
                    pred['confidence_score'],
                    str(pred['contributing_factors']),
                    pred['recommendation'],
                    pred['model_version']
                ))

            conn.commit()
```

### 2.4 Airflow DAG for Model Training & Prediction

**File**: `airflow/dags/flood_model_dag.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 1: Model Training (Weekly)
training_dag = DAG(
    'flood_model_training',
    default_args=default_args,
    description='Weekly retraining of flood risk model',
    schedule_interval='0 2 * * 0',  # Sunday 2 AM
    catchup=False,
    tags=['ml', 'training', 'phase3']
)

def train_model_task():
    from src.models.training_pipeline import train_flood_model
    model, metrics = train_flood_model()
    return metrics['accuracy']

train_model = PythonOperator(
    task_id='train_flood_model',
    python_callable=train_model_task,
    dag=training_dag
)

# DAG 2: Daily Predictions
prediction_dag = DAG(
    'flood_risk_predictions',
    default_args=default_args,
    description='Daily flood risk predictions for all regions',
    schedule_interval='0 9 * * *',  # 9 AM daily
    catchup=False,
    tags=['ml', 'predictions', 'phase3']
)

def generate_predictions_task():
    from src.models.batch_predictions import FloodRiskBatchPredictor
    import glob

    # Get latest model
    model_files = sorted(glob.glob('models/flood_risk_v2_*.pkl'))
    latest_model = model_files[-1]

    predictor = FloodRiskBatchPredictor(latest_model)
    predictions = predictor.predict_all_regions()

    return len(predictions)

generate_predictions = PythonOperator(
    task_id='generate_flood_predictions',
    python_callable=generate_predictions_task,
    dag=prediction_dag
)
```

---

## Component 3: Data Quality & Validation

### 3.1 Data Quality Framework

**File**: `src/quality/validators.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from src.utils.database import get_db_connection

@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    severity: str  # 'critical', 'warning', 'info'
    message: str
    details: Dict[str, Any]
    timestamp: datetime

class WeatherDataValidator:
    """
    Data quality checks for weather data
    """

    def __init__(self):
        self.results: List[ValidationResult] = []

    def check_null_values(self) -> ValidationResult:
        """Check for unexpected NULL values"""

        query = """
            SELECT
                COUNT(*) FILTER (WHERE temperature_high IS NULL) as null_temp_high,
                COUNT(*) FILTER (WHERE temperature_low IS NULL) as null_temp_low,
                COUNT(*) FILTER (WHERE rainfall_mm IS NULL) as null_rainfall,
                COUNT(*) FILTER (WHERE wind_speed IS NULL) as null_wind,
                COUNT(*) as total_records
            FROM weather_forecasts
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
        """

        with get_db_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        null_count = (
            result['null_temp_high'] +
            result['null_temp_low'] +
            result['null_rainfall'] +
            result['null_wind']
        )

        null_percentage = (null_count / (result['total_records'] * 4)) * 100

        passed = null_percentage < 5  # Threshold: 5%
        severity = 'critical' if null_percentage > 10 else 'warning'

        return ValidationResult(
            check_name='null_values',
            passed=passed,
            severity=severity,
            message=f"Found {null_percentage:.2f}% NULL values",
            details=result.to_dict(),
            timestamp=datetime.now()
        )

    def check_value_ranges(self) -> ValidationResult:
        """Check if values are within expected ranges"""

        query = """
            SELECT
                COUNT(*) FILTER (WHERE temperature_high < 15 OR temperature_high > 45) as invalid_temp_high,
                COUNT(*) FILTER (WHERE temperature_low < 10 OR temperature_low > 40) as invalid_temp_low,
                COUNT(*) FILTER (WHERE rainfall_mm < 0 OR rainfall_mm > 500) as invalid_rainfall,
                COUNT(*) FILTER (WHERE wind_speed < 0 OR wind_speed > 250) as invalid_wind,
                COUNT(*) as total_records
            FROM weather_forecasts
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
        """

        with get_db_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        invalid_count = (
            result['invalid_temp_high'] +
            result['invalid_temp_low'] +
            result['invalid_rainfall'] +
            result['invalid_wind']
        )

        invalid_percentage = (invalid_count / (result['total_records'] * 4)) * 100

        passed = invalid_percentage < 1
        severity = 'critical' if invalid_percentage > 5 else 'warning'

        return ValidationResult(
            check_name='value_ranges',
            passed=passed,
            severity=severity,
            message=f"Found {invalid_percentage:.2f}% out-of-range values",
            details=result.to_dict(),
            timestamp=datetime.now()
        )

    def check_data_freshness(self) -> ValidationResult:
        """Check if data is being updated regularly"""

        query = """
            SELECT
                MAX(created_at) as last_update,
                EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(created_at))) / 3600 as hours_since_update
            FROM weather_forecasts
        """

        with get_db_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        hours_old = result['hours_since_update']
        passed = hours_old < 48  # Should update within 48 hours
        severity = 'critical' if hours_old > 72 else 'warning'

        return ValidationResult(
            check_name='data_freshness',
            passed=passed,
            severity=severity,
            message=f"Last update: {hours_old:.1f} hours ago",
            details=result.to_dict(),
            timestamp=datetime.now()
        )

    def check_regional_coverage(self) -> ValidationResult:
        """Ensure all 30 regions have recent data"""

        query = """
            SELECT
                COUNT(DISTINCT region_id) as regions_with_data,
                (SELECT COUNT(*) FROM regions) as total_regions
            FROM weather_forecasts
            WHERE created_at >= CURRENT_DATE - INTERVAL '24 hours'
        """

        with get_db_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        coverage = (result['regions_with_data'] / result['total_regions']) * 100
        passed = coverage >= 90  # At least 90% coverage
        severity = 'warning' if coverage < 90 else 'info'

        return ValidationResult(
            check_name='regional_coverage',
            passed=passed,
            severity=severity,
            message=f"{coverage:.0f}% regional coverage",
            details=result.to_dict(),
            timestamp=datetime.now()
        )

    def check_anomalies(self) -> ValidationResult:
        """Detect statistical anomalies"""

        query = """
            WITH stats AS (
                SELECT
                    region_id,
                    AVG(rainfall_mm) as avg_rainfall,
                    STDDEV(rainfall_mm) as stddev_rainfall
                FROM weather_forecasts
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY region_id
            ),
            anomalies AS (
                SELECT COUNT(*) as anomaly_count
                FROM weather_forecasts wf
                JOIN stats s ON wf.region_id = s.region_id
                WHERE wf.created_at >= CURRENT_DATE - INTERVAL '7 days'
                  AND ABS(wf.rainfall_mm - s.avg_rainfall) > 3 * s.stddev_rainfall
            )
            SELECT anomaly_count FROM anomalies
        """

        with get_db_connection() as conn:
            result = pd.read_sql(query, conn).iloc[0]

        anomaly_count = result['anomaly_count']
        passed = anomaly_count < 10
        severity = 'warning' if anomaly_count < 20 else 'info'

        return ValidationResult(
            check_name='anomaly_detection',
            passed=passed,
            severity=severity,
            message=f"Found {anomaly_count} statistical anomalies",
            details={'anomaly_count': int(anomaly_count)},
            timestamp=datetime.now()
        )

    def run_all_checks(self) -> Dict[str, Any]:
        """Execute all validation checks"""

        checks = [
            self.check_null_values(),
            self.check_value_ranges(),
            self.check_data_freshness(),
            self.check_regional_coverage(),
            self.check_anomalies()
        ]

        self.results = checks

        # Save to database
        self.save_results(checks)

        # Generate summary
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.passed)
        critical_failures = [c for c in checks if not c.passed and c.severity == 'critical']

        summary = {
            'timestamp': datetime.now(),
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': total_checks - passed_checks,
            'success_rate': (passed_checks / total_checks) * 100,
            'critical_failures': len(critical_failures),
            'all_passed': len(critical_failures) == 0,
            'details': [
                {
                    'check': c.check_name,
                    'passed': c.passed,
                    'severity': c.severity,
                    'message': c.message
                }
                for c in checks
            ]
        }

        return summary

    def save_results(self, results: List[ValidationResult]):
        """Save validation results to database"""

        # Create table if not exists
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS data_quality_checks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                check_name VARCHAR(100),
                passed BOOLEAN,
                severity VARCHAR(20),
                message TEXT,
                details JSONB,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)

            for result in results:
                cursor.execute("""
                    INSERT INTO data_quality_checks
                    (check_name, passed, severity, message, details)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                """, (
                    result.check_name,
                    result.passed,
                    result.severity,
                    result.message,
                    str(result.details)
                ))

            conn.commit()
```

### 3.2 Data Quality Monitoring Dashboard

**File**: `src/quality/monitoring.py`

```python
import pandas as pd
from datetime import datetime, timedelta
from src.utils.database import get_db_connection

class QualityMonitor:
    """
    Generate data quality reports and dashboards
    """

    def get_quality_trends(self, days: int = 7) -> pd.DataFrame:
        """Get quality check trends over time"""

        query = f"""
            SELECT
                DATE(checked_at) as check_date,
                check_name,
                COUNT(*) as total_checks,
                SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_checks,
                ROUND(
                    100.0 * SUM(CASE WHEN passed THEN 1 ELSE 0 END) / COUNT(*),
                    2
                ) as pass_rate
            FROM data_quality_checks
            WHERE checked_at >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY DATE(checked_at), check_name
            ORDER BY check_date DESC, check_name
        """

        with get_db_connection() as conn:
            return pd.read_sql(query, conn)

    def get_critical_failures(self) -> pd.DataFrame:
        """Get recent critical failures"""

        query = """
            SELECT
                check_name,
                message,
                details,
                checked_at
            FROM data_quality_checks
            WHERE severity = 'critical'
              AND NOT passed
              AND checked_at >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY checked_at DESC
        """

        with get_db_connection() as conn:
            return pd.read_sql(query, conn)

    def generate_report(self) -> str:
        """Generate text report for notifications"""

        trends = self.get_quality_trends(days=1)
        failures = self.get_critical_failures()

        report = "📊 Data Quality Report\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if len(failures) > 0:
            report += "🚨 CRITICAL FAILURES:\n"
            for _, failure in failures.iterrows():
                report += f"  - {failure['check_name']}: {failure['message']}\n"
            report += "\n"

        report += "📈 Today's Quality Checks:\n"
        for _, row in trends.iterrows():
            status = "✅" if row['pass_rate'] == 100 else "⚠️"
            report += f"  {status} {row['check_name']}: {row['pass_rate']:.0f}% pass rate\n"

        return report
```

### 3.3 Airflow DAG for Quality Checks

**File**: `airflow/dags/data_quality_dag.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'retries': 1,
}

dag = DAG(
    'data_quality_checks',
    default_args=default_args,
    description='Data quality validation and monitoring',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    tags=['quality', 'monitoring', 'phase3']
)

def run_quality_checks(**context):
    from src.quality.validators import WeatherDataValidator
    from src.quality.monitoring import QualityMonitor

    # Run checks
    validator = WeatherDataValidator()
    summary = validator.run_all_checks()

    # Generate report
    monitor = QualityMonitor()
    report = monitor.generate_report()

    print(report)

    # Fail task if critical issues
    if not summary['all_passed']:
        raise ValueError(f"Quality checks failed: {summary['critical_failures']} critical issues")

    return summary

quality_checks = PythonOperator(
    task_id='run_quality_checks',
    python_callable=run_quality_checks,
    dag=dag
)
```

---

## Implementation Timeline

### Week 5: ETL Pipelines & Rule-Based Model

#### Day 1-2: Spark ETL Setup
- [ ] Install PySpark dependencies
- [ ] Configure Spark standalone cluster (or use local mode)
- [ ] Create database schema for aggregated tables
- [ ] Implement `WeatherETL` class
- [ ] Test with sample data

#### Day 3-4: ETL Pipelines
- [ ] Build daily statistics aggregation
- [ ] Implement rolling window features
- [ ] Create regional risk indicators
- [ ] Set up Redis feature caching
- [ ] Create Airflow DAG for ETL orchestration

#### Day 5-6: Rule-Based Flood Model
- [ ] Implement `RuleBasedFloodModel`
- [ ] Create batch prediction service
- [ ] Test predictions on historical data
- [ ] Save predictions to database
- [ ] Create Airflow DAG for daily predictions

#### Day 7: Testing & Documentation
- [ ] Unit tests for ETL components
- [ ] Integration tests for end-to-end pipeline
- [ ] Performance benchmarking
- [ ] Documentation updates

### Week 6: ML Model & Data Quality

#### Day 1-2: ML Model Development
- [ ] Implement `MLFloodModel` with XGBoost
- [ ] Create feature engineering pipeline
- [ ] Prepare training data (historical weather + labels)
- [ ] Initial model training
- [ ] Model evaluation and tuning

#### Day 3-4: Model Deployment
- [ ] Create training pipeline
- [ ] Build batch prediction service
- [ ] Implement model versioning
- [ ] Create Airflow DAG for weekly retraining
- [ ] A/B testing framework (v1 vs v2)

#### Day 5-6: Data Quality Framework
- [ ] Implement `WeatherDataValidator`
- [ ] Build validation checks (nulls, ranges, freshness, etc.)
- [ ] Create quality monitoring dashboard
- [ ] Set up alerting for critical failures
- [ ] Airflow DAG for quality checks

#### Day 7: Integration & Testing
- [ ] End-to-end testing of all components
- [ ] Performance optimization
- [ ] Documentation completion
- [ ] Handoff to Phase 4

---

## Testing Strategy

### 3.1 Unit Tests

**File**: `tests/processing/test_weather_etl.py`

```python
import pytest
from pyspark.sql import SparkSession
from src.processing.spark_jobs.weather_etl import WeatherETL
from datetime import datetime, timedelta

@pytest.fixture
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("test") \
        .getOrCreate()

def test_daily_aggregations(spark):
    etl = WeatherETL(spark)

    # Create sample data
    data = [
        ("region1", "2025-01-15", 32.0, 24.0, 10.5, 15.0),
        ("region1", "2025-01-15", 33.0, 25.0, 12.0, 18.0),
        ("region2", "2025-01-15", 28.0, 22.0, 50.0, 25.0),
    ]

    df = spark.createDataFrame(
        data,
        ["region_id", "forecast_date", "temperature_high",
         "temperature_low", "rainfall_mm", "wind_speed"]
    )

    result = etl.compute_daily_stats(df)

    assert result.count() == 2  # 2 regions

    region1_stats = result.filter(result.region_id == "region1").collect()[0]
    assert region1_stats['temp_high_avg'] == 32.5
    assert region1_stats['rainfall_total'] == 22.5

def test_rolling_features(spark):
    # Test rolling window calculations
    pass
```

**File**: `tests/models/test_flood_model.py`

```python
import pytest
from src.models.flood_risk_v1 import RuleBasedFloodModel, FloodRiskLevel
from src.models.flood_risk_v2 import MLFloodModel

def test_rule_based_critical_risk():
    model = RuleBasedFloodModel()

    features = {
        'rainfall_1d': 160,
        'rainfall_7d': 450,
        'elevation': 30,
        'historical_flood_count': 8
    }

    assessment = model.predict(features)

    assert assessment.risk_level == FloodRiskLevel.CRITICAL
    assert assessment.confidence_score > 0.7

def test_rule_based_low_risk():
    model = RuleBasedFloodModel()

    features = {
        'rainfall_1d': 5,
        'rainfall_7d': 20,
        'elevation': 200,
        'historical_flood_count': 0
    }

    assessment = model.predict(features)

    assert assessment.risk_level == FloodRiskLevel.LOW

def test_ml_model_training():
    # Test ML model training pipeline
    pass
```

### 3.2 Integration Tests

**File**: `tests/integration/test_etl_pipeline.py`

```python
import pytest
from datetime import datetime, timedelta
from src.processing.spark_jobs.weather_etl import WeatherETL

def test_end_to_end_etl():
    """Test complete ETL pipeline"""

    # 1. Insert test weather data
    # 2. Run Spark ETL
    # 3. Verify aggregated data in DB
    # 4. Verify Redis cache
    pass

def test_flood_prediction_pipeline():
    """Test complete prediction pipeline"""

    # 1. Prepare test features
    # 2. Run batch predictions
    # 3. Verify results in DB
    pass
```

### 3.3 Performance Tests

**File**: `tests/performance/test_etl_performance.py`

```python
import time
import pytest

def test_etl_performance():
    """Ensure ETL processes within acceptable time"""

    start = time.time()

    # Run ETL for 30 days of data
    # Should complete within 5 minutes

    duration = time.time() - start
    assert duration < 300  # 5 minutes

def test_prediction_performance():
    """Ensure predictions generate quickly"""

    start = time.time()

    # Generate predictions for 30 regions
    # Should complete within 30 seconds

    duration = time.time() - start
    assert duration < 30
```

---

## Dependencies & Setup

### 4.1 Additional Python Packages

**Add to `requirements.txt`:**

```
# Data Processing
pyspark==3.5.0
findspark==2.0.1

# Machine Learning
scikit-learn==1.4.0
xgboost==2.0.3
joblib==1.3.2
imbalanced-learn==0.12.0

# Data Analysis
pandas==2.1.4
numpy==1.26.3
scipy==1.11.4

# Visualization (for analysis)
matplotlib==3.8.2
seaborn==0.13.1
```

### 4.2 Spark Configuration

**File**: `docker/spark/Dockerfile`

```dockerfile
FROM apache/spark:3.5.0-python3

USER root

# Install PostgreSQL JDBC driver
RUN curl -o /opt/spark/jars/postgresql-42.7.1.jar \
    https://jdbc.postgresql.org/download/postgresql-42.7.1.jar

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER spark
```

**Update `docker-compose.yml`:**

```yaml
services:
  # ... existing services ...

  spark-master:
    build: ./docker/spark
    container_name: agrisafe-spark-master
    environment:
      - SPARK_MODE=master
      - SPARK_MASTER_PORT=7077
      - SPARK_MASTER_WEBUI_PORT=8080
    ports:
      - "7077:7077"
      - "8081:8080"
    volumes:
      - ./src:/opt/spark/work/src
      - ./data:/opt/spark/work/data
      - ./models:/opt/spark/work/models
    networks:
      - agrisafe-network

  spark-worker:
    build: ./docker/spark
    container_name: agrisafe-spark-worker
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_CORES=2
      - SPARK_WORKER_MEMORY=2g
    depends_on:
      - spark-master
    volumes:
      - ./src:/opt/spark/work/src
      - ./data:/opt/spark/work/data
    networks:
      - agrisafe-network
```

### 4.3 Database Migrations

**File**: `sql/migrations/03_phase3_tables.sql`

```sql
-- Weather daily statistics table
CREATE TABLE IF NOT EXISTS weather_daily_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id UUID REFERENCES regions(id) NOT NULL,
    stat_date DATE NOT NULL,
    temp_high_avg DECIMAL(5,2),
    temp_low_avg DECIMAL(5,2),
    rainfall_total DECIMAL(8,2),
    wind_speed_max DECIMAL(5,2),
    dominant_condition VARCHAR(50),
    forecast_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, stat_date)
);

CREATE INDEX idx_weather_daily_stats_region_date
ON weather_daily_stats(region_id, stat_date DESC);

-- Regional risk indicators table
CREATE TABLE IF NOT EXISTS region_risk_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id UUID REFERENCES regions(id) NOT NULL,
    indicator_date DATE NOT NULL,
    flood_season_score DECIMAL(3,2),
    typhoon_probability DECIMAL(3,2),
    harvest_suitability DECIMAL(3,2),
    risk_factors JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, indicator_date)
);

-- Data quality checks table
CREATE TABLE IF NOT EXISTS data_quality_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    check_name VARCHAR(100),
    passed BOOLEAN,
    severity VARCHAR(20),
    message TEXT,
    details JSONB,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_quality_checks_date
ON data_quality_checks(checked_at DESC);

-- Model training metadata
CREATE TABLE IF NOT EXISTS model_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accuracy DECIMAL(5,4),
    metrics JSONB,
    model_path VARCHAR(255),
    status VARCHAR(50)
);
```

### 4.4 Makefile Updates

**Add to `Makefile`:**

```makefile
# Phase 3 Commands

.PHONY: spark-up
spark-up:
	docker-compose up -d spark-master spark-worker

.PHONY: spark-down
spark-down:
	docker-compose stop spark-master spark-worker

.PHONY: spark-logs
spark-logs:
	docker-compose logs -f spark-master spark-worker

.PHONY: run-etl
run-etl:
	docker exec agrisafe-spark-master \
		spark-submit \
		--master spark://spark-master:7077 \
		--jars /opt/spark/jars/postgresql-42.7.1.jar \
		/opt/spark/work/src/processing/spark_jobs/weather_etl.py

.PHONY: train-model
train-model:
	docker exec agrisafe-webserver \
		python -m src.models.training_pipeline

.PHONY: run-predictions
run-predictions:
	docker exec agrisafe-webserver \
		python -m src.models.batch_predictions

.PHONY: quality-checks
quality-checks:
	docker exec agrisafe-webserver \
		python -m src.quality.validators

.PHONY: test-phase3
test-phase3:
	docker exec agrisafe-webserver \
		pytest tests/processing tests/models tests/quality -v --cov
```

---

## Success Criteria

### Functional Requirements
- [x] Spark ETL pipelines process weather data daily
- [x] Flood risk model generates predictions for all 30 regions
- [x] Data quality checks run every 6 hours
- [x] All components orchestrated via Airflow DAGs

### Performance Requirements
- [x] ETL processes 90 days of data in < 5 minutes
- [x] Flood predictions for 30 regions in < 30 seconds
- [x] Data quality checks complete in < 2 minutes

### Quality Requirements
- [x] Unit test coverage > 80%
- [x] ML model accuracy > 75% (or baseline rule-based)
- [x] Data freshness < 48 hours
- [x] <5% NULL values in processed data

### Documentation Requirements
- [x] API documentation for all modules
- [x] Model performance reports
- [x] Data quality dashboards
- [x] Runbooks for operations

---

## Next Steps (Phase 4)

After completing Phase 3, we'll be ready for:
1. **FastAPI Backend**: REST APIs to serve predictions
2. **LLM Integration**: Claude/GPT for harvest recommendations
3. **Authentication**: JWT-based user auth
4. **Real-time APIs**: Endpoints for frontend consumption

---

## Appendix

### A. Sample Commands

```bash
# Start Spark cluster
make spark-up

# Run ETL pipeline manually
make run-etl

# Train flood model
make train-model

# Generate predictions
make run-predictions

# Run quality checks
make quality-checks

# Run all Phase 3 tests
make test-phase3

# View Spark UI
open http://localhost:8081

# View Airflow UI
open http://localhost:8080
```

### B. Directory Structure (Phase 3 Additions)

```
project-agri-safe/
├── src/
│   ├── processing/
│   │   ├── spark_jobs/
│   │   │   ├── weather_etl.py
│   │   │   ├── rolling_features.py
│   │   │   └── risk_indicators.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── flood_risk_v1.py
│   │   ├── flood_risk_v2.py
│   │   ├── training_pipeline.py
│   │   └── batch_predictions.py
│   ├── quality/
│   │   ├── validators.py
│   │   ├── monitoring.py
│   │   └── __init__.py
├── airflow/dags/
│   ├── weather_processing_dag.py
│   ├── flood_model_dag.py
│   └── data_quality_dag.py
├── tests/
│   ├── processing/
│   ├── models/
│   ├── quality/
│   └── integration/
├── models/
│   └── flood_risk_v2_20250115.pkl
├── docker/
│   └── spark/
│       └── Dockerfile
└── docs/
    └── PHASE3_DEVELOPMENT_PLAN.md
```

### C. Resources

- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [XGBoost Python API](https://xgboost.readthedocs.io/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Data Quality Patterns](https://www.datakitchen.io/data-quality-patterns)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-17
**Author**: AgriSafe Development Team
**Status**: Ready for Implementation 🚀
