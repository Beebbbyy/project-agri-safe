# Phase 3 Airflow DAGs - Documentation

This document provides an overview of the three comprehensive Airflow DAGs created for Phase 3 orchestration.

## Overview

Three production-ready DAGs have been implemented to orchestrate Phase 3 data processing, ML operations, and quality monitoring:

1. **`weather_processing_dag.py`** - Daily weather ETL and feature engineering
2. **`flood_model_dag.py`** - Model training and prediction generation
3. **`data_quality_dag.py`** - Comprehensive data quality monitoring

---

## 1. Weather Data Processing DAG

**File:** `/home/user/project-agri-safe/airflow/dags/weather_processing_dag.py`

### Purpose
Orchestrates daily weather data processing, feature engineering, and quality validation.

### Schedule
- **Daily at 8:00 AM UTC (4:00 PM PHT)**
- Cron: `0 8 * * *`

### Tasks

| Task ID | Description | Timeout |
|---------|-------------|---------|
| `run_spark_etl` | Execute Spark ETL for weather aggregations | 15 min |
| `generate_rolling_features` | Generate rolling window features (3d, 7d, 14d, 30d) | 20 min |
| `cache_features_to_redis` | Cache latest features to Redis for fast access | 5 min |
| `data_quality_checks` | Run comprehensive quality validations | 10 min |
| `send_notification` | Send pipeline execution summary | - |
| `cleanup_old_features` | Cleanup obsolete feature data | - |

### Task Dependencies
```
run_spark_etl
    ↓
generate_rolling_features
    ↓
├── cache_features_to_redis ──┐
│                              ├→ send_notification
└── data_quality_checks ───────┘
    ↓
cleanup_old_features
```

### Key Features
- **PySpark Integration**: Distributed processing for large datasets
- **Feature Engineering**: Rainfall, temperature, wind, and derived features
- **Redis Caching**: Fast access to latest features for predictions
- **Quality Validation**: Integrated quality checks after processing
- **Error Handling**: Retry logic and comprehensive error logging
- **XCom**: Data passing between tasks for tracking

### Outputs
- `weather_daily_stats` table - Daily aggregated statistics
- `weather_features` table - Engineered features for ML
- Redis cache - Latest features with 7-day TTL
- Quality check results in `data_quality_checks` table

---

## 2. Flood Model Training & Prediction DAG

**File:** `/home/user/project-agri-safe/airflow/dags/flood_model_dag.py`

This file contains **TWO separate DAGs**:

### 2.1 Model Training DAG

**DAG ID:** `flood_model_training`

#### Schedule
- **Weekly on Sunday at 2:00 AM UTC**
- Cron: `0 2 * * 0`

#### Tasks

| Task ID | Description | Timeout |
|---------|-------------|---------|
| `check_training_readiness` | Verify sufficient data (90+ days) | - |
| `train_flood_model` | Train XGBoost model on 180 days of data | 30 min |
| `skip_training` | Placeholder when training is skipped | - |
| `validate_model` | Validate trained model quality | 10 min |
| `send_training_notification` | Send training results summary | - |

#### Task Dependencies
```
check_training_readiness (BranchOperator)
    ├→ train_flood_model → validate_model ──┐
    │                                        ├→ send_training_notification
    └→ skip_training ────────────────────────┘
```

#### Key Features
- **Conditional Execution**: BranchOperator checks data readiness
- **XGBoost Training**: 200 estimators, 6 max depth, 0.1 learning rate
- **Model Validation**: Tests model predictions and accuracy
- **Version Control**: Timestamped model files
- **Metadata Tracking**: Saves training runs to database

#### Outputs
- Model file: `models/flood_risk_v2_YYYYMMDD.pkl`
- Metrics file: `models/metrics_v2_YYYYMMDD.json`
- Training metadata in `model_training_runs` table

### 2.2 Prediction Generation DAG

**DAG ID:** `flood_risk_predictions`

#### Schedule
- **Daily at 9:00 AM UTC (5:00 PM PHT)**
- Cron: `0 9 * * *`

#### Tasks

| Task ID | Description | Timeout |
|---------|-------------|---------|
| `generate_predictions` | Generate predictions for all 30 regions | 15 min |
| `validate_predictions` | Validate prediction coverage and quality | 5 min |
| `track_performance` | Track model performance over time | 5 min |
| `send_prediction_notification` | Send prediction summary | - |

#### Task Dependencies
```
generate_predictions
    ↓
validate_predictions
    ↓
track_performance
    ↓
send_prediction_notification
```

#### Key Features
- **Dual Model Support**: Runs both rule-based (v1) and ML (v2) models
- **Auto Model Detection**: Automatically finds latest trained model
- **Coverage Validation**: Ensures all regions have predictions
- **Performance Tracking**: Monitors prediction distributions and confidence
- **Anomaly Detection**: Identifies unusual prediction patterns

#### Outputs
- Predictions in `flood_risk_assessments` table
- Risk levels: low, medium, high, critical
- Confidence scores and recommendations
- Model version tracking

---

## 3. Data Quality Monitoring DAG

**File:** `/home/user/project-agri-safe/airflow/dags/data_quality_dag.py`

### Purpose
Comprehensive data quality validation and monitoring with alerting.

### Schedule
- **Every 6 hours**
- Cron: `0 */6 * * *`

### Tasks

| Task ID | Description | Timeout |
|---------|-------------|---------|
| `run_quality_checks` | Execute all quality validators | 10 min |
| `generate_reports` | Create text and HTML reports | 5 min |
| `analyze_trends` | Analyze 30-day quality trends | 5 min |
| `generate_alerts` | Generate actionable alerts | 5 min |
| `send_notifications` | Send quality status notifications | - |
| `check_critical_failures` | Fail DAG if critical issues exist | - |
| `cleanup_old_checks` | Delete quality checks > 90 days | 5 min |
| `export_quality_metrics` | Export metrics to monitoring systems | - |

### Task Dependencies
```
run_quality_checks
    ↓
├── generate_reports ──┐
├── analyze_trends ────┤
└── generate_alerts ───┴→ send_notifications
                              ↓
                         check_critical_failures
                              ↓
                         ├── cleanup_old_checks
                         └── export_quality_metrics
```

### Quality Checks Performed

1. **Null Value Detection**
   - Threshold: < 5% null values
   - Checks: temperature, rainfall, wind, condition

2. **Value Range Validation**
   - Temperature high: 15-45°C
   - Temperature low: 10-40°C
   - Rainfall: 0-500mm
   - Wind speed: 0-250 km/h

3. **Data Freshness**
   - Threshold: < 48 hours since last update
   - Critical: > 72 hours

4. **Regional Coverage**
   - Threshold: ≥ 90% regions with data
   - Checks last 24 hours

5. **Anomaly Detection**
   - Statistical outliers (3σ threshold)
   - Threshold: < 10 anomalies

6. **Data Consistency**
   - Temperature inversion checks
   - Negative value detection
   - Threshold: < 0.5% inconsistent

### Key Features
- **Comprehensive Validation**: 6 different quality checks
- **Trend Analysis**: 30-day historical comparison
- **Alert Generation**: Actionable recommendations
- **HTML Dashboards**: Visual quality reports
- **Critical Failure Handling**: DAG fails if critical issues
- **Database Cleanup**: Automatic old data removal

### Outputs
- Quality check results in `data_quality_checks` table
- HTML dashboard: `/tmp/quality_reports/quality_dashboard_*.html`
- Text reports in logs
- Alert notifications (production: email/Slack)

---

## DAG Configuration Summary

| DAG | Schedule | Max Active Runs | Catchup | Retries |
|-----|----------|----------------|---------|---------|
| `weather_data_processing` | Daily 8:00 AM UTC | 1 | False | 2 |
| `flood_model_training` | Weekly Sun 2:00 AM UTC | 1 | False | 1 |
| `flood_risk_predictions` | Daily 9:00 AM UTC | 1 | False | 2 |
| `data_quality_monitoring` | Every 6 hours | 1 | False | 1 |

---

## Testing the DAGs

### 1. Validate DAG Syntax

```bash
# Test all DAGs for syntax errors
docker exec agrisafe-airflow-webserver airflow dags list

# Test specific DAG
docker exec agrisafe-airflow-webserver python /opt/airflow/dags/weather_processing_dag.py
docker exec agrisafe-airflow-webserver python /opt/airflow/dags/flood_model_dag.py
docker exec agrisafe-airflow-webserver python /opt/airflow/dags/data_quality_dag.py
```

### 2. Test Individual Tasks

```bash
# Test weather processing ETL task
docker exec agrisafe-airflow-webserver airflow tasks test weather_data_processing run_spark_etl 2025-01-17

# Test model training
docker exec agrisafe-airflow-webserver airflow tasks test flood_model_training train_flood_model 2025-01-17

# Test quality checks
docker exec agrisafe-airflow-webserver airflow tasks test data_quality_monitoring run_quality_checks 2025-01-17
```

### 3. Trigger DAGs Manually

```bash
# Trigger weather processing
docker exec agrisafe-airflow-webserver airflow dags trigger weather_data_processing

# Trigger model training
docker exec agrisafe-airflow-webserver airflow dags trigger flood_model_training

# Trigger predictions
docker exec agrisafe-airflow-webserver airflow dags trigger flood_risk_predictions

# Trigger quality monitoring
docker exec agrisafe-airflow-webserver airflow dags trigger data_quality_monitoring
```

### 4. View DAG Status

```bash
# List all DAGs
docker exec agrisafe-airflow-webserver airflow dags list

# View DAG runs
docker exec agrisafe-airflow-webserver airflow dags list-runs -d weather_data_processing

# View task instances
docker exec agrisafe-airflow-webserver airflow tasks list weather_data_processing
```

---

## Monitoring and Logging

### Accessing Logs

1. **Airflow Web UI**: http://localhost:8080
   - Navigate to DAGs → Select DAG → Graph View
   - Click on task → View Logs

2. **Docker Logs**:
```bash
# Follow all Airflow logs
docker-compose logs -f airflow-webserver airflow-scheduler

# View specific container logs
docker logs agrisafe-airflow-webserver -f
```

3. **Application Logs**:
   - Location: `/opt/airflow/logs/`
   - Format: `{dag_id}/{task_id}/{execution_date}/{try_number}.log`

### Key Metrics to Monitor

**Weather Processing DAG:**
- ETL duration (target: < 5 min)
- Records processed
- Feature count
- Quality check pass rate

**Model Training DAG:**
- Training accuracy (target: > 70%)
- Cross-validation score
- Training duration
- Model file size

**Prediction DAG:**
- Regional coverage (target: 100%)
- Prediction distribution
- Confidence scores
- Processing time (target: < 30 sec)

**Quality Monitoring DAG:**
- Check pass rate (target: > 95%)
- Critical failures (target: 0)
- Data freshness (target: < 24 hours)
- Regional coverage (target: > 90%)

---

## Troubleshooting

### Common Issues

#### 1. Spark ETL Fails
```bash
# Check Spark configuration
docker exec agrisafe-airflow-webserver pyspark --version

# Verify PostgreSQL JDBC driver
docker exec agrisafe-airflow-webserver ls /opt/airflow/jars/

# Check database connectivity
docker exec agrisafe-airflow-webserver python -c "from src.utils.database import get_db_connection; db = get_db_connection(); print('DB Connected!')"
```

#### 2. Model Not Found
```bash
# Check models directory
docker exec agrisafe-airflow-webserver ls -lah /opt/airflow/models/

# Verify model path
docker exec agrisafe-airflow-webserver python -c "import glob; print(glob.glob('models/flood_risk_v2_*.pkl'))"
```

#### 3. Redis Connection Issues
```bash
# Test Redis connectivity
docker exec agrisafe-redis redis-cli ping

# Check Redis keys
docker exec agrisafe-redis redis-cli --scan --pattern "weather:features:*" | head -10
```

#### 4. Quality Checks Failing
```bash
# Run validators standalone
docker exec agrisafe-airflow-webserver python -m src.quality.validators

# Check data in database
docker exec agrisafe-airflow-webserver python -c "from src.utils.database import get_db_connection; import pandas as pd; db = get_db_connection(); with db.get_connection() as conn: print(pd.read_sql('SELECT COUNT(*) FROM weather_forecasts', conn))"
```

---

## Environment Variables

Ensure these variables are set in `.env`:

```bash
# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=agrisafe_db
POSTGRES_USER=agrisafe
POSTGRES_PASSWORD=agrisafe_password

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False
```

---

## Best Practices

### 1. DAG Development
- Always test DAGs locally before deploying
- Use `provide_context=True` for tasks needing execution context
- Set appropriate timeouts to prevent hanging tasks
- Use `trigger_rule='all_done'` for notification tasks

### 2. Error Handling
- Include try-except blocks in Python callables
- Log errors comprehensively
- Don't fail notification tasks (use trigger_rule)
- Set appropriate retry counts and delays

### 3. Performance
- Use `max_active_runs=1` to prevent concurrent runs
- Set `catchup=False` to avoid backfilling
- Optimize Spark partitions based on data size
- Cache frequently accessed data in Redis

### 4. Monitoring
- Review DAG run history regularly
- Monitor task durations for performance regression
- Set up email alerts for critical failures
- Track quality metrics over time

---

## Integration with Phase 3 Architecture

### Data Flow

```
PAGASA Ingestion
    ↓
weather_forecasts (Raw Data)
    ↓
Weather Processing DAG
    ↓
weather_daily_stats (Aggregated)
    ↓
weather_features (ML Features)
    ↓
Flood Model Training DAG (Weekly)
    ↓
flood_risk_v2_*.pkl (Trained Model)
    ↓
Flood Risk Predictions DAG (Daily)
    ↓
flood_risk_assessments (Predictions)
    ↓
API/Frontend (Phase 4)

Quality Monitoring DAG (Every 6h)
    ↓
data_quality_checks (Validation Results)
```

---

## Future Enhancements

1. **Email Notifications**: Configure SMTP for email alerts
2. **Slack Integration**: Add Slack webhook for critical alerts
3. **Model A/B Testing**: Compare v1 vs v2 model performance
4. **Auto-scaling**: Adjust Spark resources based on data volume
5. **Model Drift Detection**: Automated retraining triggers
6. **Dashboard Integration**: Real-time quality dashboards
7. **SLA Monitoring**: Set and track SLAs for each DAG

---

## References

- [Phase 3 Development Plan](/home/user/project-agri-safe/docs/PHASE3_DEVELOPMENT_PLAN.md)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [XGBoost Python API](https://xgboost.readthedocs.io/)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-17
**Author**: AgriSafe Development Team
**Status**: Production Ready
