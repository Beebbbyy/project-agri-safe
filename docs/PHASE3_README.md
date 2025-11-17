# Phase 3: Data Processing & ML - README

## Overview

Phase 3 implements the data processing and machine learning infrastructure for Project Agri-Safe, including:
- **PySpark ETL pipelines** for weather data aggregation
- **Flood risk prediction models** (rule-based and ML-based)
- **Data quality validation framework**
- **Airflow orchestration** for automated workflows

## Quick Start

### 1. Setup Phase 3 Infrastructure

```bash
# Start all services including Spark
make phase3-setup

# Verify setup
make phase3-status

# Run comprehensive tests
./test_phase3.sh
```

### 2. Run Initial Data Processing

```bash
# Process weather data
make run-etl

# Generate ML features
make run-features

# Run data quality checks
make quality-checks
```

### 3. Train and Run ML Model

```bash
# Train flood risk model
make train-model

# Generate predictions for all regions
make run-predictions

# Test the models
make test-model-v1  # Rule-based
make test-model-v2  # ML-based
```

## Components

### A. PySpark ETL Pipelines

Located in: `src/processing/spark_jobs/`

#### Weather ETL (`weather_etl.py`)
- Aggregates daily weather statistics per region
- Computes rolling window features
- Caches features to Redis
- Saves to PostgreSQL

**Usage:**
```bash
make run-etl

# Or with custom dates
docker exec agrisafe-airflow-worker python -m src.processing.spark_jobs.weather_etl \
  --start-date 2025-01-01 --end-date 2025-01-31
```

#### Rolling Features (`rolling_features.py`)
- Generates 30+ ML features
- Rainfall accumulations (3d, 7d, 14d, 30d)
- Temperature statistics
- Seasonal indicators
- Historical flood risk integration

**Usage:**
```bash
make run-features
```

### B. Flood Risk Models

Located in: `src/models/`

#### Rule-Based Model v1 (`flood_risk_v1.py`)
- Expert-defined rainfall thresholds
- Geographic and historical factors
- 4 risk levels: LOW, MEDIUM, HIGH, CRITICAL
- Explainable predictions

**Features:**
- Fast predictions (<10ms per region)
- No training required
- Interpretable results
- Baseline for ML model

**Test:**
```bash
make test-model-v1
```

#### ML Model v2 (`flood_risk_v2.py`)
- XGBoost multi-class classifier
- 30+ input features
- Probability estimates
- Feature importance analysis

**Features:**
- Higher accuracy (target >75%)
- Learns from historical data
- Adapts to patterns
- Confidence scores

**Train & Test:**
```bash
make train-model   # Train on 180 days of data
make test-model-v2 # Generate predictions
```

### C. Data Quality Framework

Located in: `src/quality/`

#### Validators (`validators.py`)
- **Null value detection** - Ensures completeness
- **Value range validation** - Philippine weather norms
- **Data freshness** - Updates within 48 hours
- **Regional coverage** - All 30 regions
- **Anomaly detection** - Statistical outliers
- **Consistency checks** - Logical validation

**Usage:**
```bash
make quality-checks

# View results
docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db \
  -c "SELECT * FROM data_quality_checks ORDER BY checked_at DESC LIMIT 10;"
```

#### Monitoring (`monitoring.py`)
- Quality trends over time
- Pass rate calculations
- Alert generation
- HTML dashboard (Phase 5)

**Usage:**
```bash
make quality-report
```

### D. Airflow DAGs

Located in: `airflow/dags/`

#### 1. Weather Processing DAG
**Schedule:** Daily at 8:00 AM UTC (4:00 PM PHT)
**Tasks:**
- Run Spark ETL
- Generate features
- Cache to Redis
- Quality checks
- Send notifications

**Trigger:**
```bash
make trigger-etl
```

#### 2. Model Training DAG
**Schedule:** Weekly (Sunday 2:00 AM UTC)
**Tasks:**
- Check data readiness (90+ days)
- Train XGBoost model
- Validate performance
- Save artifacts

**Trigger:**
```bash
make trigger-training
```

#### 3. Flood Risk Predictions DAG
**Schedule:** Daily at 9:00 AM UTC (5:00 PM PHT)
**Tasks:**
- Load latest model
- Generate predictions (all regions)
- Validate coverage
- Save to database

**Trigger:**
```bash
make trigger-predictions
```

#### 4. Data Quality DAG
**Schedule:** Every 6 hours
**Tasks:**
- Run all validators
- Generate reports
- Send alerts (if critical)
- Cleanup old data

**Trigger:**
```bash
make trigger-quality
```

## Database Schema

### Phase 3 Tables

All tables created via migration: `sql/migrations/03_phase3_tables.sql`

#### 1. weather_daily_stats
Daily aggregated weather statistics per region
- Temperature averages
- Rainfall totals
- Wind speeds
- Dominant conditions

#### 2. feature_store
Pre-computed ML features
- Rolling rainfall (1d, 3d, 7d, 14d, 30d)
- Temperature statistics
- Derived features
- Seasonal indicators

#### 3. data_quality_checks
Quality validation results
- Check results
- Severity levels
- Detailed diagnostics

#### 4. model_training_runs
ML training metadata
- Model versions
- Performance metrics
- Hyperparameters
- Artifact paths

#### 5. model_predictions_log
Audit log for predictions
- Model used
- Input features
- Confidence scores
- Timestamps

#### 6. region_risk_indicators
Region-level risk scores
- Flood season scores
- Typhoon probability
- Harvest suitability

#### 7. etl_job_runs
ETL execution tracking
- Job status
- Records processed
- Execution times
- Error logs

## Makefile Commands

### Spark Management
```bash
make spark-up          # Start Spark cluster
make spark-down        # Stop Spark
make spark-logs        # View logs
make spark-status      # Check status
```

### Database
```bash
make db-migrate-phase3 # Run Phase 3 migrations
```

### ETL Operations
```bash
make run-etl           # Process weather data
make run-features      # Generate features
```

### ML Operations
```bash
make train-model       # Train XGBoost model
make run-predictions   # Generate predictions
make test-model-v1     # Test rule-based
make test-model-v2     # Test ML model
```

### Data Quality
```bash
make quality-checks    # Run validators
make quality-report    # Generate report
```

### Airflow
```bash
make list-dags         # List Phase 3 DAGs
make trigger-etl       # Trigger ETL DAG
make trigger-predictions # Trigger predictions
make trigger-quality   # Trigger quality checks
make trigger-training  # Trigger training
```

### Testing
```bash
make test-phase3       # Unit tests
make test-phase3-integration # Integration tests
./test_phase3.sh       # Infrastructure tests
```

### Complete Setup
```bash
make phase3-setup      # Infrastructure only
make phase3-init       # Full initialization with data
make phase3-status     # Check everything
```

## Testing

### Infrastructure Tests
```bash
# Run comprehensive infrastructure tests
./test_phase3.sh
```

Tests cover:
- Spark cluster (master, worker, UIs)
- Database schema (7 new tables)
- Python modules (8 modules)
- Dependencies (PySpark, XGBoost, etc.)
- Airflow DAGs (4 DAGs)
- Functional tests (predictions, validation)
- Integration tests (connectivity)

### Unit Tests
```bash
# All Phase 3 unit tests
make test-phase3

# Specific modules
pytest tests/processing -v
pytest tests/models -v
pytest tests/quality -v
```

### Integration Tests
```bash
# End-to-end workflow tests
make test-phase3-integration
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 3 DATA FLOW                           │
└─────────────────────────────────────────────────────────────────┘

PostgreSQL (weather_forecasts)
         │
         ▼
    ┌─────────────────┐
    │  Spark ETL      │ ← Airflow DAG (daily)
    │  (weather_etl)  │
    └────────┬────────┘
             │
             ├──→ weather_daily_stats (PostgreSQL)
             ├──→ feature_store (PostgreSQL)
             └──→ Redis Cache (7-day TTL)
                      │
                      ▼
            ┌─────────────────────┐
            │  Feature Engine     │
            │  (rolling_features) │
            └──────────┬──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  ML Model       │ ← Training (weekly)
              │  (XGBoost)      │
              └────────┬────────┘
                       │
                       ├──→ Predictions (daily)
                       ├──→ model_predictions_log
                       └──→ flood_risk_assessments
                                │
                                ▼
                       ┌─────────────────┐
                       │  Quality Checks │ ← Every 6 hours
                       │  (validators)   │
                       └─────────────────┘
```

## Performance

### Expected Performance
- **ETL Processing:** 90 days of data in <5 minutes
- **Feature Generation:** 30 regions in <2 minutes
- **Model Training:** 180 days in <10 minutes
- **Predictions:** All 30 regions in <30 seconds
- **Quality Checks:** Complete suite in <2 minutes

### Optimization Tips
1. **Spark Configuration:**
   - Adjust worker memory (default: 2GB)
   - Increase cores if needed (default: 2)

2. **Database:**
   - Indexes on date columns
   - Partitioning for large tables

3. **Redis:**
   - Feature caching reduces DB load
   - 7-day TTL balances freshness/performance

## Monitoring

### Spark UI
- **Master:** http://localhost:8081
- **Worker:** http://localhost:8083
- **Application:** http://localhost:4040 (when job running)

### Airflow UI
- **URL:** http://localhost:8080
- **Username:** admin
- **Password:** admin

### Database Queries
```sql
-- Check recent ETL runs
SELECT * FROM etl_job_runs
ORDER BY start_time DESC LIMIT 10;

-- View quality check results
SELECT * FROM data_quality_checks
WHERE passed = false
ORDER BY checked_at DESC;

-- Check model training history
SELECT model_name, model_version, accuracy, training_date
FROM model_training_runs
ORDER BY training_date DESC;

-- View latest predictions
SELECT r.name, fra.risk_level, fra.confidence_score
FROM flood_risk_assessments fra
JOIN regions r ON fra.region_id = r.id
WHERE fra.assessment_date = CURRENT_DATE;
```

## Troubleshooting

### Issue: Spark services not starting
**Solution:**
```bash
make spark-down
make spark-up
make spark-logs  # Check for errors
```

### Issue: DAGs not appearing in Airflow
**Solution:**
```bash
# Check DAG syntax
docker exec agrisafe-airflow-webserver python /opt/airflow/dags/weather_processing_dag.py

# Restart Airflow
make airflow-restart

# Wait 1-2 minutes for DAG refresh
```

### Issue: Model training fails
**Solution:**
```bash
# Ensure enough historical data (90+ days)
docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db \
  -c "SELECT COUNT(DISTINCT forecast_date) FROM weather_forecasts;"

# Check logs
make airflow-logs | grep -i error
```

### Issue: Import errors for Phase 3 modules
**Solution:**
```bash
# Restart Airflow worker to reload modules
docker compose restart airflow-worker

# Verify module paths
docker exec agrisafe-airflow-worker ls -R /opt/airflow/src/
```

### Issue: PostgreSQL JDBC driver not found (Spark)
**Solution:**
```bash
# Rebuild Spark containers
docker compose build spark-master spark-worker
docker compose up -d spark-master spark-worker
```

## Development Workflow

### Adding New Features
1. Implement in `src/processing/`, `src/models/`, or `src/quality/`
2. Add tests in `tests/*/`
3. Update Airflow DAGs if needed
4. Run tests: `make test-phase3`
5. Test manually with make commands

### Adding New Quality Checks
1. Add method to `WeatherDataValidator`
2. Update `data_quality_dag.py`
3. Add test in `tests/quality/test_validators.py`
4. Run: `make quality-checks`

### Updating ML Model
1. Modify `src/models/flood_risk_v2.py`
2. Retrain: `make train-model`
3. Test: `make test-model-v2`
4. Compare performance in database

## Next Steps (Phase 4)

After completing Phase 3:
1. **FastAPI Backend** - REST APIs for predictions
2. **LLM Integration** - Claude/GPT harvest recommendations
3. **Authentication** - JWT user auth
4. **API Documentation** - OpenAPI/Swagger

## Resources

- **Phase 3 Development Plan:** `docs/PHASE3_DEVELOPMENT_PLAN.md`
- **Airflow DAGs README:** `airflow/dags/PHASE3_DAGS_README.md`
- **Test Script:** `test_phase3.sh`

## Support

For issues or questions:
1. Check logs: `make airflow-logs`, `make spark-logs`
2. Run status check: `make phase3-status`
3. Review test results: `./test_phase3.sh`
4. Check database: `make db-connect`

---

**Phase 3 Status:** ✅ **COMPLETE**
**Last Updated:** 2025-01-17
**Version:** 1.0.0
