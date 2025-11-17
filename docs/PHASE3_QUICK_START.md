# Phase 3: Spark ETL - Quick Start Guide

## 🚀 Quick Start (10 Minutes)

### Prerequisites

✅ Phase 1 completed (Database schema)
✅ Phase 2 completed (PAGASA ingestion with weather data)
✅ Docker services running

### Step 1: Apply Database Schema

```bash
# Connect to PostgreSQL
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db

# Apply feature tables schema
\i /docker-entrypoint-initdb.d/02_feature_tables.sql

# Verify tables created
\dt

# Expected output should include:
# - weather_daily_stats
# - weather_rolling_features
# - flood_risk_indicators
# - feature_metadata

\q
```

Or using psql directly:

```bash
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db < sql/schema/02_feature_tables.sql
```

### Step 2: Verify Environment

```bash
# Check Python dependencies
pip list | grep -E "pyspark|redis|loguru"

# If missing, install
pip install pyspark==3.5.0 redis==5.0.1 loguru==0.7.2

# Check Redis connection
docker exec -it agrisafe-redis redis-cli ping
# Should return: PONG

# Check PostgreSQL
docker exec -it agrisafe-postgres pg_isready -U agrisafe
# Should return: accepting connections
```

### Step 3: Test Individual Jobs

#### Test 1: Daily Statistics

```bash
# Run daily stats aggregation
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07 \
    --mode append

# Expected output:
# ✓ Loaded X weather forecast records
# ✓ Computed statistics for Y region-date combinations
# ✓ Successfully saved Y records
# ✓ Cached Y records in Redis
```

#### Test 2: Rolling Features

```bash
# Run rolling features computation
python -m src.processing.jobs.rolling_features \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --windows 7,14,30 \
    --mode append

# Expected output:
# ✓ Loaded X daily statistics records
# ✓ Computing 7-day rolling features...
# ✓ Computing 14-day rolling features...
# ✓ Computing 30-day rolling features...
# ✓ Computed X rolling feature records
```

#### Test 3: Flood Risk Indicators

```bash
# Run flood risk calculation
python -m src.processing.jobs.flood_risk_indicators \
    --start-date 2025-01-01 \
    --end-date 2025-01-14 \
    --mode append

# Expected output:
# ✓ Loaded X rolling feature records
# ✓ Calculated risk indicators for Y records
# ✓ Summary: {...}
# ✓ Successfully saved Y records
```

### Step 4: Verify Data in Database

```bash
# Connect to database
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db
```

```sql
-- Check daily statistics
SELECT
    r.region_name,
    wds.stat_date,
    wds.temp_avg,
    wds.rainfall_total,
    wds.data_completeness
FROM weather_daily_stats wds
JOIN regions r ON wds.region_id = r.id
ORDER BY wds.stat_date DESC, r.region_name
LIMIT 10;

-- Check rolling features (7-day window)
SELECT
    r.region_name,
    wrf.feature_date,
    wrf.rainfall_rolling_sum,
    wrf.rainfall_heavy_days,
    wrf.consecutive_rain_days
FROM weather_rolling_features wrf
JOIN regions r ON wrf.region_id = r.id
WHERE wrf.window_days = 7
ORDER BY wrf.feature_date DESC
LIMIT 10;

-- Check flood risk indicators
SELECT
    r.region_name,
    fri.indicator_date,
    fri.flood_risk_level,
    fri.flood_risk_score,
    fri.cumulative_rainfall_7d,
    fri.is_high_risk,
    fri.alert_message
FROM flood_risk_indicators fri
JOIN regions r ON fri.region_id = r.id
WHERE fri.is_high_risk = TRUE
ORDER BY fri.flood_risk_score DESC;
```

### Step 5: Verify Redis Cache

```bash
# Connect to Redis
docker exec -it agrisafe-redis redis-cli

# List all cached keys
KEYS agrisafe:*

# Get weather stats for region 1
GET "agrisafe:weather:stats:1"

# Get rolling features for region 1 (7-day window)
GET "agrisafe:features:rolling:1:7d"

# Get risk indicators for region 1
GET "agrisafe:risk:indicators:1"

# Exit Redis
EXIT
```

### Step 6: Enable Airflow DAG

```bash
# 1. Access Airflow UI
open http://localhost:8080

# Login: admin / admin

# 2. Find DAG: spark_etl_pipeline

# 3. Toggle ON to enable

# 4. Click "Trigger DAG" to run manually

# 5. Monitor execution in Graph View
```

### Step 7: Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/processing/ -v

# Expected output:
# tests/processing/test_daily_stats.py::test_compute_daily_stats PASSED
# tests/processing/test_daily_stats.py::test_empty_dataframe PASSED
# tests/processing/test_rolling_features.py::test_compute_7day_rolling_features PASSED
# ...
# ====== X passed in Y seconds ======
```

---

## 📊 Quick Data Checks

### Check Pipeline Health

```sql
-- Feature metadata (job execution history)
SELECT
    job_name,
    job_type,
    status,
    records_processed,
    records_created,
    duration_seconds,
    created_at
FROM feature_metadata
ORDER BY created_at DESC
LIMIT 10;

-- Data freshness
SELECT
    'daily_stats' AS table_name,
    COUNT(*) AS total_records,
    MAX(stat_date) AS latest_date,
    MIN(stat_date) AS earliest_date
FROM weather_daily_stats
UNION ALL
SELECT
    'rolling_features',
    COUNT(*),
    MAX(feature_date),
    MIN(feature_date)
FROM weather_rolling_features
UNION ALL
SELECT
    'flood_risk',
    COUNT(*),
    MAX(indicator_date),
    MIN(indicator_date)
FROM flood_risk_indicators;

-- Risk distribution
SELECT
    flood_risk_level,
    COUNT(*) AS count,
    ROUND(AVG(flood_risk_score), 2) AS avg_score
FROM flood_risk_indicators
WHERE indicator_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY flood_risk_level
ORDER BY avg_score DESC;
```

---

## 🧪 Sample Queries

### Get Latest Risk Status for All Regions

```sql
SELECT
    r.region_name,
    r.province,
    fri.flood_risk_level,
    fri.flood_risk_score,
    fri.cumulative_rainfall_7d,
    fri.consecutive_rain_days,
    fri.alert_message
FROM flood_risk_indicators fri
JOIN regions r ON fri.region_id = r.id
WHERE fri.indicator_date = (
    SELECT MAX(indicator_date)
    FROM flood_risk_indicators
)
ORDER BY fri.flood_risk_score DESC;
```

### Get 7-Day Rainfall Trend for a Region

```sql
SELECT
    stat_date,
    rainfall_total,
    temp_avg,
    humidity_avg
FROM weather_daily_stats
WHERE region_id = 1
    AND stat_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY stat_date;
```

### Get Rolling Features Time Series

```sql
SELECT
    feature_date,
    rainfall_rolling_sum,
    rainfall_heavy_days,
    consecutive_rain_days,
    temp_rolling_avg
FROM weather_rolling_features
WHERE region_id = 1
    AND window_days = 7
    AND feature_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY feature_date;
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'src'"

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/user/project-agri-safe"

# Or install in development mode
cd /home/user/project-agri-safe
pip install -e .
```

### Issue: "Cannot connect to PostgreSQL/Redis"

```bash
# Check services are running
docker-compose ps

# Restart services if needed
docker-compose restart postgres redis

# Check environment variables
env | grep -E "POSTGRES|REDIS"
```

### Issue: "No data in weather_forecasts table"

```bash
# Run PAGASA ingestion first
python -m src.ingestion.pagasa_connector

# Or trigger Airflow DAG: pagasa_daily_ingestion
```

### Issue: Spark job hangs or runs slow

```bash
# Reduce data range
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-02  # Just 2 days

# Check Spark UI (if running locally)
open http://localhost:4040
```

---

## 📈 Next Steps

After successful Phase 3 setup:

1. **Monitor Daily Runs**
   - Check Airflow DAG runs daily
   - Review logs for errors
   - Validate data quality

2. **Optimize Performance**
   - Adjust Spark memory settings
   - Fine-tune Redis cache TTLs
   - Add database indexes if needed

3. **Enhance Risk Models**
   - Collect historical flood data
   - Train ML models (XGBoost, Random Forest)
   - Validate against actual flood events

4. **Build API (Phase 4)**
   - FastAPI backend
   - Expose features via REST endpoints
   - Integrate with frontend

---

## 📚 Additional Resources

- [Full Phase 3 Documentation](PHASE3_SPARK_ETL.md)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Redis Caching Patterns](https://redis.io/docs/manual/patterns/)

---

**Happy Feature Engineering!** 🌾📊
