## Phase 3: Spark ETL Pipeline - Complete Documentation

## 🚀 Overview

Phase 3 implements a comprehensive PySpark-based ETL pipeline for weather data processing, feature engineering, and flood risk assessment. The pipeline processes raw weather forecasts into actionable insights using distributed computing.

## 📊 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PAGASA Raw Data                           │
│                  (weather_forecasts)                         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Job 1: Daily Statistics                         │
│   - Aggregate hourly forecasts to daily                     │
│   - Calculate min/max/avg/stddev                            │
│   - Assess data quality                                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ weather_daily_stats │
              └─────────┬───────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│          Job 2: Rolling Window Features                      │
│   - 7-day, 14-day, 30-day windows                           │
│   - Temperature trends                                       │
│   - Rainfall patterns                                        │
│   - Extreme event indicators                                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
             ┌────────────────────────┐
             │ weather_rolling_features│
             └─────────┬──────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│         Job 3: Flood Risk Indicators                         │
│   - Rule-based risk scoring                                  │
│   - Rainfall intensity & duration                            │
│   - Alert generation                                         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ flood_risk_indicators│
              └─────────┬───────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ Redis Cache │
                  └─────────────┘
```

## 🏗️ Components

### 1. **Daily Weather Statistics Job**

**File:** `src/processing/jobs/daily_weather_stats.py`

**Purpose:** Aggregate raw weather forecasts into daily statistics per region.

**Features:**
- Computes min, max, avg, and stddev for temperature, rainfall, humidity, and wind
- Calculates data completeness metrics
- Handles missing/null values gracefully
- Caches results in Redis for fast API access

**Output Table:** `weather_daily_stats`

**Key Metrics:**
- Temperature: min, max, avg, stddev
- Rainfall: total, max, avg
- Wind: avg speed, max speed
- Humidity: min, max, avg
- Data Quality: forecast count, completeness percentage

**Example Usage:**
```bash
# Run standalone
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07 \
    --mode append

# Via Spark submit
spark-submit \
    --master local[*] \
    --driver-memory 2g \
    src/processing/jobs/daily_weather_stats.py \
    --start-date 2025-01-01
```

---

### 2. **Rolling Window Features Job**

**File:** `src/processing/jobs/rolling_features.py`

**Purpose:** Compute time-series features using rolling windows.

**Features:**
- Multi-window support (7, 14, 30 days)
- Rolling aggregations (sum, avg, max, stddev)
- Pattern detection:
  - Consecutive rainy days
  - Heavy rainfall days (> 50mm)
  - Extreme temperature days
- Temperature trend analysis

**Output Table:** `weather_rolling_features`

**Key Features:**
- **Temperature:**
  - Rolling avg, min, max, stddev
  - Trend coefficient
  - Extreme temperature day count

- **Rainfall:**
  - Cumulative sum
  - Rolling avg, max, stddev
  - Rainy days count
  - Heavy rainfall days (> 50mm)
  - Maximum consecutive rainy days

- **Wind & Humidity:**
  - Rolling averages
  - Standard deviations

**Example Usage:**
```bash
# Compute 7, 14, and 30-day features
python -m src.processing.jobs.rolling_features \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --windows 7,14,30 \
    --mode append

# Disable caching
python -m src.processing.jobs.rolling_features \
    --start-date 2025-01-01 \
    --no-cache
```

---

### 3. **Flood Risk Indicators Job**

**File:** `src/processing/jobs/flood_risk_indicators.py`

**Purpose:** Calculate flood risk levels using rule-based models.

**Features:**
- Multi-factor risk assessment
- Dynamic thresholds
- Alert message generation
- Historical percentile ranking (placeholder)

**Output Table:** `flood_risk_indicators`

**Risk Calculation:**

```python
# Risk Factors (0-1 scale)
heavy_rainfall_factor = f(cumulative_7d_rainfall)
prolonged_rain_factor = f(consecutive_rain_days)
soil_saturation_proxy = f(heavy_rain_days, consecutive_days)

# Composite Score (0-100)
flood_risk_score = (
    heavy_rainfall_factor * 0.5 +
    prolonged_rain_factor * 0.3 +
    soil_saturation_proxy * 0.2
) * 100

# Risk Levels
score >= 75:  Critical
score >= 50:  High
score >= 25:  Moderate
score < 25:   Low
```

**Thresholds:**
- Heavy rainfall: 50mm/day
- Extreme rainfall: 100mm/day
- High risk: 150mm in 7 days
- Critical risk: 250mm in 7 days
- Consecutive rain: 5+ days = high risk

**Example Usage:**
```bash
python -m src.processing.jobs.flood_risk_indicators \
    --start-date 2025-01-01 \
    --end-date 2025-01-14 \
    --mode append
```

---

### 4. **Redis Feature Cache**

**File:** `src/cache/redis_cache.py`

**Purpose:** Provide fast access to computed features via caching.

**Features:**
- Namespaced cache keys
- TTL support (configurable expiration)
- JSON serialization
- Pattern-based invalidation

**Cache Namespaces:**
- `agrisafe:weather:stats:{region_id}` - Daily statistics
- `agrisafe:features:rolling:{region_id}:{window}d` - Rolling features
- `agrisafe:risk:indicators:{region_id}` - Flood risk indicators

**Example Usage:**
```python
from src.cache.redis_cache import RedisFeatureCache

cache = RedisFeatureCache()

# Cache weather stats
stats = {
    "temp_avg": 27.5,
    "rainfall_total": 45.2,
    "date": "2025-01-15"
}
cache.cache_weather_stats(region_id=1, stats=stats, ttl=3600)

# Retrieve cached data
cached = cache.get_weather_stats(region_id=1)

# Cache rolling features
features = {
    "rainfall_sum": 150.5,
    "heavy_rain_days": 3,
    "consecutive_rain_days": 5
}
cache.cache_rolling_features(
    region_id=1,
    window_days=7,
    features=features,
    ttl=3600
)

# Invalidate region cache
cache.invalidate_region_cache(region_id=1)
```

**Cache TTLs:**
- Weather stats: 1 hour (3600s)
- Rolling features: 1 hour (3600s)
- Risk indicators: 30 minutes (1800s)

---

### 5. **Airflow Orchestration**

**File:** `airflow/dags/spark_etl_pipeline.py`

**Purpose:** Orchestrate all Spark jobs in the correct sequence.

**Schedule:** Daily at 11 PM UTC (7 AM PHT)

**Workflow:**
```
pagasa_ingestion (separate DAG)
        ↓
  daily_stats
        ↓
 rolling_features
        ↓
   flood_risk
        ↓
  ┌─────┴──────┐
  ↓            ↓
validate    cleanup
  ↓
notify
```

**Tasks:**
1. **run_daily_stats** - Aggregate raw data to daily statistics
2. **run_rolling_features** - Compute 7, 14, 30-day features
3. **run_flood_risk** - Calculate risk indicators
4. **validate_results** - Verify all jobs succeeded
5. **send_notification** - Log completion summary
6. **cleanup_old_data** - Remove stale data (optional)

**Monitoring:**
- View DAG in Airflow UI: http://localhost:8080
- Check XCom data for detailed results
- Review task logs for errors

---

## 📚 Database Schema

### weather_daily_stats

Daily aggregated statistics per region.

```sql
CREATE TABLE weather_daily_stats (
    id UUID PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id),
    stat_date DATE NOT NULL,

    -- Temperature
    temp_min DECIMAL(5, 2),
    temp_max DECIMAL(5, 2),
    temp_avg DECIMAL(5, 2),
    temp_stddev DECIMAL(5, 2),

    -- Rainfall
    rainfall_total DECIMAL(8, 2),
    rainfall_max DECIMAL(8, 2),
    rainfall_avg DECIMAL(8, 2),

    -- Wind & Humidity
    wind_speed_avg DECIMAL(6, 2),
    humidity_avg DECIMAL(5, 2),

    -- Metadata
    forecast_count INTEGER,
    data_completeness DECIMAL(5, 2),
    created_at TIMESTAMP,

    UNIQUE(region_id, stat_date)
);
```

### weather_rolling_features

Rolling window features (7, 14, 30 days).

```sql
CREATE TABLE weather_rolling_features (
    id UUID PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id),
    feature_date DATE NOT NULL,
    window_days INTEGER NOT NULL,

    -- Temperature features
    temp_rolling_avg DECIMAL(5, 2),
    temp_rolling_min DECIMAL(5, 2),
    temp_rolling_max DECIMAL(5, 2),
    temp_rolling_stddev DECIMAL(5, 2),
    temp_trend DECIMAL(7, 4),

    -- Rainfall features
    rainfall_rolling_sum DECIMAL(10, 2),
    rainfall_rolling_avg DECIMAL(8, 2),
    rainfall_days_count INTEGER,
    rainfall_heavy_days INTEGER,
    consecutive_rain_days INTEGER,

    -- Event indicators
    extreme_temp_days INTEGER,

    created_at TIMESTAMP,

    UNIQUE(region_id, feature_date, window_days)
);
```

### flood_risk_indicators

Flood risk assessments per region.

```sql
CREATE TABLE flood_risk_indicators (
    id UUID PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id),
    indicator_date DATE NOT NULL,

    -- Risk scores
    rainfall_intensity_score DECIMAL(5, 2),
    rainfall_duration_score DECIMAL(5, 2),
    flood_risk_score DECIMAL(5, 2),
    flood_risk_level VARCHAR(20),

    -- Cumulative rainfall
    cumulative_rainfall_7d DECIMAL(10, 2),
    cumulative_rainfall_14d DECIMAL(10, 2),

    -- Risk factors
    heavy_rainfall_factor DECIMAL(5, 2),
    prolonged_rain_factor DECIMAL(5, 2),
    soil_saturation_proxy DECIMAL(5, 2),

    -- Alerts
    is_high_risk BOOLEAN,
    is_critical_risk BOOLEAN,
    alert_message TEXT,

    -- Model info
    model_version VARCHAR(50),
    confidence_score DECIMAL(5, 2),

    created_at TIMESTAMP,

    UNIQUE(region_id, indicator_date)
);
```

---

## 🧪 Testing

### Running Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pyspark

# Run all processing tests
pytest tests/processing/ -v

# Run specific test file
pytest tests/processing/test_daily_stats.py -v

# Run with coverage
pytest tests/processing/ --cov=src/processing --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Coverage

- **Daily Stats Job:** ✅
  - Empty dataframe handling
  - Null value handling
  - Data completeness calculation
  - Multi-region aggregation

- **Rolling Features Job:** ✅
  - Multiple window sizes
  - Rainy days counting
  - Heavy rainfall detection
  - Extreme temperature detection
  - Consecutive rain days

---

## 🚀 Deployment

### Prerequisites

1. **Database Setup:**
```bash
# Apply schema migrations
psql -U agrisafe -d agrisafe_db -f sql/schema/02_feature_tables.sql
```

2. **Environment Variables:**
```bash
# PostgreSQL
export POSTGRES_HOST=postgres
export POSTGRES_PORT=5432
export POSTGRES_DB=agrisafe_db
export POSTGRES_USER=agrisafe
export POSTGRES_PASSWORD=your_password

# Redis
export REDIS_HOST=redis
export REDIS_PORT=6379

# Spark
export SPARK_MASTER=local[*]
```

3. **Start Services:**
```bash
# Using Docker Compose
docker-compose up -d postgres redis airflow-webserver airflow-scheduler
```

### Running the Pipeline

#### Option 1: Via Airflow (Recommended)

```bash
# 1. Access Airflow UI
open http://localhost:8080

# 2. Enable DAG: spark_etl_pipeline

# 3. Trigger manual run or wait for schedule
```

#### Option 2: Standalone Execution

```bash
# Run jobs individually
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 --end-date 2025-01-07

python -m src.processing.jobs.rolling_features \
    --start-date 2025-01-01 --end-date 2025-01-31 \
    --windows 7,14,30

python -m src.processing.jobs.flood_risk_indicators \
    --start-date 2025-01-01 --end-date 2025-01-14
```

#### Option 3: Using Makefile (if available)

```bash
make spark-etl  # Run complete pipeline
make spark-daily-stats  # Run only daily stats
make spark-features  # Run only rolling features
```

---

## 📊 Monitoring & Observability

### 1. **Airflow Monitoring**

- **DAG Runs:** Track success/failure rates
- **Task Duration:** Identify bottlenecks
- **Logs:** Debug errors in real-time

```bash
# View Airflow logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-worker
```

### 2. **Database Monitoring**

```sql
-- Check recent job executions
SELECT
    job_name,
    status,
    records_processed,
    records_created,
    duration_seconds,
    created_at
FROM feature_metadata
ORDER BY created_at DESC
LIMIT 10;

-- Check data freshness
SELECT
    'daily_stats' AS table_name,
    MAX(stat_date) AS latest_date,
    COUNT(*) AS total_records
FROM weather_daily_stats
UNION ALL
SELECT
    'rolling_features',
    MAX(feature_date),
    COUNT(*)
FROM weather_rolling_features
UNION ALL
SELECT
    'flood_risk',
    MAX(indicator_date),
    COUNT(*)
FROM flood_risk_indicators;

-- Check high-risk regions
SELECT
    r.region_name,
    fri.flood_risk_level,
    fri.flood_risk_score,
    fri.cumulative_rainfall_7d,
    fri.alert_message
FROM flood_risk_indicators fri
JOIN regions r ON fri.region_id = r.id
WHERE fri.is_high_risk = TRUE
ORDER BY fri.flood_risk_score DESC;
```

### 3. **Redis Monitoring**

```bash
# Connect to Redis CLI
docker exec -it agrisafe-redis redis-cli

# Check cache stats
INFO stats

# List all cache keys
KEYS agrisafe:*

# Check specific region cache
KEYS agrisafe:*:1
GET "agrisafe:weather:stats:1"
```

---

## 🔧 Configuration

### Spark Configuration

**File:** `src/processing/utils/spark_session.py`

```python
default_config = {
    "spark.driver.memory": "2g",
    "spark.executor.memory": "2g",
    "spark.sql.shuffle.partitions": "10",
    "spark.sql.adaptive.enabled": "true",
}
```

### Redis Configuration

**File:** `src/cache/redis_cache.py`

```python
# Default TTLs (seconds)
WEATHER_STATS_TTL = 3600  # 1 hour
ROLLING_FEATURES_TTL = 3600  # 1 hour
RISK_INDICATORS_TTL = 1800  # 30 minutes
```

### Airflow Configuration

**File:** `airflow/dags/spark_etl_pipeline.py`

```python
# Schedule: Daily at 11 PM UTC
schedule_interval='0 23 * * *'

# Retry settings
retries=2
retry_delay=timedelta(minutes=5)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Spark job fails with OutOfMemory**

```bash
# Increase driver/executor memory
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=4g

# Or modify spark_session.py config
```

**Issue: Cannot connect to PostgreSQL**

```bash
# Check connection
docker exec -it agrisafe-postgres pg_isready

# Verify environment variables
echo $POSTGRES_HOST
echo $POSTGRES_PORT
```

**Issue: Redis connection timeout**

```bash
# Check Redis status
docker exec -it agrisafe-redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

**Issue: No data in feature tables**

```bash
# Check if daily stats exist
psql -U agrisafe -d agrisafe_db -c \
    "SELECT COUNT(*) FROM weather_daily_stats;"

# Re-run pipeline
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 --mode overwrite
```

---

## 📈 Performance Benchmarks

### Expected Performance (local[*] with 4 cores, 8GB RAM)

| Job | Input Records | Output Records | Duration | Memory |
|-----|--------------|----------------|----------|---------|
| Daily Stats | 10,000 | 500 | ~15s | ~500MB |
| Rolling Features | 500 | 1,500 | ~20s | ~600MB |
| Flood Risk | 1,500 | 500 | ~10s | ~400MB |
| **Total Pipeline** | 10,000 | 2,500 | ~45s | ~600MB |

### Optimization Tips

1. **Partitioning:**
   - Use predicates when reading large tables
   - Partition by region_id for parallel processing

2. **Caching:**
   - Cache intermediate DataFrames for complex computations
   - Use Redis for frequently accessed features

3. **Broadcast Joins:**
   - Broadcast small lookup tables (regions, crop_types)

4. **Coalesce:**
   - Reduce partitions before writing to database

---

## 🔄 Next Steps (Phase 4+)

### Immediate Enhancements

- [ ] ML-based flood prediction model (XGBoost/Random Forest)
- [ ] Historical data analysis for percentile calculations
- [ ] Real-time streaming with Spark Structured Streaming
- [ ] Advanced temperature trend analysis (linear regression)

### Future Improvements

- [ ] Integration with satellite imagery for vegetation indices
- [ ] Soil moisture modeling
- [ ] Multi-source data fusion (PAGASA + OpenWeatherMap)
- [ ] Feature importance analysis
- [ ] A/B testing of risk models

---

## 📞 Support

For issues or questions about the Spark ETL pipeline:

1. Check logs: `logs/agrisafe.log`
2. Review Airflow task logs
3. Verify database connections
4. Check Redis cache status

---

**Last Updated:** 2025-11-17
**Version:** Phase 3 - Data Processing & ML
**Status:** ✅ Completed
