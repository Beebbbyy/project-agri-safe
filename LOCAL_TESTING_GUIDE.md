# Local Testing Guide for Phase 3 Spark ETL Pipeline

## 🧪 Complete Local Testing Steps

### Prerequisites Check

Before testing, ensure you have:
- ✅ Docker and Docker Compose installed
- ✅ Python 3.9+ installed
- ✅ PostgreSQL and Redis containers running
- ✅ Weather data from Phase 2 (PAGASA ingestion)

---

## Step 1: Verify Docker Services

```bash
# Check if Docker is running
docker --version

# Check running containers
docker-compose ps

# Expected output should show:
# - agrisafe-postgres (healthy)
# - agrisafe-redis (healthy)
# - agrisafe-airflow-* (healthy)

# If services aren't running, start them:
docker-compose up -d

# Wait for services to be healthy (30 seconds)
sleep 30
```

---

## Step 2: Install Python Dependencies

```bash
# Navigate to project directory
cd /home/user/project-agri-safe

# Check if you have a virtual environment
# If not, create one:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Verify PySpark installation
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"

# Verify other dependencies
python -c "import redis; print('Redis: OK')"
python -c "from loguru import logger; print('Loguru: OK')"
python -c "import psycopg2; print('psycopg2: OK')"
```

---

## Step 3: Set Environment Variables

```bash
# Option 1: Export in terminal (temporary)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agrisafe_db
export POSTGRES_USER=agrisafe
export POSTGRES_PASSWORD=agrisafe_password
export REDIS_HOST=localhost
export REDIS_PORT=6379
export PYTHONPATH="${PYTHONPATH}:/home/user/project-agri-safe"

# Option 2: Create .env file (persistent)
cat > .env.local << 'EOF'
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agrisafe_db
POSTGRES_USER=agrisafe
POSTGRES_PASSWORD=agrisafe_password
REDIS_HOST=localhost
REDIS_PORT=6379
PYTHONPATH=/home/user/project-agri-safe
EOF

# Load .env file
export $(cat .env.local | xargs)

# Verify environment variables
echo "POSTGRES_HOST: $POSTGRES_HOST"
echo "REDIS_HOST: $REDIS_HOST"
```

---

## Step 4: Apply Database Schema

```bash
# Check if PostgreSQL is accessible
docker exec -it agrisafe-postgres pg_isready -U agrisafe
# Should output: /var/run/postgresql:5432 - accepting connections

# Apply the feature tables schema
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    < sql/schema/02_feature_tables.sql

# Verify tables were created
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt"

# You should see these tables:
# - weather_daily_stats
# - weather_rolling_features
# - flood_risk_indicators
# - feature_metadata
```

---

## Step 5: Check for Weather Data (from Phase 2)

```bash
# Connect to PostgreSQL
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db

# Run this query to check weather forecasts:
```

```sql
-- Check if we have weather forecast data
SELECT
    COUNT(*) AS total_forecasts,
    MIN(forecast_date) AS earliest_date,
    MAX(forecast_date) AS latest_date,
    COUNT(DISTINCT region_id) AS unique_regions
FROM weather_forecasts;

-- Check regions
SELECT id, region_name, province
FROM regions
ORDER BY id
LIMIT 10;

-- Exit PostgreSQL
\q
```

**Important:** If you don't have weather data, run Phase 2 ingestion first:

```bash
python -m src.ingestion.pagasa_connector
```

---

## Step 6: Test Individual Spark Jobs

### Test 1: Daily Weather Statistics

```bash
echo "========================================="
echo "Testing Daily Weather Statistics Job"
echo "========================================="

# Run the job with a small date range
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07 \
    --mode append

# Expected output:
# ✓ Loading weather forecasts...
# ✓ Loaded X weather forecast records
# ✓ Computing daily statistics...
# ✓ Computed statistics for Y region-date combinations
# ✓ Saving daily statistics to PostgreSQL...
# ✓ Successfully saved Y records
# ✓ Caching results in Redis...
# ✓ Cached Y records in Redis
# ✓ Job metadata logged
# ✓ Job completed successfully in X.XX seconds
```

**Verify the results:**

```bash
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
-- Check daily stats
SELECT
    r.region_name,
    wds.stat_date,
    wds.temp_avg,
    wds.rainfall_total,
    wds.wind_speed_avg,
    wds.forecast_count,
    wds.data_completeness
FROM weather_daily_stats wds
JOIN regions r ON wds.region_id = r.id
ORDER BY wds.stat_date DESC, r.region_name
LIMIT 10;
EOF
```

---

### Test 2: Rolling Window Features

```bash
echo "========================================="
echo "Testing Rolling Features Job"
echo "========================================="

# Run with multiple windows
python -m src.processing.jobs.rolling_features \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --windows 7,14,30 \
    --mode append

# Expected output:
# ✓ Loading daily weather statistics...
# ✓ Loaded X daily statistics records
# ✓ Computing rolling features for windows: [7, 14, 30]
# ✓ Computing 7-day rolling features...
# ✓ Computing 14-day rolling features...
# ✓ Computing 30-day rolling features...
# ✓ Computed X rolling feature records
# ✓ Saving rolling features to PostgreSQL...
# ✓ Successfully saved X records
# ✓ Caching rolling features in Redis...
# ✓ Cached X feature sets in Redis
```

**Verify the results:**

```bash
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
-- Check rolling features (7-day window)
SELECT
    r.region_name,
    wrf.feature_date,
    wrf.window_days,
    wrf.rainfall_rolling_sum,
    wrf.rainfall_heavy_days,
    wrf.consecutive_rain_days,
    wrf.temp_rolling_avg
FROM weather_rolling_features wrf
JOIN regions r ON wrf.region_id = r.id
WHERE wrf.window_days = 7
ORDER BY wrf.feature_date DESC, r.region_name
LIMIT 10;

-- Check all windows
SELECT
    window_days,
    COUNT(*) AS feature_count,
    MAX(feature_date) AS latest_date
FROM weather_rolling_features
GROUP BY window_days
ORDER BY window_days;
EOF
```

---

### Test 3: Flood Risk Indicators

```bash
echo "========================================="
echo "Testing Flood Risk Indicators Job"
echo "========================================="

# Run flood risk calculation
python -m src.processing.jobs.flood_risk_indicators \
    --start-date 2025-01-01 \
    --end-date 2025-01-14 \
    --mode append

# Expected output:
# ✓ Loading 7-day rolling features...
# ✓ Loaded X rolling feature records
# ✓ Calculating flood risk scores...
# ✓ Calculated risk indicators for Y records
# ✓ Summary: {...}
# ✓ Saving flood risk indicators to PostgreSQL...
# ✓ Successfully saved Y records
# ✓ Caching risk indicators in Redis...
# ✓ Cached Y risk indicators in Redis
```

**Verify the results:**

```bash
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
-- Check flood risk indicators
SELECT
    r.region_name,
    fri.indicator_date,
    fri.flood_risk_level,
    fri.flood_risk_score,
    fri.cumulative_rainfall_7d,
    fri.is_high_risk,
    fri.is_critical_risk,
    LEFT(fri.alert_message, 50) AS alert_preview
FROM flood_risk_indicators fri
JOIN regions r ON fri.region_id = r.id
ORDER BY fri.flood_risk_score DESC
LIMIT 10;

-- Risk level distribution
SELECT
    flood_risk_level,
    COUNT(*) AS count,
    ROUND(AVG(flood_risk_score), 2) AS avg_score
FROM flood_risk_indicators
GROUP BY flood_risk_level
ORDER BY avg_score DESC;
EOF
```

---

## Step 7: Verify Redis Cache

```bash
echo "========================================="
echo "Testing Redis Cache"
echo "========================================="

# Connect to Redis
docker exec -it agrisafe-redis redis-cli

# Inside Redis CLI, run these commands:
```

```redis
# Check if Redis is working
PING
# Should return: PONG

# List all cached keys
KEYS agrisafe:*

# Check weather stats for region 1
GET "agrisafe:weather:stats:1"

# Check rolling features for region 1 (7-day window)
GET "agrisafe:features:rolling:1:7d"

# Check risk indicators for region 1
GET "agrisafe:risk:indicators:1"

# Check TTL (time to live) of a key
TTL "agrisafe:weather:stats:1"
# Should return seconds remaining (e.g., 3600 = 1 hour)

# Exit Redis
EXIT
```

---

## Step 8: Run Unit Tests

```bash
echo "========================================="
echo "Running Unit Tests"
echo "========================================="

# Install pytest if not already installed
pip install pytest pytest-cov

# Run all processing tests
pytest tests/processing/ -v

# Expected output:
# tests/processing/test_daily_stats.py::test_compute_daily_stats PASSED
# tests/processing/test_daily_stats.py::test_empty_dataframe PASSED
# tests/processing/test_daily_stats.py::test_null_handling PASSED
# tests/processing/test_daily_stats.py::test_data_completeness_calculation PASSED
# tests/processing/test_rolling_features.py::test_compute_7day_rolling_features PASSED
# tests/processing/test_rolling_features.py::test_multiple_windows PASSED
# tests/processing/test_rolling_features.py::test_rainy_days_count PASSED
# tests/processing/test_rolling_features.py::test_heavy_rainfall_days PASSED
# tests/processing/test_rolling_features.py::test_extreme_temp_days PASSED
#
# ====== 11 passed in X.XX seconds ======

# Run with coverage report
pytest tests/processing/ --cov=src/processing --cov-report=term-missing

# Generate HTML coverage report
pytest tests/processing/ --cov=src/processing --cov-report=html

# View coverage report (if on desktop)
# open htmlcov/index.html
```

---

## Step 9: Test Complete Pipeline

```bash
echo "========================================="
echo "Testing Complete ETL Pipeline"
echo "========================================="

# Create a test script
cat > test_pipeline.sh << 'EOF'
#!/bin/bash

set -e  # Exit on error

echo "Starting complete ETL pipeline test..."

# Set date range
START_DATE="2025-01-01"
END_DATE="2025-01-07"

echo ""
echo "Step 1: Running Daily Stats..."
python -m src.processing.jobs.daily_weather_stats \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --mode append

echo ""
echo "Step 2: Running Rolling Features..."
python -m src.processing.jobs.rolling_features \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --windows 7,14,30 \
    --mode append

echo ""
echo "Step 3: Running Flood Risk Indicators..."
python -m src.processing.jobs.flood_risk_indicators \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --mode append

echo ""
echo "✅ Complete pipeline test finished successfully!"
EOF

# Make it executable
chmod +x test_pipeline.sh

# Run the pipeline test
./test_pipeline.sh
```

---

## Step 10: Verify Job Metadata

```bash
# Check job execution history
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
-- View recent job executions
SELECT
    job_name,
    job_type,
    status,
    records_processed,
    records_created,
    duration_seconds,
    TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') AS executed_at
FROM feature_metadata
ORDER BY created_at DESC
LIMIT 10;
EOF
```

---

## Step 11: Test Airflow DAG (Optional)

```bash
echo "========================================="
echo "Testing Airflow DAG"
echo "========================================="

# Access Airflow UI
# Open browser: http://localhost:8080
# Login: admin / admin

# Or test via CLI:
docker exec -it agrisafe-airflow-scheduler airflow dags list | grep spark_etl

# Test DAG syntax
docker exec -it agrisafe-airflow-scheduler \
    python /opt/airflow/dags/spark_etl_pipeline.py

# Trigger DAG manually
docker exec -it agrisafe-airflow-scheduler \
    airflow dags trigger spark_etl_pipeline

# Check DAG run status
docker exec -it agrisafe-airflow-scheduler \
    airflow dags state spark_etl_pipeline
```

---

## 🐛 Troubleshooting Common Issues

### Issue 1: "No module named 'src'"

```bash
# Solution: Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/user/project-agri-safe"

# Or install in development mode
pip install -e /home/user/project-agri-safe
```

### Issue 2: "Cannot connect to PostgreSQL"

```bash
# Check if container is running
docker ps | grep postgres

# Check connection
docker exec -it agrisafe-postgres pg_isready -U agrisafe

# Restart if needed
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

### Issue 3: "Cannot connect to Redis"

```bash
# Check if container is running
docker ps | grep redis

# Test connection
docker exec -it agrisafe-redis redis-cli ping

# Restart if needed
docker-compose restart redis
```

### Issue 4: "No data in weather_forecasts"

```bash
# Run PAGASA ingestion first
python -m src.ingestion.pagasa_connector

# Or trigger Airflow DAG
docker exec -it agrisafe-airflow-scheduler \
    airflow dags trigger pagasa_daily_ingestion
```

### Issue 5: Spark job hangs

```bash
# Check available memory
free -h

# Reduce date range
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-02  # Just 2 days

# Check Spark logs in console output
```

### Issue 6: "Permission denied" errors

```bash
# Fix file permissions
sudo chown -R $USER:$USER /home/user/project-agri-safe

# Or run with sudo (not recommended)
```

---

## ✅ Success Criteria

After running all tests, you should have:

- ✅ Database tables populated with data
- ✅ Redis cache contains feature data
- ✅ All unit tests passing
- ✅ Job metadata showing successful executions
- ✅ No errors in console output

---

## 📊 Expected Results

### Sample Query Results

```sql
-- You should see data like this:

-- Daily Stats:
 region_name | stat_date  | temp_avg | rainfall_total | data_completeness
-------------+------------+----------+----------------+-------------------
 Metro Manila| 2025-01-07 |    27.50 |          45.20 |             83.33
 Cebu        | 2025-01-07 |    26.80 |          12.30 |             75.00

-- Rolling Features:
 region_name | feature_date | rainfall_sum | heavy_days | consecutive_days
-------------+--------------+--------------+------------+------------------
 Metro Manila| 2025-01-07   |       150.50 |          3 |                5
 Cebu        | 2025-01-07   |        65.20 |          1 |                2

-- Flood Risk:
 region_name | risk_level | risk_score | rainfall_7d | is_high_risk
-------------+------------+------------+-------------+--------------
 Metro Manila| High       |      72.50 |      150.50 | t
 Cebu        | Moderate   |      35.20 |       65.20 | f
```

---

## 🎉 Success!

If all tests pass, your Spark ETL pipeline is working correctly!

You can now:
1. Run jobs on a schedule via Airflow
2. Query features for API development (Phase 4)
3. Use cached data for fast dashboard responses

---

**Questions or Issues?** Check the logs:
- Application logs: `logs/agrisafe.log`
- Airflow logs: `docker-compose logs airflow-scheduler`
- PostgreSQL logs: `docker-compose logs postgres`
