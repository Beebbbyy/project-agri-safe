# Phase 2: PAGASA Connector - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database from Phase 1 running
- At least 10 regions seeded in the database

### Step 1: Ensure Services are Running

```bash
# Start all services
docker-compose up -d

# Verify services are healthy
docker-compose ps
```

Expected output:
```
agrisafe-postgres            healthy
agrisafe-redis               healthy
agrisafe-airflow-webserver   healthy
agrisafe-airflow-scheduler   healthy
agrisafe-airflow-worker      healthy
```

### Step 2: Install Python Dependencies

If running outside Docker (for development/testing):

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Set Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings (optional for development)
# The defaults should work with Docker Compose
```

### Step 4: Test the Connector Manually

```bash
# Run standalone ingestion
python -m src.ingestion.pagasa_connector
```

Expected output:
```
✓ Ingestion successful: 50 forecasts saved
```

### Step 5: Access Airflow UI

1. Open browser: http://localhost:8080
2. Login:
   - Username: `admin`
   - Password: `admin`
3. Find DAG: `pagasa_daily_ingestion`
4. Enable the DAG (toggle switch)
5. Click "Trigger DAG" to run manually

### Step 6: Verify Data in Database

```bash
# Connect to PostgreSQL
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db

# Check weather forecasts
SELECT
    r.region_name,
    wf.forecast_date,
    wf.temperature_min,
    wf.temperature_max,
    wf.weather_condition
FROM weather_forecasts wf
JOIN regions r ON wf.region_id = r.id
ORDER BY wf.forecast_date, r.region_name
LIMIT 10;
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ingestion/test_pagasa_connector.py -v

# Run with coverage
pytest tests/ingestion/test_pagasa_connector.py --cov=src/ingestion

# Run specific test
pytest tests/ingestion/test_pagasa_connector.py::TestPAGASAConnector::test_fetch_forecast_data_success -v
```

## 📊 Monitoring

### Check Airflow DAG Status

```bash
# View Airflow scheduler logs
docker-compose logs -f airflow-scheduler

# View Airflow worker logs
docker-compose logs -f airflow-worker
```

### Check Application Logs

```bash
# View latest logs
tail -f logs/agrisafe.log

# Search for errors
grep ERROR logs/agrisafe.log
```

### Check API Call Logs

```sql
-- Recent API calls
SELECT
    api_name,
    response_status,
    response_time_ms,
    created_at
FROM api_logs
ORDER BY created_at DESC
LIMIT 10;

-- Failed API calls
SELECT *
FROM api_logs
WHERE response_status >= 400 OR error_message IS NOT NULL
ORDER BY created_at DESC;
```

## 🔧 Configuration

### Schedule Configuration

Edit `airflow/dags/pagasa_daily_ingestion.py`:

```python
# Change schedule (currently 6 AM PHT / 22:00 UTC)
schedule_interval='0 22 * * *'

# Run every 6 hours
schedule_interval='0 */6 * * *'

# Run every hour
schedule_interval='0 * * * *'
```

### Retry Configuration

Edit `src/ingestion/pagasa_connector.py`:

```python
# Change connector settings
connector = PAGASAConnector(
    timeout=30,        # Increase timeout
    max_retries=5,     # More retries
    backoff_factor=2.0 # Longer backoff
)
```

### Logging Configuration

Edit `src/utils/logger.py` or set environment variable:

```bash
# Set log level
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## 🐛 Troubleshooting

### Issue: Cannot connect to database

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check connection from host
psql -h localhost -p 5432 -U agrisafe -d agrisafe_db

# Restart PostgreSQL
docker-compose restart postgres
```

### Issue: No data being fetched

```bash
# Test API connectivity
curl https://pagasa-forecast-api.vercel.app/api/pagasa-forecast

# Check if regions exist
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "SELECT COUNT(*) FROM regions;"

# If no regions, load seed data
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -f /docker-entrypoint-initdb.d/02_seed_data.sql
```

### Issue: Airflow DAG not visible

```bash
# Check DAG file for errors
python airflow/dags/pagasa_daily_ingestion.py

# Restart Airflow scheduler
docker-compose restart airflow-scheduler

# View scheduler logs
docker-compose logs airflow-scheduler
```

### Issue: Import errors when running tests

```bash
# Install project in development mode
pip install -e .

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/user/project-agri-safe"
```

## 📈 Next Steps

After successful Phase 2 implementation:

1. **Phase 3**: Data Processing & ML
   - Build Spark ETL pipelines
   - Develop flood risk prediction model
   - Optimize data transformations

2. **Enhance Phase 2**:
   - Add more data sources
   - Implement PSA crop calendar ingestion
   - Add real-time weather alerts

3. **Monitor & Optimize**:
   - Set up alerting for failed DAG runs
   - Optimize query performance
   - Implement data quality checks

## 📚 Additional Resources

- [Full Phase 2 Documentation](PHASE2_PAGASA_CONNECTOR.md)
- [Development Plan](../DEVELOPMENT_PLAN.md)
- [Database Schema](../sql/schema/01_init_schema.sql)
- [Airflow Documentation](https://airflow.apache.org/docs/)

---

**Happy Coding!** 🌾🇵🇭
