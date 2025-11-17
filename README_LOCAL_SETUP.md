# Local Development Setup Guide

## Running PAGASA Connector Locally (on Mac)

This guide helps you run the PAGASA connector on your local Mac while the database runs in Docker.

### Step 1: Install Dependencies

```bash
# Make sure your virtual environment is activated
# You should see (venv) in your terminal prompt

# Install required Python packages
pip install requests psycopg2-binary pydantic pydantic-settings loguru python-dotenv
```

Or use the automated script:

```bash
chmod +x setup_local_env.sh
./setup_local_env.sh
```

### Step 2: Configure Local Environment

```bash
# Use the local environment file
cp .env.local .env

# Or manually edit .env to use localhost:
# POSTGRES_HOST=localhost
# REDIS_HOST=localhost
```

### Step 3: Start Docker Services

Make sure PostgreSQL and other services are running:

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# You should see:
# - agrisafe-postgres (healthy)
# - agrisafe-redis (healthy)
# - agrisafe-airflow-* (healthy)
```

### Step 4: Test Database Connection

```bash
# Test connection from your Mac
psql -h localhost -p 5432 -U agrisafe -d agrisafe_db

# If it asks for password, use: agrisafe_password_change_in_production

# Inside psql, run:
\dt              # List tables
SELECT COUNT(*) FROM regions;  # Should show number of regions
\q               # Quit
```

### Step 5: Run the PAGASA Connector

```bash
# Run the connector
python -m src.ingestion.pagasa_connector
```

Expected output:
```
2025-11-17 10:00:00 | INFO     | Starting PAGASA weather data ingestion...
2025-11-17 10:00:00 | INFO     | Fetching weather forecast from PAGASA...
2025-11-17 10:00:01 | INFO     | Successfully fetched data from Vercel API
2025-11-17 10:00:01 | INFO     | Successfully parsed PAGASA response
2025-11-17 10:00:01 | INFO     | Retrieved 10 regions from database
2025-11-17 10:00:01 | INFO     | Extracted 50 regional forecasts
2025-11-17 10:00:02 | INFO     | Successfully saved 50 weather forecasts
2025-11-17 10:00:02 | INFO     | Ingestion completed successfully in 2.15s
✓ Ingestion successful: 50 forecasts saved
```

### Troubleshooting

#### Issue: "ModuleNotFoundError: No module named 'requests'"

**Solution:** Install dependencies in your virtual environment:
```bash
pip install requests psycopg2-binary pydantic loguru python-dotenv
```

#### Issue: "psycopg2.OperationalError: could not connect to server"

**Solution:**
1. Check if PostgreSQL container is running: `docker-compose ps postgres`
2. Verify you're using `localhost` in .env, not `postgres`
3. Try connecting manually: `psql -h localhost -p 5432 -U agrisafe -d agrisafe_db`

#### Issue: "No regions found in database"

**Solution:** Load seed data:
```bash
# From project root
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db < sql/seeds/02_seed_data.sql
```

#### Issue: "Table 'weather_forecasts' does not exist"

**Solution:** Database schema not initialized. Restart PostgreSQL:
```bash
docker-compose down
docker-compose up -d
# The schema should auto-initialize from sql/schema/01_init_schema.sql
```

### Verify Data Was Saved

```bash
# Connect to database
psql -h localhost -p 5432 -U agrisafe -d agrisafe_db

# Check forecasts
SELECT
    r.region_name,
    wf.forecast_date,
    wf.temperature_min,
    wf.temperature_max,
    wf.weather_condition,
    wf.created_at
FROM weather_forecasts wf
JOIN regions r ON wf.region_id = r.id
ORDER BY wf.created_at DESC
LIMIT 10;
```

### Quick Reference

| Command | Purpose |
|---------|---------|
| `docker-compose up -d` | Start services |
| `docker-compose ps` | Check service status |
| `docker-compose logs postgres` | View PostgreSQL logs |
| `python -m src.ingestion.pagasa_connector` | Run ingestion manually |
| `./test_phase2.sh` | Run automated tests |
| `psql -h localhost -U agrisafe agrisafe_db` | Connect to database |

### Environment Variables Reference

For **local development** (running Python on Mac):
```bash
POSTGRES_HOST=localhost    # Not 'postgres'
REDIS_HOST=localhost       # Not 'redis'
```

For **Docker/Airflow** (running inside containers):
```bash
POSTGRES_HOST=postgres     # Container name
REDIS_HOST=redis           # Container name
```

### Next Steps

After successful local testing:

1. **Use Airflow for automation:**
   - Access UI: http://localhost:8080
   - Enable `pagasa_daily_ingestion` DAG
   - Schedule runs at 6 AM daily

2. **Monitor logs:**
   - Application: `logs/agrisafe.log`
   - Docker: `docker-compose logs -f`

3. **View results:**
   - Check database for forecasts
   - Review API call logs

---

**Need help?** Check the full documentation in `docs/PHASE2_PAGASA_CONNECTOR.md`
