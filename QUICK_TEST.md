# Quick Test Guide - Phase 3

## 🚀 Fastest Way to Test (5 Minutes)

### Option 1: Run Automated Test Script

```bash
cd /home/user/project-agri-safe

# Run the automated test script
./test_phase3.sh
```

This script will:
- ✅ Check all dependencies
- ✅ Apply database schema
- ✅ Run all 3 Spark jobs
- ✅ Verify results in database
- ✅ Check Redis cache
- ✅ Run unit tests

**Expected Output:** All green checkmarks ✓

---

### Option 2: Manual Quick Test

```bash
# 1. Navigate to project
cd /home/user/project-agri-safe

# 2. Set environment variables
export PYTHONPATH="${PYTHONPATH}:/home/user/project-agri-safe"
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agrisafe_db
export POSTGRES_USER=agrisafe
export POSTGRES_PASSWORD=agrisafe_password
export REDIS_HOST=localhost
export REDIS_PORT=6379

# 3. Apply database schema (one-time)
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db < sql/schema/02_feature_tables.sql

# 4. Run one job to test
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07

# 5. Check results
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c \
    "SELECT COUNT(*) FROM weather_daily_stats;"
```

---

## 🔍 Quick Verification Commands

### Check Database Tables

```bash
# List all feature tables
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt"

# Count records in each table
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
SELECT 'daily_stats' AS table_name, COUNT(*) FROM weather_daily_stats
UNION ALL
SELECT 'rolling_features', COUNT(*) FROM weather_rolling_features
UNION ALL
SELECT 'flood_risk', COUNT(*) FROM flood_risk_indicators;
EOF
```

### Check Redis Cache

```bash
# List cached keys
docker exec -it agrisafe-redis redis-cli KEYS "agrisafe:*"

# Count cached items
docker exec -it agrisafe-redis redis-cli KEYS "agrisafe:*" | wc -l
```

### View Sample Data

```bash
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
-- Latest flood risk status
SELECT
    r.region_name,
    fri.flood_risk_level,
    fri.flood_risk_score,
    fri.cumulative_rainfall_7d
FROM flood_risk_indicators fri
JOIN regions r ON fri.region_id = r.id
ORDER BY fri.indicator_date DESC
LIMIT 5;
EOF
```

---

## 🐛 Quick Troubleshooting

### Issue: Services not running

```bash
# Start all services
docker-compose up -d

# Wait 30 seconds
sleep 30

# Check status
docker-compose ps
```

### Issue: "No module named 'src'"

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify
echo $PYTHONPATH
```

### Issue: No weather data

```bash
# Run PAGASA ingestion first
python -m src.ingestion.pagasa_connector
```

---

## ✅ Success Indicators

You know it's working when:

1. **Automated script shows:** All green ✓ checkmarks
2. **Database has data:** COUNT(*) > 0 for all tables
3. **Redis has cache:** Multiple "agrisafe:*" keys
4. **No errors:** All jobs complete without exceptions

---

## 📚 Full Documentation

- **Detailed Guide:** `LOCAL_TESTING_GUIDE.md`
- **Technical Docs:** `docs/PHASE3_SPARK_ETL.md`
- **Quick Start:** `docs/PHASE3_QUICK_START.md`

---

## 🎯 What Each Job Does

| Job | Input | Output | Purpose |
|-----|-------|--------|---------|
| **daily_weather_stats** | weather_forecasts | weather_daily_stats | Aggregate hourly forecasts to daily |
| **rolling_features** | weather_daily_stats | weather_rolling_features | Compute 7/14/30-day trends |
| **flood_risk_indicators** | weather_rolling_features | flood_risk_indicators | Calculate flood risk scores |

---

**Need help?** Run `./test_phase3.sh` for automated testing!
