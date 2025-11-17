# Quick Start - Local Machine (5 Minutes)

## 🚀 Super Fast Setup

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+ installed
- Git installed

---

## ⚡ 5-Minute Setup

```bash
# 1. Clone and navigate (30 seconds)
git clone https://github.com/Beebbbyy/project-agri-safe.git
cd project-agri-safe
git checkout claude/review-test-webhook-01UYm9PfNqJzqinqrvAvKG3L

# 2. Start Docker services (1 minute)
docker compose up -d postgres redis
sleep 30  # Wait for services to initialize

# 3. Set up Python (2 minutes)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Configure environment (30 seconds)
export PYTHONPATH="$(pwd)"
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agrisafe_db
export POSTGRES_USER=agrisafe
export POSTGRES_PASSWORD=agrisafe_password
export REDIS_HOST=localhost
export REDIS_PORT=6379

# 5. Initialize database (30 seconds)
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db < sql/schema/01_init_schema.sql
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db < sql/schema/02_feature_tables.sql

# 6. Run tests (30 seconds)
./test_phase3.sh
```

---

## ✅ Verify It's Working

```bash
# Should see all green checkmarks ✓
./test_phase3.sh

# Check services
docker compose ps

# Check database
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt"
```

---

## 🎯 Run Your First Job

```bash
# Run daily weather statistics
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07

# Check results
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    -c "SELECT COUNT(*) FROM weather_daily_stats;"
```

---

## 📚 Full Documentation

Need more details? See:
- **Complete Setup:** `LOCAL_SETUP_GUIDE.md`
- **Troubleshooting:** `LOCAL_SETUP_GUIDE.md#troubleshooting`
- **Phase 3 Overview:** `PHASE3_IMPLEMENTATION_SUMMARY.md`

---

## 🆘 Common Issues

### Docker not starting?
```bash
# Check Docker Desktop is running
docker --version

# Restart Docker Desktop
```

### Python errors?
```bash
# Make sure you're in virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Connection errors?
```bash
# Wait longer for services
sleep 30

# Check services are up
docker compose ps
```

---

**That's it!** You're ready to test Phase 3 locally! 🎉

For detailed instructions, see `LOCAL_SETUP_GUIDE.md`
