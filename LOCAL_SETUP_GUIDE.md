# Local Machine Setup Guide - Phase 3

## 🖥️ System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM:** 8GB minimum (16GB recommended)
- **Disk Space:** 10GB free space
- **CPU:** 4 cores recommended for PySpark

### Required Software
- **Docker Desktop:** Latest version
- **Python:** 3.11 or higher
- **Git:** Latest version

---

## 📦 Step-by-Step Setup

### Step 1: Install Docker Desktop

#### Windows
```bash
# Download from: https://www.docker.com/products/docker-desktop/
# Install and start Docker Desktop
# Ensure WSL 2 is enabled (recommended)
```

#### macOS
```bash
# Download from: https://www.docker.com/products/docker-desktop/
# Install Docker Desktop for Mac
# Start Docker Desktop from Applications
```

#### Linux (Ubuntu/Debian)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

### Step 2: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Beebbbyy/project-agri-safe.git
cd project-agri-safe

# Checkout the Phase 3 branch
git checkout claude/review-test-webhook-01UYm9PfNqJzqinqrvAvKG3L
```

### Step 3: Start Docker Services

```bash
# Start PostgreSQL and Redis
docker compose up -d postgres redis

# Wait 30 seconds for services to initialize
sleep 30

# Verify services are running
docker compose ps

# Expected output:
# NAME                  STATUS
# agrisafe-postgres     Up
# agrisafe-redis        Up
```

### Step 4: Set Up Python Environment

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n agrisafe python=3.11 -y

# Activate environment
conda activate agrisafe

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Set Environment Variables

#### On Windows (PowerShell)
```powershell
$env:PYTHONPATH = "$PWD"
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
$env:POSTGRES_DB = "agrisafe_db"
$env:POSTGRES_USER = "agrisafe"
$env:POSTGRES_PASSWORD = "agrisafe_password"
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = "6379"
```

#### On Windows (Command Prompt)
```cmd
set PYTHONPATH=%CD%
set POSTGRES_HOST=localhost
set POSTGRES_PORT=5432
set POSTGRES_DB=agrisafe_db
set POSTGRES_USER=agrisafe
set POSTGRES_PASSWORD=agrisafe_password
set REDIS_HOST=localhost
set REDIS_PORT=6379
```

#### On macOS/Linux
```bash
export PYTHONPATH="$(pwd)"
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agrisafe_db
export POSTGRES_USER=agrisafe
export POSTGRES_PASSWORD=agrisafe_password
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

**Pro Tip:** Create a `.env` file for persistent configuration:
```bash
# Create .env file
cat > .env << 'EOF'
PYTHONPATH=/path/to/project-agri-safe
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agrisafe_db
POSTGRES_USER=agrisafe
POSTGRES_PASSWORD=agrisafe_password
REDIS_HOST=localhost
REDIS_PORT=6379
EOF

# Load it automatically with:
source .env  # On macOS/Linux
# Or use python-dotenv in scripts
```

### Step 6: Initialize Database Schema

```bash
# Apply Phase 1-2 schema (base tables)
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    < sql/schema/01_init_schema.sql

# Apply Phase 3 schema (feature tables)
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    < sql/schema/02_feature_tables.sql

# Verify tables were created
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    -c "\dt"
```

Expected output:
```
                List of relations
 Schema |            Name            | Type  |  Owner
--------+----------------------------+-------+---------
 public | feature_metadata           | table | agrisafe
 public | flood_risk_indicators      | table | agrisafe
 public | regions                    | table | agrisafe
 public | weather_daily_stats        | table | agrisafe
 public | weather_forecasts          | table | agrisafe
 public | weather_rolling_features   | table | agrisafe
```

### Step 7: Seed Sample Data (Optional)

```bash
# Insert sample region data
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
INSERT INTO regions (region_name, latitude, longitude) VALUES
    ('Metro Manila', 14.5995, 120.9842),
    ('Cebu', 10.3157, 123.8854),
    ('Davao', 7.0731, 125.6128)
ON CONFLICT DO NOTHING;
EOF
```

---

## 🧪 Running Phase 3 Tests

### Quick Test (30 seconds)
```bash
# Run the automated test script
chmod +x test_phase3.sh
./test_phase3.sh
```

### Manual Testing

#### Test 1: Daily Weather Statistics
```bash
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07 \
    --mode append

# Check results
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    -c "SELECT COUNT(*) FROM weather_daily_stats;"
```

#### Test 2: Rolling Features
```bash
python -m src.processing.jobs.rolling_features \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --windows 7,14,30 \
    --mode append

# Check results
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    -c "SELECT window_days, COUNT(*) FROM weather_rolling_features GROUP BY window_days;"
```

#### Test 3: Flood Risk Indicators
```bash
python -m src.processing.jobs.flood_risk_indicators \
    --start-date 2025-01-01 \
    --end-date 2025-01-14 \
    --mode append

# Check results
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db \
    -c "SELECT flood_risk_level, COUNT(*) FROM flood_risk_indicators GROUP BY flood_risk_level;"
```

#### Test 4: Unit Tests
```bash
# Run all unit tests
pytest tests/processing/ -v

# Run with coverage
pytest tests/processing/ --cov=src/processing --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser
```

---

## 🔍 Verification Commands

### Check Docker Services
```bash
# Check service status
docker compose ps

# View PostgreSQL logs
docker compose logs postgres

# View Redis logs
docker compose logs redis

# Connect to PostgreSQL
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db

# Connect to Redis
docker exec -it agrisafe-redis redis-cli
```

### Database Queries
```bash
# Count records in each table
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
SELECT 'weather_forecasts' AS table_name, COUNT(*) FROM weather_forecasts
UNION ALL SELECT 'weather_daily_stats', COUNT(*) FROM weather_daily_stats
UNION ALL SELECT 'weather_rolling_features', COUNT(*) FROM weather_rolling_features
UNION ALL SELECT 'flood_risk_indicators', COUNT(*) FROM flood_risk_indicators
UNION ALL SELECT 'feature_metadata', COUNT(*) FROM feature_metadata;
EOF
```

### Redis Cache Check
```bash
# List all cached keys
docker exec -it agrisafe-redis redis-cli KEYS "agrisafe:*"

# Check cache statistics
docker exec -it agrisafe-redis redis-cli INFO stats

# Get a specific cached value
docker exec -it agrisafe-redis redis-cli GET "agrisafe:weather_stats:1:2025-01-01"
```

---

## 🐛 Troubleshooting

### Issue: Docker services won't start

**Solution:**
```bash
# Stop all services
docker compose down

# Remove old volumes (WARNING: deletes data)
docker compose down -v

# Restart services
docker compose up -d postgres redis

# Check logs for errors
docker compose logs
```

### Issue: "Connection refused" when connecting to PostgreSQL

**Solutions:**
```bash
# 1. Wait longer for PostgreSQL to initialize
sleep 30

# 2. Check if PostgreSQL is ready
docker exec agrisafe-postgres pg_isready -U agrisafe

# 3. Verify port is exposed
docker compose ps postgres

# 4. Check Docker networking
docker network ls
docker network inspect project-agri-safe_default
```

### Issue: "No module named 'pyspark'"

**Solutions:**
```bash
# Option 1: Install PySpark with Java bundled
pip install pyspark==3.5.0

# Option 2: If Java is missing, install it
# Ubuntu/Debian:
sudo apt-get install openjdk-11-jdk

# macOS:
brew install openjdk@11

# Windows:
# Download from: https://adoptium.net/

# Verify Java installation
java -version
```

### Issue: PySpark jobs fail with memory errors

**Solution:**
```bash
# Increase Docker Desktop memory allocation
# Docker Desktop → Settings → Resources → Memory: 8GB+

# Or reduce Spark memory in jobs:
python -m src.processing.jobs.daily_weather_stats \
    --start-date 2025-01-01 \
    --end-date 2025-01-07 \
    --spark-memory 2g
```

### Issue: "PYTHONPATH not set" errors

**Solution:**
```bash
# Always run from project root
cd /path/to/project-agri-safe

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Verify
echo $PYTHONPATH
```

### Issue: Redis connection errors

**Solution:**
```bash
# Check Redis is running
docker exec agrisafe-redis redis-cli ping
# Should return: PONG

# Check Redis port
docker compose ps redis

# Test connection from Python
python3 << 'EOF'
import redis
client = redis.Redis(host='localhost', port=6379, decode_responses=True)
print(client.ping())  # Should print True
EOF
```

### Issue: No weather data to process

**Solution:**
```bash
# First, run Phase 2 ingestion to get weather data
python -m src.ingestion.pagasa_connector

# Or insert sample data manually
docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOF'
INSERT INTO weather_forecasts (region_id, forecast_date, temperature, rainfall, wind_speed, humidity)
SELECT
    (SELECT id FROM regions LIMIT 1),
    CURRENT_DATE - INTERVAL '1 day' * s.day,
    20 + (RANDOM() * 15),
    RANDOM() * 100,
    5 + (RANDOM() * 20),
    60 + (RANDOM() * 30)
FROM generate_series(0, 30) AS s(day);
EOF
```

---

## 🚀 Performance Tips

### 1. Optimize Docker Resources
```bash
# In Docker Desktop settings, allocate:
# - CPUs: 4+ cores
# - Memory: 8GB+
# - Disk: 20GB+
```

### 2. Use Local PySpark Mode
```bash
# Jobs automatically use local[*] mode for development
# This uses all available CPU cores
```

### 3. Enable Redis Persistence (Optional)
```yaml
# Add to docker-compose.yml under redis service:
redis:
  command: redis-server --appendonly yes
  volumes:
    - redis-data:/data
```

### 4. Database Connection Pooling
```bash
# Already configured in spark_session.py
# Adjust in src/processing/utils/spark_session.py if needed
```

---

## 📊 Expected Performance

### Local Machine Benchmarks
- **Daily Stats Job:** 15-30 seconds for 10K records
- **Rolling Features:** 20-40 seconds for 500 base records
- **Flood Risk:** 10-20 seconds for 1.5K records
- **Full Pipeline:** ~1-2 minutes end-to-end

### Resource Usage
- **Memory:** 2-4GB during job execution
- **CPU:** Will use all available cores
- **Disk I/O:** Minimal (<100MB/s)

---

## ✅ Success Checklist

After setup, verify:

- [ ] Docker Desktop is running
- [ ] PostgreSQL container is up and accepting connections
- [ ] Redis container is up and responding to PING
- [ ] Database schema is applied (8+ tables exist)
- [ ] Python virtual environment is activated
- [ ] All dependencies are installed (`pip list | grep pyspark`)
- [ ] PYTHONPATH is set correctly
- [ ] Environment variables are configured
- [ ] Sample data exists (regions table has entries)
- [ ] `./test_phase3.sh` runs without errors
- [ ] Unit tests pass (`pytest tests/processing/ -v`)

---

## 🎓 Next Steps After Setup

### 1. Explore the Data
```bash
# Interactive PostgreSQL session
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db

# Try some queries:
# SELECT * FROM weather_daily_stats LIMIT 5;
# SELECT * FROM v_latest_flood_risk;
```

### 2. Run Full Pipeline
```bash
# Run all jobs in sequence
make run-etl

# Or manually:
python -m src.processing.jobs.daily_weather_stats --start-date 2025-01-01 --end-date 2025-01-31
python -m src.processing.jobs.rolling_features --start-date 2025-01-01 --end-date 2025-01-31
python -m src.processing.jobs.flood_risk_indicators --start-date 2025-01-01 --end-date 2025-01-31
```

### 3. Set Up Airflow (Optional)
```bash
# Start Airflow services
docker compose up -d airflow-webserver airflow-scheduler

# Access Airflow UI
# Open http://localhost:8080
# Username: admin
# Password: admin
```

### 4. Start Phase 4 Development
```bash
# Phase 3 is ready! Move to Phase 4 (Backend API)
# See docs/PHASE4_PLAN.md for next steps
```

---

## 📞 Getting Help

### Log Files Location
- **Spark logs:** `logs/spark/`
- **Job logs:** `logs/jobs/`
- **Docker logs:** `docker compose logs [service]`

### Common Commands Cheat Sheet
```bash
# Restart everything
docker compose restart

# Stop everything
docker compose down

# View logs
docker compose logs -f postgres redis

# Clean slate (deletes data!)
docker compose down -v && docker compose up -d

# Check disk space
docker system df

# Clean up Docker
docker system prune -a
```

### Documentation
- **Phase 3 Docs:** `docs/PHASE3_SPARK_ETL.md`
- **Quick Start:** `docs/PHASE3_QUICK_START.md`
- **Quick Test:** `QUICK_TEST.md`
- **This Guide:** `LOCAL_SETUP_GUIDE.md`

---

## 🎉 You're Ready!

Once all checklist items are complete, run:
```bash
./test_phase3.sh
```

If you see all green ✓ checkmarks, congratulations! Phase 3 is fully operational on your local machine! 🚀

---

**Happy Testing!** 🎊
