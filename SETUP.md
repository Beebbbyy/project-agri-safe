# 🚀 Project Agri-Safe - Phase 1 Setup Guide

This guide will help you set up the development environment for **Project Agri-Safe** using Docker.

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Git**

### Installing Docker

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

**macOS:**
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
```

**Windows:**
```powershell
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
# Enable WSL 2 backend for better performance
```

Verify installation:
```bash
docker --version
docker compose version
```

## 🏗️ Project Structure

```
project-agri-safe/
├── airflow/                 # Apache Airflow data pipelines
│   ├── dags/               # Airflow DAG definitions
│   ├── logs/               # Airflow execution logs
│   └── plugins/            # Custom Airflow plugins
├── backend/                # FastAPI backend application (Phase 4)
│   ├── app/               # Application code
│   └── tests/             # Backend tests
├── frontend/              # Streamlit web application (Phase 5)
├── sql/                   # Database schemas and migrations
│   ├── schema/            # Table definitions
│   ├── migrations/        # Database migrations
│   └── seeds/             # Seed data for development
├── data/                  # Data storage
│   ├── raw/              # Raw data from APIs
│   └── processed/        # Processed/cleaned data
├── docker/               # Docker-related files
├── docs/                 # Documentation
├── .env.example          # Environment variables template
├── docker-compose.yml    # Docker services configuration
└── README.md            # Project overview
```

## ⚙️ Initial Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Beebbbyy/project-agri-safe.git
cd project-agri-safe
```

### Step 2: Configure Environment Variables

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

**For development**, the default values in `.env.example` work fine. Just copy it:
```bash
# Default credentials for development:
# PostgreSQL: agrisafe / agrisafe_password
# Airflow: admin / admin
```

**For production**, you MUST change:
- All passwords
- Secret keys (generate new ones)
- API keys (add your actual keys)

Generate a new Fernet key for Airflow:
```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Step 3: Create Required Directories

The directories should already exist, but if not:
```bash
mkdir -p airflow/{dags,logs,plugins}
mkdir -p data/{raw,processed}
mkdir -p sql/{schema,migrations,seeds}
```

### Step 4: Set Proper Permissions (Linux/macOS)

Airflow needs specific permissions:
```bash
# Set your user ID for Airflow
echo -e "\nAIRFLOW_UID=$(id -u)" >> .env

# Set permissions
sudo chown -R $(id -u):$(id -g) airflow/
chmod -R 755 airflow/
```

## 🐳 Starting the Services

### Start All Services

```bash
docker compose up -d
```

This will start:
- **PostgreSQL** (Main database) - Port 5432
- **Redis** (Cache & message broker) - Port 6379
- **Airflow PostgreSQL** (Airflow metadata) - Internal
- **Airflow Webserver** - Port 8080
- **Airflow Scheduler** - Background service
- **Airflow Worker** - Background service

### Check Service Status

```bash
docker compose ps
```

All services should show as "healthy" after a few minutes.

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f postgres
docker compose logs -f airflow-webserver
docker compose logs -f redis
```

## 🔍 Accessing the Services

Once all services are running:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow UI** | http://localhost:8080 | admin / admin |
| **PostgreSQL** | localhost:5432 | agrisafe / agrisafe_password |
| **Redis** | localhost:6379 | (no password) |

### Airflow Web Interface

1. Open browser: http://localhost:8080
2. Login with: `admin` / `admin`
3. You should see the Airflow dashboard

### Database Access

Connect to PostgreSQL using any client:

```bash
# Using psql command line
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db

# Connection string
postgresql://agrisafe:agrisafe_password@localhost:5432/agrisafe_db
```

### Redis Access

```bash
# Connect to Redis CLI
docker exec -it agrisafe-redis redis-cli

# Test connection
docker exec -it agrisafe-redis redis-cli ping
# Should return: PONG
```

## 🗄️ Database Schema

The database schema is automatically created when PostgreSQL starts.

### View Tables

```bash
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt"
```

### Main Tables Created

- `users` - Farmer accounts
- `regions` - Philippine regions/provinces
- `farms` - Farm information
- `crop_types` - Catalog of crops
- `plantings` - Planting records
- `weather_forecasts` - Weather data
- `typhoon_alerts` - Typhoon warnings
- `flood_risk_assessments` - Flood predictions
- `harvest_recommendations` - AI recommendations
- `chat_conversations` - Chat history
- `chat_messages` - Individual messages
- `api_logs` - API call logs

### Sample Data

The seed data includes:
- 30 Philippine regions
- 22 crop types (rice, corn, vegetables, fruits)
- 3 test users (password: `password123`)
- 5 sample plantings

## 🧪 Testing the Setup

### 1. Check PostgreSQL

```bash
# Connect and run a query
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "SELECT COUNT(*) FROM crop_types;"
```

Expected output: 22 crop types

### 2. Check Redis

```bash
# Test Redis is working
docker exec -it agrisafe-redis redis-cli ping
docker exec -it agrisafe-redis redis-cli SET test "Hello"
docker exec -it agrisafe-redis redis-cli GET test
```

### 3. Check Airflow

1. Visit http://localhost:8080
2. Login with admin/admin
3. Check that DAGs page loads

### 4. Run Health Checks

```bash
# Check all container health
docker compose ps

# All should show "healthy" status
```

## 🛠️ Common Commands

### Stop All Services

```bash
docker compose down
```

### Stop and Remove All Data (⚠️ Careful!)

```bash
docker compose down -v
# This deletes all database data!
```

### Restart a Specific Service

```bash
docker compose restart postgres
docker compose restart airflow-webserver
```

### Rebuild Containers

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### View Resource Usage

```bash
docker stats
```

### Clean Up Docker System

```bash
# Remove unused containers, networks, images
docker system prune -a
```

## 🐛 Troubleshooting

### Airflow Won't Start

**Problem:** Airflow webserver fails to start

**Solution:**
```bash
# Check logs
docker compose logs airflow-webserver

# Ensure proper permissions
sudo chown -R $(id -u):$(id -g) airflow/

# Restart services
docker compose restart airflow-webserver
```

### PostgreSQL Connection Refused

**Problem:** Can't connect to PostgreSQL

**Solution:**
```bash
# Check if container is running
docker compose ps postgres

# Check logs
docker compose logs postgres

# Restart PostgreSQL
docker compose restart postgres
```

### Port Already in Use

**Problem:** Error: "port is already allocated"

**Solution:**
```bash
# Find process using the port (example: 5432)
sudo lsof -i :5432
# Or on some systems:
sudo netstat -tulpn | grep 5432

# Kill the process or change port in docker-compose.yml
```

### Airflow Database Migration Failed

**Problem:** Airflow complains about database migrations

**Solution:**
```bash
# Reset Airflow database
docker compose down
docker volume rm project-agri-safe_airflow_postgres_data
docker compose up -d
```

### Not Enough Disk Space

**Problem:** Docker running out of space

**Solution:**
```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a --volumes
```

### Permission Denied Errors (Linux)

**Problem:** Permission errors with Airflow logs/dags

**Solution:**
```bash
# Fix ownership
sudo chown -R $USER:$USER airflow/
chmod -R 755 airflow/

# Add AIRFLOW_UID to .env
echo "AIRFLOW_UID=$(id -u)" >> .env

# Restart
docker compose down
docker compose up -d
```

## 📊 Monitoring

### Check Container Health

```bash
docker compose ps
```

### View Resource Usage

```bash
docker stats
```

### PostgreSQL Monitoring

```bash
# Check active connections
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "SELECT count(*) FROM pg_stat_activity;"

# Check database size
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "SELECT pg_size_pretty(pg_database_size('agrisafe_db'));"
```

### Redis Monitoring

```bash
# Check Redis info
docker exec -it agrisafe-redis redis-cli INFO

# Check memory usage
docker exec -it agrisafe-redis redis-cli INFO memory
```

## 🔐 Security Notes

### For Development
- Default credentials are fine
- Services exposed on localhost only

### For Production
You MUST:
1. ✅ Change all passwords
2. ✅ Generate new secret keys
3. ✅ Use environment-specific .env files
4. ✅ Enable SSL/TLS
5. ✅ Use Docker secrets for sensitive data
6. ✅ Implement proper firewall rules
7. ✅ Regular security updates
8. ✅ Enable database backups

## 🎯 Next Steps

After completing Phase 1 setup:

1. ✅ Verify all services are running
2. ✅ Test database connections
3. ✅ Explore Airflow UI
4. 📝 Move to **Phase 2: Data Ingestion**
   - Create Airflow DAGs for weather data
   - Set up PAGASA API integration
   - Implement data storage pipelines

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review container logs: `docker compose logs [service-name]`
3. Search existing GitHub issues
4. Create a new issue with:
   - Error messages
   - Docker version
   - Operating system
   - Steps to reproduce

---

**Phase 1 Complete! 🎉** You now have a fully functional development environment ready for data pipeline development.
