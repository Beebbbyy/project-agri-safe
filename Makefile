# Project Agri-Safe - Makefile
# Convenience commands for managing Docker services

.PHONY: help setup up down restart logs status clean test db-connect redis-connect airflow-restart

# Default target
help:
	@echo "🌾 Project Agri-Safe - Available Commands"
	@echo ""
	@echo "Setup & Start:"
	@echo "  make setup          - Initial setup (copy .env, create dirs)"
	@echo "  make up             - Start all services"
	@echo "  make down           - Stop all services"
	@echo ""
	@echo "Management:"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View all logs (follow mode)"
	@echo "  make status         - Check service status"
	@echo ""
	@echo "Database:"
	@echo "  make db-connect     - Connect to PostgreSQL"
	@echo "  make db-reset       - Reset database (DANGER!)"
	@echo "  make db-backup      - Backup database"
	@echo "  make db-migrate-phase3 - Run Phase 3 migrations"
	@echo ""
	@echo "Redis:"
	@echo "  make redis-connect  - Connect to Redis CLI"
	@echo ""
	@echo "Airflow:"
	@echo "  make airflow-restart - Restart Airflow services"
	@echo "  make airflow-logs   - View Airflow logs"
	@echo "  make list-dags      - List Phase 3 DAGs"
	@echo "  make trigger-etl    - Trigger weather ETL"
	@echo "  make trigger-predictions - Trigger predictions"
	@echo ""
	@echo "Phase 3 - Spark:"
	@echo "  make spark-up       - Start Spark cluster"
	@echo "  make spark-down     - Stop Spark cluster"
	@echo "  make spark-logs     - View Spark logs"
	@echo "  make spark-status   - Check Spark status"
	@echo ""
	@echo "Phase 3 - ETL & Processing:"
	@echo "  make run-etl        - Run weather ETL pipeline"
	@echo "  make run-features   - Generate rolling features"
	@echo ""
	@echo "Phase 3 - ML Models:"
	@echo "  make train-model    - Train flood risk model"
	@echo "  make run-predictions - Generate predictions"
	@echo "  make test-model-v1  - Test rule-based model"
	@echo "  make test-model-v2  - Test ML model"
	@echo ""
	@echo "Phase 3 - Data Quality:"
	@echo "  make quality-checks - Run quality validation"
	@echo "  make quality-report - Generate quality report"
	@echo ""
	@echo "Phase 3 - Quick Start:"
	@echo "  make phase3-setup   - Setup Phase 3 infrastructure"
	@echo "  make phase3-init    - Full Phase 3 initialization"
	@echo "  make phase3-status  - Check Phase 3 status"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Stop and remove containers"
	@echo "  make clean-all      - Remove everything including volumes (DANGER!)"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run tests"
	@echo "  make test-phase3    - Run Phase 3 tests"
	@echo "  make lint           - Run linters"
	@echo ""

# Initial setup
setup:
	@echo "🔧 Setting up Project Agri-Safe..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file from .env.example"; \
	else \
		echo "ℹ️  .env file already exists"; \
	fi
	@mkdir -p airflow/{dags,logs,plugins}
	@mkdir -p data/{raw,processed}
	@mkdir -p backend/{app,tests}
	@mkdir -p frontend
	@mkdir -p sql/{schema,migrations,seeds}
	@mkdir -p docker docs
	@echo "✅ Directories created"
	@if [ "$(shell uname)" != "Darwin" ] && [ "$(shell uname)" != "Windows" ]; then \
		if ! grep -q "AIRFLOW_UID" .env; then \
			echo "AIRFLOW_UID=$(shell id -u)" >> .env; \
			echo "✅ Added AIRFLOW_UID to .env"; \
		fi; \
	fi
	@echo "✅ Setup complete! Run 'make up' to start services."

# Start all services
up:
	@echo "🚀 Starting all services..."
	docker compose up -d
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 10
	@echo "✅ Services started!"
	@echo ""
	@echo "📊 Access services at:"
	@echo "  - Airflow UI: http://localhost:8080 (admin/admin)"
	@echo "  - PostgreSQL: localhost:5432 (agrisafe/agrisafe_password)"
	@echo "  - Redis: localhost:6379"

# Stop all services
down:
	@echo "🛑 Stopping all services..."
	docker compose down
	@echo "✅ Services stopped"

# Restart all services
restart:
	@echo "🔄 Restarting all services..."
	docker compose restart
	@echo "✅ Services restarted"

# View logs
logs:
	docker compose logs -f

# Check service status
status:
	@echo "📊 Service Status:"
	@docker compose ps

# Clean up containers
clean:
	@echo "🧹 Cleaning up containers..."
	docker compose down
	@echo "✅ Cleanup complete"

# Clean everything including volumes (DANGER!)
clean-all:
	@echo "⚠️  WARNING: This will delete all data!"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		docker compose down -v; \
		echo "✅ All data removed"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Connect to PostgreSQL
db-connect:
	@echo "🗄️  Connecting to PostgreSQL..."
	docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db

# Reset database (DANGER!)
db-reset:
	@echo "⚠️  WARNING: This will reset the database!"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		docker compose down; \
		docker volume rm project-agri-safe_postgres_data || true; \
		docker compose up -d postgres; \
		echo "✅ Database reset complete"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Backup database
db-backup:
	@echo "💾 Backing up database..."
	@mkdir -p backups
	docker exec agrisafe-postgres pg_dump -U agrisafe agrisafe_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/"

# Connect to Redis
redis-connect:
	@echo "📦 Connecting to Redis..."
	docker exec -it agrisafe-redis redis-cli

# Restart Airflow services
airflow-restart:
	@echo "🔄 Restarting Airflow services..."
	docker compose restart airflow-webserver airflow-scheduler airflow-worker
	@echo "✅ Airflow restarted"

# View Airflow logs
airflow-logs:
	docker compose logs -f airflow-webserver airflow-scheduler airflow-worker

# Run tests (placeholder for future)
test:
	@echo "🧪 Running tests..."
	@echo "⚠️  Tests not yet implemented (Phase 6)"

# Run linters (placeholder for future)
lint:
	@echo "🔍 Running linters..."
	@echo "⚠️  Linting not yet configured (Phase 4)"

# ============================================================================
# PHASE 3 COMMANDS: Data Processing & ML
# ============================================================================

# Spark services
.PHONY: spark-up spark-down spark-logs spark-status

spark-up:
	@echo "⚡ Starting Spark services..."
	docker compose up -d spark-master spark-worker
	@sleep 5
	@echo "✅ Spark services started!"
	@echo "  - Spark Master UI: http://localhost:8081"
	@echo "  - Spark Worker UI: http://localhost:8083"
	@echo "  - Spark Application UI: http://localhost:4040 (when job running)"

spark-down:
	@echo "🛑 Stopping Spark services..."
	docker compose stop spark-master spark-worker
	@echo "✅ Spark stopped"

spark-logs:
	@echo "📜 Spark logs:"
	docker compose logs -f spark-master spark-worker

spark-status:
	@echo "📊 Spark Status:"
	@docker compose ps spark-master spark-worker

# Database migrations
.PHONY: db-migrate-phase3

db-migrate-phase3:
	@echo "🗄️  Running Phase 3 database migrations..."
	docker exec -i agrisafe-postgres psql -U agrisafe -d agrisafe_db < sql/migrations/03_phase3_tables.sql
	@echo "✅ Phase 3 tables created!"

# ETL Jobs
.PHONY: run-etl run-features

run-etl:
	@echo "⚙️  Running weather ETL pipeline..."
	docker exec agrisafe-airflow-worker python -m src.processing.spark_jobs.weather_etl \
		--start-date $(shell date -d '30 days ago' +%Y-%m-%d) \
		--end-date $(shell date +%Y-%m-%d)

run-features:
	@echo "🔧 Generating rolling features..."
	docker exec agrisafe-airflow-worker python -m src.processing.spark_jobs.rolling_features \
		--start-date $(shell date -d '30 days ago' +%Y-%m-%d) \
		--end-date $(shell date +%Y-%m-%d)

# ML Model commands
.PHONY: train-model run-predictions test-model-v1 test-model-v2

train-model:
	@echo "🤖 Training flood risk model..."
	docker exec agrisafe-airflow-worker python -m src.models.training_pipeline \
		--days 180 \
		--test-size 0.2
	@echo "✅ Model training complete! Check models/ directory"

run-predictions:
	@echo "🔮 Generating flood risk predictions..."
	docker exec agrisafe-airflow-worker python -m src.models.batch_predictions \
		--date $(shell date +%Y-%m-%d) \
		--model-version v2
	@echo "✅ Predictions generated for all regions"

test-model-v1:
	@echo "🧪 Testing rule-based model (v1)..."
	docker exec agrisafe-airflow-worker python -c "\
from src.models.flood_risk_v1 import RuleBasedFloodModel; \
model = RuleBasedFloodModel(); \
features = {'rainfall_1d': 120, 'rainfall_7d': 300, 'elevation': 50, 'historical_flood_count': 3}; \
result = model.predict(features); \
print(f'Risk Level: {result.risk_level}'); \
print(f'Confidence: {result.confidence_score:.2f}'); \
print(f'Recommendation: {result.recommendation}'); \
"

test-model-v2:
	@echo "🧪 Testing ML-based model (v2)..."
	@echo "⚠️  Ensure model is trained first with 'make train-model'"
	docker exec agrisafe-airflow-worker python -m src.models.batch_predictions --date $(shell date +%Y-%m-%d) --model-version v2

# Data Quality commands
.PHONY: quality-checks quality-report quality-dashboard

quality-checks:
	@echo "✅ Running data quality checks..."
	docker exec agrisafe-airflow-worker python -m src.quality.validators
	@echo "✅ Quality checks complete"

quality-report:
	@echo "📊 Generating quality report..."
	docker exec agrisafe-airflow-worker python -m src.quality.monitoring
	@echo "✅ Report generated"

quality-dashboard:
	@echo "📈 Opening quality dashboard..."
	@echo "⚠️  Dashboard feature coming in Phase 5"

# Airflow DAG management
.PHONY: trigger-etl trigger-predictions trigger-quality trigger-training list-dags

list-dags:
	@echo "📋 Available Airflow DAGs:"
	docker exec agrisafe-airflow-webserver airflow dags list | grep -E "(weather|flood|quality)"

trigger-etl:
	@echo "▶️  Triggering weather ETL DAG..."
	docker exec agrisafe-airflow-webserver airflow dags trigger weather_data_processing
	@echo "✅ DAG triggered! Check Airflow UI: http://localhost:8080"

trigger-predictions:
	@echo "▶️  Triggering flood predictions DAG..."
	docker exec agrisafe-airflow-webserver airflow dags trigger flood_risk_predictions
	@echo "✅ DAG triggered! Check Airflow UI: http://localhost:8080"

trigger-quality:
	@echo "▶️  Triggering data quality DAG..."
	docker exec agrisafe-airflow-webserver airflow dags trigger data_quality_monitoring
	@echo "✅ DAG triggered! Check Airflow UI: http://localhost:8080"

trigger-training:
	@echo "▶️  Triggering model training DAG..."
	docker exec agrisafe-airflow-webserver airflow dags trigger flood_model_training
	@echo "✅ DAG triggered! Check Airflow UI: http://localhost:8080"

# Phase 3 testing
.PHONY: test-phase3 test-phase3-integration

test-phase3:
	@echo "🧪 Running Phase 3 unit tests..."
	docker exec agrisafe-airflow-worker pytest tests/processing tests/models tests/quality -v --cov
	@echo "✅ Tests complete"

test-phase3-integration:
	@echo "🧪 Running Phase 3 integration tests..."
	docker exec agrisafe-airflow-worker pytest tests/integration/test_phase3.py -v
	@echo "✅ Integration tests complete"

# Phase 3 complete setup
.PHONY: phase3-setup phase3-init phase3-status

phase3-setup: spark-up db-migrate-phase3
	@echo "🎉 Phase 3 infrastructure ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make run-etl' to process weather data"
	@echo "  2. Run 'make train-model' to train ML model"
	@echo "  3. Run 'make run-predictions' to generate predictions"
	@echo "  4. Run 'make quality-checks' to validate data"

phase3-init: phase3-setup run-etl train-model run-predictions
	@echo "🚀 Phase 3 fully initialized with sample data!"

phase3-status:
	@echo "📊 Phase 3 Status Check"
	@echo ""
	@echo "Spark Services:"
	@docker compose ps spark-master spark-worker
	@echo ""
	@echo "Database Tables:"
	@docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt weather_daily_stats" 2>/dev/null || echo "❌ weather_daily_stats not found"
	@docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt feature_store" 2>/dev/null || echo "❌ feature_store not found"
	@docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt data_quality_checks" 2>/dev/null || echo "❌ data_quality_checks not found"
	@echo ""
	@echo "Models:"
	@ls -lh models/*.pkl 2>/dev/null || echo "❌ No trained models found"
	@echo ""
	@echo "Airflow DAGs:"
	@docker exec agrisafe-airflow-webserver airflow dags list 2>/dev/null | grep -E "(weather|flood|quality)" || echo "❌ Phase 3 DAGs not loaded"

# ============================================================================
# END PHASE 3 COMMANDS
# ============================================================================

# Development helpers
dev-setup: setup up
	@echo "🎉 Development environment ready!"

# Health check
health:
	@echo "🏥 Running health checks..."
	@echo ""
	@echo "PostgreSQL:"
	@docker exec agrisafe-postgres pg_isready -U agrisafe || echo "❌ PostgreSQL not ready"
	@echo ""
	@echo "Redis:"
	@docker exec agrisafe-redis redis-cli ping || echo "❌ Redis not ready"
	@echo ""
	@echo "Airflow:"
	@curl -s http://localhost:8080/health > /dev/null && echo "✅ Airflow webserver is healthy" || echo "❌ Airflow not ready"

# Show environment info
info:
	@echo "📋 Environment Information"
	@echo ""
	@echo "Docker version:"
	@docker --version
	@echo ""
	@echo "Docker Compose version:"
	@docker compose version
	@echo ""
	@echo "Running containers:"
	@docker compose ps
