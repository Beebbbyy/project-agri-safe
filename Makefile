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
	@echo ""
	@echo "Redis:"
	@echo "  make redis-connect  - Connect to Redis CLI"
	@echo ""
	@echo "Airflow:"
	@echo "  make airflow-restart - Restart Airflow services"
	@echo "  make airflow-logs   - View Airflow logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Stop and remove containers"
	@echo "  make clean-all      - Remove everything including volumes (DANGER!)"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run tests"
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
