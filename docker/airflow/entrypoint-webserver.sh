#!/bin/bash
set -e

echo "Installing additional Python packages..."
pip install --no-cache-dir -r /opt/airflow/requirements.txt

echo "Initializing Airflow database..."

# Try to reset the database first to clear any corrupted state
# This will fail if database doesn't exist yet, which is fine
echo "Attempting to reset database (this may fail on first run, which is expected)..."
airflow db reset --yes 2>/dev/null || echo "Database reset skipped (database may not exist yet)"

# Initialize the database with a clean slate
echo "Initializing database..."
airflow db init

# Create admin user (|| true means it won't fail if user already exists)
echo "Creating admin user..."
airflow users create \
    --username ${_AIRFLOW_WWW_USER_USERNAME} \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password ${_AIRFLOW_WWW_USER_PASSWORD} || echo "Admin user already exists"

echo "Starting Airflow webserver..."
exec airflow webserver
