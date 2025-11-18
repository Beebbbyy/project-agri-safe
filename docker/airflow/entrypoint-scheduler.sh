#!/bin/bash
set -e

echo "Installing additional Python packages..."
pip install --no-cache-dir -r /opt/airflow/requirements.txt

echo "Waiting for webserver to initialize database..."
sleep 10

echo "Starting Airflow scheduler..."
exec airflow scheduler
