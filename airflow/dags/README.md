# Airflow DAGs

This directory contains Apache Airflow Directed Acyclic Graphs (DAGs) for data pipelines.

## Phase 2: Data Ingestion DAGs (Weeks 3-4)

Upcoming DAGs to be implemented:

### 1. Weather Data Ingestion
- **File:** `weather_ingestion_dag.py`
- **Schedule:** Every 6 hours
- **Tasks:**
  - Fetch PAGASA weather forecasts
  - Fetch OpenWeatherMap data (backup)
  - Validate and clean data
  - Store in PostgreSQL `weather_forecasts` table

### 2. Typhoon Alert Monitoring
- **File:** `typhoon_alert_dag.py`
- **Schedule:** Every 2 hours
- **Tasks:**
  - Fetch PAGASA typhoon bulletins
  - Parse alert levels and affected regions
  - Store in `typhoon_alerts` table
  - Trigger notifications for high alerts

### 3. Historical Data Collection
- **File:** `historical_data_dag.py`
- **Schedule:** Daily at 2 AM
- **Tasks:**
  - Fetch historical weather data
  - Aggregate data for ML training
  - Update statistics tables

## DAG Development Guidelines

### Basic DAG Structure

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'agrisafe',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'example_dag',
    default_args=default_args,
    description='Example DAG for Project Agri-Safe',
    schedule_interval='@daily',
    catchup=False,
)

def example_task():
    print("Running example task")

task = PythonOperator(
    task_id='example_task',
    python_callable=example_task,
    dag=dag,
)
```

### Best Practices

1. **Idempotency**: Tasks should produce the same result when run multiple times
2. **Error Handling**: Always include try-except blocks and proper logging
3. **Data Validation**: Validate API responses before storing
4. **Database Connections**: Use connection pooling and proper cleanup
5. **Monitoring**: Log important metrics and data quality checks

### Testing DAGs

```bash
# Test DAG syntax
docker exec -it agrisafe-airflow-webserver airflow dags test <dag_id> <execution_date>

# Test specific task
docker exec -it agrisafe-airflow-webserver airflow tasks test <dag_id> <task_id> <execution_date>

# List all DAGs
docker exec -it agrisafe-airflow-webserver airflow dags list
```

### Environment Variables

DAGs can access environment variables from `.env`:

```python
import os

PAGASA_API_KEY = os.getenv('PAGASA_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
```

## Current Status

- ✅ Airflow infrastructure set up
- ⏳ Awaiting Phase 2 implementation
- 📝 DAG templates to be created

## Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Writing DAGs](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.html)
