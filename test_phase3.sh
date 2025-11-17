#!/bin/bash

##############################################################################
# Test Script for Phase 3: Data Processing & ML
# Project Agri-Safe
##############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_RUN++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((TESTS_FAILED++))
    ((TESTS_RUN++))
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

echo "======================================================================"
echo "  🌾 Project Agri-Safe - Phase 3 Testing"
echo "  Data Processing & ML Infrastructure Validation"
echo "======================================================================"
echo ""

log_info "Starting Phase 3 infrastructure tests..."
echo ""

##############################################################################
# 1. SPARK CLUSTER TESTS
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. SPARK CLUSTER TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1.1: Spark Master is running
log_info "Testing Spark Master container..."
if docker ps | grep -q "agrisafe-spark-master"; then
    log_success "Spark Master container is running"
else
    log_error "Spark Master container is not running"
fi

# Test 1.2: Spark Worker is running
log_info "Testing Spark Worker container..."
if docker ps | grep -q "agrisafe-spark-worker"; then
    log_success "Spark Worker container is running"
else
    log_error "Spark Worker container is not running"
fi

# Test 1.3: Spark Master UI is accessible
log_info "Testing Spark Master UI accessibility..."
if curl -sf http://localhost:8081 > /dev/null; then
    log_success "Spark Master UI is accessible at http://localhost:8081"
else
    log_error "Spark Master UI is not accessible"
fi

# Test 1.4: Spark Worker UI is accessible
log_info "Testing Spark Worker UI accessibility..."
if curl -sf http://localhost:8083 > /dev/null; then
    log_success "Spark Worker UI is accessible at http://localhost:8083"
else
    log_error "Spark Worker UI is not accessible"
fi

# Test 1.5: Spark worker is connected to master
log_info "Testing Spark Worker connection to Master..."
WORKER_STATUS=$(docker exec agrisafe-spark-master curl -s http://localhost:8081 | grep -o "Workers ([0-9]*)" | grep -o "[0-9]*")
if [ "$WORKER_STATUS" -ge "1" ]; then
    log_success "Spark Worker is connected to Master ($WORKER_STATUS worker(s))"
else
    log_error "No Spark Workers connected to Master"
fi

echo ""

##############################################################################
# 2. DATABASE SCHEMA TESTS
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  2. DATABASE SCHEMA TESTS (Phase 3 Tables)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 2.1: weather_daily_stats table exists
log_info "Testing weather_daily_stats table..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt weather_daily_stats" 2>/dev/null | grep -q "weather_daily_stats"; then
    log_success "weather_daily_stats table exists"
else
    log_error "weather_daily_stats table not found"
fi

# Test 2.2: feature_store table exists
log_info "Testing feature_store table..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt feature_store" 2>/dev/null | grep -q "feature_store"; then
    log_success "feature_store table exists"
else
    log_error "feature_store table not found"
fi

# Test 2.3: data_quality_checks table exists
log_info "Testing data_quality_checks table..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt data_quality_checks" 2>/dev/null | grep -q "data_quality_checks"; then
    log_success "data_quality_checks table exists"
else
    log_error "data_quality_checks table not found"
fi

# Test 2.4: model_training_runs table exists
log_info "Testing model_training_runs table..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt model_training_runs" 2>/dev/null | grep -q "model_training_runs"; then
    log_success "model_training_runs table exists"
else
    log_error "model_training_runs table not found"
fi

# Test 2.5: model_predictions_log table exists
log_info "Testing model_predictions_log table..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt model_predictions_log" 2>/dev/null | grep -q "model_predictions_log"; then
    log_success "model_predictions_log table exists"
else
    log_error "model_predictions_log table not found"
fi

# Test 2.6: region_risk_indicators table exists
log_info "Testing region_risk_indicators table..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt region_risk_indicators" 2>/dev/null | grep -q "region_risk_indicators"; then
    log_success "region_risk_indicators table exists"
else
    log_error "region_risk_indicators table not found"
fi

# Test 2.7: etl_job_runs table exists
log_info "Testing etl_job_runs table..."
if docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -c "\dt etl_job_runs" 2>/dev/null | grep -q "etl_job_runs"; then
    log_success "etl_job_runs table exists"
else
    log_error "etl_job_runs table not found"
fi

echo ""

##############################################################################
# 3. PYTHON MODULE TESTS
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  3. PYTHON MODULE TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 3.1: Import weather ETL module
log_info "Testing weather ETL module import..."
if docker exec agrisafe-airflow-worker python -c "from src.processing.spark_jobs.weather_etl import WeatherETL" 2>/dev/null; then
    log_success "weather_etl module imports successfully"
else
    log_error "Failed to import weather_etl module"
fi

# Test 3.2: Import rolling features module
log_info "Testing rolling features module import..."
if docker exec agrisafe-airflow-worker python -c "from src.processing.spark_jobs.rolling_features import RollingFeatureEngine" 2>/dev/null; then
    log_success "rolling_features module imports successfully"
else
    log_error "Failed to import rolling_features module"
fi

# Test 3.3: Import flood risk v1 module
log_info "Testing flood risk v1 module import..."
if docker exec agrisafe-airflow-worker python -c "from src.models.flood_risk_v1 import RuleBasedFloodModel" 2>/dev/null; then
    log_success "flood_risk_v1 module imports successfully"
else
    log_error "Failed to import flood_risk_v1 module"
fi

# Test 3.4: Import flood risk v2 module
log_info "Testing flood risk v2 module import..."
if docker exec agrisafe-airflow-worker python -c "from src.models.flood_risk_v2 import MLFloodModel" 2>/dev/null; then
    log_success "flood_risk_v2 module imports successfully"
else
    log_error "Failed to import flood_risk_v2 module"
fi

# Test 3.5: Import training pipeline module
log_info "Testing training pipeline module import..."
if docker exec agrisafe-airflow-worker python -c "from src.models.training_pipeline import FloodModelTrainingPipeline" 2>/dev/null; then
    log_success "training_pipeline module imports successfully"
else
    log_error "Failed to import training_pipeline module"
fi

# Test 3.6: Import batch predictions module
log_info "Testing batch predictions module import..."
if docker exec agrisafe-airflow-worker python -c "from src.models.batch_predictions import FloodRiskBatchPredictor" 2>/dev/null; then
    log_success "batch_predictions module imports successfully"
else
    log_error "Failed to import batch_predictions module"
fi

# Test 3.7: Import validators module
log_info "Testing validators module import..."
if docker exec agrisafe-airflow-worker python -c "from src.quality.validators import WeatherDataValidator" 2>/dev/null; then
    log_success "validators module imports successfully"
else
    log_error "Failed to import validators module"
fi

# Test 3.8: Import monitoring module
log_info "Testing monitoring module import..."
if docker exec agrisafe-airflow-worker python -c "from src.quality.monitoring import QualityMonitor" 2>/dev/null; then
    log_success "monitoring module imports successfully"
else
    log_error "Failed to import monitoring module"
fi

echo ""

##############################################################################
# 4. DEPENDENCY TESTS
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  4. DEPENDENCY TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 4.1: PySpark installed
log_info "Testing PySpark installation..."
if docker exec agrisafe-spark-master python -c "import pyspark; print(pyspark.__version__)" 2>/dev/null | grep -q "3.5"; then
    log_success "PySpark 3.5.x is installed"
else
    log_error "PySpark not properly installed"
fi

# Test 4.2: XGBoost installed
log_info "Testing XGBoost installation..."
if docker exec agrisafe-airflow-worker python -c "import xgboost; print(xgboost.__version__)" 2>/dev/null | grep -q "2.0"; then
    log_success "XGBoost 2.0.x is installed"
else
    log_error "XGBoost not properly installed"
fi

# Test 4.3: scikit-learn installed
log_info "Testing scikit-learn installation..."
if docker exec agrisafe-airflow-worker python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null | grep -q "1.4"; then
    log_success "scikit-learn 1.4.x is installed"
else
    log_error "scikit-learn not properly installed"
fi

# Test 4.4: pandas installed
log_info "Testing pandas installation..."
if docker exec agrisafe-airflow-worker python -c "import pandas; print(pandas.__version__)" 2>/dev/null | grep -q "2.1"; then
    log_success "pandas 2.1.x is installed"
else
    log_error "pandas not properly installed"
fi

# Test 4.5: PostgreSQL JDBC driver exists
log_info "Testing PostgreSQL JDBC driver..."
if docker exec agrisafe-spark-master ls /opt/spark/jars/ 2>/dev/null | grep -q "postgresql"; then
    log_success "PostgreSQL JDBC driver is present"
else
    log_error "PostgreSQL JDBC driver not found"
fi

echo ""

##############################################################################
# 5. AIRFLOW DAG TESTS
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  5. AIRFLOW DAG TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 5.1: Weather processing DAG exists
log_info "Testing weather processing DAG..."
if docker exec agrisafe-airflow-webserver python /opt/airflow/dags/weather_processing_dag.py 2>/dev/null; then
    log_success "weather_processing_dag.py syntax is valid"
else
    log_error "weather_processing_dag.py has syntax errors"
fi

# Test 5.2: Flood model DAG exists
log_info "Testing flood model DAG..."
if docker exec agrisafe-airflow-webserver python /opt/airflow/dags/flood_model_dag.py 2>/dev/null; then
    log_success "flood_model_dag.py syntax is valid"
else
    log_error "flood_model_dag.py has syntax errors"
fi

# Test 5.3: Data quality DAG exists
log_info "Testing data quality DAG..."
if docker exec agrisafe-airflow-webserver python /opt/airflow/dags/data_quality_dag.py 2>/dev/null; then
    log_success "data_quality_dag.py syntax is valid"
else
    log_error "data_quality_dag.py has syntax errors"
fi

# Test 5.4: DAGs loaded in Airflow (give it a moment)
log_info "Checking if DAGs are loaded in Airflow..."
sleep 5
if docker exec agrisafe-airflow-webserver airflow dags list 2>/dev/null | grep -q "weather_data_processing\|flood_risk_predictions\|data_quality_monitoring"; then
    log_success "Phase 3 DAGs are loaded in Airflow"
else
    log_warn "Phase 3 DAGs may not be loaded yet (this can take a minute)"
fi

echo ""

##############################################################################
# 6. FUNCTIONAL TESTS
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  6. FUNCTIONAL TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 6.1: Rule-based model can make predictions
log_info "Testing rule-based flood model prediction..."
PREDICTION_OUTPUT=$(docker exec agrisafe-airflow-worker python -c "
from src.models.flood_risk_v1 import RuleBasedFloodModel
model = RuleBasedFloodModel()
features = {'rainfall_1d': 120, 'rainfall_7d': 300, 'elevation': 50, 'historical_flood_count': 3}
result = model.predict(features)
print(result.risk_level)
" 2>/dev/null)

if [ -n "$PREDICTION_OUTPUT" ]; then
    log_success "Rule-based model generated prediction: $PREDICTION_OUTPUT"
else
    log_error "Rule-based model failed to generate prediction"
fi

# Test 6.2: Test data quality validator
log_info "Testing data quality validator..."
if docker exec agrisafe-airflow-worker python -c "
from src.quality.validators import WeatherDataValidator
validator = WeatherDataValidator()
result = validator.check_null_values()
print('Validation check executed successfully')
" 2>/dev/null | grep -q "successfully"; then
    log_success "Data quality validator executed successfully"
else
    log_error "Data quality validator failed"
fi

# Test 6.3: Directory structure exists
log_info "Testing directory structure..."
if docker exec agrisafe-airflow-worker test -d /opt/airflow/src/processing/spark_jobs; then
    log_success "Processing directory structure exists"
else
    log_error "Processing directory structure missing"
fi

if docker exec agrisafe-airflow-worker test -d /opt/airflow/src/models; then
    log_success "Models directory structure exists"
else
    log_error "Models directory structure missing"
fi

if docker exec agrisafe-airflow-worker test -d /opt/airflow/src/quality; then
    log_success "Quality directory structure exists"
else
    log_error "Quality directory structure missing"
fi

echo ""

##############################################################################
# 7. INTEGRATION TESTS
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  7. INTEGRATION TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 7.1: Redis connectivity from Python
log_info "Testing Redis connectivity from Python..."
if docker exec agrisafe-airflow-worker python -c "
import redis
r = redis.Redis(host='redis', port=6379)
r.ping()
print('Redis connected')
" 2>/dev/null | grep -q "connected"; then
    log_success "Redis connectivity works from Python"
else
    log_error "Redis connectivity failed from Python"
fi

# Test 7.2: PostgreSQL connectivity from Python
log_info "Testing PostgreSQL connectivity from Python..."
if docker exec agrisafe-airflow-worker python -c "
from src.utils.database import get_db_connection
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM regions')
    count = cursor.fetchone()[0]
    print(f'Connected, found {count} regions')
" 2>/dev/null | grep -q "Connected"; then
    log_success "PostgreSQL connectivity works from Python"
else
    log_error "PostgreSQL connectivity failed from Python"
fi

# Test 7.3: Verify test infrastructure
log_info "Checking test infrastructure..."
if docker exec agrisafe-airflow-worker test -d /opt/airflow/tests/processing; then
    log_success "Test directory structure exists"
else
    log_error "Test directory structure missing"
fi

echo ""

##############################################################################
# SUMMARY
##############################################################################

echo "======================================================================"
echo "  📊 TEST SUMMARY"
echo "======================================================================"
echo ""
echo "  Total Tests Run:    ${TESTS_RUN}"
echo "  Tests Passed:       ${GREEN}${TESTS_PASSED}${NC}"
echo "  Tests Failed:       ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ ${TESTS_FAILED} -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "🎉 Phase 3 infrastructure is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Run 'make run-etl' to process weather data"
    echo "  2. Run 'make train-model' to train the ML model"
    echo "  3. Run 'make run-predictions' to generate predictions"
    echo "  4. Run 'make quality-checks' to validate data quality"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please fix the failing tests before proceeding."
    echo ""
    exit 1
fi
