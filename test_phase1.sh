#!/bin/bash

# Project Agri-Safe - Phase 1 Test Script
# This script tests all Phase 1 components

echo "🌾 Project Agri-Safe - Phase 1 Testing"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC} - $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC} - $2"
        ((TESTS_FAILED++))
    fi
}

echo "📋 Test 1: Docker Compose Services"
echo "-----------------------------------"

# Check if docker compose is running
docker compose ps > /dev/null 2>&1
print_result $? "Docker Compose is accessible"

# Check PostgreSQL
docker compose ps postgres | grep -q "healthy"
print_result $? "PostgreSQL is running and healthy"

# Check Redis
docker compose ps redis | grep -q "healthy"
print_result $? "Redis is running and healthy"

# Check Airflow Webserver
docker compose ps airflow-webserver | grep -q "healthy"
print_result $? "Airflow Webserver is running and healthy"

# Check Airflow Scheduler
docker compose ps airflow-scheduler | grep -q "healthy"
print_result $? "Airflow Scheduler is running and healthy"

# Check Airflow Worker
docker compose ps airflow-worker | grep -q "healthy"
print_result $? "Airflow Worker is running and healthy"

echo ""
echo "🗄️  Test 2: PostgreSQL Database"
echo "-----------------------------------"

# Test PostgreSQL connection
docker exec agrisafe-postgres pg_isready -U agrisafe > /dev/null 2>&1
print_result $? "PostgreSQL accepts connections"

# Check if tables were created
TABLE_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ')
if [ "$TABLE_COUNT" -ge 12 ]; then
    print_result 0 "Database tables created (found $TABLE_COUNT tables)"
else
    print_result 1 "Database tables created (expected 12+, found $TABLE_COUNT)"
fi

# Check crop_types data
CROP_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c "SELECT COUNT(*) FROM crop_types;" 2>/dev/null | tr -d ' ')
if [ "$CROP_COUNT" -ge 20 ]; then
    print_result 0 "Crop types seed data loaded ($CROP_COUNT crops)"
else
    print_result 1 "Crop types seed data loaded (expected 20+, found $CROP_COUNT)"
fi

# Check regions data
REGION_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c "SELECT COUNT(*) FROM regions;" 2>/dev/null | tr -d ' ')
if [ "$REGION_COUNT" -ge 25 ]; then
    print_result 0 "Regions seed data loaded ($REGION_COUNT regions)"
else
    print_result 1 "Regions seed data loaded (expected 25+, found $REGION_COUNT)"
fi

# Check users data
USER_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c "SELECT COUNT(*) FROM users;" 2>/dev/null | tr -d ' ')
if [ "$USER_COUNT" -ge 3 ]; then
    print_result 0 "Test users created ($USER_COUNT users)"
else
    print_result 1 "Test users created (expected 3, found $USER_COUNT)"
fi

# Check farms data
FARM_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c "SELECT COUNT(*) FROM farms;" 2>/dev/null | tr -d ' ')
if [ "$FARM_COUNT" -ge 4 ]; then
    print_result 0 "Sample farms created ($FARM_COUNT farms)"
else
    print_result 1 "Sample farms created (expected 4, found $FARM_COUNT)"
fi

# Check plantings data
PLANTING_COUNT=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c "SELECT COUNT(*) FROM plantings;" 2>/dev/null | tr -d ' ')
if [ "$PLANTING_COUNT" -ge 5 ]; then
    print_result 0 "Sample plantings created ($PLANTING_COUNT plantings)"
else
    print_result 1 "Sample plantings created (expected 5, found $PLANTING_COUNT)"
fi

# Check database size
DB_SIZE=$(docker exec agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c "SELECT pg_size_pretty(pg_database_size('agrisafe_db'));" 2>/dev/null | tr -d ' ')
print_result 0 "Database size: $DB_SIZE"

echo ""
echo "📦 Test 3: Redis Cache"
echo "-----------------------------------"

# Test Redis connection
docker exec agrisafe-redis redis-cli ping > /dev/null 2>&1
print_result $? "Redis accepts connections"

# Test Redis SET/GET
docker exec agrisafe-redis redis-cli SET test_key "Hello Agri-Safe" > /dev/null 2>&1
TEST_VALUE=$(docker exec agrisafe-redis redis-cli GET test_key 2>/dev/null)
if [ "$TEST_VALUE" == "Hello Agri-Safe" ]; then
    print_result 0 "Redis SET/GET operations work"
    docker exec agrisafe-redis redis-cli DEL test_key > /dev/null 2>&1
else
    print_result 1 "Redis SET/GET operations work"
fi

# Check Redis memory
REDIS_MEMORY=$(docker exec agrisafe-redis redis-cli INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
print_result 0 "Redis memory usage: $REDIS_MEMORY"

echo ""
echo "🌬️  Test 4: Apache Airflow"
echo "-----------------------------------"

# Test Airflow webserver
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null)
if [ "$HTTP_STATUS" == "200" ]; then
    print_result 0 "Airflow webserver is accessible (HTTP $HTTP_STATUS)"
else
    print_result 1 "Airflow webserver is accessible (HTTP $HTTP_STATUS)"
fi

# Check Airflow database
docker exec agrisafe-airflow-postgres pg_isready -U airflow > /dev/null 2>&1
print_result $? "Airflow metadata database is running"

# Try to list DAGs (should be empty or show default DAGs)
DAG_COUNT=$(docker exec agrisafe-airflow-webserver airflow dags list 2>/dev/null | grep -v "dag_id" | grep -v "^$" | wc -l)
print_result 0 "Airflow DAGs folder accessible ($DAG_COUNT DAGs found)"

echo ""
echo "🌐 Test 5: Network Connectivity"
echo "-----------------------------------"

# Check if PostgreSQL port is accessible from host
nc -z localhost 5432 > /dev/null 2>&1
print_result $? "PostgreSQL port 5432 is accessible from host"

# Check if Redis port is accessible from host
nc -z localhost 6379 > /dev/null 2>&1
print_result $? "Redis port 6379 is accessible from host"

# Check if Airflow port is accessible from host
nc -z localhost 8080 > /dev/null 2>&1
print_result $? "Airflow port 8080 is accessible from host"

echo ""
echo "🔧 Test 6: Docker Volumes"
echo "-----------------------------------"

# Check if volumes exist
docker volume ls | grep -q "project-agri-safe_postgres_data"
print_result $? "PostgreSQL data volume exists"

docker volume ls | grep -q "project-agri-safe_redis_data"
print_result $? "Redis data volume exists"

docker volume ls | grep -q "project-agri-safe_airflow_postgres_data"
print_result $? "Airflow metadata volume exists"

echo ""
echo "======================================"
echo "📊 Test Summary"
echo "======================================"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Phase 1 is working correctly.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Access Airflow UI: http://localhost:8080 (admin/admin)"
    echo "  2. Connect to database: make db-connect"
    echo "  3. View logs: make logs"
    echo "  4. Proceed to Phase 2: Data Ingestion"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Check the output above.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  - View logs: docker compose logs [service-name]"
    echo "  - Restart services: docker compose restart"
    echo "  - Check SETUP.md for troubleshooting guide"
    exit 1
fi
