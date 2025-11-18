# Airflow Troubleshooting Guide

## Issue: Database Migration Error

### Error Message
```
alembic.util.exc.CommandError: Can't locate revision identified by '88344c1d9134'
ERROR: You need to upgrade the database. Please run `airflow db upgrade`.
```

### Root Cause
The Airflow metadata database (PostgreSQL) has an inconsistent or corrupted schema state from previous startup attempts. This can happen when:
- Airflow version was changed
- Database initialization was interrupted
- Multiple conflicting migrations were attempted

### Solution

**✨ AUTO-FIX AVAILABLE (Latest Version):** As of the latest update, the Airflow webserver now includes an entrypoint script that automatically detects and fixes corrupted database states. Simply restart your containers:

```bash
docker-compose down
docker-compose up
```

The webserver will automatically:
1. Detect if the database is corrupted
2. Reset the database if needed
3. Initialize it with a clean schema
4. Create the admin user
5. Start the webserver

If this doesn't work, try the manual options below:

#### Option 1: Reset Airflow Database Volume (Manual)

This will completely reset the Airflow metadata database. **Note: This will delete all Airflow DAG runs, task history, and user accounts.**

```bash
# Stop all containers
docker-compose down

# Remove the Airflow database volume
docker volume rm project-agri-safe_airflow_postgres_data

# Start containers again (database will be initialized fresh)
docker-compose up
```

#### Option 2: Manual Database Reset

If you want to keep other volumes intact:

```bash
# Stop all containers
docker-compose down

# Remove only the Airflow PostgreSQL container and volume
docker rm agrisafe-airflow-postgres
docker volume rm project-agri-safe_airflow_postgres_data

# Start containers again
docker-compose up
```

#### Option 3: Keep Existing Data (Advanced)

If you need to preserve DAG history, you can try to fix the database manually:

```bash
# Access the Airflow webserver container
docker exec -it agrisafe-airflow-webserver bash

# Reset and reinitialize the database
airflow db reset --yes
airflow db init

# Recreate admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Exit the container
exit

# Restart the container
docker-compose restart airflow-webserver
```

### Verification

After applying any of the above solutions, verify that Airflow is running:

```bash
# Check container status
docker-compose ps

# Check Airflow webserver logs
docker-compose logs -f airflow-webserver

# Access Airflow UI
# Open http://localhost:8080 in your browser
# Login with username: admin, password: admin
```

### Prevention

To avoid this issue in the future:
1. Always use `docker-compose down` before making changes to Airflow configuration
2. Don't manually modify the Airflow database
3. Use proper version control for Airflow upgrades
4. Keep backups of important DAG runs if needed

## Other Common Issues

### Issue: Dependency Conflicts

If you see errors related to package version conflicts, ensure you're using `requirements-airflow.txt` instead of the general `requirements.txt` for Airflow containers.

The docker-compose.yml should mount:
```yaml
- ./requirements-airflow.txt:/opt/airflow/requirements.txt
```

### Issue: Port Already in Use

If port 8080 is already in use:
```bash
# Check what's using the port
lsof -i :8080

# Either stop that service or change the port in docker-compose.yml:
# ports:
#   - "8081:8080"  # Use 8081 instead
```

### Issue: Container Exits Immediately

Check the logs:
```bash
docker-compose logs airflow-webserver
docker-compose logs airflow-postgres
```

Common causes:
- Database not ready (check airflow-postgres health)
- Missing environment variables
- Permission issues with mounted volumes
