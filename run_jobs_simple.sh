#!/bin/bash

# Quick test without metadata logging issues

export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agrisafe_db
export POSTGRES_USER=agrisafe
export POSTGRES_PASSWORD='agrisafe_password_change_in_production'
export PYTHONPATH="/home/user/project-agri-safe:$PYTHONPATH"

# Get dates from daily stats (which already has data)
START_DATE=$(docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT MIN(stat_date)::text FROM weather_daily_stats;" | tr -d ' \n\r')
END_DATE=$(docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db -t -c \
    "SELECT MAX(stat_date)::text FROM weather_daily_stats;" | tr -d ' \n\r')

echo "Processing dates: $START_DATE to $END_DATE"
echo ""

echo "========================================="
echo "Job 2/3: Rolling Window Features"
echo "========================================="
python3 << EOF
import sys
sys.path.insert(0, '/home/user/project-agri-safe')

from src.processing.jobs.rolling_features import RollingFeaturesJob
from src.processing.utils.spark_session import get_spark_session, stop_spark_session

spark = get_spark_session(app_name="AgriSafe-RollingFeatures")

try:
    job = RollingFeaturesJob(spark, use_cache=True)

    # Load data
    print("Loading daily statistics...")
    daily_stats = job.load_daily_stats('$START_DATE', '$END_DATE')
    print(f"Loaded {daily_stats.count()} records")

    # Compute features
    print("Computing rolling features...")
    features_df = job.compute_rolling_features(daily_stats, [7, 14, 30])
    print(f"Computed {features_df.count()} feature records")

    # Save
    print("Saving to PostgreSQL...")
    job.save_to_postgres(features_df, mode='overwrite')
    print("✅ Rolling features job completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    stop_spark_session(spark)
EOF

echo ""
echo "========================================="
echo "Job 3/3: Flood Risk Indicators"
echo "========================================="
python3 << EOF
import sys
sys.path.insert(0, '/home/user/project-agri-safe')

from src.processing.jobs.flood_risk_indicators import FloodRiskIndicatorsJob
from src.processing.utils.spark_session import get_spark_session, stop_spark_session

spark = get_spark_session(app_name="AgriSafe-FloodRisk")

try:
    job = FloodRiskIndicatorsJob(spark, use_cache=True)

    # Load data
    print("Loading rolling features...")
    features_df = job.load_rolling_features('$START_DATE', '$END_DATE', window_days=7)
    print(f"Loaded {features_df.count()} records")

    # Calculate risk
    print("Calculating risk indicators...")
    indicators_df = job.calculate_risk_scores(features_df)
    print(f"Calculated {indicators_df.count()} risk indicators")

    # Summary
    summary = job.generate_summary_stats(indicators_df)
    print(f"Summary: {summary}")

    # Save
    print("Saving to PostgreSQL...")
    job.save_to_postgres(indicators_df, mode='overwrite')
    print("✅ Flood risk indicators job completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    stop_spark_session(spark)
EOF

echo ""
echo "========================================="
echo "✅ All jobs complete!"
echo "========================================="

# Show results
docker exec -it agrisafe-postgres psql -U agrisafe -d agrisafe_db << 'EOSQL'
SELECT
    'daily_stats' AS table_name,
    COUNT(*) AS records
FROM weather_daily_stats
UNION ALL
SELECT
    'rolling_features',
    COUNT(*)
FROM weather_rolling_features
UNION ALL
SELECT
    'flood_risk',
    COUNT(*)
FROM flood_risk_indicators;
EOSQL
