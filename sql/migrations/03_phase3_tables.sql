-- Phase 3: Data Processing & ML Tables
-- Migration: 03_phase3_tables.sql
-- Created: 2025-01-17
-- Description: Creates tables for weather aggregations, risk indicators, quality checks, and model tracking

-- ============================================================================
-- 1. WEATHER DAILY STATISTICS TABLE
-- ============================================================================
-- Stores aggregated daily weather statistics per region

CREATE TABLE IF NOT EXISTS weather_daily_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id UUID NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    stat_date DATE NOT NULL,

    -- Temperature metrics
    temp_high_avg DECIMAL(5,2),
    temp_low_avg DECIMAL(5,2),
    temp_range DECIMAL(5,2),  -- High - Low

    -- Rainfall metrics
    rainfall_total DECIMAL(8,2),
    rainfall_max DECIMAL(8,2),
    rainfall_min DECIMAL(8,2),

    -- Wind metrics
    wind_speed_max DECIMAL(5,2),
    wind_speed_avg DECIMAL(5,2),

    -- Weather condition
    dominant_condition VARCHAR(50),

    -- Metadata
    forecast_count INTEGER,  -- Number of forecasts aggregated
    data_quality_score DECIMAL(3,2),  -- 0-1 score

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(region_id, stat_date)
);

CREATE INDEX idx_weather_daily_stats_region_date ON weather_daily_stats(region_id, stat_date DESC);
CREATE INDEX idx_weather_daily_stats_date ON weather_daily_stats(stat_date DESC);

COMMENT ON TABLE weather_daily_stats IS 'Daily aggregated weather statistics per region';

-- ============================================================================
-- 2. REGION RISK INDICATORS TABLE
-- ============================================================================
-- Stores computed risk indicators for each region

CREATE TABLE IF NOT EXISTS region_risk_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id UUID NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    indicator_date DATE NOT NULL,

    -- Risk scores (0-1)
    flood_season_score DECIMAL(3,2),
    typhoon_probability DECIMAL(3,2),
    harvest_suitability DECIMAL(3,2),
    drought_risk_score DECIMAL(3,2),

    -- Contributing factors (JSON)
    risk_factors JSONB,

    -- Metadata
    model_version VARCHAR(50),
    confidence_score DECIMAL(3,2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(region_id, indicator_date)
);

CREATE INDEX idx_region_risk_indicators_region_date ON region_risk_indicators(region_id, indicator_date DESC);
CREATE INDEX idx_region_risk_indicators_flood_score ON region_risk_indicators(flood_season_score DESC);
CREATE INDEX idx_region_risk_indicators_factors ON region_risk_indicators USING GIN(risk_factors);

COMMENT ON TABLE region_risk_indicators IS 'Region-level risk indicators and scores';

-- ============================================================================
-- 3. DATA QUALITY CHECKS TABLE
-- ============================================================================
-- Stores results of automated data quality validation checks

CREATE TABLE IF NOT EXISTS data_quality_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    check_name VARCHAR(100) NOT NULL,
    check_category VARCHAR(50),  -- 'completeness', 'accuracy', 'consistency', 'timeliness'

    passed BOOLEAN NOT NULL,
    severity VARCHAR(20) NOT NULL,  -- 'critical', 'warning', 'info'

    message TEXT,
    details JSONB,

    -- Statistics
    records_checked INTEGER,
    records_failed INTEGER,

    -- Context
    table_name VARCHAR(100),
    column_name VARCHAR(100),

    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_quality_checks_date ON data_quality_checks(checked_at DESC);
CREATE INDEX idx_quality_checks_name ON data_quality_checks(check_name);
CREATE INDEX idx_quality_checks_severity ON data_quality_checks(severity, passed);
CREATE INDEX idx_quality_checks_table ON data_quality_checks(table_name);

COMMENT ON TABLE data_quality_checks IS 'Automated data quality validation results';

-- ============================================================================
-- 4. MODEL TRAINING RUNS TABLE
-- ============================================================================
-- Tracks ML model training sessions and metrics

CREATE TABLE IF NOT EXISTS model_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),  -- 'classification', 'regression', 'rule-based'

    -- Training metrics
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_duration_seconds INTEGER,

    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),

    -- Additional metrics (JSON)
    metrics JSONB,

    -- Training configuration
    hyperparameters JSONB,
    features_used JSONB,

    -- Dataset info
    training_samples INTEGER,
    validation_samples INTEGER,
    test_samples INTEGER,

    -- Artifacts
    model_path VARCHAR(255),
    artifacts_path VARCHAR(255),

    -- Status
    status VARCHAR(50) DEFAULT 'completed',  -- 'completed', 'failed', 'in_progress'
    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_training_runs_name_version ON model_training_runs(model_name, model_version);
CREATE INDEX idx_model_training_runs_date ON model_training_runs(training_date DESC);
CREATE INDEX idx_model_training_runs_accuracy ON model_training_runs(accuracy DESC);

COMMENT ON TABLE model_training_runs IS 'ML model training sessions and performance metrics';

-- ============================================================================
-- 5. MODEL PREDICTIONS LOG TABLE
-- ============================================================================
-- Logs all predictions made by ML models for auditing and monitoring

CREATE TABLE IF NOT EXISTS model_predictions_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    -- Prediction context
    region_id UUID REFERENCES regions(id) ON DELETE SET NULL,
    prediction_date DATE NOT NULL,

    -- Prediction results
    predicted_class VARCHAR(50),
    confidence_score DECIMAL(5,4),
    probability_distribution JSONB,

    -- Input features
    input_features JSONB,

    -- Output reference
    output_table VARCHAR(100),  -- e.g., 'flood_risk_assessments'
    output_record_id UUID,

    -- Performance tracking
    prediction_time_ms INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_predictions_log_model ON model_predictions_log(model_name, model_version);
CREATE INDEX idx_model_predictions_log_region_date ON model_predictions_log(region_id, prediction_date DESC);
CREATE INDEX idx_model_predictions_log_date ON model_predictions_log(created_at DESC);

COMMENT ON TABLE model_predictions_log IS 'Audit log for all ML model predictions';

-- ============================================================================
-- 6. FEATURE STORE TABLE
-- ============================================================================
-- Stores pre-computed features for ML models (complementing Redis cache)

CREATE TABLE IF NOT EXISTS feature_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    region_id UUID NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    feature_date DATE NOT NULL,

    -- Rolling rainfall features
    rainfall_1d DECIMAL(8,2),
    rainfall_3d DECIMAL(8,2),
    rainfall_7d DECIMAL(8,2),
    rainfall_14d DECIMAL(8,2),
    rainfall_30d DECIMAL(8,2),

    -- Temperature features
    temp_avg_7d DECIMAL(5,2),
    temp_variance_7d DECIMAL(5,2),
    temp_min_7d DECIMAL(5,2),
    temp_max_7d DECIMAL(5,2),

    -- Wind features
    wind_speed_max_7d DECIMAL(5,2),
    wind_speed_avg_7d DECIMAL(5,2),

    -- Derived features
    soil_moisture_proxy DECIMAL(8,2),
    rainfall_intensity DECIMAL(5,2),
    rainy_days_7d INTEGER,

    -- Seasonal features
    season_typhoon BOOLEAN,
    month_of_year INTEGER,
    week_of_year INTEGER,

    -- Historical features
    historical_flood_count INTEGER,
    region_vulnerability_score DECIMAL(3,2),

    -- All features as JSON (for flexibility)
    all_features JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(region_id, feature_date)
);

CREATE INDEX idx_feature_store_region_date ON feature_store(region_id, feature_date DESC);
CREATE INDEX idx_feature_store_date ON feature_store(feature_date DESC);
CREATE INDEX idx_feature_store_all_features ON feature_store USING GIN(all_features);

COMMENT ON TABLE feature_store IS 'Pre-computed ML features for all regions';

-- ============================================================================
-- 7. ETL JOB RUNS TABLE
-- ============================================================================
-- Tracks ETL pipeline executions

CREATE TABLE IF NOT EXISTS etl_job_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    job_name VARCHAR(100) NOT NULL,
    job_type VARCHAR(50),  -- 'spark_etl', 'aggregation', 'feature_engineering'

    -- Execution info
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds INTEGER,

    status VARCHAR(50) NOT NULL,  -- 'running', 'completed', 'failed'

    -- Data processed
    records_read INTEGER,
    records_written INTEGER,
    records_failed INTEGER,

    -- Configuration
    parameters JSONB,

    -- Results
    output_tables JSONB,
    metrics JSONB,

    -- Error handling
    error_message TEXT,
    error_stack_trace TEXT,

    -- Metadata
    triggered_by VARCHAR(100),  -- 'airflow', 'manual', 'scheduled'
    airflow_dag_id VARCHAR(100),
    airflow_task_id VARCHAR(100),
    airflow_run_id VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_etl_job_runs_name ON etl_job_runs(job_name);
CREATE INDEX idx_etl_job_runs_start_time ON etl_job_runs(start_time DESC);
CREATE INDEX idx_etl_job_runs_status ON etl_job_runs(status);

COMMENT ON TABLE etl_job_runs IS 'ETL pipeline execution tracking and monitoring';

-- ============================================================================
-- TRIGGERS FOR UPDATED_AT
-- ============================================================================

-- Trigger function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_weather_daily_stats_updated_at BEFORE UPDATE ON weather_daily_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_region_risk_indicators_updated_at BEFORE UPDATE ON region_risk_indicators
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_training_runs_updated_at BEFORE UPDATE ON model_training_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feature_store_updated_at BEFORE UPDATE ON feature_store
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Latest features for all regions
CREATE OR REPLACE VIEW latest_features AS
SELECT
    fs.*,
    r.name as region_name,
    r.elevation
FROM feature_store fs
JOIN regions r ON fs.region_id = r.id
WHERE fs.feature_date = (
    SELECT MAX(feature_date)
    FROM feature_store fs2
    WHERE fs2.region_id = fs.region_id
);

COMMENT ON VIEW latest_features IS 'Latest ML features for all regions';

-- View: Recent data quality summary
CREATE OR REPLACE VIEW data_quality_summary AS
SELECT
    check_name,
    check_category,
    COUNT(*) as total_checks,
    SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_count,
    SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed_count,
    ROUND(100.0 * SUM(CASE WHEN passed THEN 1 ELSE 0 END) / COUNT(*), 2) as pass_rate,
    MAX(checked_at) as last_check
FROM data_quality_checks
WHERE checked_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY check_name, check_category
ORDER BY pass_rate ASC, check_name;

COMMENT ON VIEW data_quality_summary IS '7-day data quality check summary';

-- View: Model performance comparison
CREATE OR REPLACE VIEW model_performance_comparison AS
SELECT
    model_name,
    model_version,
    model_type,
    training_date,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    training_samples,
    status,
    ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY training_date DESC) as version_rank
FROM model_training_runs
WHERE status = 'completed'
ORDER BY model_name, training_date DESC;

COMMENT ON VIEW model_performance_comparison IS 'Model performance metrics across versions';

-- ============================================================================
-- SAMPLE QUERIES (Commented out - for documentation)
-- ============================================================================

/*
-- Get latest weather statistics for a region
SELECT * FROM weather_daily_stats
WHERE region_id = 'some-uuid'
ORDER BY stat_date DESC
LIMIT 30;

-- Get features for today's predictions
SELECT * FROM latest_features;

-- Check data quality status
SELECT * FROM data_quality_summary
WHERE pass_rate < 95;

-- Get best performing model version
SELECT * FROM model_performance_comparison
WHERE version_rank = 1;

-- Get ETL job execution history
SELECT
    job_name,
    status,
    duration_seconds,
    records_written,
    start_time
FROM etl_job_runs
WHERE start_time >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY start_time DESC;
*/

-- ============================================================================
-- GRANTS (Optional - adjust based on your security requirements)
-- ============================================================================

-- Grant permissions to application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO agrisafe;
-- GRANT SELECT ON ALL VIEWS IN SCHEMA public TO agrisafe;

-- ============================================================================
-- END OF MIGRATION
-- ============================================================================

-- Migration completed successfully
SELECT 'Phase 3 tables created successfully!' as status;
