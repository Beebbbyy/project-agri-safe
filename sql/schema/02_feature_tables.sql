-- Feature Tables for Aggregated Weather Data
-- Phase 3: Data Processing & ML

-- =====================================================
-- Table: weather_daily_stats
-- Description: Daily aggregated weather statistics per region
-- =====================================================
CREATE TABLE IF NOT EXISTS weather_daily_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    stat_date DATE NOT NULL,

    -- Temperature statistics
    temp_min DECIMAL(5, 2),
    temp_max DECIMAL(5, 2),
    temp_avg DECIMAL(5, 2),
    temp_stddev DECIMAL(5, 2),

    -- Rainfall statistics
    rainfall_total DECIMAL(8, 2),
    rainfall_max DECIMAL(8, 2),
    rainfall_avg DECIMAL(8, 2),

    -- Wind statistics
    wind_speed_avg DECIMAL(6, 2),
    wind_speed_max DECIMAL(6, 2),

    -- Humidity statistics
    humidity_avg DECIMAL(5, 2),
    humidity_min DECIMAL(5, 2),
    humidity_max DECIMAL(5, 2),

    -- Data quality
    forecast_count INTEGER,  -- Number of forecasts aggregated
    data_completeness DECIMAL(5, 2),  -- Percentage of expected data points

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(region_id, stat_date)
);

CREATE INDEX idx_weather_daily_stats_region_date ON weather_daily_stats(region_id, stat_date);
CREATE INDEX idx_weather_daily_stats_date ON weather_daily_stats(stat_date);

-- =====================================================
-- Table: weather_rolling_features
-- Description: Rolling window features (7, 14, 30 days)
-- =====================================================
CREATE TABLE IF NOT EXISTS weather_rolling_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    feature_date DATE NOT NULL,
    window_days INTEGER NOT NULL,  -- 7, 14, or 30

    -- Rolling temperature features
    temp_rolling_avg DECIMAL(5, 2),
    temp_rolling_min DECIMAL(5, 2),
    temp_rolling_max DECIMAL(5, 2),
    temp_rolling_stddev DECIMAL(5, 2),
    temp_trend DECIMAL(7, 4),  -- Linear trend coefficient

    -- Rolling rainfall features
    rainfall_rolling_sum DECIMAL(10, 2),
    rainfall_rolling_avg DECIMAL(8, 2),
    rainfall_rolling_max DECIMAL(8, 2),
    rainfall_rolling_stddev DECIMAL(8, 2),
    rainfall_days_count INTEGER,  -- Days with rainfall > 0
    rainfall_heavy_days INTEGER,  -- Days with rainfall > 50mm

    -- Rolling wind features
    wind_rolling_avg DECIMAL(6, 2),
    wind_rolling_max DECIMAL(6, 2),

    -- Rolling humidity features
    humidity_rolling_avg DECIMAL(5, 2),
    humidity_rolling_stddev DECIMAL(5, 2),

    -- Extreme event indicators
    extreme_temp_days INTEGER,  -- Days with temp > 35°C or < 10°C
    consecutive_rain_days INTEGER,  -- Max consecutive rainy days

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(region_id, feature_date, window_days)
);

CREATE INDEX idx_rolling_features_region_date ON weather_rolling_features(region_id, feature_date);
CREATE INDEX idx_rolling_features_window ON weather_rolling_features(window_days);
CREATE INDEX idx_rolling_features_region_window ON weather_rolling_features(region_id, feature_date, window_days);

-- =====================================================
-- Table: flood_risk_indicators
-- Description: Calculated flood risk indicators per region
-- =====================================================
CREATE TABLE IF NOT EXISTS flood_risk_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    indicator_date DATE NOT NULL,

    -- Rainfall-based indicators
    rainfall_intensity_score DECIMAL(5, 2),  -- 0-100
    rainfall_duration_score DECIMAL(5, 2),   -- 0-100
    cumulative_rainfall_7d DECIMAL(10, 2),
    cumulative_rainfall_14d DECIMAL(10, 2),

    -- Flood risk scores
    flood_risk_score DECIMAL(5, 2),  -- 0-100 composite score
    flood_risk_level VARCHAR(20),    -- 'Low', 'Moderate', 'High', 'Critical'

    -- Risk factors
    heavy_rainfall_factor DECIMAL(5, 2),  -- 0-1
    prolonged_rain_factor DECIMAL(5, 2),  -- 0-1
    soil_saturation_proxy DECIMAL(5, 2),  -- 0-1 (based on recent rainfall)

    -- Historical comparison
    percentile_vs_historical DECIMAL(5, 2),  -- Percentile rank

    -- Alert flags
    is_high_risk BOOLEAN DEFAULT FALSE,
    is_critical_risk BOOLEAN DEFAULT FALSE,
    alert_message TEXT,

    -- Model metadata
    model_version VARCHAR(50),
    confidence_score DECIMAL(5, 2),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(region_id, indicator_date)
);

CREATE INDEX idx_flood_risk_region_date ON flood_risk_indicators(region_id, indicator_date);
CREATE INDEX idx_flood_risk_level ON flood_risk_indicators(flood_risk_level, indicator_date);
CREATE INDEX idx_flood_risk_alerts ON flood_risk_indicators(is_high_risk, is_critical_risk);

-- =====================================================
-- Table: feature_metadata
-- Description: Metadata about feature computation runs
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(100) NOT NULL,
    job_type VARCHAR(50) NOT NULL,  -- 'daily_stats', 'rolling_features', 'risk_indicators'

    -- Execution details
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    status VARCHAR(20),  -- 'running', 'success', 'failed'

    -- Processing stats
    records_processed INTEGER,
    records_created INTEGER,
    records_updated INTEGER,
    records_failed INTEGER,

    -- Date range processed
    date_from DATE,
    date_to DATE,

    -- Error info
    error_message TEXT,
    error_traceback TEXT,

    -- Spark details
    spark_app_id VARCHAR(200),
    executor_memory VARCHAR(20),
    driver_memory VARCHAR(20),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feature_metadata_job ON feature_metadata(job_name, job_type);
CREATE INDEX idx_feature_metadata_status ON feature_metadata(status, created_at);
CREATE INDEX idx_feature_metadata_dates ON feature_metadata(date_from, date_to);

-- =====================================================
-- Views for easy querying
-- =====================================================

-- View: Latest flood risk by region
CREATE OR REPLACE VIEW v_latest_flood_risk AS
SELECT DISTINCT ON (r.id)
    r.id AS region_id,
    r.region_name,
    r.province,
    fri.indicator_date,
    fri.flood_risk_score,
    fri.flood_risk_level,
    fri.is_high_risk,
    fri.is_critical_risk,
    fri.cumulative_rainfall_7d,
    fri.cumulative_rainfall_14d,
    fri.alert_message,
    fri.created_at
FROM regions r
LEFT JOIN flood_risk_indicators fri ON r.id = fri.region_id
ORDER BY r.id, fri.indicator_date DESC, fri.created_at DESC;

-- View: Regional weather summary (last 30 days)
CREATE OR REPLACE VIEW v_regional_weather_summary AS
SELECT
    r.id AS region_id,
    r.region_name,
    r.province,
    COUNT(DISTINCT wds.stat_date) AS days_with_data,
    AVG(wds.temp_avg) AS avg_temperature,
    SUM(wds.rainfall_total) AS total_rainfall,
    MAX(wds.rainfall_max) AS max_daily_rainfall,
    AVG(wds.wind_speed_avg) AS avg_wind_speed,
    AVG(wds.humidity_avg) AS avg_humidity
FROM regions r
LEFT JOIN weather_daily_stats wds
    ON r.id = wds.region_id
    AND wds.stat_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY r.id, r.region_name, r.province;

-- View: Rolling features with risk indicators
CREATE OR REPLACE VIEW v_features_with_risk AS
SELECT
    wrf.region_id,
    r.region_name,
    r.province,
    wrf.feature_date,
    wrf.window_days,
    wrf.rainfall_rolling_sum,
    wrf.rainfall_heavy_days,
    wrf.consecutive_rain_days,
    fri.flood_risk_level,
    fri.flood_risk_score,
    fri.is_high_risk,
    wrf.created_at
FROM weather_rolling_features wrf
JOIN regions r ON wrf.region_id = r.id
LEFT JOIN flood_risk_indicators fri
    ON wrf.region_id = fri.region_id
    AND wrf.feature_date = fri.indicator_date
WHERE wrf.window_days = 7  -- Default to 7-day window
ORDER BY wrf.feature_date DESC, r.region_name;

COMMENT ON TABLE weather_daily_stats IS 'Daily aggregated weather statistics per region';
COMMENT ON TABLE weather_rolling_features IS 'Rolling window features for time-series analysis';
COMMENT ON TABLE flood_risk_indicators IS 'Calculated flood risk indicators and alerts';
COMMENT ON TABLE feature_metadata IS 'Metadata tracking for ETL job executions';
