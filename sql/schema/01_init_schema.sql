-- Project Agri-Safe Database Schema
-- PostgreSQL 15+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable PostGIS for geospatial data (optional, for future enhancements)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- =====================================================
-- Table: users
-- Description: Farmer/user accounts
-- =====================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    phone_number VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- =====================================================
-- Table: regions
-- Description: Philippine administrative regions
-- =====================================================
CREATE TABLE IF NOT EXISTS regions (
    id SERIAL PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL UNIQUE,
    region_code VARCHAR(20) UNIQUE,
    province VARCHAR(100),
    municipality VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_regions_code ON regions(region_code);

-- =====================================================
-- Table: farms
-- Description: Farm/field information
-- =====================================================
CREATE TABLE IF NOT EXISTS farms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    region_id INTEGER REFERENCES regions(id) ON DELETE SET NULL,
    farm_name VARCHAR(255) NOT NULL,
    area_hectares DECIMAL(10, 2),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    soil_type VARCHAR(100),
    irrigation_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_farms_user_id ON farms(user_id);
CREATE INDEX idx_farms_region_id ON farms(region_id);

-- =====================================================
-- Table: crop_types
-- Description: Catalog of crop types
-- =====================================================
CREATE TABLE IF NOT EXISTS crop_types (
    id SERIAL PRIMARY KEY,
    crop_name VARCHAR(100) NOT NULL UNIQUE,
    crop_category VARCHAR(50), -- rice, corn, vegetables, etc.
    typical_growth_days INTEGER, -- typical days from planting to harvest
    min_growth_days INTEGER,
    max_growth_days INTEGER,
    optimal_temp_min DECIMAL(5, 2),
    optimal_temp_max DECIMAL(5, 2),
    water_requirement VARCHAR(50), -- low, medium, high
    flood_tolerance VARCHAR(50), -- low, medium, high
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_crop_types_category ON crop_types(crop_category);

-- =====================================================
-- Table: plantings
-- Description: Individual planting records
-- =====================================================
CREATE TABLE IF NOT EXISTS plantings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farm_id UUID NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    crop_type_id INTEGER NOT NULL REFERENCES crop_types(id) ON DELETE RESTRICT,
    planting_date DATE NOT NULL,
    expected_harvest_date DATE,
    actual_harvest_date DATE,
    area_planted_hectares DECIMAL(10, 2),
    status VARCHAR(50) DEFAULT 'active', -- active, harvested, damaged, abandoned
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_plantings_farm_id ON plantings(farm_id);
CREATE INDEX idx_plantings_crop_type_id ON plantings(crop_type_id);
CREATE INDEX idx_plantings_status ON plantings(status);
CREATE INDEX idx_plantings_planting_date ON plantings(planting_date);

-- =====================================================
-- Table: weather_forecasts
-- Description: Weather forecast data from PAGASA
-- =====================================================
CREATE TABLE IF NOT EXISTS weather_forecasts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    forecast_date DATE NOT NULL,
    forecast_created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature_min DECIMAL(5, 2),
    temperature_max DECIMAL(5, 2),
    temperature_avg DECIMAL(5, 2),
    humidity_percent DECIMAL(5, 2),
    rainfall_mm DECIMAL(8, 2),
    wind_speed_kph DECIMAL(6, 2),
    weather_condition VARCHAR(100), -- sunny, cloudy, rainy, stormy, etc.
    weather_description TEXT,
    data_source VARCHAR(100) DEFAULT 'PAGASA',
    raw_data JSONB, -- store original API response
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, forecast_date, forecast_created_at)
);

CREATE INDEX idx_weather_region_date ON weather_forecasts(region_id, forecast_date);
CREATE INDEX idx_weather_forecast_date ON weather_forecasts(forecast_date);

-- =====================================================
-- Table: typhoon_alerts
-- Description: Typhoon and tropical storm alerts
-- =====================================================
CREATE TABLE IF NOT EXISTS typhoon_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    typhoon_name VARCHAR(100),
    alert_level INTEGER, -- 1-5 (TCWS levels)
    affected_regions INTEGER[] NOT NULL, -- array of region IDs
    alert_start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    alert_end_date TIMESTAMP WITH TIME ZONE,
    max_wind_speed_kph DECIMAL(6, 2),
    expected_rainfall_mm DECIMAL(8, 2),
    description TEXT,
    advisory_text TEXT,
    data_source VARCHAR(100) DEFAULT 'PAGASA',
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_typhoon_start_date ON typhoon_alerts(alert_start_date);
CREATE INDEX idx_typhoon_affected_regions ON typhoon_alerts USING GIN(affected_regions);

-- =====================================================
-- Table: flood_risk_assessments
-- Description: Calculated flood risk predictions
-- =====================================================
CREATE TABLE IF NOT EXISTS flood_risk_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    assessment_date DATE NOT NULL,
    risk_level VARCHAR(50) NOT NULL, -- low, moderate, high, critical
    risk_score DECIMAL(5, 2), -- 0-100 score
    rainfall_forecast_mm DECIMAL(8, 2),
    historical_flood_probability DECIMAL(5, 2),
    soil_saturation_index DECIMAL(5, 2),
    river_level_status VARCHAR(50),
    model_version VARCHAR(50),
    confidence_score DECIMAL(5, 2), -- 0-100
    factors JSONB, -- detailed breakdown of risk factors
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, assessment_date, model_version)
);

CREATE INDEX idx_flood_risk_region_date ON flood_risk_assessments(region_id, assessment_date);
CREATE INDEX idx_flood_risk_level ON flood_risk_assessments(risk_level);

-- =====================================================
-- Table: harvest_recommendations
-- Description: AI-generated harvest timing recommendations
-- =====================================================
CREATE TABLE IF NOT EXISTS harvest_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    planting_id UUID NOT NULL REFERENCES plantings(id) ON DELETE CASCADE,
    recommendation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    recommended_harvest_window_start DATE NOT NULL,
    recommended_harvest_window_end DATE NOT NULL,
    urgency_level VARCHAR(50), -- normal, moderate, urgent, critical
    reason TEXT NOT NULL,
    weather_factors JSONB,
    flood_risk_factors JSONB,
    crop_maturity_status VARCHAR(50), -- immature, optimal, overripe
    confidence_score DECIMAL(5, 2), -- 0-100
    model_version VARCHAR(50),
    user_feedback VARCHAR(50), -- helpful, not_helpful, followed, ignored
    user_feedback_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_harvest_rec_planting_id ON harvest_recommendations(planting_id);
CREATE INDEX idx_harvest_rec_date ON harvest_recommendations(recommendation_date);
CREATE INDEX idx_harvest_rec_urgency ON harvest_recommendations(urgency_level);

-- =====================================================
-- Table: chat_conversations
-- Description: Chat conversation history with LLM
-- =====================================================
CREATE TABLE IF NOT EXISTS chat_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    planting_id UUID REFERENCES plantings(id) ON DELETE SET NULL,
    conversation_title VARCHAR(255),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_chat_user_id ON chat_conversations(user_id);
CREATE INDEX idx_chat_planting_id ON chat_conversations(planting_id);

-- =====================================================
-- Table: chat_messages
-- Description: Individual chat messages
-- =====================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- user, assistant, system
    content TEXT NOT NULL,
    metadata JSONB, -- store context like crop info, weather data used
    token_count INTEGER,
    model_used VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);

-- =====================================================
-- Table: api_logs
-- Description: Log external API calls for monitoring
-- =====================================================
CREATE TABLE IF NOT EXISTS api_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_name VARCHAR(100) NOT NULL, -- PAGASA, OpenAI, etc.
    endpoint VARCHAR(255),
    request_method VARCHAR(10),
    response_status INTEGER,
    response_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_logs_name_date ON api_logs(api_name, created_at);
CREATE INDEX idx_api_logs_status ON api_logs(response_status);

-- =====================================================
-- Function: Update updated_at timestamp
-- =====================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- =====================================================
-- Triggers: Auto-update updated_at columns
-- =====================================================
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_farms_updated_at BEFORE UPDATE ON farms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_plantings_updated_at BEFORE UPDATE ON plantings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_typhoon_alerts_updated_at BEFORE UPDATE ON typhoon_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Initial comments
-- =====================================================
COMMENT ON TABLE users IS 'Farmer/user account information';
COMMENT ON TABLE regions IS 'Philippine administrative regions and locations';
COMMENT ON TABLE farms IS 'Farm/field details owned by users';
COMMENT ON TABLE crop_types IS 'Catalog of crop types and their characteristics';
COMMENT ON TABLE plantings IS 'Individual crop planting records';
COMMENT ON TABLE weather_forecasts IS 'Weather forecast data from PAGASA';
COMMENT ON TABLE typhoon_alerts IS 'Typhoon and tropical storm alerts';
COMMENT ON TABLE flood_risk_assessments IS 'Calculated flood risk predictions';
COMMENT ON TABLE harvest_recommendations IS 'AI-generated harvest timing recommendations';
COMMENT ON TABLE chat_conversations IS 'LLM chat conversation sessions';
COMMENT ON TABLE chat_messages IS 'Individual messages in chat conversations';
COMMENT ON TABLE api_logs IS 'External API call logs for monitoring';
