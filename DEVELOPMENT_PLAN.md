# Project Agri-Safe - Development Plan

## 🌾 Project Overview

**Domain:** Agriculture (Philippines)
**Target Users:** Filipino farmers
**Core Value:** Flood risk forecasting + AI-powered harvest timing recommendations

### Key Features
1. **5-Day Flood Risk Forecast** - Regional weather predictions with risk assessment
2. **Harvest Advisor Chatbot** - LLM-powered recommendations based on planting date, crop type, and weather forecasts

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│              (Streamlit/Flask Web App)                      │
│   - Regional Dashboard (Flood Risk Charts)                  │
│   - Harvest Advisor Chat (LLM Interface)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  APPLICATION LAYER                          │
│   - FastAPI/Flask Backend                                   │
│   - LLM Integration (OpenAI/Anthropic/Local)               │
│   - Business Logic & Query Processing                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              DATA WAREHOUSE (PostgreSQL)                    │
│   - Dim_Region, Dim_Crop, Fact_Weather_Forecast            │
│   - Optimized for analytical queries                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              DATA PROCESSING LAYER                          │
│   - Apache Spark (PySpark)                                  │
│   - ETL Pipelines                                           │
│   - ML Models (Flood Risk Classification)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              DATA INGESTION LAYER                           │
│   - PAGASA API Connector (Daily Batch)                     │
│   - PSA Data Loader (One-time Setup)                       │
│   - Orchestration (Apache Airflow)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies
| Component | Technology | Reasoning |
|-----------|-----------|-----------|
| **Web Framework** | Streamlit | Rapid prototyping, built-in data viz |
| **Backend API** | FastAPI | Modern, async, auto-documentation |
| **Database** | PostgreSQL 15+ | Robust, excellent for analytics |
| **Data Processing** | Apache Spark (PySpark) | Scalable ETL and ML |
| **Orchestration** | Apache Airflow | Workflow management, scheduling |
| **LLM** | OpenAI GPT-4 / Claude | Natural language interface |
| **Containerization** | Docker + Docker Compose | Environment consistency |
| **Caching** | Redis | API response caching |

### Python Libraries
- **Data Processing:** `pyspark`, `pandas`, `numpy`
- **Database:** `psycopg2`, `SQLAlchemy`
- **Web:** `streamlit`, `fastapi`, `uvicorn`
- **ML:** `scikit-learn`, `xgboost` (for flood prediction)
- **API:** `requests`, `httpx`
- **LLM:** `openai`, `anthropic`, `langchain`

---

## 📊 Database Schema Design

### Dimensional Model (Star Schema)

#### Dimension Tables

**Dim_Region**
```sql
CREATE TABLE dim_region (
    region_id SERIAL PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL,
    province VARCHAR(100) NOT NULL,
    municipality VARCHAR(100),
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    elevation_meters INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_name, province, municipality)
);

-- Example data:
-- region_id: 1, region_name: 'Central Luzon', province: 'Bulacan', municipality: 'Malolos'
```

**Dim_Crop**
```sql
CREATE TABLE dim_crop (
    crop_id SERIAL PRIMARY KEY,
    crop_name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50), -- 'Grain', 'Vegetable', 'Fruit'
    avg_maturity_days INTEGER NOT NULL,
    min_maturity_days INTEGER,
    max_maturity_days INTEGER,
    optimal_harvest_moisture_pct DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Example data:
-- crop_id: 1, crop_name: 'Rice', category: 'Grain', avg_maturity_days: 120
-- crop_id: 2, crop_name: 'Corn', category: 'Grain', avg_maturity_days: 90
```

**Dim_Date**
```sql
CREATE TABLE dim_date (
    date_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    quarter INTEGER,
    week_of_year INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    month_name VARCHAR(20),
    season VARCHAR(20) -- 'Wet', 'Dry'
);
```

#### Fact Tables

**Fact_Weather_Forecast**
```sql
CREATE TABLE fact_weather_forecast (
    forecast_id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES dim_region(region_id),
    date_id INTEGER REFERENCES dim_date(date_id),
    forecast_date DATE NOT NULL,
    predicted_rainfall_mm DECIMAL(6,2),
    predicted_max_temp_c DECIMAL(4,1),
    predicted_min_temp_c DECIMAL(4,1),
    wind_speed_kph DECIMAL(5,1),
    typhoon_signal_level INTEGER DEFAULT 0, -- 0-5
    flood_risk_level VARCHAR(20), -- 'Low', 'Medium', 'High', 'Critical'
    flood_risk_score DECIMAL(3,2), -- 0.00 - 1.00
    data_source VARCHAR(50) DEFAULT 'PAGASA',
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, forecast_date, data_source)
);

CREATE INDEX idx_forecast_region_date ON fact_weather_forecast(region_id, forecast_date);
CREATE INDEX idx_forecast_risk ON fact_weather_forecast(flood_risk_level, forecast_date);
```

**Fact_User_Plantings**
```sql
CREATE TABLE fact_user_plantings (
    planting_id SERIAL PRIMARY KEY,
    user_id INTEGER, -- Future: user authentication
    region_id INTEGER REFERENCES dim_region(region_id),
    crop_id INTEGER REFERENCES dim_crop(crop_id),
    planting_date DATE NOT NULL,
    expected_harvest_date DATE,
    actual_harvest_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_planting_user ON fact_user_plantings(user_id);
CREATE INDEX idx_planting_harvest ON fact_user_plantings(expected_harvest_date);
```

---

## 📅 Development Phases

### **Phase 1: Foundation Setup** (Week 1-2)

#### Objectives
- Set up development environment
- Initialize Docker containerization
- Create database schema
- Load initial seed data

#### Tasks
1. **Project Structure**
   ```
   project-agri-safe/
   ├── docker/
   │   ├── docker-compose.yml
   │   ├── postgres/
   │   ├── airflow/
   │   ├── spark/
   │   └── webapp/
   ├── data/
   │   ├── raw/              # PSA CSVs, sample data
   │   ├── processed/
   │   └── seeds/            # Initial DB seeds
   ├── src/
   │   ├── ingestion/        # Data connectors
   │   ├── processing/       # Spark jobs
   │   ├── models/           # SQLAlchemy models
   │   ├── api/              # FastAPI endpoints
   │   ├── webapp/           # Streamlit app
   │   └── utils/
   ├── sql/
   │   ├── schema/
   │   └── migrations/
   ├── notebooks/            # Jupyter for exploration
   ├── tests/
   ├── .env.example
   ├── requirements.txt
   └── README.md
   ```

2. **Docker Setup**
   - PostgreSQL container (port 5432)
   - Redis container (port 6379)
   - PgAdmin container (port 5050) for database management
   - Airflow container (port 8080)

3. **Database Initialization**
   - Run schema creation scripts
   - Load Philippine region data (provinces, municipalities)
   - Load crop data from PSA resources

#### Deliverables
- ✅ Docker Compose environment running
- ✅ Database schema created and seeded
- ✅ Basic project documentation

---

### **Phase 2: Data Ingestion** (Week 3-4)

#### Objectives
- Connect to PAGASA API
- Create ETL pipelines for weather data
- Load PSA crop calendar data

#### Tasks
1. **PAGASA API Integration**
   - Research PAGASA API documentation
   - Create Python connector class
   - Implement error handling and retries
   - Store raw JSON responses for auditing

2. **PSA Data Processing**
   - Download PSA crop calendar CSVs
   - Create one-time import scripts
   - Validate and clean data

3. **Airflow DAGs**
   ```python
   # Example DAG structure
   dag_pagasa_daily_ingestion:
     - fetch_weather_forecast
     - validate_data
     - load_to_staging
     - run_data_quality_checks
   ```

#### Deliverables
- ✅ PAGASA API connector with daily scheduling
- ✅ PSA data loaded into Dim_Crop
- ✅ Airflow DAG for automated data ingestion

---

### **Phase 3: Data Processing & ML** (Week 5-6)

#### Objectives
- Build Spark ETL pipelines
- Develop flood risk prediction model
- Optimize data transformations

#### Tasks
1. **Spark ETL Pipeline**
   - Read from staging tables
   - Transform and clean weather data
   - Calculate flood risk scores

2. **Flood Risk Model**
   - **Rule-Based (v1):**
     ```python
     IF predicted_rainfall_mm > 100 AND duration >= 2 days:
         flood_risk = 'High'
     ELIF predicted_rainfall_mm > 50:
         flood_risk = 'Medium'
     ELSE:
         flood_risk = 'Low'
     ```
   - **ML-Based (v2):**
     - Features: rainfall, historical flood data, terrain elevation
     - Model: Random Forest or XGBoost classifier
     - Training data: Historical PAGASA data + flood incidents

3. **Data Quality**
   - Implement data validation checks
   - Create alerting for data anomalies

#### Deliverables
- ✅ Spark jobs running daily
- ✅ Flood risk predictions in database
- ✅ Model evaluation metrics (if ML approach)

---

### **Phase 4: Backend API** (Week 7-8)

#### Objectives
- Build FastAPI backend
- Create LLM integration
- Implement business logic for harvest advisor

#### Tasks
1. **FastAPI Endpoints**
   ```python
   GET  /api/regions              # List all regions
   GET  /api/crops                # List all crops
   GET  /api/forecast/{region_id} # Get 5-day forecast
   POST /api/harvest-advisor      # Chat endpoint
   POST /api/plantings            # Record planting
   ```

2. **LLM Integration**
   - Set up OpenAI/Anthropic API client
   - Create prompt templates
   - Implement context retrieval:
     ```python
     def get_harvest_recommendation(crop, planting_date, region):
         # 1. Get crop maturity days
         # 2. Calculate target harvest date
         # 3. Query weather forecast for harvest window
         # 4. Build LLM prompt with context
         # 5. Return recommendation
     ```

3. **Caching Layer**
   - Cache weather forecasts in Redis (TTL: 6 hours)
   - Cache LLM responses for common queries

#### Deliverables
- ✅ RESTful API with documentation (FastAPI auto-docs)
- ✅ Harvest advisor logic implemented
- ✅ API tests with >80% coverage

---

### **Phase 5: Web Application** (Week 9-10)

#### Objectives
- Build user-facing Streamlit app
- Create interactive dashboards
- Implement chat interface

#### Tasks
1. **Dashboard Pages**
   - **Home:** Project overview
   - **Flood Forecast:** Regional selection + 5-day chart
   - **Harvest Advisor:** Chat interface
   - **My Plantings:** User planting tracker (future auth)

2. **Visualizations**
   - Rainfall bar charts (Plotly/Altair)
   - Flood risk heat maps
   - Crop calendar timelines

3. **Chat Interface**
   ```python
   # Streamlit chat example
   user_input = st.chat_input("Ask about harvest timing...")
   if user_input:
       with st.chat_message("user"):
           st.write(user_input)

       response = call_harvest_advisor_api(user_input)

       with st.chat_message("assistant"):
           st.write(response)
   ```

#### Deliverables
- ✅ Functional web app with all features
- ✅ Mobile-responsive design
- ✅ User documentation/FAQ

---

### **Phase 6: Testing & Deployment** (Week 11-12)

#### Objectives
- Comprehensive testing
- Production deployment setup
- Monitoring and logging

#### Tasks
1. **Testing**
   - Unit tests for all modules
   - Integration tests for API
   - End-to-end tests for critical workflows
   - Load testing (simulate 100+ concurrent users)

2. **Deployment**
   - Choose hosting: AWS EC2, DigitalOcean, or Railway
   - Set up CI/CD pipeline (GitHub Actions)
   - Configure environment variables
   - SSL certificates for HTTPS

3. **Monitoring**
   - Application logging (Python logging module)
   - Database query performance monitoring
   - API response time tracking
   - Error alerting (email/Slack notifications)

#### Deliverables
- ✅ Test coverage >80%
- ✅ Production environment live
- ✅ Monitoring dashboards

---

## 🔒 Security Considerations

1. **API Keys:** Store PAGASA API keys, LLM API keys in `.env` (never commit)
2. **Database:** Use strong passwords, restrict network access
3. **Input Validation:** Sanitize all user inputs to prevent SQL injection
4. **Rate Limiting:** Implement rate limits on API endpoints
5. **CORS:** Configure proper CORS policies for web app

---

## 🚀 Future Enhancements (Post-MVP)

### Phase 7: Advanced Features
- **User Authentication:** Login system for personalized planting records
- **Mobile App:** React Native or Flutter app
- **SMS Alerts:** Twilio integration for critical flood warnings
- **Multi-language Support:** Tagalog, Cebuano, Ilocano translations
- **Community Forum:** Farmers sharing best practices
- **Satellite Imagery:** Integrate NASA/ESA data for drought monitoring
- **Marketplace Integration:** Connect farmers with buyers

### Technical Improvements
- **Real-time Streaming:** Apache Kafka for live weather updates
- **Advanced ML:** Time-series forecasting (LSTM/Prophet)
- **Data Lake:** Store historical data in S3/MinIO
- **GraphQL API:** More flexible data querying
- **Kubernetes:** Container orchestration for scaling

---

## 📊 Success Metrics

### Technical KPIs
- API response time < 200ms (p95)
- Database query performance < 100ms
- System uptime > 99.5%
- Data freshness: < 24 hours from PAGASA source

### User-Centric KPIs
- User acquisition rate
- Chat advisor usage frequency
- Forecast accuracy validation
- User feedback scores

---

## 📚 Learning Resources

### Data Engineering
- [PAGASA Website](https://www.pagasa.dost.gov.ph/)
- [PSA OpenSTAT](https://openstat.psa.gov.ph/)
- Apache Spark Documentation
- Apache Airflow Best Practices

### LLM Integration
- LangChain Documentation
- OpenAI API Guide
- Prompt Engineering Guide

### Philippine Agriculture
- Philippine Crop Calendar (PSA)
- IRRI Rice Knowledge Bank
- DA-BAR Agricultural Resources

---

## 🎯 Development Priorities

### Must-Have (MVP)
- [x] Database schema
- [ ] PAGASA data ingestion
- [ ] Basic flood risk calculation
- [ ] Simple web dashboard
- [ ] Harvest advisor chat (basic)

### Should-Have (v1.1)
- [ ] ML-based flood prediction
- [ ] User planting records
- [ ] Mobile-responsive design
- [ ] Caching layer

### Nice-to-Have (v2.0)
- [ ] SMS alerts
- [ ] Multi-language support
- [ ] Advanced visualizations
- [ ] Community features

---

## 🤝 Contributing Guidelines

(For future collaborators)

1. Fork the repository
2. Create feature branch: `git checkout -b feature/harvest-alert-sms`
3. Write tests for new features
4. Follow PEP 8 style guide
5. Submit pull request with clear description

---

## 📞 Contact & Support

**Project Maintainer:** [Your Name]
**Email:** [Your Email]
**GitHub:** [Your GitHub Profile]

---

**Last Updated:** 2025-11-17
**Version:** 1.0 (Development Plan)
