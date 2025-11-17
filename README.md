# 🌾 Project Agri-Safe

**Flood Risk & Harvest-Window Advisor for Filipino Farmers**

## Overview

Project Agri-Safe is a web application designed to help Filipino farmers make data-driven decisions about harvest timing by combining weather forecasting, flood risk assessment, and AI-powered recommendations.

### Key Features

- **5-Day Flood Risk Forecast**: Regional weather predictions with automated risk assessment
- **Harvest Advisor Chat**: LLM-powered chatbot that recommends optimal harvest timing based on:
  - Crop type and maturity period
  - Planting date
  - Upcoming weather conditions (typhoons, heavy rainfall)
  - PAGASA (Philippine Atmospheric, Geophysical and Astronomical Services Administration) forecasts

### Target Users

Filipino farmers growing crops like rice, corn, and vegetables who need to plan harvest windows around unpredictable weather patterns and typhoon seasons.

## Project Status

**Current Phase:** Planning & Setup
**Development Branch:** `claude/hi-for-thi-01LJFTAH6iFoWmv8HdK3V9bX`

## Technology Stack

- **Backend:** FastAPI, PostgreSQL, Redis
- **Data Processing:** Apache Spark, Apache Airflow
- **Frontend:** Streamlit
- **AI/ML:** OpenAI/Anthropic LLM, scikit-learn, XGBoost
- **Infrastructure:** Docker, Docker Compose

## Quick Start

(Coming soon - after Phase 1 completion)

```bash
# Clone repository
git clone https://github.com/Beebbbyy/project-agri-safe.git
cd project-agri-safe

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start Docker containers
docker-compose up -d

# Access the application
# Web App: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## Documentation

- [Development Plan](DEVELOPMENT_PLAN.md) - Comprehensive 12-week development roadmap
- [Database Schema](sql/schema/) - (Coming soon)
- [API Documentation](docs/api/) - (Coming soon)

## Development Phases

1. **Phase 1**: Foundation Setup (Weeks 1-2)
2. **Phase 2**: Data Ingestion (Weeks 3-4)
3. **Phase 3**: Data Processing & ML (Weeks 5-6)
4. **Phase 4**: Backend API (Weeks 7-8)
5. **Phase 5**: Web Application (Weeks 9-10)
6. **Phase 6**: Testing & Deployment (Weeks 11-12)

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for detailed information.

## Data Sources

- **PAGASA**: Philippine weather and typhoon forecasts
- **PSA (Philippine Statistics Authority)**: Crop calendar and agricultural statistics

## Contributing

This is currently a personal project. Contribution guidelines will be added in the future.

## License

(To be determined)

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Built with care for Filipino farmers** 🇵🇭