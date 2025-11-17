# Backend API (FastAPI)

**Status:** Phase 4 (Weeks 7-8) - Not yet implemented

## Overview

The backend will be a FastAPI application providing REST API endpoints for:
- User authentication and management
- Farm and crop management
- Weather data queries
- Flood risk assessments
- Harvest recommendations
- LLM chat interface

## Planned Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection
│   ├── models/              # SQLAlchemy ORM models
│   ├── schemas/             # Pydantic schemas
│   ├── api/                 # API route handlers
│   │   ├── v1/
│   │   │   ├── auth.py
│   │   │   ├── farms.py
│   │   │   ├── weather.py
│   │   │   ├── floods.py
│   │   │   └── chat.py
│   ├── services/            # Business logic
│   ├── ml/                  # ML model integration
│   └── utils/               # Helper functions
├── tests/
│   ├── test_api/
│   ├── test_services/
│   └── conftest.py
├── Dockerfile
└── requirements.txt
```

## Technology Stack

- **Framework:** FastAPI 0.104+
- **Database:** PostgreSQL with SQLAlchemy ORM
- **Authentication:** JWT tokens
- **Validation:** Pydantic
- **Testing:** pytest
- **Documentation:** Auto-generated OpenAPI/Swagger

## API Endpoints (Planned)

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh token

### Farms & Crops
- `GET /api/v1/farms` - List user's farms
- `POST /api/v1/farms` - Create new farm
- `GET /api/v1/plantings` - List plantings
- `POST /api/v1/plantings` - Create new planting

### Weather & Forecasts
- `GET /api/v1/weather/forecast` - Get weather forecast
- `GET /api/v1/weather/region/{region_id}` - Regional forecast

### Flood Risk
- `GET /api/v1/flood-risk/region/{region_id}` - Get flood risk
- `GET /api/v1/flood-risk/farm/{farm_id}` - Farm-specific risk

### Harvest Recommendations
- `GET /api/v1/recommendations/planting/{id}` - Get recommendations
- `POST /api/v1/recommendations/generate` - Generate new recommendation

### Chat Interface
- `POST /api/v1/chat/conversations` - Start conversation
- `POST /api/v1/chat/messages` - Send message
- `GET /api/v1/chat/history/{conversation_id}` - Get history

## Coming in Phase 4

Phase 4 (Weeks 7-8) will implement:
1. FastAPI application setup
2. Database models and schemas
3. Authentication system
4. CRUD operations for all entities
5. Integration with ML models
6. LLM chat interface
7. API documentation
8. Unit and integration tests

## Local Development (Future)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/
```

## Current Status

⏳ Awaiting Phase 4 implementation
