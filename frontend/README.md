# Frontend Web Application (Streamlit)

**Status:** Phase 5 (Weeks 9-10) - Not yet implemented

## Overview

The frontend will be a Streamlit web application providing an intuitive interface for Filipino farmers to:
- View 5-day weather and flood risk forecasts
- Manage farms and plantings
- Get harvest timing recommendations
- Chat with AI advisor for personalized advice
- Track harvest history

## Planned Structure

```
frontend/
├── app.py                   # Main Streamlit application
├── pages/                   # Multi-page app structure
│   ├── 1_🏠_Dashboard.py
│   ├── 2_🌾_My_Farms.py
│   ├── 3_🌦️_Weather_Forecast.py
│   ├── 4_💬_Chat_Advisor.py
│   └── 5_📊_Analytics.py
├── components/              # Reusable UI components
│   ├── weather_card.py
│   ├── flood_risk_gauge.py
│   ├── crop_timeline.py
│   └── chat_interface.py
├── utils/                   # Helper functions
│   ├── api_client.py       # Backend API calls
│   ├── formatting.py       # Data formatting
│   └── validators.py       # Input validation
├── assets/                  # Static assets
│   ├── images/
│   ├── styles/
│   └── icons/
├── config.py               # App configuration
├── Dockerfile
└── requirements.txt
```

## Features

### 1. Dashboard (Home Page)
- Quick overview of all farms and active plantings
- Current weather conditions
- Active flood risk alerts
- Urgent harvest recommendations
- Recent chat conversations

### 2. My Farms
- List all farms with details
- Add/edit/delete farms
- View plantings per farm
- Add new planting records
- Track planting progress

### 3. Weather Forecast
- 5-day regional weather forecast
- Interactive map of Philippine regions
- Rainfall predictions
- Typhoon alerts
- Temperature and humidity trends

### 4. Flood Risk Assessment
- Regional flood risk levels (low/moderate/high/critical)
- Risk factors breakdown
- Historical flood data
- Early warning notifications
- Affected area visualization

### 5. Chat Advisor (AI-Powered)
- Natural language conversation
- Context-aware recommendations
- Crop-specific advice
- Weather integration
- Harvest timing suggestions
- Multi-lingual support (English, Tagalog)

### 6. Analytics & History
- Harvest history
- Yield tracking
- Weather patterns analysis
- Decision history
- Performance metrics

## Technology Stack

- **Framework:** Streamlit 1.28+
- **Charts:** Plotly, Altair
- **Maps:** Folium for geographic visualization
- **HTTP Client:** httpx for API calls
- **State Management:** Streamlit session state
- **Styling:** Custom CSS

## User Interface Design

### Color Scheme
- Primary: Green (agriculture theme)
- Warning: Yellow/Orange (moderate risk)
- Danger: Red (high/critical risk)
- Info: Blue (information)
- Success: Green (positive actions)

### Language Support
- English (default)
- Tagalog (Filipino)
- Bilingual interface elements

### Mobile-Responsive
- Optimized for mobile devices
- Touch-friendly controls
- Simplified navigation for small screens

## Sample Pages (To Be Implemented)

### Dashboard Example
```python
import streamlit as st
from utils.api_client import get_farms, get_weather_summary

st.set_page_config(
    page_title="Agri-Safe Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agri-Safe Dashboard")

# Weather summary
weather = get_weather_summary()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Temperature", f"{weather['temp']}°C")
with col2:
    st.metric("Rainfall", f"{weather['rain']} mm")
with col3:
    st.metric("Flood Risk", weather['risk_level'])

# Farms overview
farms = get_farms()
st.subheader("My Farms")
for farm in farms:
    st.write(f"🏞️ {farm['name']} - {farm['area']} hectares")
```

## Coming in Phase 5

Phase 5 (Weeks 9-10) will implement:
1. Multi-page Streamlit application
2. All main features (Dashboard, Farms, Weather, Chat)
3. API integration with FastAPI backend
4. Interactive charts and visualizations
5. User-friendly forms
6. Real-time updates
7. Mobile-responsive design
8. Multi-language support (EN/TL)

## Local Development (Future)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Access at http://localhost:8501
```

## Environment Variables

```bash
# Backend API URL
BACKEND_API_URL=http://localhost:8000

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_PRIMARY_COLOR=#2E7D32
```

## Design Mockups

(To be added in Phase 5)

## Current Status

⏳ Awaiting Phase 5 implementation

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Plotly Documentation](https://plotly.com/python/)
