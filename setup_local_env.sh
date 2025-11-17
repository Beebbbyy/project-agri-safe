#!/bin/bash
# ==================================================
# Local Development Environment Setup
# ==================================================

echo "=========================================="
echo "Setting up local development environment"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: No virtual environment detected"
    echo ""
    echo "Please create and activate a virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Install core dependencies
echo "Installing core dependencies..."
pip install --upgrade pip

# Install required packages for Phase 2
pip install \
    requests \
    psycopg2-binary \
    pydantic \
    pydantic-settings \
    loguru \
    python-dotenv

echo ""
echo "✓ Core dependencies installed"
echo ""

# Verify installations
echo "Verifying installations..."
python -c "import requests; print('✓ requests')"
python -c "import psycopg2; print('✓ psycopg2')"
python -c "import pydantic; print('✓ pydantic')"
python -c "import loguru; print('✓ loguru')"
python -c "from dotenv import load_dotenv; print('✓ python-dotenv')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure Docker services are running: docker-compose up -d"
echo "  2. Set up .env file: cp .env.example .env"
echo "  3. Test the connector: python -m src.ingestion.pagasa_connector"
echo ""
