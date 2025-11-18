"""
Database Initialization Script for AgriSafe API

This script initializes the database schema required for the FastAPI backend.
Run this before starting the API server or running tests.

Usage:
    python init_database.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import text
from src.api.core.database import engine, Base
from src.api.models.user import User
from src.api.models.chat import ChatConversation, ChatMessage


async def init_database():
    """Initialize database schema"""
    print("=" * 60)
    print("AgriSafe Database Initialization")
    print("=" * 60)
    print()

    # Test database connection
    print("1. Testing database connection...")
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"   ✓ Connected to PostgreSQL")
            print(f"   Version: {version[:50]}...")
    except Exception as e:
        print(f"   ✗ Database connection failed: {e}")
        print()
        print("Please ensure:")
        print("  1. Docker services are running: docker-compose up -d")
        print("  2. PostgreSQL is accessible at localhost:5432")
        print("  3. Your .env file is configured correctly")
        return False

    print()

    # Create tables
    print("2. Creating database tables...")
    try:
        async with engine.begin() as conn:
            # Drop all tables (for clean initialization)
            # await conn.run_sync(Base.metadata.drop_all)
            # print("   ✓ Dropped existing tables")

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            print("   ✓ Created tables:")
            print("      - users")
            print("      - chat_sessions")
            print("      - chat_messages")
    except Exception as e:
        print(f"   ✗ Failed to create tables: {e}")
        return False

    print()

    # Verify tables
    print("3. Verifying tables...")
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            print(f"   ✓ Found {len(tables)} tables:")
            for table in tables:
                print(f"      - {table}")
    except Exception as e:
        print(f"   ✗ Failed to verify tables: {e}")
        return False

    print()
    print("=" * 60)
    print("Database initialization completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Start the API server: uvicorn src.api.main:app --reload")
    print("  2. Run tests: python test_advisor_api.py")
    print()

    return True


if __name__ == "__main__":
    result = asyncio.run(init_database())
    sys.exit(0 if result else 1)
