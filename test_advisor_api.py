#!/usr/bin/env python3
"""
Test script for AgriSafe Advisor API endpoints
Make sure the API is running at http://localhost:8000
"""

import requests
import json
import sys

API_BASE = "http://localhost:8000/api/v1"

def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)

def print_response(response):
    """Pretty print JSON response"""
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print(f"Status Code: {response.status_code}")

def check_api_running():
    """Check if API server is running"""
    print_section("Checking API server status")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✓ API server is running")
        print_response(response)
        return True
    except requests.exceptions.ConnectionError:
        print("✗ API server is not running!")
        print("\nPlease start the API server first:")
        print("  uvicorn src.api.main:app --reload")
        print("\nOr using:")
        print("  python -m src.api.main")
        return False
    except Exception as e:
        print(f"✗ Error connecting to API: {e}")
        return False

def main():
    print_section("AgriSafe Advisor API Testing")

    # Check if API is running
    if not check_api_running():
        sys.exit(1)

    # Step 1: Register (or skip if user exists)
    print_section("1. Registering test user")
    try:
        response = requests.post(
            f"{API_BASE}/auth/register",
            json={
                "email": "testuser@example.com",
                "password": "testpass123",
                "username": "testuser"
            }
        )
        print_response(response)

        if response.status_code == 500:
            print("\n⚠️  Database error detected!")
            print("\nLikely causes:")
            print("  1. Database tables not initialized")
            print("  2. Database connection issue")
            print("\nSolution:")
            print("  Run: python init_database.py")
            print("\nThen make sure:")
            print("  - Docker services are running: docker-compose up -d")
            print("  - PostgreSQL is accessible: psql -h localhost -U agrisafe agrisafe_db")
            sys.exit(1)
        elif response.status_code == 400:
            print("\n✓ User already exists (this is OK)")
    except Exception as e:
        print(f"❌ Registration error: {e}")
        sys.exit(1)

    # Step 2: Login
    print_section("2. Logging in")
    response = requests.post(
        f"{API_BASE}/auth/login",
        json={
            "email": "testuser@example.com",
            "password": "testpass123"
        }
    )
    print_response(response)

    if response.status_code != 200:
        print("❌ Login failed. Please check credentials.")
        sys.exit(1)

    access_token = response.json()["access_token"]
    print(f"\n✅ Access token obtained: {access_token[:20]}...")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Step 3: Create a conversation
    print_section("3. Creating a new conversation")
    response = requests.post(
        f"{API_BASE}/advisor/conversations",
        headers=headers,
        json={"conversation_title": "Rice Harvest Advice"}
    )
    print_response(response)

    if response.status_code != 201:
        print("❌ Failed to create conversation")
        sys.exit(1)

    conversation_id = response.json()["id"]
    print(f"\n✅ Conversation ID: {conversation_id}")

    # Step 4: Send a chat message
    print_section("4. Sending a chat message")
    response = requests.post(
        f"{API_BASE}/advisor/chat",
        headers=headers,
        json={
            "conversation_id": conversation_id,
            "message": "When is the best time to harvest rice?"
        }
    )
    print_response(response)

    # Step 5: Get conversation messages
    print_section("5. Getting conversation history")
    response = requests.get(
        f"{API_BASE}/advisor/conversations/{conversation_id}/messages",
        headers=headers
    )
    print_response(response)

    # Step 6: List all conversations
    print_section("6. Listing all conversations")
    response = requests.get(
        f"{API_BASE}/advisor/conversations",
        headers=headers
    )
    print_response(response)

    # Step 7: Get specific conversation
    print_section("7. Getting conversation details")
    response = requests.get(
        f"{API_BASE}/advisor/conversations/{conversation_id}",
        headers=headers
    )
    print_response(response)

    # Step 8: Delete conversation (optional)
    print_section("8. Deleting conversation (optional)")
    print("Uncomment the code below to test deletion:")
    print(f"""
    response = requests.delete(
        f"{API_BASE}/advisor/conversations/{conversation_id}",
        headers=headers
    )
    print_response(response)
    """)

    print_section("✅ Testing complete!")
    print("\nTo test the recommendations endpoint, you need a valid planting_id.")
    print("Example:")
    print(f"""
    response = requests.post(
        f"{API_BASE}/advisor/recommendations",
        headers=headers,
        json={{"planting_id": "YOUR_PLANTING_ID_HERE"}}
    )
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
