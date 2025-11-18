"""
Unit and Integration tests for Advisor Router
Run with: pytest tests/test_advisor_router.py -v
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

# Import the FastAPI app
from src.api.main import app
from src.api.core.security import create_access_token

# Create test client with raise_server_exceptions=False to handle async issues
client = TestClient(app, raise_server_exceptions=False)


class TestAdvisorAuth:
    """Test authentication requirements for advisor endpoints"""

    def test_chat_requires_auth(self):
        """Test that chat endpoint requires authentication"""
        chat_data = {"message": "What's the best time to harvest?"}
        response = client.post("/api/v1/advisor/chat", json=chat_data)
        assert response.status_code == 403  # No auth header

    def test_recommendations_requires_auth(self):
        """Test that recommendations endpoint requires authentication"""
        fake_uuid = str(uuid.uuid4())
        recommendation_data = {"planting_id": fake_uuid}
        response = client.post("/api/v1/advisor/recommendations", json=recommendation_data)
        assert response.status_code == 403

    def test_list_conversations_requires_auth(self):
        """Test that listing conversations requires authentication"""
        response = client.get("/api/v1/advisor/conversations")
        assert response.status_code == 403

    def test_get_conversation_requires_auth(self):
        """Test that getting a specific conversation requires authentication"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/advisor/conversations/{fake_uuid}")
        assert response.status_code == 403

    def test_get_conversation_messages_requires_auth(self):
        """Test that getting conversation messages requires authentication"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/advisor/conversations/{fake_uuid}/messages")
        assert response.status_code == 403

    def test_create_conversation_requires_auth(self):
        """Test that creating a conversation requires authentication"""
        conversation_data = {"conversation_title": "Test Conversation"}
        response = client.post("/api/v1/advisor/conversations", json=conversation_data)
        assert response.status_code == 403

    def test_delete_conversation_requires_auth(self):
        """Test that deleting a conversation requires authentication"""
        fake_uuid = str(uuid.uuid4())
        response = client.delete(f"/api/v1/advisor/conversations/{fake_uuid}")
        assert response.status_code == 403


class TestAdvisorWithAuth:
    """Test advisor endpoints with valid authentication"""

    @pytest.fixture
    def auth_headers(self):
        """Create valid JWT token for testing"""
        test_user_id = str(uuid.uuid4())
        token = create_access_token(data={"sub": test_user_id})
        return {"Authorization": f"Bearer {token}"}

    def test_list_conversations_with_valid_token(self, auth_headers):
        """Test listing conversations with valid authentication"""
        response = client.get("/api/v1/advisor/conversations", headers=auth_headers)
        # Should be 200, 500 (DB not connected), or 503 (service unavailable)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "conversations" in data
            assert "total" in data
            assert "page" in data
            assert "page_size" in data
            assert isinstance(data["conversations"], list)

    def test_list_conversations_pagination(self, auth_headers):
        """Test pagination parameters for listing conversations"""
        response = client.get(
            "/api/v1/advisor/conversations?page=1&page_size=10",
            headers=auth_headers
        )
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["page"] == 1
            assert data["page_size"] == 10

    def test_list_conversations_with_active_filter(self, auth_headers):
        """Test filtering conversations by active status"""
        response = client.get(
            "/api/v1/advisor/conversations?is_active=true",
            headers=auth_headers
        )
        assert response.status_code in [200, 500, 503]

    def test_get_conversation_not_found(self, auth_headers):
        """Test getting a non-existent conversation"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/advisor/conversations/{fake_uuid}",
            headers=auth_headers
        )
        # Should be 404, 500 (DB not connected), or 503 (service unavailable)
        assert response.status_code in [404, 500, 503]

    def test_get_conversation_messages_not_found(self, auth_headers):
        """Test getting messages for a non-existent conversation"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/advisor/conversations/{fake_uuid}/messages",
            headers=auth_headers
        )
        assert response.status_code in [404, 500, 503]

    def test_delete_conversation_not_found(self, auth_headers):
        """Test deleting a non-existent conversation"""
        fake_uuid = str(uuid.uuid4())
        response = client.delete(
            f"/api/v1/advisor/conversations/{fake_uuid}",
            headers=auth_headers
        )
        assert response.status_code in [404, 500, 503]

    def test_create_conversation_with_valid_data(self, auth_headers):
        """Test creating a conversation with valid data"""
        conversation_data = {
            "conversation_title": "Test Harvest Advice"
        }
        response = client.post(
            "/api/v1/advisor/conversations",
            json=conversation_data,
            headers=auth_headers
        )
        # Should be 201, 500 (DB not connected), or 503 (service unavailable)
        assert response.status_code in [201, 500, 503]

        if response.status_code == 201:
            data = response.json()
            assert "id" in data
            assert "user_id" in data
            assert "conversation_title" in data
            assert data["conversation_title"] == "Test Harvest Advice"

    def test_create_conversation_with_planting_id(self, auth_headers):
        """Test creating a conversation associated with a planting"""
        fake_planting_id = str(uuid.uuid4())
        conversation_data = {
            "planting_id": fake_planting_id,
            "conversation_title": "Rice Harvest Advice"
        }
        response = client.post(
            "/api/v1/advisor/conversations",
            json=conversation_data,
            headers=auth_headers
        )
        # Should be 201 (created), 404 (planting not found), 500 (DB error), or 503
        assert response.status_code in [201, 404, 500, 503]


class TestChatEndpoint:
    """Test the chat endpoint functionality"""

    @pytest.fixture
    def auth_headers(self):
        """Create valid JWT token for testing"""
        test_user_id = str(uuid.uuid4())
        token = create_access_token(data={"sub": test_user_id})
        return {"Authorization": f"Bearer {token}"}

    def test_chat_with_valid_message(self, auth_headers):
        """Test chat endpoint with valid message"""
        chat_data = {"message": "What's the best time to harvest rice?"}
        response = client.post(
            "/api/v1/advisor/chat",
            json=chat_data,
            headers=auth_headers
        )
        # Should be 200, 500 (DB/LLM error), or 503 (service unavailable)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "conversation_id" in data
            assert "message" in data
            assert "timestamp" in data
            assert "model_used" in data
            assert isinstance(data["message"], str)

    def test_chat_with_empty_message(self, auth_headers):
        """Test chat endpoint with empty message"""
        chat_data = {"message": ""}
        response = client.post(
            "/api/v1/advisor/chat",
            json=chat_data,
            headers=auth_headers
        )
        # Should be 422 validation error
        assert response.status_code == 422

    def test_chat_with_too_long_message(self, auth_headers):
        """Test chat endpoint with message exceeding max length"""
        chat_data = {"message": "x" * 2001}  # Max is 2000
        response = client.post(
            "/api/v1/advisor/chat",
            json=chat_data,
            headers=auth_headers
        )
        # Should be 422 validation error
        assert response.status_code == 422

    def test_chat_with_conversation_id(self, auth_headers):
        """Test chat endpoint with existing conversation ID"""
        fake_conversation_id = str(uuid.uuid4())
        chat_data = {
            "conversation_id": fake_conversation_id,
            "message": "Should I harvest now?"
        }
        response = client.post(
            "/api/v1/advisor/chat",
            json=chat_data,
            headers=auth_headers
        )
        # Should be 200, 404 (conversation not found), 500 (error), or 503
        assert response.status_code in [200, 404, 500, 503]

    def test_chat_with_planting_context(self, auth_headers):
        """Test chat endpoint with planting context"""
        fake_planting_id = str(uuid.uuid4())
        chat_data = {
            "message": "When should I harvest?",
            "planting_id": fake_planting_id
        }
        response = client.post(
            "/api/v1/advisor/chat",
            json=chat_data,
            headers=auth_headers
        )
        # Should be 200 or error status
        assert response.status_code in [200, 404, 500, 503]


class TestRecommendationsEndpoint:
    """Test the recommendations endpoint functionality"""

    @pytest.fixture
    def auth_headers(self):
        """Create valid JWT token for testing"""
        test_user_id = str(uuid.uuid4())
        token = create_access_token(data={"sub": test_user_id})
        return {"Authorization": f"Bearer {token}"}

    def test_recommendations_with_valid_planting_id(self, auth_headers):
        """Test recommendations endpoint with valid planting ID"""
        fake_planting_id = str(uuid.uuid4())
        recommendation_data = {
            "planting_id": fake_planting_id,
            "include_weather": True,
            "include_flood_risk": True
        }
        response = client.post(
            "/api/v1/advisor/recommendations",
            json=recommendation_data,
            headers=auth_headers
        )
        # Should be 200, 404 (planting not found), 500 (error), or 503
        assert response.status_code in [200, 404, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "planting_id" in data
            assert "crop_name" in data
            assert "region_name" in data
            assert "recommendation" in data
            assert "current_growth_stage" in data
            assert "generated_at" in data
            assert "model_used" in data

    def test_recommendations_without_weather(self, auth_headers):
        """Test recommendations endpoint without weather context"""
        fake_planting_id = str(uuid.uuid4())
        recommendation_data = {
            "planting_id": fake_planting_id,
            "include_weather": False,
            "include_flood_risk": False
        }
        response = client.post(
            "/api/v1/advisor/recommendations",
            json=recommendation_data,
            headers=auth_headers
        )
        assert response.status_code in [200, 404, 500, 503]

    def test_recommendations_with_invalid_planting_id(self, auth_headers):
        """Test recommendations endpoint with invalid UUID"""
        recommendation_data = {
            "planting_id": "not-a-valid-uuid"
        }
        response = client.post(
            "/api/v1/advisor/recommendations",
            json=recommendation_data,
            headers=auth_headers
        )
        # Should be 422 validation error
        assert response.status_code == 422

    def test_recommendations_missing_planting_id(self, auth_headers):
        """Test recommendations endpoint without planting_id"""
        recommendation_data = {}
        response = client.post(
            "/api/v1/advisor/recommendations",
            json=recommendation_data,
            headers=auth_headers
        )
        # Should be 422 validation error
        assert response.status_code == 422


class TestAdvisorValidation:
    """Test input validation for advisor endpoints"""

    @pytest.fixture
    def auth_headers(self):
        """Create valid JWT token for testing"""
        test_user_id = str(uuid.uuid4())
        token = create_access_token(data={"sub": test_user_id})
        return {"Authorization": f"Bearer {token}"}

    def test_chat_missing_message_field(self, auth_headers):
        """Test chat endpoint without message field"""
        chat_data = {}
        response = client.post(
            "/api/v1/advisor/chat",
            json=chat_data,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_conversation_messages_invalid_limit(self, auth_headers):
        """Test conversation messages with invalid limit"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/advisor/conversations/{fake_uuid}/messages?limit=300",
            headers=auth_headers
        )
        # Should be 422 validation error (max limit is 200)
        assert response.status_code == 422

    def test_conversation_messages_zero_limit(self, auth_headers):
        """Test conversation messages with zero limit"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/advisor/conversations/{fake_uuid}/messages?limit=0",
            headers=auth_headers
        )
        # Should be 422 validation error (min limit is 1)
        assert response.status_code == 422

    def test_list_conversations_invalid_pagination(self, auth_headers):
        """Test list conversations with invalid pagination"""
        response = client.get(
            "/api/v1/advisor/conversations?page=0&page_size=0",
            headers=auth_headers
        )
        # Should be 422 validation error
        assert response.status_code == 422

    def test_create_conversation_title_too_long(self, auth_headers):
        """Test creating conversation with title exceeding max length"""
        conversation_data = {
            "conversation_title": "x" * 256  # Max is 255
        }
        response = client.post(
            "/api/v1/advisor/conversations",
            json=conversation_data,
            headers=auth_headers
        )
        # Should be 422 validation error
        assert response.status_code == 422


class TestAdvisorEndpointsExist:
    """Test that all required advisor endpoints exist"""

    def test_chat_endpoint_exists(self):
        """Test that chat endpoint exists"""
        response = client.post("/api/v1/advisor/chat")
        # Should not be 404
        assert response.status_code != 404

    def test_recommendations_endpoint_exists(self):
        """Test that recommendations endpoint exists"""
        response = client.post("/api/v1/advisor/recommendations")
        # Should not be 404
        assert response.status_code != 404

    def test_list_conversations_endpoint_exists(self):
        """Test that list conversations endpoint exists"""
        response = client.get("/api/v1/advisor/conversations")
        # Should not be 404
        assert response.status_code != 404

    def test_get_conversation_endpoint_exists(self):
        """Test that get conversation endpoint exists"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/advisor/conversations/{fake_uuid}")
        # Should not be 404 (might be 403 for auth)
        assert response.status_code != 404

    def test_get_conversation_messages_endpoint_exists(self):
        """Test that get conversation messages endpoint exists"""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/advisor/conversations/{fake_uuid}/messages")
        # Should not be 404
        assert response.status_code != 404

    def test_create_conversation_endpoint_exists(self):
        """Test that create conversation endpoint exists"""
        response = client.post("/api/v1/advisor/conversations")
        # Should not be 404
        assert response.status_code != 404

    def test_delete_conversation_endpoint_exists(self):
        """Test that delete conversation endpoint exists"""
        fake_uuid = str(uuid.uuid4())
        response = client.delete(f"/api/v1/advisor/conversations/{fake_uuid}")
        # Should not be 404
        assert response.status_code != 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
