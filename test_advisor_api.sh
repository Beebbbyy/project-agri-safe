#!/bin/bash
# Test script for AgriSafe Advisor API endpoints
# Make sure the API is running at http://localhost:8000

set -e

API_BASE="http://localhost:8000/api/v1"
echo "Testing AgriSafe Advisor API..."
echo "================================"

# Step 1: Register a test user (or skip if you already have one)
echo -e "\n1. Registering test user..."
REGISTER_RESPONSE=$(curl -s -X POST "$API_BASE/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "testuser@example.com",
    "password": "testpass123",
    "username": "testuser"
  }' || echo '{"error": "User may already exist"}')
echo "$REGISTER_RESPONSE" | jq '.' || echo "$REGISTER_RESPONSE"

# Step 2: Login and get access token
echo -e "\n2. Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "$API_BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "testuser@example.com",
    "password": "testpass123"
  }')

echo "$LOGIN_RESPONSE" | jq '.'

# Extract access token
ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token')

if [ "$ACCESS_TOKEN" = "null" ] || [ -z "$ACCESS_TOKEN" ]; then
  echo "Error: Could not get access token. Check login credentials."
  exit 1
fi

echo "Access token obtained: ${ACCESS_TOKEN:0:20}..."

# Step 3: Create a conversation
echo -e "\n3. Creating a new conversation..."
CONVERSATION_RESPONSE=$(curl -s -X POST "$API_BASE/advisor/conversations" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_title": "Rice Harvest Advice"
  }')

echo "$CONVERSATION_RESPONSE" | jq '.'

CONVERSATION_ID=$(echo "$CONVERSATION_RESPONSE" | jq -r '.id')
echo "Conversation ID: $CONVERSATION_ID"

# Step 4: Send a chat message
echo -e "\n4. Sending a chat message..."
CHAT_RESPONSE=$(curl -s -X POST "$API_BASE/advisor/chat" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"conversation_id\": \"$CONVERSATION_ID\",
    \"message\": \"When is the best time to harvest rice?\"
  }")

echo "$CHAT_RESPONSE" | jq '.'

# Step 5: Get conversation messages
echo -e "\n5. Getting conversation history..."
MESSAGES_RESPONSE=$(curl -s -X GET "$API_BASE/advisor/conversations/$CONVERSATION_ID/messages" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "$MESSAGES_RESPONSE" | jq '.'

# Step 6: List all conversations
echo -e "\n6. Listing all conversations..."
CONVERSATIONS_RESPONSE=$(curl -s -X GET "$API_BASE/advisor/conversations" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "$CONVERSATIONS_RESPONSE" | jq '.'

# Step 7: Get specific conversation
echo -e "\n7. Getting conversation details..."
CONVERSATION_DETAIL=$(curl -s -X GET "$API_BASE/advisor/conversations/$CONVERSATION_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "$CONVERSATION_DETAIL" | jq '.'

# Optional: Test recommendations endpoint (requires a planting_id)
echo -e "\n8. Testing recommendations endpoint..."
echo "Note: This requires a valid planting_id from your database."
echo "To test, replace PLANTING_ID with an actual UUID from your plantings table."
echo ""
echo "Example command:"
echo "curl -X POST \"$API_BASE/advisor/recommendations\" \\"
echo "  -H \"Authorization: Bearer $ACCESS_TOKEN\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"planting_id\": \"YOUR_PLANTING_ID_HERE\"}'"

echo -e "\n✅ Testing complete!"
echo "================================"
