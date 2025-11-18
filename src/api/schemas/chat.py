from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


# Conversation Management Schemas
class ConversationCreate(BaseModel):
    """Schema for creating a new conversation"""
    planting_id: Optional[UUID] = Field(None, description="Optional planting to associate with conversation")
    conversation_title: Optional[str] = Field(None, max_length=255, description="Title for the conversation")


class ConversationResponse(BaseModel):
    """Schema for conversation response"""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    planting_id: Optional[UUID]
    conversation_title: Optional[str]
    started_at: datetime
    last_message_at: datetime
    is_active: bool


class ConversationList(BaseModel):
    """Schema for paginated conversation list"""
    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int


# Chat Message Schemas
class ChatMessageCreate(BaseModel):
    """Schema for creating a chat message"""
    conversation_id: Optional[UUID] = Field(None, description="Conversation ID (will create new if not provided)")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    planting_id: Optional[UUID] = Field(None, description="Optional planting context for the conversation")


class ChatMessageResponse(BaseModel):
    """Schema for individual chat message"""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: str
    content: str
    message_metadata: Optional[Dict[str, Any]] = Field(None, serialization_alias="metadata")
    token_count: Optional[int]
    model_used: Optional[str]
    created_at: datetime


class ChatResponse(BaseModel):
    """Schema for chat completion response"""
    conversation_id: UUID
    message: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime
    model_used: str


class MessageHistoryResponse(BaseModel):
    """Schema for message history"""
    messages: List[ChatMessageResponse]
    total: int
    conversation_id: UUID


# Recommendation Schemas
class RecommendationRequest(BaseModel):
    """Schema for harvest recommendation request"""
    planting_id: UUID = Field(..., description="Planting ID to get recommendations for")
    include_weather: bool = Field(True, description="Include weather forecast in recommendations")
    include_flood_risk: bool = Field(True, description="Include flood risk assessment")


class RecommendationResponse(BaseModel):
    """Schema for harvest recommendation response"""
    planting_id: UUID
    crop_name: str
    region_name: str
    planting_date: str
    days_since_planting: int
    expected_harvest_date: Optional[str]
    days_until_harvest: Optional[int]
    current_growth_stage: str
    recommendation: str
    weather_context: Optional[Dict[str, Any]]
    flood_risk_context: Optional[Dict[str, Any]]
    confidence: Optional[str]
    generated_at: datetime
    model_used: str
