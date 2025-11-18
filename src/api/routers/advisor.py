from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime, timezone, timedelta
import logging

from src.api.core.database import get_db
from src.api.core.security import get_current_user_id
from src.api.core.llm import get_llm_client, LLMClient
from src.api.core.cache import CacheService
from src.api.models.chat import ChatConversation, ChatMessage
from src.api.models.planting import Planting
from src.api.models.farm import Farm
from src.api.models.crop import CropType
from src.api.models.region import Region
from src.api.schemas.chat import (
    ChatMessageCreate,
    ChatResponse,
    ConversationCreate,
    ConversationResponse,
    ConversationList,
    MessageHistoryResponse,
    RecommendationRequest,
    RecommendationResponse,
    ChatMessageResponse
)
from src.api.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# System prompt for the agricultural advisor
ADVISOR_SYSTEM_PROMPT = """You are an expert agricultural advisor for Filipino farmers specializing in harvest timing and crop management.
Your role is to provide actionable, practical advice based on:
- Current crop growth stages
- Weather forecasts and flood risks
- Local agricultural practices in the Philippines
- Optimal harvest timing for maximum yield and quality

Always provide:
1. Clear, actionable recommendations
2. Risk assessments (weather, flood, disease)
3. Optimal harvest timing windows
4. Post-harvest handling advice when relevant

Be concise, practical, and empathetic to the challenges Filipino farmers face.
Use simple language and avoid overly technical jargon unless asked."""


async def get_planting_context(
    planting_id: UUID,
    user_id: str,
    db: AsyncSession
) -> Dict[str, Any]:
    """Gather context about a planting for LLM recommendations"""

    # Get planting with related data
    result = await db.execute(
        select(Planting, Farm, CropType, Region)
        .join(Farm, Planting.farm_id == Farm.id)
        .join(CropType, Planting.crop_type_id == CropType.id)
        .join(Region, Farm.region_id == Region.id)
        .where(
            and_(
                Planting.id == planting_id,
                Farm.user_id == UUID(user_id)
            )
        )
    )
    row = result.first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Planting not found or does not belong to user"
        )

    planting, farm, crop, region = row

    # Calculate days since planting
    days_since_planting = (datetime.now().date() - planting.planting_date).days

    # Calculate days until expected harvest
    days_until_harvest = None
    if planting.expected_harvest_date:
        days_until_harvest = (planting.expected_harvest_date - datetime.now().date()).days

    # Determine growth stage based on days since planting and crop type
    growth_stage = _estimate_growth_stage(crop.name, days_since_planting)

    context = {
        "planting_id": str(planting.id),
        "crop_name": crop.name,
        "crop_category": crop.category,
        "region_name": region.name,
        "farm_name": farm.name,
        "planting_date": planting.planting_date.isoformat(),
        "expected_harvest_date": planting.expected_harvest_date.isoformat() if planting.expected_harvest_date else None,
        "days_since_planting": days_since_planting,
        "days_until_harvest": days_until_harvest,
        "growth_stage": growth_stage,
        "area_hectares": float(planting.area_planted_hectares) if planting.area_planted_hectares else None,
        "status": planting.status,
        "notes": planting.notes
    }

    return context


def _estimate_growth_stage(crop_name: str, days_since_planting: int) -> str:
    """Estimate crop growth stage based on days since planting"""

    # Simplified growth stage estimation (can be enhanced with actual crop data)
    crop_name_lower = crop_name.lower()

    if "rice" in crop_name_lower or "palay" in crop_name_lower:
        if days_since_planting < 30:
            return "Vegetative (Tillering)"
        elif days_since_planting < 65:
            return "Reproductive (Flowering)"
        elif days_since_planting < 100:
            return "Ripening"
        else:
            return "Mature (Ready for Harvest)"

    elif "corn" in crop_name_lower or "mais" in crop_name_lower:
        if days_since_planting < 30:
            return "Vegetative"
        elif days_since_planting < 60:
            return "Tasseling/Silking"
        elif days_since_planting < 90:
            return "Grain Filling"
        else:
            return "Mature"

    else:
        # Generic stages
        if days_since_planting < 30:
            return "Early Growth"
        elif days_since_planting < 60:
            return "Vegetative"
        elif days_since_planting < 90:
            return "Flowering/Fruiting"
        else:
            return "Mature"


async def get_conversation_history(
    conversation_id: UUID,
    user_id: str,
    db: AsyncSession,
    limit: int = 20
) -> List[Dict[str, str]]:
    """Get recent conversation history formatted for LLM"""

    # Verify conversation belongs to user
    conv_result = await db.execute(
        select(ChatConversation).where(
            and_(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == UUID(user_id)
            )
        )
    )
    conversation = conv_result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Get recent messages
    messages_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    )
    messages = messages_result.scalars().all()

    # Format for LLM (OpenAI/Anthropic format)
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]

    return history


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_advisor(
    chat_request: ChatMessageCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with the agricultural advisor AI

    Creates a new conversation if conversation_id is not provided.
    Optionally associate with a planting for context-aware advice.
    """

    llm_client = get_llm_client()
    conversation_id = chat_request.conversation_id

    # Create new conversation if not provided
    if not conversation_id:
        conversation = ChatConversation(
            user_id=UUID(user_id),
            planting_id=chat_request.planting_id,
            conversation_title=chat_request.message[:50] + ("..." if len(chat_request.message) > 50 else ""),
            is_active=True
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        conversation_id = conversation.id
        history = []
    else:
        # Get existing conversation history
        history = await get_conversation_history(conversation_id, user_id, db)

    # Get conversation to check for planting context
    conv_result = await db.execute(
        select(ChatConversation).where(ChatConversation.id == conversation_id)
    )
    conversation = conv_result.scalar_one()

    # Build context-aware system prompt
    system_prompt = ADVISOR_SYSTEM_PROMPT
    metadata = {}

    # Add planting context if available
    if conversation.planting_id or chat_request.planting_id:
        planting_id = conversation.planting_id or chat_request.planting_id
        try:
            planting_context = await get_planting_context(planting_id, user_id, db)
            metadata["planting_context"] = planting_context

            context_str = f"""

Current Planting Context:
- Crop: {planting_context['crop_name']} ({planting_context['crop_category']})
- Region: {planting_context['region_name']}
- Planted: {planting_context['planting_date']} ({planting_context['days_since_planting']} days ago)
- Growth Stage: {planting_context['growth_stage']}
- Expected Harvest: {planting_context['expected_harvest_date']} ({planting_context['days_until_harvest']} days)
- Status: {planting_context['status']}
"""
            system_prompt += context_str
        except HTTPException:
            # Planting not found or not owned by user - continue without context
            pass

    # Build messages for LLM
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": chat_request.message})

    # Save user message
    user_message = ChatMessage(
        conversation_id=conversation_id,
        role="user",
        content=chat_request.message,
        message_metadata=metadata
    )
    db.add(user_message)

    # Get AI response
    try:
        ai_response = await llm_client.chat_completion(messages)

        # Save assistant message
        assistant_message = ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content=ai_response,
            model_used=settings.LLM_MODEL,
            message_metadata=metadata
        )
        db.add(assistant_message)

        # Update conversation last_message_at
        conversation.last_message_at = datetime.now(timezone.utc)

        await db.commit()

        return ChatResponse(
            conversation_id=conversation_id,
            message=ai_response,
            metadata=metadata,
            timestamp=datetime.now(timezone.utc),
            model_used=settings.LLM_MODEL
        )

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/recommendations", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
async def generate_harvest_recommendation(
    request: RecommendationRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate AI-powered harvest recommendations for a specific planting

    Analyzes crop growth stage, weather forecast, and flood risk to provide
    optimal harvest timing and handling recommendations.
    """

    llm_client = get_llm_client()
    cache_service = CacheService()

    # Check cache first
    cache_key = f"recommendation:{request.planting_id}:{user_id}"
    cached = await cache_service.get(cache_key)
    if cached:
        logger.info(f"Cache hit for recommendation: {request.planting_id}")
        return RecommendationResponse(**cached)

    # Get planting context
    planting_context = await get_planting_context(request.planting_id, user_id, db)

    # Build recommendation prompt
    prompt = f"""Provide a comprehensive harvest recommendation for the following crop:

Crop: {planting_context['crop_name']} ({planting_context['crop_category']})
Region: {planting_context['region_name']}
Planting Date: {planting_context['planting_date']}
Days Since Planting: {planting_context['days_since_planting']}
Current Growth Stage: {planting_context['growth_stage']}
Expected Harvest Date: {planting_context['expected_harvest_date']}
Days Until Expected Harvest: {planting_context['days_until_harvest']}
Area: {planting_context['area_hectares']} hectares
Current Status: {planting_context['status']}

Please provide:
1. Assessment of current growth stage and readiness for harvest
2. Recommended harvest timing (optimal window)
3. Pre-harvest preparations needed
4. Expected yield considerations
5. Post-harvest handling recommendations
6. Any weather or environmental concerns

Format your response as a practical, actionable recommendation for a Filipino farmer."""

    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        recommendation_text = await llm_client.chat_completion(messages)

        # Determine confidence based on data availability
        confidence = "high" if planting_context['expected_harvest_date'] else "medium"

        response = RecommendationResponse(
            planting_id=request.planting_id,
            crop_name=planting_context['crop_name'],
            region_name=planting_context['region_name'],
            planting_date=planting_context['planting_date'],
            days_since_planting=planting_context['days_since_planting'],
            expected_harvest_date=planting_context['expected_harvest_date'],
            days_until_harvest=planting_context['days_until_harvest'],
            current_growth_stage=planting_context['growth_stage'],
            recommendation=recommendation_text,
            weather_context=None,  # Can be enhanced with weather API integration
            flood_risk_context=None,  # Can be enhanced with flood risk API integration
            confidence=confidence,
            generated_at=datetime.now(timezone.utc),
            model_used=settings.LLM_MODEL
        )

        # Cache the recommendation
        await cache_service.set(
            cache_key,
            response.model_dump(mode='json'),
            ttl=settings.CACHE_TTL_LLM
        )

        return response

    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendation: {str(e)}"
        )


@router.get("/conversations", response_model=ConversationList)
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    is_active: Optional[bool] = None,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    List all conversations for the authenticated user with pagination

    Optionally filter by active status
    """

    # Build query
    query = select(ChatConversation).where(ChatConversation.user_id == UUID(user_id))

    if is_active is not None:
        query = query.where(ChatConversation.is_active == is_active)

    # Get total count
    count_query = select(func.count()).select_from(ChatConversation).where(
        ChatConversation.user_id == UUID(user_id)
    )
    if is_active is not None:
        count_query = count_query.where(ChatConversation.is_active == is_active)

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(desc(ChatConversation.last_message_at))

    # Execute query
    result = await db.execute(query)
    conversations = result.scalars().all()

    return ConversationList(
        conversations=conversations,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific conversation

    Only returns conversation if it belongs to the authenticated user
    """

    result = await db.execute(
        select(ChatConversation).where(
            and_(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == UUID(user_id)
            )
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return conversation


@router.get("/conversations/{conversation_id}/messages", response_model=MessageHistoryResponse)
async def get_conversation_messages(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Get message history for a specific conversation

    Returns messages in chronological order (oldest first)
    """

    # Verify conversation belongs to user
    conv_result = await db.execute(
        select(ChatConversation).where(
            and_(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == UUID(user_id)
            )
        )
    )
    conversation = conv_result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Get messages
    messages_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    )
    messages = messages_result.scalars().all()

    # Get total count
    count_result = await db.execute(
        select(func.count())
        .select_from(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
    )
    total = count_result.scalar()

    return MessageHistoryResponse(
        messages=messages,
        total=total,
        conversation_id=conversation_id
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a conversation and all its messages

    Only deletes conversations that belong to the authenticated user
    """

    # Get conversation and verify ownership
    result = await db.execute(
        select(ChatConversation).where(
            and_(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == UUID(user_id)
            )
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    await db.delete(conversation)
    await db.commit()


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_data: ConversationCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new conversation

    Optionally associate with a planting for context-aware recommendations
    """

    # If planting_id provided, verify it belongs to user
    if conversation_data.planting_id:
        planting_result = await db.execute(
            select(Planting)
            .join(Farm, Planting.farm_id == Farm.id)
            .where(
                and_(
                    Planting.id == conversation_data.planting_id,
                    Farm.user_id == UUID(user_id)
                )
            )
        )
        planting = planting_result.scalar_one_or_none()

        if not planting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Planting not found or does not belong to user"
            )

    # Create conversation
    conversation = ChatConversation(
        user_id=UUID(user_id),
        planting_id=conversation_data.planting_id,
        conversation_title=conversation_data.conversation_title or "New Conversation",
        is_active=True
    )

    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    return conversation
