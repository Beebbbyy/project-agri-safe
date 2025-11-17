from sqlalchemy import Column, String, Date, Text, DateTime, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from src.api.core.database import Base


class HarvestRecommendation(Base):
    __tablename__ = "harvest_recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    planting_id = Column(UUID(as_uuid=True), ForeignKey("plantings.id", ondelete="CASCADE"), nullable=False, index=True)
    recommendation_date = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    recommended_harvest_window_start = Column(Date, nullable=False)
    recommended_harvest_window_end = Column(Date, nullable=False)
    urgency_level = Column(String(50), index=True)
    reason = Column(Text, nullable=False)
    weather_factors = Column(JSONB)
    flood_risk_factors = Column(JSONB)
    crop_maturity_status = Column(String(50))
    confidence_score = Column(Numeric(5, 2))
    model_version = Column(String(50))
    user_feedback = Column(String(50))
    user_feedback_notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<HarvestRecommendation {self.planting_id} - {self.urgency_level}>"
