from sqlalchemy import Column, String, Integer, Numeric, Date, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from src.api.core.database import Base


class Planting(Base):
    __tablename__ = "plantings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id = Column(UUID(as_uuid=True), ForeignKey("farms.id", ondelete="CASCADE"), nullable=False, index=True)
    crop_type_id = Column(Integer, ForeignKey("crop_types.id", ondelete="RESTRICT"), nullable=False, index=True)
    planting_date = Column(Date, nullable=False, index=True)
    expected_harvest_date = Column(Date)
    actual_harvest_date = Column(Date)
    area_planted_hectares = Column(Numeric(10, 2))
    status = Column(String(50), default="active", index=True)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Planting {self.id}>"
