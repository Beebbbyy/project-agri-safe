from sqlalchemy import Column, String, Integer, Numeric, Text, DateTime
from sqlalchemy.sql import func

from src.api.core.database import Base


class CropType(Base):
    __tablename__ = "crop_types"

    id = Column(Integer, primary_key=True, autoincrement=True)
    crop_name = Column(String(100), unique=True, nullable=False)
    crop_category = Column(String(50))
    typical_growth_days = Column(Integer)
    min_growth_days = Column(Integer)
    max_growth_days = Column(Integer)
    optimal_temp_min = Column(Numeric(5, 2))
    optimal_temp_max = Column(Numeric(5, 2))
    water_requirement = Column(String(50))
    flood_tolerance = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<CropType {self.crop_name}>"
