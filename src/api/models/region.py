from sqlalchemy import Column, String, Integer, Numeric, DateTime
from sqlalchemy.sql import func

from src.api.core.database import Base


class Region(Base):
    __tablename__ = "regions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    region_name = Column(String(100), unique=True, nullable=False)
    region_code = Column(String(20), unique=True)
    province = Column(String(100))
    municipality = Column(String(100))
    latitude = Column(Numeric(10, 8))
    longitude = Column(Numeric(11, 8))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Region {self.region_name}>"
