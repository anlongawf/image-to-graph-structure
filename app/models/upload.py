from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from app.core.database import Base
from datetime import datetime

class UploadHistory(Base):
    __tablename__ = "upload_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True) # ForeignKey could be added
    filename = Column(String, index=True)
    original_path = Column(String)
    result_json_path = Column(String, nullable=True)
    result_image_path = Column(String, nullable=True)
    status = Column(String, default="processing") # processing, success, failed
    error_message = Column(String, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
