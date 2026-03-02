from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.upload import UploadHistory

router = APIRouter()

@router.get("/")
def get_user_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    history = db.query(UploadHistory)\
                .filter(UploadHistory.user_id == current_user.id)\
                .order_by(UploadHistory.created_at.desc())\
                .limit(50)\
                .all()
    return {"status": "success", "data": history}

from fastapi import HTTPException
@router.get("/shared/{history_id}")
def get_shared_history(history_id: int, db: Session = Depends(get_db)):
    record = db.query(UploadHistory).filter(UploadHistory.id == history_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử chia sẻ này")
    return {"status": "success", "data": record}
