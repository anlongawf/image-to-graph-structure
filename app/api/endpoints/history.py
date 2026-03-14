from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.deps import get_current_user, get_optional_user
from app.models.user import User
from app.models.upload import UploadHistory
import json
import os

router = APIRouter()

@router.get("/")
def get_user_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    history = db.query(UploadHistory)\
                .filter(UploadHistory.user_id == current_user.id)\
                .order_by(UploadHistory.created_at.desc())\
                .limit(50)\
                .all()
    
    # Chuyển đổi sang dict
    result = []
    for h in history:
        result.append({
            "id": h.id,
            "filename": h.filename,
            "created_at": h.created_at,
            "status": h.status,
            "execution_time_ms": h.execution_time_ms,
            "result_image_path": h.result_image_path,
            "is_public": h.is_public
        })
        
    return {"status": "success", "data": result}

@router.get("/shared/{history_id}")
def get_shared_history(
    history_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user)
):
    record = db.query(UploadHistory).filter(UploadHistory.id == history_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử này")
    
    # Quyền truy cập:
    # 1. Là chủ sở hữu (record.user_id == current_user.id)
    # 2. Hoặc record.is_public == True
    is_owner = current_user and record.user_id == current_user.id
    if not (is_owner or record.is_public):
        raise HTTPException(
            status_code=403, 
            detail="Bạn không có quyền xem kết quả này. Vui lòng liên hệ chủ sở hữu để được chia sẻ."
        )
    
    # Read JSON data from file
    graph_data = {}
    if record.result_json_path and os.path.exists(record.result_json_path):
        with open(record.result_json_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
            
    return {
        "status": "success", 
        "data": graph_data,
        "history": {
            "id": record.id,
            "filename": record.filename,
            "original_path": record.original_path,
            "result_image_path": record.result_image_path,
            "created_at": record.created_at,
            "is_public": record.is_public,
            "is_owner": is_owner
        }
    }

@router.post("/{history_id}/toggle-share")
def toggle_share(
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    record = db.query(UploadHistory).filter(
        UploadHistory.id == history_id,
        UploadHistory.user_id == current_user.id
    ).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi hoặc bạn không có quyền.")
    
    record.is_public = not record.is_public
    db.commit()
    db.refresh(record)
    
    return {
        "status": "success", 
        "is_public": record.is_public,
        "message": "Đã cập nhật trạng thái chia sẻ" if record.is_public else "Đã hủy chia sẻ"
    }
