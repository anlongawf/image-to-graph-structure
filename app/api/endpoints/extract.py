from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import shutil
import os
import time
from app.core.config import settings
from app.core.database import get_db
from app.models.upload import UploadHistory
from app.algorithm import UniversalGraphRecognizer
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/")
async def extract_graph(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh (.png, .jpg, .jpeg)")
    
    file_id = int(time.time() * 1000)
    save_filename = f"{file_id}_{file.filename}"
    original_path = os.path.join(settings.UPLOAD_DIR, save_filename)
    
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    start_time = time.time()
    
    # Chạy thuật toán
    basename = os.path.join(settings.UPLOAD_DIR, str(file_id))
    try:
        recognizer = UniversalGraphRecognizer(original_path)
        recognizer.process()
        
        # Lưu kết quả
        json_data, result_image_path = recognizer.export(basename)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Lưu vào SQLite Database
        upload_record = UploadHistory(
            user_id=current_user.id,
            filename=save_filename,
            original_path=original_path,
            result_json_path=f"{basename}.json",
            result_image_path=result_image_path,
            status="success",
            execution_time_ms=execution_time_ms
        )
        db.add(upload_record)
        db.commit()
        db.refresh(upload_record)
        
        return {
            "status": "success",
            "id": upload_record.id,
            "data": json_data,
            "result_image_path": "/" + result_image_path,
            "execution_time_ms": execution_time_ms
        }
        
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        upload_record = UploadHistory(
            filename=save_filename,
            original_path=original_path,
            status="failed",
            error_message=str(e),
            execution_time_ms=execution_time_ms
        )
        db.add(upload_record)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý thuật toán: {str(e)}")
