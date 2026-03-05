from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import shutil
import os
import time
import tempfile
from app.algorithm import UniversalGraphRecognizer
from app.api.deps import verify_api_key

router = APIRouter()


@router.post("/extract")
async def public_extract_graph(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Public API: upload ảnh đồ thị → trả JSON kết quả.
    
    Xác thực bằng header X-API-Key. Không lưu file hay history trên server.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh (.png, .jpg, .jpeg)")

    # Lưu file tạm
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()

        start_time = time.time()

        # Chạy thuật toán
        recognizer = UniversalGraphRecognizer(tmp.name)
        recognizer.process()

        # Lấy kết quả (không cần lưu file)
        data = {
            "nodes": [
                {"id": i, "label": n['label']}
                for i, n in enumerate(recognizer.nodes)
            ],
            "edges": [
                {
                    "from": recognizer.nodes[e['from']]['label'],
                    "to": recognizer.nodes[e['to']]['label'],
                    "weight": e['weight']
                }
                for e in recognizer.edges
            ],
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "status": "success",
            "data": data,
            "execution_time_ms": round(execution_time_ms, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
    finally:
        # Luôn xoá file tạm
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
