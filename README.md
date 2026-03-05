# Image-to-Graph Recognition Web App

Hệ thống trích xuất và nhận diện cấu trúc đồ thị từ hình ảnh, phục vụ cho đề tài Nghiên cứu Khoa học. Sử dụng OpenCV, Tesseract OCR, FastAPI và SQLite.

## Cài đặt và Chạy (Local)

### Kiến trúc hệ thống
- **Backend:** Python 3.10+, FastAPI, SQLite.
- **Frontend:** HTML5, Vanilla JavaScript, CSS.
- **Dependencies hệ thống:** `tesseract-ocr`, `libgl1-mesa-glx` (cho OpenCV).

### Cài đặt
```bash
pip install -r requirements.txt
```

> **Lưu ý:** Cần cài **Tesseract OCR** trên máy. Trên Mac: `brew install tesseract`

### Khởi chạy
```bash
python main.py
```
Truy cập: `http://localhost:8000` | Swagger docs: `http://localhost:8000/docs`

---

## Public API (cho bên ngoài tích hợp)

Bên thứ ba có thể gọi API để nhận diện đồ thị mà không cần đăng ký tài khoản — chỉ cần API Key.

### Endpoint
```
POST /api/v1/public/extract
```

### Cách gọi
```bash
curl -X POST http://localhost:8000/api/v1/public/extract \
  -H "X-API-Key: test_key_123" \
  -F "file=@graph_image.png"
```

### Response
```json
{
  "status": "success",
  "data": {
    "nodes": [{"id": 0, "label": "A"}, {"id": 1, "label": "B"}],
    "edges": [{"from": "A", "to": "B", "weight": 5}]
  },
  "execution_time_ms": 1234.56
}
```

> API Key mặc định: `test_key_123`. Sửa trong `app/core/config.py` (`API_KEYS`).
> Xem hướng dẫn tích hợp chi tiết tại [`docs/api_guide.md`](docs/api_guide.md).

---

## Triển khai Docker

```bash
docker build -t image-to-graph-app .
docker run -p 8000:8000 image-to-graph-app
```
Truy cập `http://<IP-VPS>:8000`.

---

## Cấu trúc dự án

```
├── main.py                          # Entry point (FastAPI + CORS)
├── app/
│   ├── algorithm/                   # Thuật toán nhận diện đồ thị
│   │   ├── recognizer.py            #   Orchestrator chính
│   │   ├── node_detection.py        #   Phát hiện đỉnh (3 phương pháp)
│   │   ├── ocr.py                   #   OCR đọc nhãn đỉnh
│   │   ├── weight_detection.py      #   Phát hiện trọng số
│   │   ├── edge_detection.py        #   Phát hiện cạnh
│   │   ├── preprocessing.py         #   Tiền xử lý ảnh
│   │   └── utils.py                 #   Hàm tiện ích
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── extract.py           #   API nhận diện (yêu cầu login)
│   │   │   ├── public.py            #   API public (yêu cầu API Key)
│   │   │   ├── auth.py              #   Đăng ký / Đăng nhập
│   │   │   └── history.py           #   Lịch sử phân tích
│   │   ├── router.py
│   │   └── deps.py                  #   Auth dependencies
│   ├── core/
│   │   ├── config.py                #   Cấu hình (API Keys, JWT, ...)
│   │   ├── database.py              #   SQLite connection
│   │   └── security.py              #   JWT + bcrypt
│   └── models/                      #   SQLAlchemy models
├── templates/                       #   HTML pages
├── static/                          #   CSS, JS, uploads
├── docs/                            #   Tài liệu
│   ├── algorithm_documentation.md   #   Chi tiết thuật toán
│   └── api_guide.md                 #   Hướng dẫn tích hợp API
├── requirements.txt
└── Dockerfile
```

---

## Tài liệu
- [Chi tiết thuật toán](docs/algorithm_documentation.md) — mô tả các module, hàm, và giải thuật
- [Hướng dẫn tích hợp API](docs/api_guide.md) — ví dụ code cURL, JavaScript, Python
