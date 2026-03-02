# Image-to-Graph Recognition Web App

Đây là hệ thống trích xuất và nhận diện cấu trúc đồ thị từ hình ảnh, phục vụ cho đề tài Nghiên cứu Khoa học. Dự án sử dụng OpenCV, Tesseract OCR kết hợp với backend FastAPI và SQLite.

## Cài đặt và Chạy thử nghiệm (Local)

### 1. Kiến trúc hệ thống
- **Backend:** Python 3.10+, FastAPI, SQLite.
- **Frontend:** HTML5, Vanilla JavaScript, CSS phong cách học thuật.
- **Dependencies hệ thống (Bắt buộc):** `tesseract-ocr`, `libgl1-mesa-glx` (cho OpenCV).

### 2. Cài đặt thư viện Python (Nên dùng Virtual Environment)
```bash
pip install -r requirements.txt
```

**Lưu ý:** Bạn cần phải tải và cài đặt phần mềm **Tesseract OCR** trên máy tính của bạn và hãy đảm bảo đường dẫn biến môi trường (PATH) đã thiết lập để lệnh `tesseract` có thể hoạt động ở terminal.
(Trên Mac: `brew install tesseract`)

### 3. Khởi chạy Server
```bash
python main.py
```
Máy chủ sẽ chạy tại địa chỉ: `http://localhost:8000`. Vào trình duyệt và tận hưởng kết quả.

---

## Triển khai lên Production (Cài đặt bằng Docker)

Cách dễ nhất và chuẩn nhất để chạy 100% giống nhau trên cả máy cá nhân và VPS (Linux/Ubuntu) là chạy bằng Docker, bởi vì Docker sẽ tự cài `tesseract-ocr` bên trong tự động.

Chạy lệnh sau ngay tại thư mục chứa code:
```bash
docker build -t image-to-graph-app .
docker run -p 8000:8000 image-to-graph-app
```
Truy cập `http://<IP-VPS>:8000`.
