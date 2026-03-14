# Tiêu Chuẩn Hình Ảnh Cho Thuật Toán Image2Graph

Để đạt được kết quả nhận diện đồ thị chính xác nhất, hình ảnh đầu vào cần tuân thủ các tiêu chuẩn kỹ thuật dưới đây. Thuật toán của chúng tôi sử dụng kết hợp xử lý thị giác máy tính (OpenCV) và nhận diện ký tự quang học (Tesseract OCR).

## 1. Yêu Cầu Tổng Quan
- **Định dạng file**: `.png`, `.jpg`, `.jpeg`. Khuyên dùng `.png` để tránh nhiễu nén (compression artifacts).
- **Độ phân giải**: Chiều rộng/cao nên nằm trong khoảng **1000px - 2500px**. Ảnh quá nhỏ sẽ làm mất chi tiết ký tự, ảnh quá lớn làm chậm tốc độ xử lý và tăng nhiễu.
- **Độ tương phản**: Hình ảnh cần có độ tương phản cao giữa đối tượng (nodes, edges) và nền.

## 2. Tiêu Chuẩn Cho Đỉnh (Nodes)
- **Hình dạng**: Đỉnh phải là hình tròn hoặc gần tròn (ellipse). Thuật toán kiểm tra độ tròn (`circularity > 0.5`) và độ đặc (`solidity > 0.8`).
- **Kích thước**: Bán kính lý tưởng của đỉnh từ **15px đến 60px**.
- **Khoảng cách**: Tâm các đỉnh nên cách nhau ít nhất **30px** để tránh bị nhận diện nhầm là một cụm.
- **Màu sắc**: 
  - Hỗ trợ tốt nhất là đỉnh có viền đen trên nền trắng (hoặc ngược lại).
  - Có cơ chế lọc màu đặc thù cho các đỉnh màu: **Đỏ, Vàng, Xanh lá, Xanh dương, Tím**. Nên sử dụng màu thuần (vibrant colors) để tăng độ chính xác.

## 3. Tiêu Chuẩn Cho Cạnh (Edges)
- **Đường kẻ**: Nét vẽ nên rõ ràng, không bị đứt quãng quá **20px**.
- **Kết nối**: Các đường kẻ phải hướng về phía tâm của đỉnh và kết thúc gần biên của đỉnh (cách tâm không quá **120px**).
- **Kiểu cạnh**: Thuật toán hiện tại tối ưu cho cạnh thẳng và cạnh cong nhẹ. Các cạnh quá ngoằn ngoèo có thể không được phát hiện đầy đủ.

## 4. Tiêu Chuẩn Cho Nhãn và Trọng Số (Labels & Weights)
- **Nhãn trong Node (Labels)**: 
  - Nên viết hoa, rõ ràng.
  - Độ dài tối ưu: **1-3 ký tự** (VD: A, B, C1, N1).
  - Tránh viết nhãn quá sát viền trong của đỉnh.
- **Trọng số trên Cạnh (Weights)**:
  - Phải là **chữ số (0-9)**.
  - Vị trí: Đặt gần cạnh (cách cạnh không quá **60px**) và tốt nhất là ở gần điểm giữa của cạnh.
  - Hướng chữ: Chữ nên nằm ngang. Thuật toán cố gắng xoay ảnh để đọc nhưng kết quả tốt nhất vẫn là chữ không bị nghiêng quá nhiều.

## 5. Những Điều Nên Tránh (Anti-patterns)
- [ ] **Nền có họa tiết**: Tránh sử dụng giấy có dòng kẻ, ô ly hoặc ảnh chụp có bóng đổ phức tạp.
- [ ] **Nét vẽ quá mảnh**: Nét vẽ dưới 2px có thể bị bộ lọc nhiễu loại bỏ.
- [ ] **Chữ viết tay quá cẩu thả**: Tesseract OCR hoạt động tốt nhất với font chữ in hoặc viết tay rõ ràng, không dính nét.
- [ ] **Chồng lấn**: Tránh để các nhãn hoặc trọng số đè lên đường kẻ hoặc biên của đỉnh.

---
> [!TIP]
> Nếu bạn sử dụng các công cụ vẽ sơ đồ chuyên nghiệp (như Draw.io, Lucidchart) và xuất ảnh PNG, thuật toán sẽ đạt độ chính xác gần như 100%.
