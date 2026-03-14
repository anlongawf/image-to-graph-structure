# Plan Prompt Tạo Ảnh Đồ Thị Phù Hợp Thuật Toán

## 1) Mục tiêu tài liệu
- Chuẩn hóa cách viết prompt cho AI tạo ảnh để pipeline nhận diện trong dự án xử lý tốt.
- Giảm lỗi ở các bước: detect node, OCR nhãn node, detect trọng số, detect cạnh.
- Tạo bộ prompt có thể dùng lặp lại cho dataset train/test.

## 2) Thuật toán hiện tại đang “đọc” ảnh như thế nào

### 2.1 Node detection
- Kết hợp 3 cách: lọc màu HSV, contour, Hough Circle.
- Ngưỡng quan trọng:
  - Diện tích node: khoảng `400 -> 20000`.
  - Hough radius: khoảng `15 -> 60`.
  - Circularity > 0.5, solidity > 0.8.
- Suy ra ảnh đầu vào nên có node hình tròn rõ, biên kín, không méo mạnh.

### 2.2 OCR nhãn node
- OCR bằng Tesseract với whitelist chữ/số và ký hiệu `∞`.
- Ưu tiên ảnh có nhãn ngắn (1 ký tự hoặc rất ngắn), chữ in rõ, tương phản cao.
- Hệ thống có map sửa lỗi OCR phổ biến (B/C/E/F/G/A...), nhưng không nên phụ thuộc hoàn toàn.

### 2.3 Weight detection
- Loại vùng node, tách thành phần chữ số bên ngoài node.
- Tìm contour theo bbox hợp lệ, rồi OCR số với whitelist `0123456789`.
- Trọng số nên đặt gần cạnh, nằm ngoài node, không đè lên đường kẻ.

### 2.4 Edge detection
- Dùng Canny + HoughLinesP.
- Nối cạnh dựa trên đoạn thẳng tìm được và node gần nhất.
- Suy ra cạnh nên là nét đủ thẳng, đủ dài, không quá đứt đoạn hoặc quá mờ.

## 3) Tiêu chuẩn ảnh cần đạt (để pipeline nhận diện tốt)

### 3.1 Bố cục tổng thể
- Nền sáng đồng nhất (ưu tiên trắng/xám rất nhạt).
- Đồ thị nằm giữa khung hình, không cắt mép.
- Tỷ lệ ảnh nên từ 1:1 đến 4:3, độ phân giải từ 1024px trở lên.

### 3.2 Node
- Hình tròn đều, đường viền rõ, đường kính tương đối đồng nhất.
- Khoảng cách tâm-tâm giữa các node đủ xa để không chồng biên.
- Số node vừa phải (khuyến nghị 4–12 node/ảnh cho độ ổn định cao).

### 3.3 Nhãn node
- Mỗi node 1 nhãn ngắn: `A, B, C...` hoặc `N1, N2...`.
- Font sans-serif rõ, không viết tay.
- Không xoay chữ, không dùng hiệu ứng bóng, không làm mờ.

### 3.4 Cạnh
- Cạnh nét liền, dày vừa phải, tương phản tốt với nền.
- Tránh quá nhiều cạnh cắt chéo nhau tại 1 vùng.
- Nếu có hướng, mũi tên rõ ràng và không che nhãn/trọng số.

### 3.5 Trọng số
- Chỉ dùng số nguyên dương (ví dụ 1–99).
- Đặt gần trung điểm cạnh, lệch nhẹ khỏi đường kẻ.
- Không đặt sát node và không chồng lên label node.

## 4) Quy tắc viết prompt cho AI tạo ảnh

### 4.1 Prompt khung chuẩn (dùng cho mọi model)
```text
Create a clean 2D graph diagram on a plain white background.
Use 6 circular nodes with clear black outlines and high contrast.
Place uppercase labels inside nodes: A, B, C, D, E, F (sans-serif font).
Connect nodes with straight black edges, medium thickness, no blur.
Add integer edge weights (1-20) near edge midpoints, outside nodes.
Keep spacing uniform, avoid overlaps between labels, edges, and weights.
No perspective distortion, no shadows, no watermark, no handwritten text.
High-resolution, sharp, technical diagram style.
```

### 4.2 Negative prompt khuyến nghị
```text
blurry, low contrast, noisy background, hand-drawn, sketch style,
perspective distortion, curved text, overlapping labels, heavy texture,
watermark, logo, glow effects, gradient background, cluttered scene
```

### 4.3 Prompt cho đồ thị có hướng
```text
Generate a directed weighted graph diagram, clean vector style.
White background, black circular nodes, labels A-F centered in each node.
Draw directed edges with clear arrowheads.
Edge weights are integers placed near edges, not touching nodes.
No overlapping text, no blur, no artistic effects.
```

### 4.4 Prompt cho đồ thị vô hướng
```text
Generate an undirected weighted graph, 8 nodes, clean textbook style.
Circular nodes with clear boundaries, labels are uppercase single letters.
Straight edges with integer weights near midpoints.
High contrast black on white, crisp lines, no noise.
```

## 5) Plan tạo dataset ảnh phù hợp

### Giai đoạn A: Baseline dễ nhận diện
- Tạo 100 ảnh với điều kiện “đẹp”:
  - nền trắng, node tròn đều, cạnh thẳng rõ, nhãn 1 ký tự.
  - số node: 4–8.
  - trọng số: 1–20.

### Giai đoạn B: Tăng độ khó có kiểm soát
- Tạo thêm 200 ảnh:
  - tăng số node lên 8–12.
  - tăng giao cắt cạnh mức vừa.
  - dùng cả đồ thị có hướng và vô hướng.

### Giai đoạn C: Stress test biên
- Tạo 100 ảnh biên:
  - nhãn dài hơn (N10, N11...).
  - trọng số 2 chữ số nhiều hơn.
  - cạnh dày/mảnh khác nhau trong khoảng nhỏ.

## 6) Checklist kiểm thử chất lượng ảnh trước khi đưa vào app
- Node có tròn và tách biệt không.
- Nhãn node có đọc được bằng mắt, không chồng lấn.
- Cạnh có liền nét và nối đúng node.
- Trọng số có nằm ngoài node và gần cạnh.
- Không có hiệu ứng làm giảm OCR (blur, shadow, texture nền).

## 7) Công thức prompt nhanh cho team
- Mẫu:
```text
[type: directed/undirected] weighted graph, [N] circular nodes,
labels [style], edges [style], weights [range],
white background, high contrast, clean technical diagram,
no overlap, no blur, no watermark.
```
- Ví dụ:
```text
Undirected weighted graph, 7 circular nodes, labels A-G,
straight medium black edges, weights 1-30 near edge midpoints,
white background, high contrast, clean technical diagram,
no overlap, no blur, no watermark.
```

## 8) Kết luận áp dụng thực tế
- Nếu bám đúng tiêu chuẩn trên, pipeline hiện tại sẽ nhận diện ổn định hơn rõ rệt.
- Ưu tiên giữ ảnh “sơ đồ kỹ thuật sạch” thay vì “ảnh nghệ thuật”.
- Mỗi lần thay model tạo ảnh, chạy lại checklist mục 6 trước khi đưa vào production.

## 9) Prompt V2 Copy-Paste Dùng Ngay

### V2.1 Prompt mặc định (ổn định nhất)
```text
Create a clean 2D weighted graph diagram for computer vision OCR.
White plain background, high contrast, no texture.
7 circular nodes with black outlines, evenly spaced.
Node labels: A, B, C, D, E, F, G (uppercase, centered, sans-serif, sharp).
Straight black edges, medium thickness, clear connections between nodes.
Add integer edge weights (1-25) near edge midpoints, outside nodes.
No overlap between labels, weights, edges, or nodes.
No blur, no shadow, no watermark, no handwritten style, no perspective distortion.
Technical textbook style, sharp lines, 1536x1536 resolution.
```

### V2.2 Prompt đồ thị có hướng (directed)
```text
Generate a directed weighted graph diagram in clean technical style.
Plain white background, high contrast.
8 circular nodes with uppercase labels A-H centered inside each node.
Directed edges with clear arrowheads, straight lines only.
Edge weights are integer numbers (1-30), placed near edge centers, not touching nodes.
Keep all text horizontal and highly legible.
No overlap, no blur, no artistic effects, no watermark, no noise.
High-resolution image suitable for OCR and line detection.
```

### V2.3 Prompt đồ thị vô hướng (undirected)
```text
Generate an undirected weighted graph, clean black-and-white diagram.
White background, no gradient.
6 circular nodes, labels A-F, centered and crisp.
Straight edges only, medium stroke width, no broken lines.
Weights are integers from 1 to 20, placed slightly offset from each edge midpoint.
Balanced layout with enough spacing between nodes and edge labels.
No overlap, no blur, no sketch style, no watermark.
```

### V2.4 Prompt tạo batch dataset (nhiều ảnh cùng chuẩn)
```text
Create a dataset-style graph diagram image with these constraints:
- clean 2D technical graph
- white plain background
- circular nodes only
- uppercase node labels (single characters)
- straight edges with integer weights
- high contrast, sharp text for OCR
- no overlap, no blur, no perspective, no watermark
Randomize graph topology while preserving readability.
```

### V2.5 Negative prompt (dán thêm nếu tool hỗ trợ)
```text
blurry, low contrast, noisy background, textured paper, hand-drawn, sketch,
watermark, logo, glow, shadow, perspective view, tilted text, curved text,
overlapping labels, overlapping edges, cluttered layout, gradient background
```

### V2.6 Mẫu điền nhanh theo ý bạn
```text
Create a [directed/undirected] weighted graph with [N] circular nodes.
Node labels: [A..].
Weight range: [min-max].
Style: clean technical black-and-white diagram, plain white background.
Constraints: no overlap, straight edges, sharp OCR-friendly text, no blur, no watermark.
Resolution: [width]x[height].
```
