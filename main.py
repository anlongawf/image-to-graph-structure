import cv2
import numpy as np
import pytesseract
import math
import json

IMAGE = "4.jpg"

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


class UniversalGraphRecognizer:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Không đọc được: {image_path}")

        self.img_vis = self.img.copy()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.nodes = []
        self.edges = []
        self.weights = []

    # ==========================================
    # BƯỚC 1: PHÁT HIỆN NODES (Universal)

    def detect_nodes_universal(self, min_area=400, max_area=6000):
        """
        Phát hiện nodes bằng NHIỀU PHƯƠNG PHÁP:
        1. Color detection (nếu có màu)
        2. Contour detection (threshold)
        3. Circle detection (Hough)
        """
        print("🔍 Phát hiện nodes (universal)...")

        candidates = []

        # --- METHOD 1: Color detection ---
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Detect màu sắc đa dạng (nhiều range)
        color_ranges = [
            ([0, 100, 100], [25, 255, 255]),     # Đỏ/Cam
            ([25, 100, 100], [40, 255, 255]),    # Vàng
            ([40, 50, 50], [80, 255, 255]),      # Xanh lá
            ([80, 50, 50], [130, 255, 255]),     # Xanh dương
            ([130, 50, 50], [180, 255, 255]),    # Tím/Hồng
        ]

        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        (x, y), r = cv2.minEnclosingCircle(cnt)
                        candidates.append({
                            'center': (cx, cy),
                            'radius': int(r),
                            'method': 'color'
                        })

        # --- METHOD 2: Threshold + Contour ---
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Thử cả BINARY và BINARY_INV
        for thresh_type in [cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY]:
            _, th = cv2.threshold(blur, 127, 255, thresh_type)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    # Kiểm tra hình dạng (chỉ lấy gần tròn/vuông)
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h > 0 else 0

                    if 0.6 < aspect_ratio < 1.5:
                        M = cv2.moments(cnt)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            (px, py), r = cv2.minEnclosingCircle(cnt)
                            candidates.append({
                                'center': (cx, cy),
                                'radius': int(r),
                                'method': 'threshold'
                            })

        # --- METHOD 3: Hough Circle Detection ---
        circles = cv2.HoughCircles(
            self.gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=100, param2=30, minRadius=15, maxRadius=60
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, r = circle
                candidates.append({
                    'center': (int(cx), int(cy)),
                    'radius': int(r),
                    'method': 'hough'
                })

        # Loại bỏ duplicate (nodes trùng nhau)
        self.nodes = self._remove_duplicate_nodes(candidates)

        print(f"Tìm thấy {len(self.nodes)} nodes")
        return self.nodes

    def _remove_duplicate_nodes(self, candidates, threshold=30):
        """Gộp các nodes trùng lặp"""
        if not candidates:
            return []

        unique = []
        for cand in candidates:
            is_duplicate = False
            for node in unique:
                if dist(cand['center'], node['center']) < threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(cand)

        return unique

    # ==========================================
    # BƯỚC 2: OCR LABELS
    # ==========================================

    def ocr_node_labels(self):
        """OCR đọc nhãn trong nodes"""
        print("OCR nhãn nodes...")

        for i, node in enumerate(self.nodes):
            cx, cy = node['center']
            r = node['radius']

            # Crop ROI
            padding = int(r * 0.1)
            y1 = max(0, cy - r - padding)
            y2 = min(self.img.shape[0], cy + r + padding)
            x1 = max(0, cx - r - padding)
            x2 = min(self.img.shape[1], cx + r + padding)

            roi = self.img[y1:y2, x1:x2]

            if roi.size == 0:
                node['label'] = f'N{i}'
                continue

            # Enhance ROI
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Thử cả 2 cách threshold
            _, th1 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th2 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Resize
            scale = 3
            th1 = cv2.resize(th1, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            th2 = cv2.resize(th2, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # OCR với whitelist rộng
            config = "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789∞"

            label = None
            for th in [th1, th2]:
                try:
                    text = pytesseract.image_to_string(th, config=config).strip()
                    text = text.replace(' ', '').replace('\n', '')

                    # Xử lý ký tự đặc biệt
                    if text in ['8', 'oo', '00', 'OO']:
                        text = '∞'

                    if text and len(text) <= 3:
                        label = text
                        break
                except:
                    pass

            node['label'] = label if label else f'N{i}'

            # Vẽ
            cv2.circle(self.img_vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.putText(self.img_vis, node['label'], (cx-15, cy-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"   Labels: {[n['label'] for n in self.nodes]}")

    # ==========================================
    # BƯỚC 3: PHÁT HIỆN TRỌNG SỐ
    # ==========================================

    def detect_weights(self):
        """Phát hiện số bên ngoài nodes (trọng số)"""
        print("Phát hiện trọng số...")

        # Tạo mask loại bỏ nodes
        mask = np.ones_like(self.gray) * 255
        for node in self.nodes:
            cx, cy = node['center']
            r = node['radius']
            cv2.circle(mask, (cx, cy), r + 15, 0, -1)

        masked = cv2.bitwise_and(self.gray, mask)

        # Threshold để tìm text
        _, th = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphology để gộp text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        # Tìm contours
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if 30 < area < 1500:
                x, y, w, h = cv2.boundingRect(cnt)

                # Kiểm tra aspect ratio (text thường có tỷ lệ nhất định)
                aspect = w / float(h) if h > 0 else 0
                if 0.3 < aspect < 3:
                    roi = self.img[y:y+h, x:x+w]

                    if roi.size == 0:
                        continue

                    # OCR
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.resize(roi_gray, None, fx=3, fy=3)
                    _, roi_th = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789∞"

                    try:
                        text = pytesseract.image_to_string(roi_th, config=config).strip()
                        text = text.replace(' ', '').replace('\n', '')

                        if text.isdigit() or text == '∞':
                            center = (x + w//2, y + h//2)
                            value = int(text) if text.isdigit() else text

                            self.weights.append({
                                'value': value,
                                'center': center,
                                'bbox': (x, y, w, h)
                            })

                            # Vẽ
                            cv2.rectangle(self.img_vis, (x, y), (x+w, y+h), (0, 255, 255), 1)
                    except:
                        pass

        print(f"Tìm thấy {len(self.weights)} trọng số: {[w['value'] for w in self.weights]}")

    # ==========================================
    # BƯỚC 4: PHÁT HIỆN EDGES + GÁN TRỌNG SỐ
    # ==========================================

    def detect_edges(self):
        """Phát hiện edges và gán trọng số"""
        print("Phát hiện edges...")

        # Mask loại nodes
        mask = np.ones_like(self.gray) * 255
        for node in self.nodes:
            cx, cy = node['center']
            r = node['radius']
            cv2.circle(mask, (cx, cy), r + 5, 0, -1)

        masked = cv2.bitwise_and(self.gray, mask)

        # Canny
        edges_img = cv2.Canny(masked, 30, 100)

        # Hough Lines
        lines = cv2.HoughLinesP(edges_img, 1, np.pi/180, 40, minLineLength=20, maxLineGap=20)

        if lines is None:
            print("Không tìm thấy edges")
            return

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Tìm nearest nodes
            n1, d1 = self._find_nearest_node((x1, y1))
            n2, d2 = self._find_nearest_node((x2, y2))

            if n1 is not None and n2 is not None and d1 < 100 and d2 < 100:
                # Tính midpoint
                mid = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Tìm weight gần nhất
                weight = self._find_nearest_weight(mid)

                self.edges.append({
                    'from': n1,
                    'to': n2,
                    'weight': weight,
                    'line': (x1, y1, x2, y2)
                })

                # Vẽ
                cv2.line(self.img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        print(f"Tìm thấy {len(self.edges)} edges")

    def _find_nearest_node(self, point):
        """Tìm node gần nhất"""
        min_d = float('inf')
        nearest = None

        for i, node in enumerate(self.nodes):
            d = dist(point, node['center'])
            if d < min_d:
                min_d = d
                nearest = i

        return nearest, min_d

    def _find_nearest_weight(self, point, max_dist=50):
        """Tìm trọng số gần nhất"""
        min_d = float('inf')
        nearest = None

        for w in self.weights:
            d = dist(point, w['center'])
            if d < min_d and d < max_dist:
                min_d = d
                nearest = w['value']

        return nearest

    # ==========================================
    # PIPELINE + OUTPUT
    # ==========================================

    def process(self):
        """Pipeline đầy đủ"""
        print("\n" + "="*50)
        print("BẮT ĐẦU XỬ LÝ")
        print("="*50 + "\n")

        self.detect_nodes_universal()
        self.ocr_node_labels()
        self.detect_weights()
        self.detect_edges()

        return self.nodes, self.edges

    def output_results(self):
        """In kết quả"""
        print("\n" + "="*50)
        print("KẾT QUẢ")
        print("="*50)

        print("\nNODES:")
        for i, node in enumerate(self.nodes):
            print(f"   {i}. {node['label']}")

        print("\nEDGES:")
        if not self.edges:
            print("   (không tìm thấy)")
        else:
            for edge in self.edges:
                from_label = self.nodes[edge['from']]['label']
                to_label = self.nodes[edge['to']]['label']
                weight = edge['weight'] if edge['weight'] else ''

                if weight:
                    print(f"   {from_label} → {to_label} ({weight})")
                else:
                    print(f"   {from_label} → {to_label}")

    def export(self, basename="graph_output"):
        """Export files"""
        # JSON
        data = {
            "nodes": [{"id": i, "label": n['label']} for i, n in enumerate(self.nodes)],
            "edges": [
                {
                    "from": self.nodes[e['from']]['label'],
                    "to": self.nodes[e['to']]['label'],
                    "weight": e['weight']
                }
                for e in self.edges
            ]
        }

        with open(f"{basename}.json", "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Visualization
        cv2.imwrite(f"{basename}_visualization.jpg", self.img_vis)

        print("\nĐã lưu:")
        print(f"{basename}.json")
        print(f"{basename}_visualization.jpg")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    try:
        recognizer = UniversalGraphRecognizer(IMAGE)
        recognizer.process()
        recognizer.output_results()
        recognizer.export("universal_graph")

        print("\nHOÀN THÀNH!")

    except Exception as e:
        print(f"\nLỖI: {e}")
        import traceback
        traceback.print_exc()