import cv2
import numpy as np
import pytesseract
import math
import json

IMAGE = "/Users/anphan/Desktop/image-to-graph-structure/img/A.png"


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


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
        self.logs = []
        
    def _log(self, msg):
        print(msg)
        import datetime
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{time_str}] {msg}")

    def preprocess_image(self):
        """Tạo nhiều phiên bản ảnh binary để tăng khả năng detect"""
        binaries = []
        
        # 1. Gray + Blur + Simple Threshold
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        _, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        binaries.append(('simple_thresh', th1))
        
        _, th2 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
        binaries.append(('simple_thresh_inv', th2))

        # 2. Adaptive Threshold
        th3 = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        binaries.append(('adaptive', th3))
        
        th4 = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        binaries.append(('adaptive_inv', th4))
        
        # 3. OTSU
        _, th5 = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaries.append(('otsu', th5))
        
        _, th6 = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binaries.append(('otsu_inv', th6))

        return binaries

    # ==========================================
    # BƯỚC 1: PHÁT HIỆN NODES (Universal)

    def detect_nodes_universal(self, min_area=400, max_area=20000): # Tăng max_area lên 20000
        """
        Phát hiện nodes bằng NHIỀU PHƯƠNG PHÁP:
        1. Color detection (nếu có màu)
        2. Contour detection (nhiều thresholds)
        3. Circle detection (Hough)
        """
        self._log("🔍 Phát hiện nodes (universal)...")

        candidates = []

        # --- METHOD 1: Color detection ---
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        color_ranges = [
            ([0, 100, 100], [25, 255, 255]),  # Đỏ/Cam
            ([25, 100, 100], [40, 255, 255]),  # Vàng
            ([40, 50, 50], [80, 255, 255]),  # Xanh lá
            ([80, 50, 50], [130, 255, 255]),  # Xanh dương
            ([130, 50, 50], [180, 255, 255]),  # Tím/Hồng
        ]

        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if min_area < cv2.contourArea(cnt) < max_area:
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    candidates.append({
                        'center': (int(x), int(y)),
                        'radius': int(r),
                        'method': 'color'
                    })

        # --- METHOD 2: Advanced Contour Detection ---
        binaries = self.preprocess_image()
        
        for name, th in binaries:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    # Kiểm tra độ tròn
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                    
                    # Kiểm tra độ lồi (Convexity)
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0

                    # Điều kiện lọc node
                    if circularity > 0.5 and solidity > 0.8:
                        (x, y), r = cv2.minEnclosingCircle(cnt)
                        candidates.append({
                            'center': (int(x), int(y)),
                            'radius': int(r),
                            'method': f'contour_{name}'
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

        # Loại bỏ duplicate
        self.nodes = self._remove_duplicate_nodes(candidates)
        self._log(f"Tìm thấy {len(self.nodes)} nodes")
        return self.nodes

    def _remove_duplicate_nodes(self, candidates, threshold=30):
        """Gộp các nodes trùng lặp thông minh hơn"""
        if not candidates:
            return []

        # Sắp xếp theo bán kính giảm dần để ưu tiên circle to hơn (thường ổn định hơn)
        candidates.sort(key=lambda x: x['radius'], reverse=True)
        
        unique = []
        for cand in candidates:
            is_duplicate = False
            for i, node in enumerate(unique):
                d = dist(cand['center'], node['center'])
                # Nếu tâm quá gần hoặc chồng lấn quá nhiều
                if d < max(cand['radius'], node['radius']): 
                    is_duplicate = True
                    # Nếu candidate hiện tại 'tròn' hơn hoặc tốt hơn thì có thể replace (ở đây giữ simple)
                    break 

            if not is_duplicate:
                unique.append(cand)

        return unique

    # ==========================================
    # BƯỚC 2: OCR LABELS
    # ==========================================

    def ocr_node_labels(self):
        """OCR đọc nhãn trong nodes - Cải thiện độ chính xác"""
        self._log("OCR nhãn nodes...")

        for i, node in enumerate(self.nodes):
            cx, cy = node['center']
            r = node['radius']

            # Crop ROI - crop chặt hơn vào center
            padding = int(r * 0.3)  # Giảm padding để crop chặt hơn
            y1 = max(0, cy - r + padding)
            y2 = min(self.img.shape[0], cy + r - padding)
            x1 = max(0, cx - r + padding)
            x2 = min(self.img.shape[1], cx + r - padding)

            roi = self.img[y1:y2, x1:x2]

            if roi.size == 0:
                node['label'] = f'N{i}'
                continue

            # Thử OCR với nhiều góc xoay (Rotation Invariance)
            angles = [0, -5, 5, -10, 10]
            
            label = None
            best_confidence = 0

            for angle in angles:
                if angle != 0:
                    M = cv2.getRotationMatrix2D((roi.shape[1]//2, roi.shape[0]//2), angle, 1.0)
                    rotated_roi = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]), 
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                else:
                    rotated_roi = roi

                # Preprocess cho biến thể xoay này
                roi_gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)
                roi_variants = []
                
                # Basic thresholds
                _, th1 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                roi_variants.append(th1)
                _, th2 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                roi_variants.append(th2)
                
                # Resize để tăng độ phân giải
                scale = 4
                roi_variants = [cv2.resize(v, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) for v in roi_variants]

                whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789∞"
                psm_modes = ["--psm 10", "--psm 6", "--psm 7"]

                for v in roi_variants:
                    for psm_config in psm_modes:
                        config = f"{psm_config} --oem 3 -c tessedit_char_whitelist={whitelist}"
                        try:
                            data = pytesseract.image_to_data(v, config=config, output_type=pytesseract.Output.DICT)
                            texts = [t.strip() for t in data['text'] if t.strip()]
                            confidences = [c for c, t in zip(data['conf'], data['text']) if t.strip()]

                            if texts:
                                max_conf_idx = confidences.index(max(confidences))
                                text = texts[max_conf_idx]
                                conf = confidences[max_conf_idx]
                                
                                # Clean text
                                text = text.replace(' ', '').replace('\n', '').replace('\t', '')

                                # Post-processing: Sửa các lỗi phổ biến
                                text = self._correct_ocr_errors(text)

                                # Chỉ chấp nhận nếu confidence > 30 và text hợp lệ
                                if text and len(text) <= 3 and conf > best_confidence and conf > 30:
                                    best_confidence = conf
                                    label = text
                        except:
                            pass
                
                if best_confidence > 80:
                    break

            node['label'] = label if label else f'N{i}'

            # Vẽ nhãn Premium
            label_text = node['label']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            
            # Tọa độ nhãn (phía trên node)
            tx = cx - text_size[0] // 2
            ty = cy - r - 15
            
            # Vẽ nền cho nhãn cho sang trọng
            cv2.rectangle(self.img_vis, (tx - 5, ty - text_size[1] - 5), 
                          (tx + text_size[0] + 5, ty + 5), (0, 0, 0), -1)
            cv2.putText(self.img_vis, label_text, (tx, ty), font, font_scale, (255, 255, 255), thickness)
            cv2.circle(self.img_vis, (cx, cy), r, (0, 255, 0), 2)

        self._log(f"   Labels: {[n['label'] for n in self.nodes]}")

    def _correct_ocr_errors(self, text):
        """Sửa các lỗi OCR phổ biến"""
        if not text:
            return text

        text = text.upper().strip()

        # Mapping các lỗi phổ biến
        corrections = {
            'BY': 'B',
            'S': 'B',
            '5': 'B',
            'SS': 'B',
            'R': 'C', 
            'RA': 'A',
            'A': 'A',
            'C': 'C',
            'E': 'E',
            'B': 'B',
            'F': 'F',
            '0': 'C',
            'Q': 'C',
            '8': 'B',
            '3': 'E',
            '#': 'F',
            '6': 'G',
            'G': 'G', # Ensure G stays G
            'G ': 'G',
            'C ': 'C',
            'E ': 'E',
            'F ': 'F',
            'BY ': 'B',
            'BY\n': 'B',
            'rA': 'A',
            'rA ': 'A',
            'rA\n': 'A',
            'CA': 'C',
            'CA ': 'C',
            'CA\n': 'C',
            '8': '∞',
            'OO': '∞',
            '00': '∞',
            '0': 'O',
            '1': 'I',
            '4': 'B',  # Fix 4 -> B
            'M': 'C',  # Fix M -> C (looks like turned C)
            'W': 'C',
            'U': 'C',
        }


        # Áp dụng corrections
        if text in corrections:
            return corrections[text]

        # Nếu text có 2 ký tự và bắt đầu bằng ký tự đúng, lấy ký tự đầu
        if len(text) == 2:
            # Nếu ký tự đầu là chữ cái hợp lệ và ký tự thứ 2 là lỗi
            if text[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and text[1] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                return text[0]
            # Nếu ký tự thứ 2 là chữ cái hợp lệ và ký tự đầu là lỗi
            if text[1] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and text[0] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                return text[1]

        # Chỉ lấy ký tự đầu nếu là chữ cái hợp lệ
        if len(text) > 1 and text[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            cand = text[0]
            if cand in corrections:
                return corrections[cand]
            return cand

        return text

    # ==========================================
    # BƯỚC 3: PHÁT HIỆN TRỌNG SỐ
    # ==========================================

    def detect_weights(self):
        """Phát hiện số bên ngoài nodes (trọng số) theo phương pháp Component-based"""
        self._log("Phát hiện trọng số (Component-based)...")

        # 1. Mask loại bỏ nodes
        mask_nodes = np.ones_like(self.gray) * 255
        for node in self.nodes:
            cv2.circle(mask_nodes, node['center'], int(node['radius'] + 5), 0, -1)
        
        # Ảnh không có nodes
        no_nodes = cv2.bitwise_and(self.gray, mask_nodes)
        
        # 2. Xử lý ảnh để tách số và nét vẽ
        # Threshold
        blur = cv2.GaussianBlur(no_nodes, (3, 3), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)

        # Loại bỏ các đường kẻ dài (Edges) bằng Morphology
        # Kernel ngang và dọc để bắt các đường kẻ (tăng kích thước để không bắt nhầm số 1)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
        
        detected_lines = cv2.add(cv2.morphologyEx(th, cv2.MORPH_OPEN, horizontal_kernel),
                                 cv2.morphologyEx(th, cv2.MORPH_OPEN, vertical_kernel))
        
        # Xóa các đường kẻ khỏi ảnh threshold
        # Dùng dilate lines một chút để xóa sạch
        detected_lines = cv2.dilate(detected_lines, np.ones((3,3), np.uint8), iterations=1)
        
        # Ảnh chỉ còn lại các thành phần nhỏ (chữ số, nhiễu)
        text_components = cv2.bitwise_and(th, cv2.bitwise_not(detected_lines))
        
        # Noise removal (chấm nhỏ quá)
        text_components = cv2.morphologyEx(text_components, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

        # 3. Find Contours các vùng nghi là số
        cnts, _ = cv2.findContours(text_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter kích thước: nới lỏng hơn nữa (đặc biệt area nhỏ)
            if 20 < area < 4000 and 0.1 < w/h < 6.0:
                # Merge các bounding box gần nhau (ví dụ số 14 gồm 1 và 4)
                candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'center': (x + w//2, y + h//2)})

        # Merge các box gần nhau (nhóm các chữ số của cùng 1 số tự nhiên)
        merged_candidates = self._merge_close_boxes(candidates, distance_th=35)

        # 4. OCR từng vùng
        self._log(f"   Tìm thấy {len(merged_candidates)} vùng tiềm năng, bắt đầu OCR từng vùng...")
        
        whitelist = "-c tessedit_char_whitelist=0123456789"
        psm_modes = ["--psm 10", "--psm 7", "--psm 6"] # prioritized single char/line

        for cand in merged_candidates:
            x, y, w, h = cand['x'], cand['y'], cand['w'], cand['h']
            
            # Crop và thêm padding
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(self.img.shape[1], x + w + pad)
            y2 = min(self.img.shape[0], y + h + pad)
            
            roi = self.gray[y1:y2, x1:x2]
            
            # Preprocess ROI
            # Resize to make it big enough for Tesseract
            scale = 3
            roi_big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Threshold ROI
            _, roi_th = cv2.threshold(roi_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert nếu cần (Tesseract thích đen trên trắng)
            # Ở đây roi_th là trắng trên đen (do THRESH_BINARY trên nền đen).
            # Tesseract hoạt động tốt với chữ đen nền trắng => Invert lại
            roi_th = cv2.bitwise_not(roi_th)

            best_conf = 0
            best_text = None

            for psm in psm_modes:
                config = f"{psm} --oem 3 {whitelist}"
                try: 
                    data = pytesseract.image_to_data(roi_th, config=config, output_type=pytesseract.Output.DICT)
                    texts = [t.strip() for t in data['text'] if t.strip()]
                    confs = [c for c, t in zip(data['conf'], data['text']) if t.strip()]
                    
                    if texts:
                        # Lấy text có confidence cao nhất
                        idx = confs.index(max(confs))
                        c_text = texts[idx]
                        c_conf = confs[idx]
                        
                        if c_text.isdigit() and c_conf > best_conf:
                            best_conf = c_conf
                            best_text = c_text
                except:
                    pass

            if best_text and best_conf > 30: # Ngưỡng tự tin thấp hơn chút vì crop nhỏ
                 value = int(best_text)
                 
                 # Check duplicate
                 is_dup = False
                 for w_exist in self.weights:
                     if dist(cand['center'], w_exist['center']) < 10:
                         is_dup = True
                         break
                 
                 if not is_dup:
                     self.weights.append({
                        'value': value,
                        'center': cand['center'],
                        'bbox': (x, y, w, h)
                     })
                     
                     # Draw Vis
                     cv2.rectangle(self.img_vis, (x, y), (x+w, y+h), (0, 165, 255), 2)
                     cv2.putText(self.img_vis, str(value), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        self._log(f"Tìm thấy {len(self.weights)} trọng số: {[w['value'] for w in self.weights]}")

    def _merge_close_boxes(self, candidates, distance_th=20):
        """Gộp các bounding box gần nhau (ví dụ số có 2 chữ số)"""
        if not candidates:
            return []
            
        # Đơn giản: lặp component connected
        # Nhưng để code ngắn gọn, dùng logic merge 2 box gần nhau lặp đi lặp lại
        
        while True:
            merged = False
            new_candidates = []
            used_indices = set()
            
            for i in range(len(candidates)):
                if i in used_indices:
                    continue
                
                cur = candidates[i]
                merged_box = cur.copy()
                
                for j in range(i + 1, len(candidates)):
                    if j in used_indices:
                        continue
                    
                    n = candidates[j]
                    
                    # Tính khoảng cách giữa 2 center
                    d = math.hypot(cur['center'][0] - n['center'][0], cur['center'][1] - n['center'][1])
                    
                    # Hoặc check overlap / distance giữa các cạnh
                    # Ở đây dùng center distance cho đơn giản + check alignment
                    # Số thường nằm ngang cạnh nhau
                    dx = abs(cur['center'][0] - n['center'][0])
                    dy = abs(cur['center'][1] - n['center'][1])
                    
                    if dx < distance_th and dy < distance_th/2: # Gần nhau theo phương ngang
                        # Merge
                        x_min = min(merged_box['x'], n['x'])
                        y_min = min(merged_box['y'], n['y'])
                        x_max = max(merged_box['x'] + merged_box['w'], n['x'] + n['w'])
                        y_max = max(merged_box['y'] + merged_box['h'], n['y'] + n['h'])
                        
                        merged_box = {
                            'x': x_min,
                            'y': y_min,
                            'w': x_max - x_min,
                            'h': y_max - y_min,
                            'center': ((x_min + x_max)//2, (y_min + y_max)//2)
                        }
                        used_indices.add(j)
                        merged = True
                
                new_candidates.append(merged_box)
                
            if not merged:
                break
            candidates = new_candidates
            
        return candidates

    # ==========================================
    # BƯỚC 4: PHÁT HIỆN EDGES + GÁN TRỌNG SỐ
    # ==========================================

    def detect_edges(self):
        """Phát hiện edges và gán trọng số"""
        self._log("Phát hiện edges...")

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
        lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, 40, minLineLength=20, maxLineGap=20)

        if lines is None:
            self._log("Không tìm thấy edges")
            return

        detected_pairs = {} # (n1, n2) -> {weight, line}

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Tìm nearest nodes
            n1, d1 = self._find_nearest_node((x1, y1))
            n2, d2 = self._find_nearest_node((x2, y2))

            if n1 is not None and n2 is not None and n1 != n2: # Avoid self-loops
                if d1 < 120 and d2 < 120: 
                    # Sắp xếp để không phân biệt n1->n2 và n2->n1
                    pair = tuple(sorted((n1, n2)))
                    
                    # Tính độ dài line detect được
                    line_len = math.hypot(x1 - x2, y1 - y2)
                    
                    # Tính khoảng cách thực tế giữa 2 node
                    c1 = self.nodes[n1]['center']
                    c2 = self.nodes[n2]['center']
                    node_dist = dist(c1, c2)
                    
                    # Chỉ chấp nhận nếu line detect được có độ dài tương đối so với khoảng cách node
                    # (tránh noise nhỏ kết nối 2 node xa)
                    if line_len > node_dist * 0.3:
                        if pair not in detected_pairs:
                            detected_pairs[pair] = {
                                'from': n1,
                                'to': n2,
                                'weight': None,
                                'line': (x1, y1, x2, y2) # Keep the detected line segment
                            }
                        
                        # Vẽ (debug)
                        cv2.line(self.img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Gán trọng số cho edges đã tìm thấy
        for pair, data in detected_pairs.items():
            n1, n2 = pair
            c1 = self.nodes[n1]['center']
            c2 = self.nodes[n2]['center']
            
            best_weight = None
            min_dist = float('inf')

            # Vector đoạn thẳng nối 2 tâm node
            edge_len = dist(c1, c2)

            for w in self.weights:
                wx, wy = w['center']
                
                # 1. Khoảng cách vuông góc đến đoạn thẳng
                d_perpendicular = self._point_line_segment_distance(wx, wy, c1[0], c1[1], c2[0], c2[1])
                
                # 2. Khoảng cách đến trung điểm đoạn thẳng (fallback)
                mx, my = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
                d_midpoint = math.hypot(wx - mx, wy - my)
                
                # Điều kiện chấp nhận:
                # - Vuông góc thấp (< 60) HOẶC
                # - Gần trung điểm (< 1/3 độ dài cạnh)
                
                valid_candidate = False
                current_dist = float('inf')

                if d_perpendicular < 60:
                    valid_candidate = True
                    current_dist = d_perpendicular
                elif d_midpoint < edge_len * 0.4: # Nới lỏng check trung điểm
                    valid_candidate = True
                    current_dist = d_midpoint * 0.5 # Ưu tiên midpoint nếu perpendicular fail
                
                if valid_candidate and current_dist < min_dist:
                    min_dist = current_dist
                    best_weight = w['value']

            data['weight'] = best_weight

        self.edges = list(detected_pairs.values())
        self._log(f"Tìm thấy {len(self.edges)} edges")

    def _point_line_segment_distance(self, px, py, x1, y1, x2, y2):
        """Tính khoảng cách từ điểm (px,py) đến đoạn thẳng (x1,y1)-(x2,y2)"""
        # Vector AB
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)

        # Project point onto line (parameter t)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

        # Clamp t to segment [0, 1]
        t = max(0, min(1, t))

        # Closest point on segment
        nx = x1 + t * dx
        ny = y1 + t * dy

        return math.hypot(px - nx, py - ny)

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
    
    # Removed _find_nearest_weight as it is integrated into detect_edges now

    # ==========================================
    # PIPELINE + OUTPUT
    # ==========================================

    def process(self):
        """Pipeline đầy đủ"""
        self._log("\n" + "=" * 50)
        self._log("BẮT ĐẦU XỬ LÝ")
        self._log("=" * 50 + "\n")

        self.detect_nodes_universal()
        self.ocr_node_labels()
        self.detect_weights()
        self.detect_edges()

        return self.nodes, self.edges

    def output_results(self):
        """In kết quả"""
        print("\n" + "=" * 50)
        print("KẾT QUẢ")
        print("=" * 50)

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
            ],
            "logs": self.logs
        }

        with open(f"{basename}.json", "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Visualization
        viz_path = f"{basename}_visualization.jpg"
        cv2.imwrite(viz_path, self.img_vis)

        print("\nĐã lưu:")
        print(f"{basename}.json")
        print(viz_path)
        
        return data, viz_path


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