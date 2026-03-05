"""Phát hiện trọng số (weights) bên ngoài nodes."""

import math

import cv2
import numpy as np
import pytesseract


def detect_weights(img: np.ndarray, gray: np.ndarray, img_vis: np.ndarray,
                   nodes: list) -> list:
    """Phát hiện số bên ngoài nodes (trọng số) theo Component-based.

    Args:
        img: Ảnh gốc BGR.
        gray: Ảnh grayscale.
        img_vis: Ảnh visualization (sẽ vẽ bounding box lên đây).
        nodes: List of node dicts.

    Returns:
        List of weight dicts: {'value': int, 'center': (x,y), 'bbox': (x,y,w,h)}
    """
    weights = []

    # 1. Mask loại bỏ nodes
    mask_nodes = np.ones_like(gray) * 255
    for node in nodes:
        cv2.circle(mask_nodes, node['center'], int(node['radius'] + 5), 0, -1)

    no_nodes = cv2.bitwise_and(gray, mask_nodes)

    # 2. Threshold + loại bỏ đường kẻ dài
    blur = cv2.GaussianBlur(no_nodes, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))

    detected_lines = cv2.add(
        cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel)
    )
    detected_lines = cv2.dilate(detected_lines, np.ones((3, 3), np.uint8), iterations=1)

    text_components = cv2.bitwise_and(th, cv2.bitwise_not(detected_lines))
    text_components = cv2.morphologyEx(text_components, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # 3. Find contours
    cnts, _ = cv2.findContours(text_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 20 < area < 4000 and 0.1 < w / h < 6.0:
            candidates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'center': (x + w // 2, y + h // 2)
            })

    merged = _merge_close_boxes(candidates, distance_th=35)

    # 4. OCR từng vùng
    whitelist = "-c tessedit_char_whitelist=0123456789"
    psm_modes = ["--psm 10", "--psm 7", "--psm 6"]

    for cand in merged:
        x, y, w, h = cand['x'], cand['y'], cand['w'], cand['h']

        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        roi = gray[y1:y2, x1:x2]

        scale = 3
        roi_big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, roi_th = cv2.threshold(roi_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        roi_th = cv2.bitwise_not(roi_th)

        best_conf = 0
        best_text = None

        for psm in psm_modes:
            config = f"{psm} --oem 3 {whitelist}"
            try:
                data = pytesseract.image_to_data(
                    roi_th, config=config, output_type=pytesseract.Output.DICT
                )
                texts = [t.strip() for t in data['text'] if t.strip()]
                confs = [c for c, t in zip(data['conf'], data['text']) if t.strip()]

                if texts:
                    idx = confs.index(max(confs))
                    c_text = texts[idx]
                    c_conf = confs[idx]

                    if c_text.isdigit() and c_conf > best_conf:
                        best_conf = c_conf
                        best_text = c_text
            except Exception:
                pass

        if best_text and best_conf > 30:
            value = int(best_text)

            # Check duplicate
            is_dup = any(
                math.hypot(cand['center'][0] - w_exist['center'][0],
                           cand['center'][1] - w_exist['center'][1]) < 10
                for w_exist in weights
            )

            if not is_dup:
                weights.append({
                    'value': value,
                    'center': cand['center'],
                    'bbox': (x, y, w, h)
                })

                # Vẽ visualization
                cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(
                    img_vis, str(value), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                )

    return weights


def _merge_close_boxes(candidates: list, distance_th: int = 20) -> list:
    """Gộp các bounding box gần nhau (ví dụ số có 2 chữ số)."""
    if not candidates:
        return []

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
                dx = abs(cur['center'][0] - n['center'][0])
                dy = abs(cur['center'][1] - n['center'][1])

                if dx < distance_th and dy < distance_th / 2:
                    x_min = min(merged_box['x'], n['x'])
                    y_min = min(merged_box['y'], n['y'])
                    x_max = max(merged_box['x'] + merged_box['w'], n['x'] + n['w'])
                    y_max = max(merged_box['y'] + merged_box['h'], n['y'] + n['h'])

                    merged_box = {
                        'x': x_min, 'y': y_min,
                        'w': x_max - x_min, 'h': y_max - y_min,
                        'center': ((x_min + x_max) // 2, (y_min + y_max) // 2)
                    }
                    used_indices.add(j)
                    merged = True

            new_candidates.append(merged_box)

        if not merged:
            break
        candidates = new_candidates

    return candidates
