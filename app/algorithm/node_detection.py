"""Phát hiện nodes (đỉnh) trong ảnh đồ thị."""

import cv2
import numpy as np

from .preprocessing import preprocess_image
from .utils import dist


def detect_nodes(img: np.ndarray, gray: np.ndarray,
                 min_area: int = 400, max_area: int = 20000) -> list:
    """Phát hiện nodes bằng 3 phương pháp kết hợp.

    1. Color detection (nếu có màu)
    2. Contour detection (nhiều thresholds)
    3. Circle detection (Hough)

    Args:
        img: Ảnh gốc BGR.
        gray: Ảnh grayscale.
        min_area: Diện tích tối thiểu của node.
        max_area: Diện tích tối đa của node.

    Returns:
        List of node dicts: {'center': (x,y), 'radius': r, 'method': str}
    """
    candidates = []

    # --- METHOD 1: Color detection ---
    candidates.extend(_detect_by_color(img, min_area, max_area))

    # --- METHOD 2: Advanced Contour Detection ---
    candidates.extend(_detect_by_contour(gray, min_area, max_area))

    # --- METHOD 3: Hough Circle Detection ---
    candidates.extend(_detect_by_hough(gray))

    # Loại bỏ duplicate
    return _remove_duplicate_nodes(candidates)


def _detect_by_color(img: np.ndarray, min_area: int, max_area: int) -> list:
    """Detect nodes bằng lọc màu HSV."""
    candidates = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_ranges = [
        ([0, 100, 100], [25, 255, 255]),    # Đỏ/Cam
        ([25, 100, 100], [40, 255, 255]),    # Vàng
        ([40, 50, 50], [80, 255, 255]),      # Xanh lá
        ([80, 50, 50], [130, 255, 255]),     # Xanh dương
        ([130, 50, 50], [180, 255, 255]),    # Tím/Hồng
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

    return candidates


def _detect_by_contour(gray: np.ndarray, min_area: int, max_area: int) -> list:
    """Detect nodes bằng contour trên nhiều binary thresholds."""
    candidates = []
    binaries = preprocess_image(gray)

    for name, th in binaries:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                circularity = (
                    4 * np.pi * (area / (perimeter * perimeter))
                    if perimeter > 0 else 0
                )

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                if circularity > 0.5 and solidity > 0.8:
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    candidates.append({
                        'center': (int(x), int(y)),
                        'radius': int(r),
                        'method': f'contour_{name}'
                    })

    return candidates


def _detect_by_hough(gray: np.ndarray) -> list:
    """Detect nodes bằng Hough Circle Transform."""
    candidates = []
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
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

    return candidates


def _remove_duplicate_nodes(candidates: list, threshold: int = 30) -> list:
    """Gộp các nodes trùng lặp — ưu tiên circle lớn hơn."""
    if not candidates:
        return []

    candidates.sort(key=lambda x: x['radius'], reverse=True)

    unique = []
    for cand in candidates:
        is_duplicate = False
        for node in unique:
            d = dist(cand['center'], node['center'])
            if d < max(cand['radius'], node['radius']):
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(cand)

    return unique
