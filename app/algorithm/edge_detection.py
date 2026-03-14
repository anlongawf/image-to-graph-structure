"""Phát hiện edges (cạnh) và gán trọng số."""

import math

import cv2
import numpy as np

from .utils import dist, find_nearest_node, point_line_segment_distance


def detect_edges(gray: np.ndarray, img_vis: np.ndarray,
                 nodes: list, weights: list) -> list:
    """Phát hiện edges bằng Canny + HoughLinesP, rồi gán trọng số.

    Args:
        gray: Ảnh grayscale.
        img_vis: Ảnh visualization (sẽ vẽ edges lên đây).
        nodes: List of node dicts.
        weights: List of weight dicts.

    Returns:
        List of edge dicts: {'from': int, 'to': int, 'weight': int|None}
    """
    # Mask loại nodes
    mask = np.ones_like(gray) * 255
    for node in nodes:
        cx, cy = node['center']
        r = node['radius']
        # Mask slightly smaller than radius to keep line endpoints touching the boundary
        cv2.circle(mask, (cx, cy), r - 5, 0, -1)

    masked = cv2.bitwise_and(gray, mask)

    # Canny + Hough Lines
    edges_img = cv2.Canny(masked, 30, 100)
    lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, 20, minLineLength=5, maxLineGap=80)

    detected_pairs = {}  # (n1, n2) -> edge data

    if lines is None:
        return []

    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            n1, n2 = i, j
            c1 = nodes[n1]['center']
            c2 = nodes[n2]['center']
            edge_len = dist(c1, c2)

            if edge_len < 10:
                continue

            score = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                seg_len = dist((x1, y1), (x2, y2))
                if seg_len < 3:
                    continue

                # Tính khoảng cách từ 2 mút đoạn thẳng đến cạnh "lý thuyết" n1-n2
                d1 = point_line_segment_distance(x1, y1, c1[0], c1[1], c2[0], c2[1])
                d2 = point_line_segment_distance(x2, y2, c1[0], c1[1], c2[0], c2[1])

                # Endpoints must be close to the ideal line
                if d1 < 30 and d2 < 30:
                    # Kiểm tra độ song song
                    dot = (x2 - x1) * (c2[0] - c1[0]) + (y2 - y1) * (c2[1] - c1[1])
                    cos_val = dot / (seg_len * edge_len)

                    if abs(cos_val) > 0.85:  # Góc lệch < ~30 độ
                        score += seg_len

            # Nếu tổng chiều dài các đoạn trùng lắp đủ lớn (Cải tiến: Yêu cầu cả % chiều dài cạnh)
            if score > max(35, edge_len * 0.25):
                pair = tuple(sorted((n1, n2)))
                # Vẽ cạnh sạch sẽ từ tâm đến tâm
                cv2.line(img_vis, c1, c2, (255, 0, 0), 2)
                
                detected_pairs[pair] = {
                    'from': n1,
                    'to': n2,
                    'weight': None,
                    'line': (c1[0], c1[1], c2[0], c2[1])
                }

    # Gán trọng số cho edges
    for pair, data in detected_pairs.items():
        n1, n2 = pair
        c1 = nodes[n1]['center']
        c2 = nodes[n2]['center']
        edge_len = dist(c1, c2)

        best_weight = None
        min_dist_val = float('inf')

        for w in weights:
            wx, wy = w['center']

            d_perp = point_line_segment_distance(wx, wy, c1[0], c1[1], c2[0], c2[1])
            mx, my = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
            d_mid = math.hypot(wx - mx, wy - my)

            valid = False
            current_dist = float('inf')

            if d_perp < 60:
                valid = True
                current_dist = d_perp
            elif d_mid < edge_len * 0.4:
                valid = True
                current_dist = d_mid * 0.5

            if valid and current_dist < min_dist_val:
                min_dist_val = current_dist
                best_weight = w['value']

        data['weight'] = best_weight

    return list(detected_pairs.values())
