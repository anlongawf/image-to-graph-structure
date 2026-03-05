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
        cv2.circle(mask, (cx, cy), r + 5, 0, -1)

    masked = cv2.bitwise_and(gray, mask)

    # Canny + Hough Lines
    edges_img = cv2.Canny(masked, 30, 100)
    lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, 40, minLineLength=20, maxLineGap=20)

    if lines is None:
        return []

    detected_pairs = {}  # (n1, n2) -> edge data

    for line in lines:
        x1, y1, x2, y2 = line[0]

        n1, d1 = find_nearest_node((x1, y1), nodes)
        n2, d2 = find_nearest_node((x2, y2), nodes)

        if n1 is not None and n2 is not None and n1 != n2:
            if d1 < 120 and d2 < 120:
                pair = tuple(sorted((n1, n2)))

                line_len = math.hypot(x1 - x2, y1 - y2)
                c1 = nodes[n1]['center']
                c2 = nodes[n2]['center']
                node_dist = dist(c1, c2)

                if line_len > node_dist * 0.3:
                    if pair not in detected_pairs:
                        detected_pairs[pair] = {
                            'from': n1,
                            'to': n2,
                            'weight': None,
                            'line': (x1, y1, x2, y2)
                        }

                    cv2.line(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
