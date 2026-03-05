"""Utility functions dùng chung cho algorithm modules."""

import math


def dist(a: tuple, b: tuple) -> float:
    """Tính khoảng cách Euclidean giữa 2 điểm."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_line_segment_distance(px, py, x1, y1, x2, y2) -> float:
    """Tính khoảng cách từ điểm (px, py) đến đoạn thẳng (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    # Project point onto line (parameter t)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp to segment [0, 1]

    # Closest point on segment
    nx = x1 + t * dx
    ny = y1 + t * dy

    return math.hypot(px - nx, py - ny)


def find_nearest_node(point: tuple, nodes: list) -> tuple:
    """Tìm node gần nhất với point.

    Returns:
        (index, distance) — index của node gần nhất và khoảng cách.
    """
    min_d = float('inf')
    nearest = None

    for i, node in enumerate(nodes):
        d = dist(point, node['center'])
        if d < min_d:
            min_d = d
            nearest = i

    return nearest, min_d
