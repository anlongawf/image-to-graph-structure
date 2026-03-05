"""Tiền xử lý ảnh: tạo nhiều phiên bản binary để tăng khả năng detect."""

import cv2
import numpy as np


def preprocess_image(gray: np.ndarray) -> list:
    """Tạo nhiều phiên bản ảnh binary từ ảnh grayscale.

    Args:
        gray: Ảnh grayscale (numpy array).

    Returns:
        List of (name, binary_image) tuples.
    """
    binaries = []

    # 1. Gray + Blur + Simple Threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    binaries.append(('simple_thresh', th1))

    _, th2 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    binaries.append(('simple_thresh_inv', th2))

    # 2. Adaptive Threshold
    th3 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    binaries.append(('adaptive', th3))

    th4 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    binaries.append(('adaptive_inv', th4))

    # 3. OTSU
    _, th5 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaries.append(('otsu', th5))

    _, th6 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(('otsu_inv', th6))

    return binaries
