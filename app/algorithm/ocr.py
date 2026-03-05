"""OCR: đọc nhãn nodes và sửa lỗi OCR phổ biến."""

import cv2
import numpy as np
import pytesseract


# Mapping lỗi OCR phổ biến
_OCR_CORRECTIONS = {
    'BY': 'B', 'S': 'B', '5': 'B', 'SS': 'B',
    'R': 'C', 'RA': 'A',
    'A': 'A', 'C': 'C', 'E': 'E', 'B': 'B', 'F': 'F',
    '0': 'C', 'Q': 'C',
    '8': 'B', '3': 'E', '#': 'F', '6': 'G',
    'G': 'G', 'G ': 'G', 'C ': 'C', 'E ': 'E', 'F ': 'F',
    'BY ': 'B', 'BY\n': 'B',
    'rA': 'A', 'rA ': 'A', 'rA\n': 'A',
    'CA': 'C', 'CA ': 'C', 'CA\n': 'C',
    'OO': '∞', '00': '∞',
    '1': 'I', '4': 'B',
    'M': 'C', 'W': 'C', 'U': 'C',
}

_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789∞"
_PSM_MODES = ["--psm 10", "--psm 6", "--psm 7"]


def correct_ocr_errors(text: str) -> str:
    """Sửa các lỗi OCR phổ biến."""
    if not text:
        return text

    text = text.upper().strip()

    if text in _OCR_CORRECTIONS:
        return _OCR_CORRECTIONS[text]

    # Nếu 2 ký tự: giữ ký tự hợp lệ
    if len(text) == 2:
        valid_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if text[0] in valid_chars and text[1] not in valid_chars + '0123456789':
            return text[0]
        if text[1] in valid_chars and text[0] not in valid_chars + '0123456789':
            return text[1]

    # Lấy ký tự đầu nếu là chữ cái hợp lệ
    if len(text) > 1 and text[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        cand = text[0]
        return _OCR_CORRECTIONS.get(cand, cand)

    return text


def ocr_node_labels(img: np.ndarray, img_vis: np.ndarray, nodes: list) -> None:
    """OCR đọc nhãn trong nodes — cập nhật trực tiếp vào list nodes.

    Args:
        img: Ảnh gốc BGR.
        img_vis: Ảnh visualization (sẽ vẽ labels lên đây).
        nodes: List of node dicts (sẽ thêm key 'label').
    """
    for i, node in enumerate(nodes):
        cx, cy = node['center']
        r = node['radius']

        # Crop ROI chặt vào center
        padding = int(r * 0.3)
        y1 = max(0, cy - r + padding)
        y2 = min(img.shape[0], cy + r - padding)
        x1 = max(0, cx - r + padding)
        x2 = min(img.shape[1], cx + r - padding)

        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            node['label'] = f'N{i}'
            continue

        label = _ocr_single_node(roi)
        node['label'] = label if label else f'N{i}'

        # Vẽ nhãn lên img_vis
        _draw_label(img_vis, node)


def _ocr_single_node(roi: np.ndarray) -> str | None:
    """OCR một ROI node, thử nhiều góc xoay."""
    angles = [0, -5, 5, -10, 10]
    best_confidence = 0
    label = None

    for angle in angles:
        if angle != 0:
            center = (roi.shape[1] // 2, roi.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                roi, M, (roi.shape[1], roi.shape[0]),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
            )
        else:
            rotated = roi

        roi_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        # Tạo variants
        _, th1 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th2 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        scale = 4
        variants = [
            cv2.resize(th1, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
            cv2.resize(th2, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
        ]

        for v in variants:
            for psm in _PSM_MODES:
                config = f"{psm} --oem 3 -c tessedit_char_whitelist={_WHITELIST}"
                try:
                    data = pytesseract.image_to_data(
                        v, config=config, output_type=pytesseract.Output.DICT
                    )
                    texts = [t.strip() for t in data['text'] if t.strip()]
                    confidences = [c for c, t in zip(data['conf'], data['text']) if t.strip()]

                    if texts:
                        max_idx = confidences.index(max(confidences))
                        text = texts[max_idx].replace(' ', '').replace('\n', '').replace('\t', '')
                        conf = confidences[max_idx]

                        text = correct_ocr_errors(text)

                        if text and len(text) <= 3 and conf > best_confidence and conf > 30:
                            best_confidence = conf
                            label = text
                except Exception:
                    pass

        if best_confidence > 80:
            break

    return label


def _draw_label(img_vis: np.ndarray, node: dict) -> None:
    """Vẽ nhãn node lên ảnh visualization."""
    cx, cy = node['center']
    r = node['radius']
    label_text = node['label']

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]

    tx = cx - text_size[0] // 2
    ty = cy - r - 15

    # Nền đen cho nhãn
    cv2.rectangle(
        img_vis,
        (tx - 5, ty - text_size[1] - 5),
        (tx + text_size[0] + 5, ty + 5),
        (0, 0, 0), -1
    )
    cv2.putText(img_vis, label_text, (tx, ty), font, font_scale, (255, 255, 255), thickness)
    cv2.circle(img_vis, (cx, cy), r, (0, 255, 0), 2)
