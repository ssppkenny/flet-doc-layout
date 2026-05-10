"""
Word detection using doctr fast_base via cv2.dnn.

Model: fast_base.onnx  (~63 MB)
  - Exported from doctr with exportable=True (post-processing stripped)
  - Input:  [1, 3, 1024, 1024]  float32, normalised with fast_base mean/std, NCHW
  - Output: [1, 1, 1024, 1024]  float32, pre-sigmoid logits (probability map)

Post-processing is reimplemented here in pure NumPy/cv2 (no pyclipper, no shapely),
matching doctr's FASTPostProcessor with assume_straight_pages=True.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# fast_base normalisation constants (from doctr default_cfgs)
_MEAN = np.array([0.798, 0.785, 0.772], dtype=np.float32)
_STD  = np.array([0.264, 0.2749, 0.287], dtype=np.float32)

INPUT_SIZE   = 1024
BIN_THRESH   = 0.1   # probability threshold to binarise the map
BOX_THRESH   = 0.1   # minimum mean probability inside a box to keep it
UNCLIP_RATIO = 1.0   # fast_base default (expand each box by this factor)


def load_model(onnx_path: str) -> cv2.dnn.Net:
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def _letterbox(img_rgb: np.ndarray, target: int = INPUT_SIZE):
    """Resize keeping aspect ratio, pad to square — same as doctr's preserve_aspect_ratio=True."""
    h, w = img_rgb.shape[:2]
    scale = target / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    pad_top  = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def _unclip_rect(x: int, y: int, w: int, h: int, ratio: float, img_w: int, img_h: int):
    """Expand a bounding rect by unclip_ratio using the area/perimeter formula.

    Matches doctr's polygon_to_box logic for straight pages:
        distance = area * ratio / perimeter
    then expands each side by `distance` pixels.
    """
    area = w * h
    perimeter = 2 * (w + h)
    if perimeter == 0:
        return x, y, w, h
    distance = int(area * ratio / perimeter)
    x1 = max(0, x - distance)
    y1 = max(0, y - distance)
    x2 = min(img_w, x + w + distance)
    y2 = min(img_h, y + h + distance)
    return x1, y1, x2 - x1, y2 - y1


def detect_words(
    net: cv2.dnn.Net,
    pil_image: Image.Image,
) -> List[Tuple[int, int, int, int]]:
    """Detect word bounding boxes in a PIL image.

    Returns a list of (x1, y1, x2, y2) tuples in original image pixel coordinates.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]

    # --- pre-process ---
    letterboxed, scale, pad_left, pad_top = _letterbox(img_rgb, INPUT_SIZE)

    # uint8 → float32, normalise
    inp = letterboxed.astype(np.float32) / 255.0
    inp = (inp - _MEAN) / _STD
    inp = np.transpose(inp, (2, 0, 1))          # HWC → CHW
    blob = inp[np.newaxis, ...]                  # → NCHW (1,3,1024,1024)

    # --- forward pass ---
    net.setInput(blob)
    logits = net.forward()                       # (1, 1, 1024, 1024)
    prob_map = 1.0 / (1.0 + np.exp(-logits[0, 0].clip(-88, 88)))  # sigmoid → (1024, 1024)

    # --- post-process (FASTPostProcessor, straight pages) ---
    bitmap = (prob_map >= BIN_THRESH).astype(np.uint8)

    contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        # Skip tiny contours
        pts = contour[:, 0]
        if np.any(pts.max(axis=0) - pts.min(axis=0) < 2):
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Score = mean probability inside the bounding rect
        score = float(prob_map[y:y + h, x:x + w].mean())
        if score < BOX_THRESH:
            continue

        # Unclip (expand) the rect
        x, y, w, h = _unclip_rect(x, y, w, h, UNCLIP_RATIO, INPUT_SIZE, INPUT_SIZE)
        if w <= 0 or h <= 0:
            continue

        # Undo letterbox: remove padding offset, undo scale
        x1 = int((x - pad_left) / scale)
        y1 = int((y - pad_top)  / scale)
        x2 = int((x + w - pad_left) / scale)
        y2 = int((y + h - pad_top)  / scale)

        # Clamp to original image bounds
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append((x1, y1, x2, y2))

    return boxes
