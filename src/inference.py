"""
ONNX inference wrapper for DocLayout-YOLO using cv2.dnn.

Model: doclayout_yolo_docstructbench_imgsz1024.onnx  (post-processing stripped)
Input:  [1, 3, 1024, 1024]   float32, normalised [0, 1], RGB, NCHW
Output: [1, 21504, 14]        float32
        cols 0-3  : x1, y1, x2, y2  (letterbox coords)
        cols 4-13 : class scores (sigmoid not yet applied)
Post-processing is done here in Python (top-K by max class score, threshold, NMS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image

# DocLayout-YOLO DocStructBench class names (index == class_id)
CLASS_NAMES = [
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
]

# Distinct colours per class (BGR for cv2, RGB for PIL)
_PALETTE_RGB = [
    (220, 50,  50),   # title          – red
    ( 50, 120, 220),  # plain text     – blue
    (160, 160, 160),  # abandon        – grey
    ( 50, 200,  50),  # figure         – green
    (100, 220, 100),  # figure_caption – light green
    (220, 150,  50),  # table          – orange
    (240, 190,  80),  # table_caption  – yellow-orange
    (240, 220, 100),  # table_footnote – yellow
    (180,  50, 220),  # isolate_formula– purple
    (220, 130, 220),  # formula_caption– pink
]

INPUT_SIZE = 1024


@dataclass
class Detection:
    label: str
    class_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def color_rgb(self):
        return _PALETTE_RGB[self.class_id % len(_PALETTE_RGB)]


def load_model(onnx_path: str) -> cv2.dnn.Net:
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def _letterbox(image_rgb: np.ndarray, target: int = INPUT_SIZE):
    """Resize keeping aspect ratio, pad to square with grey (114)."""
    h, w = image_rgb.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_top = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def detect(
    net: cv2.dnn.Net,
    pil_image: Image.Image,
    conf_thresh: float = 0.51,
    top_k: int = 300,
) -> List[Detection]:
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]

    letterboxed, scale, pad_left, pad_top = _letterbox(img_rgb, INPUT_SIZE)

    # HWC uint8 → NCHW float32 [0,1]
    blob = cv2.dnn.blobFromImage(
        letterboxed,
        scalefactor=1.0 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        mean=(0, 0, 0),
        swapRB=False,   # already RGB
        crop=False,
    )

    net.setInput(blob)
    output = net.forward()          # shape: (1, 21504, 14)
    preds = output[0]               # shape: (21504, 14)

    boxes_lb = preds[:, :4]         # letterbox coords
    class_scores = preds[:, 4:]     # raw logits, shape (21504, 10)

    # Sigmoid + max class score per anchor
    class_scores = 1.0 / (1.0 + np.exp(-class_scores.astype(np.float32)))
    max_scores = class_scores.max(axis=1)   # (21504,)
    class_ids = class_scores.argmax(axis=1) # (21504,)

    # Keep top-K by score, then threshold
    if len(max_scores) > top_k:
        top_idx = np.argpartition(max_scores, -top_k)[-top_k:]
    else:
        top_idx = np.arange(len(max_scores))

    detections: List[Detection] = []
    for i in top_idx:
        conf = float(max_scores[i])
        if conf < conf_thresh:
            continue
        class_id = int(class_ids[i])
        x1, y1, x2, y2 = boxes_lb[i]

        # Undo letterbox padding and scale
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        # Clamp to image bounds
        x1 = max(0, min(int(x1), orig_w - 1))
        y1 = max(0, min(int(y1), orig_h - 1))
        x2 = max(0, min(int(x2), orig_w - 1))
        y2 = max(0, min(int(y2), orig_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        detections.append(Detection(
            label=label,
            class_id=class_id,
            conf=conf,
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    # Sort by confidence descending
    detections.sort(key=lambda d: d.conf, reverse=True)

    # Simple NMS: suppress boxes with IoU > threshold against higher-conf boxes
    keep: List[Detection] = []
    suppressed = [False] * len(detections)
    for i, di in enumerate(detections):
        if suppressed[i]:
            continue
        keep.append(di)
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            dj = detections[j]
            if di.class_id != dj.class_id:
                continue
            # Compute IoU
            ix1 = max(di.x1, dj.x1); iy1 = max(di.y1, dj.y1)
            ix2 = min(di.x2, dj.x2); iy2 = min(di.y2, dj.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_i = (di.x2 - di.x1) * (di.y2 - di.y1)
            area_j = (dj.x2 - dj.x1) * (dj.y2 - dj.y1)
            iou = inter / (area_i + area_j - inter)
            if iou > 0.45:
                suppressed[j] = True
    # Group raw detections into semantic blocks (figure+caption, table+caption,
    # titled_block pairs, plain-text merging, etc.)
    import block_grouping
    orig_w, orig_h = pil_image.size
    return block_grouping.group_detections(keep, img_w=orig_w, img_h=orig_h)
