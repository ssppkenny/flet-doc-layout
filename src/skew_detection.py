"""
Skew Detection and Correction Module

Implementation of MCCSD (Modified Cross-Correlation Skew Detection) algorithm
based on the paper: "A Robust Skew Detection Algorithm for Grayscale Document Image"
by Ming Chen and Xiaoqing Ding, Tsinghua University.

Ported from the segmentation project; adapted to use Detection namedtuples
(x1, y1, x2, y2, label, score) instead of shapely geometries.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _vcc(image: np.ndarray, d: int, s_range: int) -> np.ndarray:
    """Vertical cross-correlation."""
    height, width = image.shape
    R = np.zeros(2 * s_range + 1, dtype=np.float64)
    for s_idx, s in enumerate(range(-s_range, s_range + 1)):
        corr = 0.0
        count = 0
        for x0 in range(width - d):
            y_start = 0 if s >= 0 else -s
            y_end = height - s if s >= 0 else height
            if y_end > y_start:
                l1 = image[y_start:y_end, x0].astype(np.float64)
                l2 = image[y_start + s:y_end + s, x0 + d].astype(np.float64)
                m1, s1 = l1.mean(), l1.std()
                m2, s2 = l2.mean(), l2.std()
                if s1 > 1e-10 and s2 > 1e-10:
                    corr += np.sum((l1 - m1) / s1 * ((l2 - m2) / s2))
                    count += 1
        R[s_idx] = corr / max(count, 1)
    return R


def _hcc(image: np.ndarray, d: int, s_range: int) -> np.ndarray:
    """Horizontal cross-correlation."""
    height, width = image.shape
    R = np.zeros(2 * s_range + 1, dtype=np.float64)
    for s_idx, s in enumerate(range(-s_range, s_range + 1)):
        corr = 0.0
        count = 0
        for y0 in range(height - d):
            x_start = 0 if s >= 0 else -s
            x_end = width - s if s >= 0 else width
            if x_end > x_start:
                l1 = image[y0, x_start:x_end].astype(np.float64)
                l2 = image[y0 + d, x_start + s:x_end + s].astype(np.float64)
                m1, s1 = l1.mean(), l1.std()
                m2, s2 = l2.mean(), l2.std()
                if s1 > 1e-10 and s2 > 1e-10:
                    corr += np.sum((l1 - m1) / s1 * ((l2 - m2) / s2))
                    count += 1
        R[s_idx] = corr / max(count, 1)
    return R


def _total_variation(R: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(R))))


def _find_peaks(R: np.ndarray, s_range: int) -> list:
    peaks = []
    for i in range(1, len(R) - 1):
        if R[i] > R[i - 1] and R[i] > R[i + 1]:
            peaks.append((i - s_range, R[i]))
    if not peaks:
        max_idx = int(np.argmax(R))
        peaks.append((max_idx - s_range, R[max_idx]))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks


def _detect_skew_in_region(region: np.ndarray, d: int, s_range: int,
                            d_prime: Optional[int] = None) -> Optional[float]:
    R_V = _vcc(region, d, s_range)
    R_H = _hcc(region, d, s_range)
    dV = _total_variation(R_V)
    dH = _total_variation(R_H)

    if dV < 10.0 and dH < 10.0:
        return None

    is_horizontal = dV >= dH
    R_sel = R_V if is_horizontal else R_H
    peaks = _find_peaks(R_sel, s_range)
    if not peaks:
        return None

    s_p = peaks[0][0]
    angle = 0.0 if s_p == 0 else float(np.degrees(np.arctan(s_p / d)))

    if len(peaks) > 1 and d_prime is not None and d_prime != d:
        R2 = _vcc(region, d_prime, s_range) if is_horizontal else _hcc(region, d_prime, s_range)
        peaks2 = _find_peaks(R2, s_range)
        if peaks2:
            s_p2 = peaks2[0][0]
            angle2 = float(np.degrees(np.arctan(s_p2 / d_prime)))
            if abs(angle - angle2) < 1.0:
                angle = (angle + angle2) / 2.0

    return angle


def detect_skew_in_text_regions(image: np.ndarray,
                                 detections: list,
                                 d: int = 75,
                                 s_range: int = 25,
                                 d_prime: int = 50,
                                 region_size: int = 150,
                                 num_regions: int = 20,
                                 max_attempts: int = 200) -> float:
    """
    Detect skew using only plain-text detection boxes.

    detections: list of Detection namedtuples with .x1 .y1 .x2 .y2 .label
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    height, width = gray.shape

    text_boxes = [d for d in detections if d.label == "plain text"]
    if not text_boxes:
        return _detect_skew_full(gray, d, s_range, d_prime, region_size, num_regions, max_attempts)

    detected_angles: List[float] = []
    attempts = 0

    while len(detected_angles) < num_regions and attempts < max_attempts:
        attempts += 1
        det = random.choice(text_boxes)
        bw = det.x2 - det.x1
        bh = det.y2 - det.y1

        if bw < region_size or bh < region_size:
            if bw > 50 and bh > 50:
                region = gray[det.y1:det.y2, det.x1:det.x2]
                angle = _detect_skew_in_region(region, d, s_range, d_prime)
                if angle is not None:
                    detected_angles.append(angle)
            continue

        x = random.randint(det.x1, min(det.x2 - region_size, det.x2 - 1))
        y = random.randint(det.y1, min(det.y2 - region_size, det.y2 - 1))
        x = max(0, min(x, width - region_size))
        y = max(0, min(y, height - region_size))
        region = gray[y:y + region_size, x:x + region_size]
        angle = _detect_skew_in_region(region, d, s_range, d_prime)
        if angle is not None:
            detected_angles.append(angle)

    if not detected_angles:
        return 0.0

    if len(detected_angles) < 5:
        return 0.0

    arr = np.array(detected_angles)
    mean, std = arr.mean(), arr.std()
    clipped = arr[np.abs(arr - mean) <= 2 * std] if std > 0 else arr
    if len(clipped) == 0:
        clipped = arr

    final_angle = float(np.median(clipped))
    angle_std = float(np.std(clipped))

    if angle_std > 1.0:
        return 0.0

    return final_angle


def _detect_skew_full(gray: np.ndarray, d: int, s_range: int, d_prime: int,
                      region_size: int, num_regions: int, max_attempts: int) -> float:
    height, width = gray.shape
    if height < region_size or width < region_size:
        angle = _detect_skew_in_region(gray, d, s_range, d_prime)
        return angle if angle is not None else 0.0

    detected_angles: List[float] = []
    attempts = 0
    while len(detected_angles) < num_regions and attempts < max_attempts:
        attempts += 1
        x = random.randint(0, width - region_size)
        y = random.randint(0, height - region_size)
        region = gray[y:y + region_size, x:x + region_size]
        angle = _detect_skew_in_region(region, d, s_range, d_prime)
        if angle is not None:
            detected_angles.append(angle)

    if not detected_angles:
        angle = _detect_skew_in_region(gray, d, s_range, d_prime)
        return angle if angle is not None else 0.0

    return float(np.median(detected_angles))


def rotate_image(image: np.ndarray, angle: float,
                 background_color: tuple = (255, 255, 255)) -> np.ndarray:
    """Rotate image to correct skew. angle in degrees (positive = CCW)."""
    if abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    bg = background_color if len(image.shape) == 3 else background_color[0]
    return cv2.warpAffine(image, M, (new_w, new_h), borderValue=bg)
