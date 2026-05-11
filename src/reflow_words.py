"""
reflow_words.py — word-level reflow engine (Android-compatible subset).

Adapted from ocr_reflow/reflow_words.py with the following changes:
  - pytesseract / pyphen removed (lang=None only for now)
  - shapely removed; divide_conquer_4d replaced with O(n²) enclosed-rect filter
  - Only dependencies: cv2, numpy (both available on Android)

Public API:
    create_page_word_reflow(lines, original_image, zoom_factor, new_page_width, ...)
    words_to_wordlines(lines)
"""

from __future__ import annotations

import cv2
import numpy as np
import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Word:
    """A word bounding box in the original image coordinate space."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    bl: int = 0    # descender offset: pixels from ymax UP to the text baseline
    above: int = 0 # above-baseline height in original pixels

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


@dataclass
class _PlacedWord:
    """Internal: a word ready to be rendered on an output line."""
    word: Word
    space_before: int
    synth_image: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# clamp_word_boxes — vertical clamping to suppress inter-line bleed
# ---------------------------------------------------------------------------

def clamp_word_boxes(
    boxes: List[Tuple[int, int, int, int]],
    slack: float = 1.15,
) -> List[Tuple[int, int, int, int]]:
    """
    Clamp each word box's vertical extent to suppress bleed from adjacent lines.

    Strategy:
      1. Group boxes into lines by y-center proximity (gap < median_h * 0.6).
      2. For each line compute the median box height.
      3. Clamp each box to [cy - median_h/2 * slack, cy + median_h/2 * slack]
         where cy is the box's y-center.

    slack=1.15 allows 15% extra for ascenders/descenders/diacritics while
    still cutting boxes that span two lines (which are typically 2× too tall).

    Boxes that are already within the clamped range are returned unchanged.
    """
    if not boxes:
        return boxes

    heights = [y2 - y1 for x1, y1, x2, y2 in boxes]
    median_h = float(np.median(heights))

    # Group into lines: sort by y-center, start new line when gap > median_h * 0.6
    sorted_boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    lines: List[List[Tuple[int, int, int, int]]] = []
    current_line: List[Tuple[int, int, int, int]] = []
    prev_cy = None
    for box in sorted_boxes:
        cy = (box[1] + box[3]) / 2
        if prev_cy is None or cy - prev_cy < median_h * 0.6:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
        prev_cy = cy
    if current_line:
        lines.append(current_line)

    # Clamp each box using its line's median height,
    # but never use a line median larger than the global median
    # (handles isolated oversized boxes that form their own "line")
    result: List[Tuple[int, int, int, int]] = []
    for line in lines:
        line_heights = [y2 - y1 for x1, y1, x2, y2 in line]
        line_median_h = min(float(np.median(line_heights)), median_h)
        half = line_median_h / 2 * slack
        for x1, y1, x2, y2 in line:
            cy = (y1 + y2) / 2
            new_y1 = int(cy - half)
            new_y2 = int(cy + half)
            # Only clamp if the box is actually oversized — never expand
            clamped_y1 = max(y1, new_y1)
            clamped_y2 = min(y2, new_y2)
            # Safety: don't produce empty boxes
            if clamped_y2 <= clamped_y1:
                result.append((x1, y1, x2, y2))
            else:
                result.append((x1, clamped_y1, x2, clamped_y2))

    return result


# ---------------------------------------------------------------------------
# find_rects — letter-component detection (ported from segmentation/main.py)
# ---------------------------------------------------------------------------

def _remove_enclosed(rects: list) -> list:
    """
    Remove rectangles that are fully enclosed by another rectangle.
    O(n²) replacement for divide_conquer_4d.
    rects: list of (xmin, ymin, xmax, ymax) in full-image coords.
    """
    n = len(rects)
    enclosed = [False] * n
    for i in range(n):
        ax1, ay1, ax2, ay2 = rects[i]
        for j in range(n):
            if i == j or enclosed[j]:
                continue
            bx1, by1, bx2, by2 = rects[j]
            # j encloses i?
            if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
                enclosed[i] = True
                break
    return [r for i, r in enumerate(rects) if not enclosed[i]]


def find_rects(img: np.ndarray, line_words: list) -> list:
    """
    For each word box in line_words, find connected-component letter boxes.
    line_words: list of (xmin, ymin, xmax, ymax) or (xmin, ymin, xmax, ymax, conf).
    Returns list of (xmin, ymin, xmax, ymax) in full-image coords (with 2px padding).
    """
    rects = []
    for word in line_words:
        if len(word) == 5:
            xmin, ymin, xmax, ymax, _ = word
        else:
            xmin, ymin, xmax, ymax = word
        word_height = ymax - ymin
        word_width = xmax - xmin

        r = img[ymin:ymax, xmin:xmax, :].copy()
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r, 8, cv2.CV_32S)

        # Step 1: identify main letter components (≥30% of word height)
        main_components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if w < 2 or h < 2 or area < 4:
                continue
            if h >= word_height * 0.3:
                main_components.append(i)

        # Step 2: collect valid components (main + nearby small ones)
        valid_components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if w < 2 or h < 2 or area < 4:
                continue
            if i in main_components:
                valid_components.append((x, y, w, h))
                continue
            cy = y + h / 2
            is_near_main = False
            for main_idx in main_components:
                main_x = stats[main_idx, cv2.CC_STAT_LEFT]
                main_y = stats[main_idx, cv2.CC_STAT_TOP]
                main_w = stats[main_idx, cv2.CC_STAT_WIDTH]
                main_h = stats[main_idx, cv2.CC_STAT_HEIGHT]
                main_bottom = main_y + main_h
                max_distance_above = word_height * 0.4
                if y < main_bottom + max_distance_above and y + h > main_y - max_distance_above:
                    horizontal_gap = max(0, max(x - (main_x + main_w), main_x - (x + w)))
                    if horizontal_gap < word_width * 0.3:
                        is_near_main = True
                        break
            if is_near_main:
                valid_components.append((x, y, w, h))

        if len(valid_components) == 0 and num_labels > 1:
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                if w >= 2 and h >= 2:
                    valid_components.append((x, y, w, h))

        # Step 3: merge diacritics with base letters
        if valid_components:
            component_heights = [h for x, y, w, h in valid_components]
            median_height = float(np.median(component_heights))
        else:
            median_height = word_height * 0.5

        dots_to_merge = []
        main_letters_to_merge = []
        for comp_idx, (x, y, w, h) in enumerate(valid_components):
            is_diacritic = (
                h < median_height * 0.4 and
                w < median_height * 0.8 and
                w * h < (median_height ** 2) * 0.3 and
                h < word_height * 0.25 and
                w < word_width * 0.5
            )
            if is_diacritic:
                dots_to_merge.append((comp_idx, x, y, w, h))
            else:
                main_letters_to_merge.append((comp_idx, x, y, w, h))

        # Merge horizontally adjacent main letter components
        if len(main_letters_to_merge) > 1:
            merged_main_indices = set()
            merged_main_components = []
            for i, (idx_i, x_i, y_i, w_i, h_i) in enumerate(main_letters_to_merge):
                if idx_i in merged_main_indices:
                    continue
                merge_group = [(idx_i, x_i, y_i, w_i, h_i)]
                for j, (idx_j, x_j, y_j, w_j, h_j) in enumerate(main_letters_to_merge):
                    if i == j or idx_j in merged_main_indices:
                        continue
                    horizontal_gap = max(0, max(x_i - (x_j + w_j), x_j - (x_i + w_i)))
                    vertical_overlap = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)
                    min_height = min(h_i, h_j)
                    height_ratio = max(h_i, h_j) / max(min_height, 1)
                    should_merge = (
                        horizontal_gap < median_height * 0.3 and
                        vertical_overlap > min_height * 0.5 and
                        height_ratio < 1.5
                    )
                    if should_merge:
                        merge_group.append((idx_j, x_j, y_j, w_j, h_j))
                        merged_main_indices.add(idx_j)
                if len(merge_group) > 1:
                    all_x = [x for _, x, y, w, h in merge_group]
                    all_y = [y for _, x, y, w, h in merge_group]
                    all_right = [x + w for _, x, y, w, h in merge_group]
                    all_bottom = [y + h for _, x, y, w, h in merge_group]
                    merged_main_components.append((
                        idx_i,
                        min(all_x), min(all_y),
                        max(all_right) - min(all_x),
                        max(all_bottom) - min(all_y),
                    ))
                    merged_main_indices.add(idx_i)
                else:
                    merged_main_components.append((idx_i, x_i, y_i, w_i, h_i))
                    merged_main_indices.add(idx_i)
            main_letters_to_merge = merged_main_components

        # Merge diacritics with base letters
        merged_indices = set()
        merged_components = []
        for dot_idx, dx, dy, dw, dh in dots_to_merge:
            dot_cx = dx + dw / 2
            dot_bottom = dy + dh
            dot_left = dx
            dot_right = dx + dw
            matching_components = []
            for main_idx, mx, my, mw, mh in main_letters_to_merge:
                if main_idx in merged_indices:
                    continue
                main_cx = mx + mw / 2
                main_top = my
                main_left = mx
                main_right = mx + mw
                main_bottom = my + mh
                vertical_gap = main_top - dot_bottom
                if vertical_gap < median_height:
                    horizontal_overlap = min(dot_right, main_right) - max(dot_left, main_left)
                    is_horizontally_aligned = horizontal_overlap > 0
                    if is_horizontally_aligned:
                        horizontal_center_distance = abs(dot_cx - main_cx)
                        is_center_aligned = horizontal_center_distance < mw * 0.5
                    else:
                        is_center_aligned = False
                    is_vertically_ok = dot_bottom <= main_bottom
                    if is_horizontally_aligned and is_center_aligned and is_vertically_ok:
                        matching_components.append((main_idx, mx, my, mw, mh))
            if matching_components:
                all_comps = [(dx, dy, dw, dh)] + [(mx, my, mw, mh) for _, mx, my, mw, mh in matching_components]
                merged_x = min(x for x, y, w, h in all_comps)
                merged_y = min(y for x, y, w, h in all_comps)
                merged_right = max(x + w for x, y, w, h in all_comps)
                merged_bottom = max(y + h for x, y, w, h in all_comps)
                merged_components.append((merged_x, merged_y, merged_right - merged_x, merged_bottom - merged_y))
                merged_indices.add(dot_idx)
                for main_idx, _, _, _, _ in matching_components:
                    merged_indices.add(main_idx)

        for main_idx, mx, my, mw, mh in main_letters_to_merge:
            if main_idx not in merged_indices:
                merged_components.append((mx, my, mw, mh))
                merged_indices.add(main_idx)

        skip_indices = set(merged_indices)
        for comp_idx, (x, y, w, h) in enumerate(valid_components):
            if comp_idx not in skip_indices:
                merged_components.append((x, y, w, h))

        final_components = merged_components

        # Filter small fragments
        if len(final_components) > 2:
            component_areas = [w * h for x, y, w, h in final_components]
            median_area = float(np.median(component_areas))
            filtered = []
            for x, y, w, h in final_components:
                if w * h >= median_area * 0.25 or h >= median_height * 0.5:
                    filtered.append((x, y, w, h))
            if len(filtered) >= 2:
                final_components = filtered

        # Add with padding in full-image coords
        padding = 2
        for x, y, w, h in final_components:
            padded_x = max(0, x - padding)
            padded_y = max(0, y - padding)
            padded_w = min(w + 2 * padding, word_width - padded_x)
            padded_h = min(h + 2 * padding, word_height - padded_y)
            rects.append((
                padded_x + xmin, padded_y + ymin,
                padded_x + padded_w + xmin, padded_y + padded_h + ymin,
            ))

    rects = _remove_enclosed(rects)
    return rects


# ---------------------------------------------------------------------------
# Word splitting
# ---------------------------------------------------------------------------

def _synthesize_hyphen(
    ref_word: Word,
    zoom_factor: float,
    background_color: tuple,
) -> np.ndarray:
    """Return a small BGR image of a hyphen glyph sized to match the rendered word."""
    img_h = max(4, int(ref_word.height * zoom_factor))
    stroke_h = max(2, round(img_h * 0.07))
    stroke_w = max(4, round(img_h * 0.35))
    pad = 2
    img_w = stroke_w + 2 * pad

    img = np.ones((img_h, img_w, 3), dtype=np.uint8)
    img[:] = background_color

    y_mid = img_h // 2
    y0 = max(0, y_mid - stroke_h // 2)
    y1 = min(img_h, y0 + stroke_h)
    x0 = pad
    x1 = pad + stroke_w
    img[y0:y1, x0:x1] = (30, 30, 30)
    return img


def _find_split_x(
    rects_sorted: list,
    word_xmin: int,
    target_x_in_word: int,
    PADDING: int = 2,
) -> Optional[int]:
    """
    Return the absolute x of the best inter-letter cut that fits within
    target_x_in_word (relative to word_xmin), or None.
    """
    cut_x = None
    for rx1, _ry1, rx2, _ry2 in rects_sorted:
        ink_right = rx2 - PADDING
        rel_ink_right = ink_right - word_xmin
        if rel_ink_right <= target_x_in_word:
            cut_x = ink_right
    return cut_x


def _split_word(
    word: Word,
    remaining_px: int,
    zoom_factor: float,
    original_image: np.ndarray,
) -> Tuple[Optional[Word], Word]:
    """
    Split `word` into (left_half, right_half) such that left_half fits within
    `remaining_px` scaled pixels.  Returns (None, word) if no valid cut found.
    """
    if remaining_px <= 0:
        return None, word

    target_x_in_word = int(remaining_px / zoom_factor)

    word_box = [(word.xmin, word.ymin, word.xmax, word.ymax)]
    try:
        rects = find_rects(original_image, word_box)
    except Exception as e:
        logger.warning(f"find_rects failed during word split: {e}")
        return None, word

    if len(rects) < 2:
        return None, word

    rects_sorted = sorted(rects, key=lambda r: r[0])
    cut_x = _find_split_x(rects_sorted, word.xmin, target_x_in_word)

    if cut_x is None or cut_x <= word.xmin:
        return None, word

    cut_x = min(cut_x, word.xmax - 1)
    left  = Word(word.xmin, word.ymin, cut_x,     word.ymax, bl=word.bl, above=word.above)
    right = Word(cut_x,     word.ymin, word.xmax, word.ymax, bl=word.bl, above=word.above)
    return left, right


# ---------------------------------------------------------------------------
# Paragraph detection
# ---------------------------------------------------------------------------

def _detect_paragraphs(lines: List[List[Word]]) -> List[int]:
    if not lines:
        return [0]

    first_xmins = [min(w.xmin for w in line) for line in lines if line]
    if not first_xmins:
        return [0]

    avg = sum(first_xmins) / len(first_xmins)

    if len(first_xmins) > 1:
        variance = sum((x - avg) ** 2 for x in first_xmins) / len(first_xmins)
        std = math.sqrt(variance)
        threshold = avg + 1.5 * std
    else:
        threshold = float('inf')

    para_starts_1 = {0}
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        if min(w.xmin for w in line) > threshold:
            para_starts_1.add(i)

    para_starts_2 = {0}
    prev_xmin = first_xmins[0]
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        xmin = min(w.xmin for w in line)
        if xmin > prev_xmin + 20:
            para_starts_2.add(i)
        prev_xmin = xmin

    result = para_starts_1 if len(para_starts_1) >= len(para_starts_2) else para_starts_2
    return sorted(result)


# ---------------------------------------------------------------------------
# Hyphen detection
# ---------------------------------------------------------------------------

def _ends_with_hyphen(word: Word, image: np.ndarray) -> bool:
    h = word.ymax - word.ymin
    w = word.xmax - word.xmin
    if h <= 0 or w <= 0:
        return False
    crop = image[word.ymin:word.ymax, word.xmin:word.xmax]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    x0 = max(0, w - w // 4)
    y0 = h // 3
    y1 = h - h // 3
    if y1 <= y0 or x0 >= w:
        return False
    zone = bw[y0:y1, x0:]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(zone, 8, cv2.CV_32S)
    if num_labels < 2:
        return False
    zone_h = y1 - y0
    best_zone_x = -1
    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if cw < 1 or ch < 1 or ch < 3:
            continue
        if cw < max(2, int(w * 0.06)):
            continue
        if ch > zone_h * 0.6 or ch > h * 0.30:
            continue
        if cw / ch < 2.0:
            continue
        cx = stats[i, cv2.CC_STAT_LEFT]
        if cx > best_zone_x:
            best_zone_x = cx
    return best_zone_x >= 0


def _strip_trailing_hyphen(word: Word, image: np.ndarray) -> Word:
    h = word.ymax - word.ymin
    w = word.xmax - word.xmin
    if h <= 0 or w <= 0:
        return word
    crop = image[word.ymin:word.ymax, word.xmin:word.xmax]
    if crop.size == 0:
        return word
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    x0 = max(0, w - w // 4)
    y0 = h // 3
    y1 = h - h // 3
    if y1 <= y0 or x0 >= w:
        return word
    zone = bw[y0:y1, x0:]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(zone, 8, cv2.CV_32S)
    if num_labels < 2:
        return word
    zone_h = y1 - y0
    best_zone_x = -1
    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if cw < 1 or ch < 1 or ch < 3:
            continue
        if cw < max(2, int(w * 0.06)):
            continue
        if ch > zone_h * 0.6 or ch > h * 0.30:
            continue
        if cw / ch < 2.0:
            continue
        cx = stats[i, cv2.CC_STAT_LEFT]
        if cx > best_zone_x:
            best_zone_x = cx
    if best_zone_x < 0:
        return word
    hyphen_left_in_crop = x0 + best_zone_x
    new_xmax = word.xmin + hyphen_left_in_crop - 2
    new_xmax = max(word.xmin + 1, new_xmax)
    return Word(word.xmin, word.ymin, new_xmax, word.ymax, bl=word.bl, above=word.above)


# ---------------------------------------------------------------------------
# Inter-word gap
# ---------------------------------------------------------------------------

def _inter_word_gap(
    prev_word: Word,
    curr_word: Word,
    zoom_factor: float,
    avg_word_space: int,
) -> int:
    gap = curr_word.xmin - prev_word.xmax
    if gap > 0:
        return int(gap * zoom_factor)
    if gap >= -10:
        return max(1, int((gap + 10) * zoom_factor // 2))
    return avg_word_space


# ---------------------------------------------------------------------------
# Main reflow function
# ---------------------------------------------------------------------------

def create_page_word_reflow(
    lines: List[List[Word]],
    original_image: np.ndarray,
    zoom_factor: float,
    new_page_width: int,
    left_margin: int = 50,
    right_margin: int = 50,
    top_margin: int = 50,
    bottom_margin: int = 50,
    preserve_line_breaks: bool = False,
    background_color: tuple = (220, 220, 220),
    is_title: bool = False,
) -> np.ndarray:
    """
    Reflow a list of word lines onto a new page of width `new_page_width`.

    Words that overflow a line are split at inter-letter gaps when possible,
    otherwise wrapped whole to the next line.

    Args:
        lines:            List of lines; each line is a list of Word objects.
        original_image:   Source image (BGR numpy array).
        zoom_factor:      Scale factor applied to all crops.
        new_page_width:   Width of the output image in pixels.
        left_margin, right_margin, top_margin, bottom_margin: margins in px.
        preserve_line_breaks: If True, honour original line boundaries exactly.
        background_color: BGR tuple for the page background.
        is_title:         If True, suppress paragraph detection / indentation.

    Returns:
        Output page as a numpy BGR image.
    """
    empty_h = top_margin + bottom_margin + 100
    if not lines or all(not line for line in lines):
        page = np.ones((empty_h, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    available_width = new_page_width - left_margin - right_margin

    if is_title:
        para_line_starts = {0}
    else:
        para_line_starts = set(_detect_paragraphs(lines))

    all_words_flat = [w for line in lines for w in line]
    if all_words_flat:
        avg_word_w = sum(w.width for w in all_words_flat) / len(all_words_flat)
        all_heights = sorted(w.height for w in all_words_flat)
        median_word_h = all_heights[len(all_heights) // 2]
        avg_word_space = int(median_word_h * zoom_factor * 0.30)
    else:
        avg_word_w = 50
        avg_word_space = 20

    output_lines: List[dict] = []

    current_words: List[_PlacedWord] = []
    current_width = 0

    def _flush(para_start: bool):
        nonlocal current_words, current_width
        if current_words:
            output_lines.append({'words': current_words, 'para_start': para_start})
        current_words = []
        current_width = 0

    prev_line_ends_with_hyphen = False

    for line_idx, line in enumerate(lines):
        if not line:
            continue

        is_para_start_line = line_idx in para_line_starts
        sorted_words = sorted(line, key=lambda w: w.xmin)

        if preserve_line_breaks:
            _flush(is_para_start_line)
            placed = [
                _PlacedWord(
                    word=w,
                    space_before=0 if j == 0 else _inter_word_gap(sorted_words[j-1], w, zoom_factor, avg_word_space),
                )
                for j, w in enumerate(sorted_words)
            ]
            output_lines.append({'words': placed, 'para_start': is_para_start_line})
            prev_line_ends_with_hyphen = False
            continue

        this_line_ends_with_hyphen = _ends_with_hyphen(sorted_words[-1], original_image)
        if this_line_ends_with_hyphen:
            sorted_words[-1] = _strip_trailing_hyphen(sorted_words[-1], original_image)

        for word_idx, word in enumerate(sorted_words):
            scaled_w = int(word.width * zoom_factor)

            if not current_words and not output_lines:
                space = 0
            elif not current_words:
                space = 0
            elif word_idx == 0:
                space = 0 if prev_line_ends_with_hyphen else avg_word_space
            else:
                prev_word = sorted_words[word_idx - 1]
                space = _inter_word_gap(prev_word, word, zoom_factor, avg_word_space)

            indent = 0
            if is_para_start_line and word_idx == 0 and not is_title:
                indent = int(avg_word_w * zoom_factor * 0.5)

            effective_available = available_width - indent

            would_overflow = (
                current_words
                and current_width + space + scaled_w > effective_available
            )

            # Case 1: word overflows current line (line already has words)
            if would_overflow:
                remaining = effective_available - current_width - space
                left_half, right_half = _split_word(
                    word, remaining, zoom_factor, original_image
                )
                if left_half is not None:
                    # Place left half with synthesized hyphen on current line
                    hyphen_img = _synthesize_hyphen(left_half, zoom_factor, background_color)
                    current_words.append(_PlacedWord(word=left_half, space_before=space, synth_image=hyphen_img))
                    current_width += space + int(left_half.width * zoom_factor) + hyphen_img.shape[1]
                    _flush(False)
                    # Right half starts next line
                    current_words.append(_PlacedWord(word=right_half, space_before=0))
                    current_width = int(right_half.width * zoom_factor)
                else:
                    # Can't split — wrap whole word
                    _flush(is_para_start_line and word_idx == 0)
                    current_words.append(_PlacedWord(word=word, space_before=0))
                    current_width = scaled_w

            # Case 2: word is first on line but wider than available width
            elif not current_words and scaled_w > effective_available:
                left_half, right_half = _split_word(
                    word, effective_available, zoom_factor, original_image
                )
                if left_half is not None:
                    hyphen_img = _synthesize_hyphen(left_half, zoom_factor, background_color)
                    eff_space = indent if not output_lines else 0
                    current_words.append(_PlacedWord(word=left_half, space_before=eff_space, synth_image=hyphen_img))
                    current_width = eff_space + int(left_half.width * zoom_factor) + hyphen_img.shape[1]
                    _flush(False)
                    current_words.append(_PlacedWord(word=right_half, space_before=0))
                    current_width = int(right_half.width * zoom_factor)
                else:
                    eff_space = indent if not current_words else space
                    current_words.append(_PlacedWord(word=word, space_before=eff_space))
                    current_width += eff_space + scaled_w

            else:
                effective_space = space + indent if not current_words else space
                current_words.append(_PlacedWord(word=word, space_before=effective_space))
                current_width += effective_space + scaled_w

        prev_line_ends_with_hyphen = this_line_ends_with_hyphen

    _flush(False)

    if not output_lines:
        page = np.ones((empty_h, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Line height: 95th percentile of scaled cap-heights × 1.3  (≈130% leading)
    all_above = [
        int(pw.word.above * zoom_factor)
        for ol in output_lines
        for pw in ol['words']
        if pw.word.above > 0
    ]
    if all_above:
        p95_above = int(np.percentile(all_above, 95))
        line_height = int(p95_above * 1.3)
    else:
        line_height = 60

    para_spacing = int(line_height * 0.5)

    total_height = top_margin
    for ol in output_lines:
        if ol['para_start'] and total_height > top_margin:
            total_height += para_spacing
        total_height += line_height
    total_height += bottom_margin

    page = np.ones((total_height, new_page_width, 3), dtype=np.uint8)
    page[:] = background_color

    current_y = top_margin

    for ol in output_lines:
        if ol['para_start'] and current_y > top_margin:
            current_y += para_spacing

        above_vals = [int(pw.word.above * zoom_factor) for pw in ol['words']]
        max_above = max(above_vals) if above_vals else line_height
        baseline_y = current_y + max_above

        current_x = left_margin

        for pw in ol['words']:
            current_x += pw.space_before
            w = pw.word
            scaled_w = int(w.width * zoom_factor)
            scaled_h = int(w.height * zoom_factor)
            if scaled_w <= 0 or scaled_h <= 0:
                continue

            crop = original_image[w.ymin:w.ymax, w.xmin:w.xmax]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            scaled_bl = int(w.bl * zoom_factor)
            word_above = int(w.height * zoom_factor) - scaled_bl
            y_start = baseline_y - word_above
            y_end = y_start + scaled_h
            x_start = current_x
            x_end = current_x + scaled_w

            y_start_c = max(0, y_start)
            y_end_c = min(total_height, y_end)
            x_start_c = max(0, x_start)
            x_end_c = min(new_page_width - right_margin, x_end)

            if y_end_c > y_start_c and x_end_c > x_start_c:
                crop_y0 = y_start_c - y_start
                crop_y1 = crop_y0 + (y_end_c - y_start_c)
                crop_x0 = x_start_c - x_start
                crop_x1 = crop_x0 + (x_end_c - x_start_c)
                page[y_start_c:y_end_c, x_start_c:x_end_c] = resized[crop_y0:crop_y1, crop_x0:crop_x1]

            current_x += scaled_w

            # Render synthesized hyphen if present
            if pw.synth_image is not None:
                hi = pw.synth_image
                h_h, h_w = hi.shape[:2]
                # Centre hyphen within the above-baseline span of the word
                # (mirrors segmentation project logic exactly)
                scaled_bl = int(pw.word.bl * zoom_factor)
                word_above = int(pw.word.height * zoom_factor) - scaled_bl
                y_start_word = baseline_y - word_above
                y_offset = max(0, (word_above - h_h) // 2)
                hy_start = y_start_word + y_offset
                hy_end = hy_start + h_h
                hx_start = current_x
                hx_end = hx_start + h_w

                hy_start_c = max(0, hy_start)
                hy_end_c = min(total_height, hy_end)
                hx_start_c = max(0, hx_start)
                hx_end_c = min(new_page_width - right_margin, hx_end)

                if hy_end_c > hy_start_c and hx_end_c > hx_start_c:
                    hy0 = hy_start_c - hy_start
                    hy1 = hy0 + (hy_end_c - hy_start_c)
                    hx0 = hx_start_c - hx_start
                    hx1 = hx0 + (hx_end_c - hx_start_c)
                    page[hy_start_c:hy_end_c, hx_start_c:hx_end_c] = hi[hy0:hy1, hx0:hx1]

                current_x += h_w

        current_y += line_height

    return page


# ---------------------------------------------------------------------------
# Utility: convert word tuples → List[List[Word]]
# ---------------------------------------------------------------------------

def _robust_linear_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    epsilon: float,
) -> Optional[Tuple[float, float]]:
    if len(xs) < 2:
        return None
    a, b = np.polyfit(xs, ys, 1)
    residuals = np.abs(ys - (a * xs + b))
    inlier_mask = residuals <= epsilon
    if inlier_mask.sum() < 2:
        return None
    a, b = np.polyfit(xs[inlier_mask], ys[inlier_mask], 1)
    return float(a), float(b)


def words_to_wordlines(
    lines: List[List[Tuple[int, int, int, int]]],
) -> List[List[Word]]:
    """
    Convert list of lines (each line = list of (xmin,ymin,xmax,ymax)) into
    List[List[Word]] with baseline (bl) and above-baseline (above) fields set.
    """
    result = []
    for line_idx, line in enumerate(lines):
        if not line:
            result.append([])
            continue

        ymaxes = np.array([ymax for (_, _, _, ymax) in line], dtype=float)
        ymins  = np.array([ymin for (_, ymin, _, _) in line], dtype=float)
        xctrs  = np.array([(xmin + xmax) / 2.0 for (xmin, _, xmax, _) in line], dtype=float)
        heights = ymaxes - ymins

        median_h = float(np.median(heights))
        epsilon = 0.3 * median_h

        use_fit = False
        bl_fit = cap_fit = None

        if len(line) >= 4:
            bl_fit  = _robust_linear_fit(xctrs, ymaxes, epsilon)
            cap_fit = _robust_linear_fit(xctrs, ymins,  epsilon)
            if bl_fit is not None and cap_fit is not None:
                x_range = float(xctrs.max() - xctrs.min())
                predicted_range = abs(bl_fit[0]) * x_range
                if predicted_range > 0.15 * median_h:
                    use_fit = True

        if not use_fit:
            if len(ymaxes) >= 4:
                baseline_ymax = int(np.percentile(ymaxes, 10))
            else:
                baseline_ymax = int(ymaxes.min())
            if len(ymins) >= 4:
                ref_ymin = int(np.percentile(ymins, 10))
            else:
                ref_ymin = int(ymins.min())
            line_above = max(1, baseline_ymax - ref_ymin)

        words = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(line):
            height = ymax - ymin
            if use_fit:
                xc = xctrs[i]
                fitted_bl  = bl_fit[0]  * xc + bl_fit[1]
                fitted_cap = cap_fit[0] * xc + cap_fit[1]
                word_above = max(1, int(round(fitted_bl - fitted_cap)))
                word_bl    = max(0, min(int(round(ymax - fitted_bl)), height // 2))
            else:
                word_above = line_above
                word_bl    = max(0, min(ymax - baseline_ymax, height // 2))
            words.append(Word(xmin, ymin, xmax, ymax, bl=word_bl, above=word_above))
        result.append(words)
    return result
