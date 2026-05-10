"""
line_grouping.py — group word bounding boxes into text lines.

Pure NumPy/Python implementation; no shapely or divide_conquer_4d required.

Algorithm (adapted from ocr_reflow/main.py):
  1. Union-Find clustering: each word's right-side neighborhood (width=2×height,
     height=word height) captures words on the same line.
  2. merge_close_lines: merge clusters whose Y-centres are very close
     (handles superscripts / subscripts).
  3. Assign each word to its cluster line via Y-centre proximity.

Public API:
    group_words_into_lines(word_boxes) -> List[List[Tuple[int,int,int,int]]]
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


def _union_find_lines(
    words: List[Tuple[int, int, int, int]],
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Cluster word boxes into lines using Union-Find.

    Each word's neighborhood extends from its right edge rightward by 2×height,
    and vertically spans the word's own ymin..ymax.  Any other word whose
    Y-centre falls inside this neighborhood is merged into the same line.

    Returns lines sorted top-to-bottom, each line sorted left-to-right.
    """
    n = len(words)
    if n == 0:
        return []

    # Filter tiny words (height < 60% of median) — subscripts / noise
    heights = [ymax - ymin for (xmin, ymin, xmax, ymax) in words]
    median_h = float(np.median(heights))
    threshold_h = median_h * 0.60

    entities = []
    for i, (xmin, ymin, xmax, ymax) in enumerate(words):
        h = ymax - ymin
        if h >= threshold_h:
            entities.append((i, xmin, ymin, xmax, ymax, h))

    m = len(entities)
    if m == 0:
        return []

    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, (_, xmin_i, ymin_i, xmax_i, ymax_i, h_i) in enumerate(entities):
        nb_xmin = xmax_i
        nb_xmax = xmax_i + 2 * h_i
        nb_ymin = ymin_i
        nb_ymax = ymax_i
        for j, (_, xmin_j, ymin_j, xmax_j, ymax_j, _) in enumerate(entities):
            if i == j:
                continue
            mid_y_j = (ymin_j + ymax_j) / 2.0
            x_overlap = not (xmax_j < nb_xmin or xmin_j > nb_xmax)
            y_in = nb_ymin <= mid_y_j <= nb_ymax
            if x_overlap and y_in:
                union(i, j)

    # Group by cluster root
    clusters: dict[int, list[int]] = {}
    for i in range(m):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Build lines: sort clusters top-to-bottom, words left-to-right
    def cluster_top(indices: list[int]) -> float:
        return min(entities[i][2] for i in indices)  # min ymin

    lines = []
    for _, indices in sorted(clusters.items(), key=lambda kv: cluster_top(kv[1])):
        line_words = sorted(
            [words[entities[i][0]] for i in indices],
            key=lambda w: w[0],  # sort by xmin
        )
        lines.append(line_words)

    return lines


def _merge_close_lines(
    lines: List[List[Tuple[int, int, int, int]]],
    y_threshold: int = 30,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Merge lines whose Y-centres are within y_threshold of each other.
    Handles superscripts / subscripts that form spurious separate clusters.
    """
    if len(lines) <= 1:
        return lines

    def line_y_centre(line: list) -> float:
        ymins = [w[1] for w in line]
        ymaxes = [w[3] for w in line]
        return (min(ymins) + max(ymaxes)) / 2.0

    centres = [line_y_centre(ln) for ln in lines]

    # Adaptive threshold: min(y_threshold, 0.8 × avg gap)
    gaps = [centres[i + 1] - centres[i] for i in range(len(centres) - 1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else y_threshold
    adaptive = min(y_threshold, avg_gap * 0.8)

    merged = True
    result = [list(ln) for ln in lines]
    for _ in range(10):
        if not merged:
            break
        merged = False
        new_result = []
        i = 0
        while i < len(result):
            if i + 1 < len(result):
                c1 = line_y_centre(result[i])
                c2 = line_y_centre(result[i + 1])
                if abs(c2 - c1) <= adaptive:
                    combined = sorted(result[i] + result[i + 1], key=lambda w: w[0])
                    new_result.append(combined)
                    i += 2
                    merged = True
                    continue
            new_result.append(result[i])
            i += 1
        result = new_result

    return result


def group_words_into_lines(
    word_boxes: List[Tuple[int, int, int, int]],
    merge_threshold: int = 30,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Group flat list of word bounding boxes into text lines.

    Args:
        word_boxes: List of (xmin, ymin, xmax, ymax) in pixel coords.
        merge_threshold: Max Y-distance to merge adjacent clusters (pixels).

    Returns:
        List of lines; each line is a list of (xmin, ymin, xmax, ymax) tuples
        sorted left-to-right.  Lines are sorted top-to-bottom.
    """
    if not word_boxes:
        return []
    lines = _union_find_lines(word_boxes)
    lines = _merge_close_lines(lines, y_threshold=merge_threshold)
    return lines
