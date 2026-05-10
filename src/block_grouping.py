"""
block_grouping.py — group raw DocLayout-YOLO detections into semantic blocks.

Port of segmentation/src/ocr_reflow/layout.py:find_grouped_bounding_boxes()
with shapely and networkx replaced by plain AABB arithmetic and Union-Find.

Public API:
    group_detections(detections, img_w, img_h) -> List[Detection]

New synthetic labels produced:
    "figure_and_caption"
    "isolate_formula_and_caption"
    "table_and_caption"
    "titled_block_title"
    "titled_block_body"
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Tuple

from inference import Detection


# ---------------------------------------------------------------------------
# AABB helpers (replace all shapely calls)
# ---------------------------------------------------------------------------

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _cx(b: Box) -> float:
    return (b[0] + b[2]) / 2.0


def _cy(b: Box) -> float:
    return (b[1] + b[3]) / 2.0


def _centroid_dist(a: Box, b: Box) -> float:
    return math.hypot(_cx(a) - _cx(b), _cy(a) - _cy(b))


def _y_mid_dist(a: Box, b: Box) -> float:
    """Vertical distance between y-midpoints (for formula-caption pairing)."""
    return abs(_cy(a) - _cy(b))


def _union_box(boxes: List[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersects(a: Box, b: Box) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _intersection(a: Box, b: Box) -> Box:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


# ---------------------------------------------------------------------------
# Union-Find (same pattern as line_grouping.py)
# ---------------------------------------------------------------------------

def _make_uf(n: int) -> List[int]:
    return list(range(n))


def _find(parent: List[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: List[int], x: int, y: int) -> None:
    px, py = _find(parent, x), _find(parent, y)
    if px != py:
        parent[px] = py


def _connected_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    parent = _make_uf(n)
    for i, j in edges:
        _union(parent, i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(parent, i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


# ---------------------------------------------------------------------------
# Helper: build a synthetic Detection from a box + label
# ---------------------------------------------------------------------------

def _det(box: Box, label: str, conf: float = 1.0) -> Detection:
    from inference import CLASS_NAMES
    class_id = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
    return Detection(
        label=label,
        class_id=class_id,
        conf=conf,
        x1=box[0], y1=box[1], x2=box[2], y2=box[3],
    )


def _det_box(d: Detection) -> Box:
    return (d.x1, d.y1, d.x2, d.y2)


# ---------------------------------------------------------------------------
# Main grouping function
# ---------------------------------------------------------------------------

def group_detections(
    detections: List[Detection],
    img_w: int,
    img_h: int,
) -> List[Detection]:
    """
    Group raw detections into semantic blocks.

    Mirrors segmentation/layout.py:find_grouped_bounding_boxes() exactly,
    but uses plain AABB math instead of shapely/networkx.

    Plain-text boxes are expanded by 5 px (clamped to image bounds) before
    grouping, matching the behaviour of segmentation/layout.py:layout().

    Returns a list of Detection objects (some with synthetic labels).
    """

    # --- Expand plain-text boxes by 5 px (clamped) ---
    expanded: List[Detection] = []
    for d in detections:
        if d.label == "plain text":
            from inference import Detection as Det
            expanded.append(Det(
                label=d.label,
                class_id=d.class_id,
                conf=d.conf,
                x1=max(0, d.x1 - 5),
                y1=max(0, d.y1 - 5),
                x2=min(img_w, d.x2 + 5),
                y2=min(img_h, d.y2 + 5),
            ))
        else:
            expanded.append(d)
    detections = expanded

    # Index by label
    by_label: dict[str, List[int]] = defaultdict(list)
    for idx, d in enumerate(detections):
        by_label[d.label].append(idx)

    boxes = [_det_box(d) for d in detections]
    confs = [d.conf for d in detections]

    result: List[Tuple[Box, str, float]] = []  # (box, label, conf)

    # ------------------------------------------------------------------
    # Step 2: figure + figure_caption
    # ------------------------------------------------------------------
    figures   = set(by_label.get("figure", []))
    fig_caps  = set(by_label.get("figure_caption", []))
    used_figs = set()
    used_fcaps = set()

    for cap_idx in fig_caps:
        cap_box = boxes[cap_idx]
        best_dist, best_fig = float("inf"), None
        for fig_idx in figures - used_figs:
            d = _centroid_dist(cap_box, boxes[fig_idx])
            if d < best_dist:
                best_dist, best_fig = d, fig_idx
        if best_fig is not None:
            merged = _union_box([boxes[best_fig], cap_box])
            result.append((merged, "figure_and_caption", max(confs[best_fig], confs[cap_idx])))
            used_figs.add(best_fig)
            used_fcaps.add(cap_idx)

    for fig_idx in figures - used_figs:
        result.append((boxes[fig_idx], "figure", confs[fig_idx]))
    for cap_idx in fig_caps - used_fcaps:
        result.append((boxes[cap_idx], "figure_caption", confs[cap_idx]))

    # ------------------------------------------------------------------
    # Step 3: isolate_formula + formula_caption  (match by y-midpoint)
    # ------------------------------------------------------------------
    formulas   = set(by_label.get("isolate_formula", []))
    form_caps  = set(by_label.get("formula_caption", []))
    used_forms = set()
    used_fcaps2 = set()

    for cap_idx in form_caps:
        cap_box = boxes[cap_idx]
        best_dist, best_form = float("inf"), None
        for form_idx in formulas - used_forms:
            d = _y_mid_dist(cap_box, boxes[form_idx])
            if d < best_dist:
                best_dist, best_form = d, form_idx
        if best_form is not None:
            merged = _union_box([boxes[best_form], cap_box])
            result.append((merged, "isolate_formula_and_caption", max(confs[best_form], confs[cap_idx])))
            used_forms.add(best_form)
            used_fcaps2.add(cap_idx)

    for form_idx in formulas - used_forms:
        result.append((boxes[form_idx], "isolate_formula", confs[form_idx]))
    for cap_idx in form_caps - used_fcaps2:
        result.append((boxes[cap_idx], "formula_caption", confs[cap_idx]))

    # ------------------------------------------------------------------
    # Step 4: table + table_caption + table_footnote
    # ------------------------------------------------------------------
    tables      = set(by_label.get("table", []))
    tbl_caps    = set(by_label.get("table_caption", []))
    tbl_fnotes  = set(by_label.get("table_footnote", []))
    used_tables = set()
    used_tcaps  = set()
    used_tfnotes = set()

    # Pair each caption to nearest table
    cap_to_table: dict[int, int] = {}
    for cap_idx in tbl_caps:
        best_dist, best_tbl = float("inf"), None
        for tbl_idx in tables - used_tables:
            d = _centroid_dist(boxes[cap_idx], boxes[tbl_idx])
            if d < best_dist:
                best_dist, best_tbl = d, tbl_idx
        if best_tbl is not None:
            cap_to_table[cap_idx] = best_tbl
            used_tables.add(best_tbl)
            used_tcaps.add(cap_idx)

    # Build table groups: table_idx -> [table_idx, cap_idx?, footnote_idx*]
    table_groups: dict[int, List[int]] = {tbl_idx: [tbl_idx] for tbl_idx in tables}
    for cap_idx, tbl_idx in cap_to_table.items():
        table_groups[tbl_idx].append(cap_idx)

    # Pair each footnote to nearest table (by centroid)
    for fn_idx in tbl_fnotes:
        best_dist, best_tbl = float("inf"), None
        for tbl_idx in tables:
            d = _centroid_dist(boxes[fn_idx], boxes[tbl_idx])
            if d < best_dist:
                best_dist, best_tbl = d, tbl_idx
        if best_tbl is not None:
            table_groups[best_tbl].append(fn_idx)
            used_tfnotes.add(fn_idx)

    for tbl_idx, group_indices in table_groups.items():
        group_boxes = [boxes[i] for i in group_indices]
        group_conf  = max(confs[i] for i in group_indices)
        if len(group_indices) > 1:
            result.append((_union_box(group_boxes), "table_and_caption", group_conf))
        else:
            result.append((boxes[tbl_idx], "table", confs[tbl_idx]))

    for cap_idx in tbl_caps - used_tcaps:
        result.append((boxes[cap_idx], "table_caption", confs[cap_idx]))
    for fn_idx in tbl_fnotes - used_tfnotes:
        result.append((boxes[fn_idx], "table_footnote", confs[fn_idx]))

    # ------------------------------------------------------------------
    # Step 5: plain text — merge overlapping boxes via Union-Find
    # ------------------------------------------------------------------
    pt_indices = by_label.get("plain text", [])
    plain_text_boxes: List[Box] = []

    if pt_indices:
        n = len(pt_indices)
        edges = []
        for ii in range(n):
            for jj in range(ii + 1, n):
                i, j = pt_indices[ii], pt_indices[jj]
                if not _intersects(boxes[i], boxes[j]):
                    continue
                inter = _intersection(boxes[i], boxes[j])
                inter_area = _area(inter)
                if inter_area <= 0:
                    continue
                smaller = min(_area(boxes[i]), _area(boxes[j]))
                if smaller > 0 and inter_area / smaller >= 0.1:
                    edges.append((ii, jj))

        for component in _connected_components(n, edges):
            comp_boxes = [boxes[pt_indices[k]] for k in component]
            plain_text_boxes.append(_union_box(comp_boxes))

    # ------------------------------------------------------------------
    # Step 6: split plain text boxes around overlapping non-text regions
    # ------------------------------------------------------------------
    # Collect all non-text regions: already in result + raw non-plain-text dets
    non_text_regions: List[Box] = [b for b, _, _ in result]
    for i, d in enumerate(detections):
        if d.label != "plain text":
            non_text_regions.append(boxes[i])

    MIN_PIECE_H = 20

    for pt_box in plain_text_boxes:
        px1, py1, px2, py2 = pt_box

        cut_bands: List[Tuple[float, float]] = []
        for nt_box in non_text_regions:
            if not _intersects(pt_box, nt_box):
                continue
            inter = _intersection(pt_box, nt_box)
            if _area(inter) <= 0:
                continue
            overlap_h = inter[3] - inter[1]
            if overlap_h < 10:
                continue
            cut_bands.append((inter[1], inter[3]))

        if not cut_bands:
            result.append((pt_box, "plain text", 1.0))
            continue

        # Merge overlapping cut bands
        cut_bands.sort()
        merged_bands = [list(cut_bands[0])]
        for band_top, band_bot in cut_bands[1:]:
            if band_top <= merged_bands[-1][1]:
                merged_bands[-1][1] = max(merged_bands[-1][1], band_bot)
            else:
                merged_bands.append([band_top, band_bot])

        # Emit text-only slices between cut bands
        slice_tops = [py1] + [b[1] for b in merged_bands]
        slice_bots = [b[0] for b in merged_bands] + [py2]

        for top, bot in zip(slice_tops, slice_bots):
            if bot - top >= MIN_PIECE_H:
                result.append(((px1, int(top), px2, int(bot)), "plain text", 1.0))

    # ------------------------------------------------------------------
    # Step 7: all other labels (title, abandon, …) as individual boxes
    # ------------------------------------------------------------------
    HANDLED = {
        "figure", "figure_caption",
        "isolate_formula", "formula_caption",
        "table", "table_caption", "table_footnote",
        "plain text",
    }
    for label, indices in by_label.items():
        if label in HANDLED:
            continue
        for idx in indices:
            result.append((boxes[idx], label, confs[idx]))

    # ------------------------------------------------------------------
    # Step 8: pair adjacent title + plain text → titled_block_title / body
    # ------------------------------------------------------------------
    title_entries = [(i, b, t) for i, (b, t, _) in enumerate(result) if t == "title"]
    pt_entries    = [(i, b, t) for i, (b, t, _) in enumerate(result) if t == "plain text"]

    paired = set()
    new_pairs: List[Tuple[int, int, Box, Box]] = []

    for ti, geom_t, _ in title_entries:
        tx1, ty1, tx2, ty2 = geom_t
        best_gap, best = float("inf"), None
        for pi, geom_p, _ in pt_entries:
            if pi in paired:
                continue
            px1, py1, px2, py2 = geom_p
            gap = py1 - ty2
            if gap < 0 or gap > 50:
                continue
            overlap = min(tx2, px2) - max(tx1, px1)
            narrower = min(tx2 - tx1, px2 - px1)
            if narrower <= 0 or overlap / narrower < 0.5:
                continue
            if gap < best_gap:
                best_gap, best = gap, (pi, geom_p)
        if best is not None:
            pi, geom_p = best
            paired.add(ti)
            paired.add(pi)
            new_pairs.append((ti, pi, geom_t, geom_p))

    if new_pairs:
        pair_by_title_idx = {ti: (gt, gp) for ti, pi, gt, gp in new_pairs}
        new_result: List[Tuple[Box, str, float]] = []
        for i, (b, t, c) in enumerate(result):
            if i in paired:
                if i in pair_by_title_idx:
                    gt, gp = pair_by_title_idx[i]
                    new_result.append((gt, "titled_block_title", c))
                    new_result.append((gp, "titled_block_body", c))
                # plain text partner already added above
            else:
                new_result.append((b, t, c))
        result = new_result

    # ------------------------------------------------------------------
    # Deduplicate plain text boxes (drop those fully covered by another)
    # ------------------------------------------------------------------
    plain_indexed = [(i, b) for i, (b, t, _) in enumerate(result) if t == "plain text"]
    drop: set[int] = set()
    for ii, (i, bi) in enumerate(plain_indexed):
        xi1, yi1, xi2, yi2 = bi
        for jj, (j, bj) in enumerate(plain_indexed):
            if ii == jj:
                continue
            xj1, yj1, xj2, yj2 = bj
            if yi1 >= yj1 - 2 and yi2 <= yj2 + 2:
                x_overlap = min(xi2, xj2) - max(xi1, xj1)
                if x_overlap > 0.7 * (xi2 - xi1):
                    drop.add(i)
                    break

    final = [(b, t, c) for i, (b, t, c) in enumerate(result) if i not in drop]

    # ------------------------------------------------------------------
    # Convert to Detection objects
    # ------------------------------------------------------------------
    out: List[Detection] = []
    for b, label, conf in final:
        out.append(_det(b, label, conf))
    return out
