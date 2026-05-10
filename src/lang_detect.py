"""
lang_detect.py — script detection from word image crops.

No OCR required.  Uses connected-component topology to classify each word
crop as Cyrillic, Latin, Arabic, CJK, or Unknown, then majority-votes across
all words in the page to pick a script family, and maps that to a default
ISO 639-1 language code.

Public API:
    detect_script(word_boxes, img_bgr) -> str   # ISO 639-1 code, e.g. "ru"
    LANGUAGES                                   # list of (code, display_name) sorted by popularity
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Language list (30 entries, sorted by global usage / document frequency)
# ---------------------------------------------------------------------------

LANGUAGES: List[Tuple[str, str]] = [
    ("ru", "Russian"),
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pl", "Polish"),
    ("uk", "Ukrainian"),
    ("bg", "Bulgarian"),
    ("cs", "Czech"),
    ("sk", "Slovak"),
    ("nl", "Dutch"),
    ("pt", "Portuguese"),
    ("it", "Italian"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("nb", "Norwegian"),
    ("hu", "Hungarian"),
    ("ro", "Romanian"),
    ("hr", "Croatian"),
    ("sr", "Serbian"),
    ("sl", "Slovenian"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("et", "Estonian"),
    ("be", "Belarusian"),
    ("ca", "Catalan"),
    ("eu", "Basque"),
    ("gl", "Galician"),
    ("ga", "Irish"),
]

# ---------------------------------------------------------------------------
# Script → default language mapping
# ---------------------------------------------------------------------------

_SCRIPT_TO_LANG = {
    "cyrillic": "ru",
    "latin":    "en",
    "arabic":   "ar",
    "cjk":      "zh",
    "unknown":  "en",
}


def script_to_lang(script: str) -> str:
    return _SCRIPT_TO_LANG.get(script, "en")


# ---------------------------------------------------------------------------
# Per-crop script classification
# ---------------------------------------------------------------------------

def _classify_crop(crop_bgr: np.ndarray) -> str:
    """
    Classify a single word crop as 'cyrillic', 'latin', 'arabic', 'cjk', or 'unknown'.

    Heuristics (purely geometric, no OCR):
      - Binarize (Otsu), find connected components.
      - Compute median aspect ratio (w/h) and vertical-stroke density.
      - CJK: many square-ish components (aspect ratio near 1), high density.
      - Arabic: many small components with low median height ratio, right-to-left
        tendency (components cluster in lower half).
      - Cyrillic vs Latin: Cyrillic letters tend to have more vertical strokes
        per unit width (denser) and slightly higher median component height
        relative to word height.  Latin has more ascenders/descenders giving
        a wider height spread.
      - Falls back to 'unknown' when evidence is weak.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return "unknown"

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    if num_labels < 2:
        return "unknown"

    comp_ws = []
    comp_hs = []
    comp_ys = []  # top y of each component (relative to crop)
    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cy = stats[i, cv2.CC_STAT_TOP]
        area = stats[i, cv2.CC_STAT_AREA]
        if cw < 2 or ch < 2 or area < 4:
            continue
        comp_ws.append(cw)
        comp_hs.append(ch)
        comp_ys.append(cy)

    if not comp_hs:
        return "unknown"

    median_ch = float(np.median(comp_hs))
    median_cw = float(np.median(comp_ws))
    n_comps = len(comp_hs)

    # Aspect ratio of median component
    aspect = median_cw / max(median_ch, 1)

    # Height ratio: median component height vs word crop height
    height_ratio = median_ch / max(h, 1)

    # Component density: components per unit width
    density = n_comps / max(w, 1)

    # Fraction of components in lower half of crop (Arabic tendency)
    lower_half = sum(1 for cy in comp_ys if cy > h * 0.5)
    lower_frac = lower_half / max(n_comps, 1)

    # --- CJK: square-ish, dense, tall ---
    if aspect > 0.6 and aspect < 1.8 and height_ratio > 0.55 and density > 0.04:
        return "cjk"

    # --- Arabic: many small fragments, lower-half clustering ---
    if lower_frac > 0.6 and height_ratio < 0.45 and n_comps > 3:
        return "arabic"

    # --- Cyrillic vs Latin ---
    # Cyrillic: height_ratio typically 0.45–0.75, density slightly higher
    # Latin: more ascenders → wider height spread, lower median height_ratio
    if height_ratio > 0.42 and density > 0.025:
        return "cyrillic"

    if height_ratio > 0.25:
        return "latin"

    return "unknown"


# ---------------------------------------------------------------------------
# Page-level script detection
# ---------------------------------------------------------------------------

def detect_script(
    word_boxes: List[Tuple[int, int, int, int]],
    img_bgr: np.ndarray,
    max_words: int = 60,
) -> str:
    """
    Detect the dominant script in the page by classifying up to `max_words`
    word crops and majority-voting.

    Args:
        word_boxes: list of (xmin, ymin, xmax, ymax) in image coords.
        img_bgr:    full page image (BGR numpy array).
        max_words:  cap on how many crops to examine (for speed).

    Returns:
        ISO 639-1 language code string (e.g. "ru", "en", "ar", "zh").
    """
    if not word_boxes or img_bgr is None or img_bgr.size == 0:
        return "en"

    img_h, img_w = img_bgr.shape[:2]
    votes: dict = {}

    # Sample evenly across the word list
    step = max(1, len(word_boxes) // max_words)
    sampled = word_boxes[::step][:max_words]

    for xmin, ymin, xmax, ymax in sampled:
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w, xmax)
        ymax = min(img_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        crop = img_bgr[ymin:ymax, xmin:xmax]
        script = _classify_crop(crop)
        votes[script] = votes.get(script, 0) + 1

    if not votes:
        return "en"

    # Remove 'unknown' from voting unless it's the only option
    known_votes = {k: v for k, v in votes.items() if k != "unknown"}
    if known_votes:
        best_script = max(known_votes, key=known_votes.__getitem__)
    else:
        best_script = "unknown"

    return script_to_lang(best_script)
