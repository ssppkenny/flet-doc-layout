"""
lang_detect.py — language detection using multi-script OCR + langid.

Pipeline:
  1. Sample word crops from pages (middle of document).
  2. Run three CRNN models on each crop:
       - Latin CRNN  (crnn_mobilenet_v3_small, doctr)
       - Cyrillic CRNN (PP-OCRv5 mobile rec, PaddleOCR)
       - Greek CRNN    (PP-OCRv5 mobile rec, PaddleOCR)
  3. Concatenate all recognized text and feed to langid (pure Python).
  4. Map ISO 639-1 code to app language code.

Public API:
    load_models(models_dir) -> LangDetector | None
    LangDetector.detect(word_boxes_per_page, images_per_page) -> (lang_code, confidence)

    LANGUAGES        — list of (app_code, display_name) for the dropdown
    APP_LANG_DEFAULT — fallback language code
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Language list for the UI dropdown
# ---------------------------------------------------------------------------

LANGUAGES: List[Tuple[str, str]] = [
    ("en", "English"),
    ("ru", "Russian"),
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
    ("el", "Greek"),
]

APP_LANG_DEFAULT = "en"

# langid ISO 639-1 → app language code
# Only languages present in LANGUAGES above are mapped.
_LID_TO_APP: Dict[str, str] = {
    "en": "en", "ru": "ru", "de": "de", "fr": "fr", "es": "es",
    "pl": "pl", "uk": "uk", "bg": "bg", "cs": "cs", "sk": "sk",
    "nl": "nl", "pt": "pt", "it": "it", "sv": "sv", "da": "da",
    "fi": "fi", "nb": "nb", "no": "nb", "hu": "hu", "ro": "ro",
    "hr": "hr", "sr": "sr", "sl": "sl", "lt": "lt", "lv": "lv",
    "et": "et", "be": "be", "ca": "ca", "eu": "eu", "gl": "gl",
    "ga": "ga", "el": "el",
}

# ---------------------------------------------------------------------------
# CRNN constants
# ---------------------------------------------------------------------------

# Latin CRNN (doctr crnn_mobilenet_v3_small)
_LATIN_VOCAB = (
    "0123456789abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿'
    "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"
)
_LATIN_H, _LATIN_W = 32, 128
_LATIN_MEAN = np.array([0.694, 0.695, 0.693], dtype=np.float32)
_LATIN_STD  = np.array([0.299, 0.296, 0.301], dtype=np.float32)

# PaddleOCR CRNN input
_PADDLE_H, _PADDLE_W = 48, 320


# ---------------------------------------------------------------------------
# CTC decoders
# ---------------------------------------------------------------------------

def _ctc_decode_latin(logits: np.ndarray) -> str:
    """Greedy CTC decode for Latin CRNN. logits: (T, vocab+1), last class = blank."""
    indices = logits.argmax(axis=-1)
    blank = len(_LATIN_VOCAB)
    chars, prev = [], blank
    for idx in indices:
        if idx != blank and idx != prev and idx < len(_LATIN_VOCAB):
            chars.append(_LATIN_VOCAB[idx])
        prev = idx
    return "".join(chars)


def _ctc_decode_paddle(logits: np.ndarray, vocab: List[str]) -> str:
    """Greedy CTC decode for PaddleOCR CRNN.
    vocab is the raw character_dict from inference.yml (without blank).
    Blank = index 0 (prepended by PaddleOCR at runtime).
    """
    indices = logits.argmax(axis=-1)
    blank = 0
    # full_vocab: blank at 0, then vocab chars, then space
    full_vocab = [""] + vocab + [" "]
    chars, prev = [], blank
    for idx in indices:
        if idx != blank and idx != prev and idx < len(full_vocab):
            chars.append(full_vocab[idx])
        prev = idx
    return "".join(chars)


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_latin(crop_pil: Image.Image) -> np.ndarray:
    """Preprocess crop for Latin CRNN. Returns (1, 3, 32, 128) float32."""
    img = crop_pil.convert("RGB").resize((_LATIN_W, _LATIN_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _LATIN_MEAN) / _LATIN_STD
    return arr.transpose(2, 0, 1)[np.newaxis]


def _preprocess_paddle(crop_pil: Image.Image) -> np.ndarray:
    """Preprocess crop for PaddleOCR CRNN. Returns (1, 3, 48, 320) float32 BGR."""
    img = crop_pil.convert("RGB")
    # Convert to BGR numpy
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    h, w = arr.shape[:2]
    new_w = max(1, int(w * _PADDLE_H / h))
    arr = cv2.resize(arr, (new_w, _PADDLE_H))
    if new_w < _PADDLE_W:
        pad = np.ones((_PADDLE_H, _PADDLE_W - new_w, 3), dtype=np.uint8) * 255
        arr = np.concatenate([arr, pad], axis=1)
    else:
        arr = arr[:, :_PADDLE_W]
    inp = arr.astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5
    return inp.transpose(2, 0, 1)[np.newaxis]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LangDetector:
    """Language detector: multi-script CRNN → langid."""

    def __init__(
        self,
        latin_net: cv2.dnn.Net,
        cyrillic_net: cv2.dnn.Net,
        greek_net: cv2.dnn.Net,
        cyrillic_vocab: List[str],
        greek_vocab: List[str],
    ):
        self._latin = latin_net
        self._cyrillic = cyrillic_net
        self._greek = greek_net
        self._cyr_vocab = cyrillic_vocab
        self._grk_vocab = greek_vocab

    def _recognize_crop(self, crop_pil: Image.Image) -> str:
        """Run all three CRNNs on a crop and return concatenated recognized text."""
        parts = []

        # Latin
        try:
            blob = _preprocess_latin(crop_pil)
            self._latin.setInput(blob)
            raw = self._latin.forward()  # (1, T, vocab+1)
            word = _ctc_decode_latin(raw[0])
            if word.strip():
                parts.append(word)
        except Exception:
            pass

        # Cyrillic
        try:
            blob = _preprocess_paddle(crop_pil)
            self._cyrillic.setInput(blob)
            raw = self._cyrillic.forward()  # (1, T, 852)
            word = _ctc_decode_paddle(raw[0], self._cyr_vocab)
            if word.strip():
                parts.append(word)
        except Exception:
            pass

        # Greek
        try:
            blob = _preprocess_paddle(crop_pil)
            self._greek.setInput(blob)
            raw = self._greek.forward()  # (1, T, 356)
            word = _ctc_decode_paddle(raw[0], self._grk_vocab)
            if word.strip():
                parts.append(word)
        except Exception:
            pass

        return " ".join(parts)

    def detect(
        self,
        word_boxes_per_page: List[List[Tuple[int, int, int, int]]],
        images_per_page: List[Image.Image],
        max_words_per_page: int = 20,
    ) -> Tuple[str, float]:
        """
        Detect language from word crops across multiple pages.

        Returns:
            (app_lang_code, confidence)  e.g. ("ru", 0.991)
        """
        all_text_parts: List[str] = []

        for word_boxes, pil_img in zip(word_boxes_per_page, images_per_page):
            if not word_boxes or pil_img is None:
                continue
            img_w, img_h = pil_img.size

            n = len(word_boxes)
            start = max(0, n // 4)
            end   = min(n, 3 * n // 4)
            candidates = word_boxes[start:end] if end > start else word_boxes
            step = max(1, len(candidates) // max_words_per_page)
            sampled = candidates[::step][:max_words_per_page]

            page_words: List[str] = []
            for x1, y1, x2, y2 in sampled:
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = pil_img.crop((x1, y1, x2, y2))
                text = self._recognize_crop(crop)
                if text.strip():
                    page_words.append(text)

            if page_words:
                all_text_parts.append(" ".join(page_words))

        if not all_text_parts:
            return APP_LANG_DEFAULT, 0.0

        combined = " ".join(all_text_parts)
        print(f"[lang_detect] OCR text sample ({len(combined)} chars): {combined[:120]!r}")

        try:
            import langid
            lang_code, score = langid.classify(combined)
            # score is a log-probability (negative); convert to a rough confidence
            conf = min(1.0, max(0.0, 1.0 + score / 500.0))
            app_code = _LID_TO_APP.get(lang_code, APP_LANG_DEFAULT)
            print(f"[lang_detect] langid: {lang_code} ({score:.1f}) -> {app_code}")
            return app_code, conf
        except Exception as e:
            print(f"[lang_detect] langid error: {e}")
            return APP_LANG_DEFAULT, 0.0


def load_models(models_dir: str) -> Optional[LangDetector]:
    """
    Load all models from models_dir:
      - crnn_mobilenet_v3_small_v11.onnx  (Latin CRNN)
      - cyrillic_rec_sim.onnx             (Cyrillic CRNN)
      - greek_rec_sim.onnx                (Greek CRNN)
      - cyrillic_vocab.json
      - greek_vocab.json

    Returns None if any file is missing or fails to load.
    """
    d = Path(models_dir)
    try:
        # Verify langid is importable (pure Python, no native deps)
        import langid  # noqa: F401

        # Load CRNN nets via cv2.dnn
        def load_net(name: str) -> cv2.dnn.Net:
            path = str(d / name)
            net = cv2.dnn.readNetFromONNX(path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return net

        latin_net    = load_net("crnn_mobilenet_v3_small_v11.onnx")
        cyrillic_net = load_net("cyrillic_rec_sim.onnx")
        greek_net    = load_net("greek_rec_sim.onnx")

        # Load vocabs
        with open(d / "cyrillic_vocab.json", encoding="utf-8") as f:
            cyrillic_vocab: List[str] = json.load(f)
        with open(d / "greek_vocab.json", encoding="utf-8") as f:
            greek_vocab: List[str] = json.load(f)

        return LangDetector(
            latin_net, cyrillic_net, greek_net,
            cyrillic_vocab, greek_vocab,
        )
    except Exception as e:
        import traceback
        print(f"[lang_detect] Failed to load models: {e}\n{traceback.format_exc()}")
        raise
