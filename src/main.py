"""
Flow Reader: DjVu/PDF viewer with word-level reflow.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import flet as ft
from PIL import Image
import inference
import doctr_inference
import lang_detect
import line_grouping
import reflow_words
import server_inference
import skew_detection

# ---------------------------------------------------------------------------
# Model download configuration
# ---------------------------------------------------------------------------
_RELEASE_BASE = (
    "https://github.com/ssppkenny/flet-doc-layout/releases/download/models"
)
MODEL_URLS = {
    "doclayout.onnx":              (f"{_RELEASE_BASE}/doclayout.onnx",              75512430),
    "fast_base.onnx":              (f"{_RELEASE_BASE}/fast_base.onnx",              65436251),
    "crnn_mobilenet_v3_small_v11.onnx": (f"{_RELEASE_BASE}/crnn_mobilenet_v3_small_v11.onnx", 8336543),
    "cyrillic_rec_sim.onnx":       (f"{_RELEASE_BASE}/cyrillic_rec_sim.onnx",       8005590),
    "greek_rec_sim.onnx":          (f"{_RELEASE_BASE}/greek_rec_sim.onnx",          7765526),
    "cyrillic_vocab.json":         (f"{_RELEASE_BASE}/cyrillic_vocab.json",         5333),
    "greek_vocab.json":            (f"{_RELEASE_BASE}/greek_vocab.json",            2166),
}


def _models_dir() -> Path:
    storage = os.environ.get("FLET_APP_STORAGE_DATA")
    if storage:
        return Path(storage) / "models"
    # fallback for local dev runs
    return Path(__file__).parent / "assets" / "models"


def _models_ok() -> bool:
    d = _models_dir()
    for name, (_, expected_size) in MODEL_URLS.items():
        f = d / name
        if not f.exists() or f.stat().st_size != expected_size:
            return False
    return True


def _settings_path() -> Path:
    storage = os.environ.get("FLET_APP_STORAGE_DATA")
    if storage:
        return Path(storage) / "settings.json"
    return Path(__file__).parent / "assets" / "settings.json"


def _load_settings() -> dict:
    p = _settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_settings(data: dict):
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


_TEXT_CLASSES    = {"plain text", "title", "titled_block_body"}
_NON_TEXT_LABELS = {
    "figure", "figure_and_caption", "figure_caption",
    "table", "table_and_caption", "table_caption", "table_footnote",
    "isolate_formula", "isolate_formula_and_caption", "formula_caption",
}
ZOOM_INCREMENT = 1.2  # each zoom button press multiplies by this factor

LANGUAGES = lang_detect.LANGUAGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def render_djvu_page(doc_path: str, page_index: int, dpi: int = 150) -> Image.Image:
    import djvu.decode as djvu_decode
    ctx = djvu_decode.Context()
    doc = ctx.new_document(djvu_decode.FileURI(doc_path))
    doc.decoding_job.wait()
    page = doc.pages[page_index]
    job = page.decode(wait=True)
    native_w, native_h = job.size
    scale = dpi / page.dpi
    out_w = int(native_w * scale)
    out_h = int(native_h * scale)
    fmt = djvu_decode.PixelFormatRgb()
    fmt.rows_top_to_bottom = 1
    fmt.y_top_to_bottom = 1
    data = job.render(
        djvu_decode.RENDER_COLOR,
        (0, 0, out_w, out_h),
        (0, 0, out_w, out_h),
        fmt,
    )
    return Image.frombytes("RGB", (out_w, out_h), data)


def render_pdf_page(pdf_doc, page_index: int, dpi: int = 150) -> Image.Image:
    import pypdfium2 as pdfium  # noqa: F401 — imported here to keep top-level imports light
    page = pdf_doc[page_index]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()




def _do_reflow(
    dets: list,
    all_word_boxes: List[Tuple[int, int, int, int]],
    page_img: Image.Image,
    zoom_factor: float,
    new_page_width: int,
    lang: str = "en",
    force_left_margin: int = None,
    force_right_margin: int = None,
) -> Image.Image:
    """
    Per-region reflow. Runs in a thread executor (no Flet state access).

    Text blocks (plain text / title / titled_block_body):
      - Narrow (< 65% of median plain-text width) → proportionally scaled crop.
      - Normal → word reflow.
      - Before word detection, non-text regions that overlap are masked with bg.

    Non-text blocks (figure, table, formula, …):
      - Zoomed crop, scaled down if wider than available width, centered.

    titled_block_title:
      - Buffered until the matching titled_block_body arrives.
      - If body is narrow → stack title+body as one zoomed crop.
      - If body is not narrow → emit title as zoomed crop, then reflow body.

    Gaps between blocks are proportional to the original page spacing × zoom_factor.
    """
    img_np = np.array(page_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    page_h, page_w = img_bgr.shape[:2]

    # Background colour: median pixel of the full page
    bg = tuple(int(x) for x in np.median(img_bgr.reshape(-1, 3), axis=0))

    # Sort all detections top-to-bottom
    all_dets = sorted(dets, key=lambda d: d.y1)

    if not all_dets:
        h = int(page_img.size[1] * zoom_factor)
        blank = np.ones((h, new_page_width, 3), dtype=np.uint8)
        blank[:] = bg
        return Image.fromarray(blank)

    # Median width of reflowable plain-text regions (for narrow-block detection)
    plain_widths = [
        d.x2 - d.x1 for d in all_dets
        if d.label in ("plain text", "titled_block_body")
    ]
    median_plain_w = float(np.median(plain_widths)) if plain_widths else page_w

    if force_left_margin is not None:
        left_margin  = force_left_margin
        right_margin = force_right_margin if force_right_margin is not None else force_left_margin
    else:
        left_margin  = max(10, int(20 * zoom_factor / 2))
        right_margin = left_margin
    available_w  = new_page_width - left_margin - right_margin
    min_gap      = 4

    # ------------------------------------------------------------------
    # Helper: zoom a BGR crop to fit available_w, return (h, w, strip)
    # ------------------------------------------------------------------
    def _zoom_crop(crop: np.ndarray) -> np.ndarray:
        ch, cw = crop.shape[:2]
        zh = int(ch * zoom_factor)
        zw = int(cw * zoom_factor)
        if zw > available_w:
            scale = available_w / zw
            zw = available_w
            zh = max(1, int(zh * scale))
        zh = max(1, zh)
        zw = max(1, zw)
        resized = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_LINEAR)
        strip = np.empty((zh, new_page_width, 3), dtype=np.uint8)
        strip[:] = bg
        x_off = left_margin + (available_w - zw) // 2
        x_off = max(0, min(x_off, new_page_width - zw))
        strip[:, x_off:x_off + zw] = resized
        return strip

    # ------------------------------------------------------------------
    # Helper: reflow one text detection, return strip or None
    # ------------------------------------------------------------------
    def _reflow_text(det, box_img: np.ndarray) -> Optional[np.ndarray]:
        """box_img is already masked (non-text regions filled with bg)."""
        if box_img.size == 0:
            return None

        is_title = (det.label == "title")
        rw = det.x2 - det.x1
        rh = det.y2 - det.y1

        local_boxes = []
        for (x1, y1, x2, y2) in all_word_boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if det.x1 <= cx <= det.x2 and det.y1 <= cy <= det.y2:
                local_boxes.append((
                    x1 - det.x1, y1 - det.y1,
                    x2 - det.x1, y2 - det.y1,
                ))

        if not local_boxes:
            return None

        local_boxes = reflow_words.clamp_word_boxes(local_boxes)

        padding = 35 if is_title else 5
        local_boxes = [
            (
                max(0,  lx1 - padding),
                max(0,  ly1 - padding),
                min(rw, lx2 + padding),
                min(rh, ly2 + padding),
            )
            for (lx1, ly1, lx2, ly2) in local_boxes
        ]

        if is_title and len(local_boxes) > 1:
            extra = 20
            local_boxes = [(
                max(0,  min(b[0] for b in local_boxes) - extra),
                max(0,  min(b[1] for b in local_boxes) - extra),
                min(rw, max(b[2] for b in local_boxes) + extra),
                min(rh, max(b[3] for b in local_boxes) + extra),
            )]

        lines_raw  = line_grouping.group_words_into_lines(local_boxes)
        word_lines = reflow_words.words_to_wordlines(lines_raw)

        return reflow_words.create_page_word_reflow(
            word_lines,
            box_img,
            zoom_factor=zoom_factor,
            new_page_width=new_page_width,
            top_margin=0,
            bottom_margin=0,
            left_margin=left_margin,
            right_margin=right_margin,
            background_color=bg,
            is_title=is_title,
        )

    # ------------------------------------------------------------------
    # Helper: mask non-text regions out of a text crop
    # ------------------------------------------------------------------
    def _masked_crop(det) -> np.ndarray:
        crop = img_bgr[det.y1:det.y2, det.x1:det.x2].copy()
        for other in all_dets:
            if other.label in _TEXT_CLASSES:
                continue
            # AABB intersection in page coords
            ix1 = max(det.x1, other.x1)
            iy1 = max(det.y1, other.y1)
            ix2 = min(det.x2, other.x2)
            iy2 = min(det.y2, other.y2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            # Convert to local coords
            lx1 = max(0, ix1 - det.x1)
            ly1 = max(0, iy1 - det.y1)
            lx2 = min(crop.shape[1], ix2 - det.x1)
            ly2 = min(crop.shape[0], iy2 - det.y1)
            if lx2 > lx1 and ly2 > ly1:
                crop[ly1:ly2, lx1:lx2] = bg
        return crop

    # ------------------------------------------------------------------
    # Phase 1: build ordered work units (sequential — resolves gaps and
    # titled_block_title buffering which require positional context).
    # Each unit is a callable() -> (strip, gap_before, gap_after) | None.
    # titled pairs are kept as a single callable (sequential internally).
    # ------------------------------------------------------------------
    _LINE_H_PX = 25  # original-space pixels
    _max_nontext_gap = max(min_gap, int(_LINE_H_PX * zoom_factor * 2))

    def _last_placed_label(idx: int) -> Optional[str]:
        """Return label of last detection that was actually placed (skip titled_block_title / abandon)."""
        for i in range(idx - 1, -1, -1):
            if all_dets[i].label not in ("titled_block_title", "abandon"):
                return all_dets[i].label
        return None

    # work_units: list of callables, each returns (strip, gap_before, gap_after) | None
    work_units = []

    pending_title_det = None
    pending_title_gap = 0

    for idx, det in enumerate(all_dets):
        prev_y2 = all_dets[idx - 1].y2 if idx > 0 else 0
        next_y1 = all_dets[idx + 1].y1 if idx < len(all_dets) - 1 else page_h
        gap_before = max(min_gap, int((det.y1 - prev_y2) * zoom_factor))
        gap_after  = max(min_gap, int((next_y1 - det.y2) * zoom_factor))

        label = det.label

        if label in _TEXT_CLASSES:
            last_placed = _last_placed_label(idx)
            if last_placed in _NON_TEXT_LABELS:
                gap_before = min(gap_before, _max_nontext_gap)

        if label == "abandon":
            continue

        if label == "titled_block_title":
            last_placed = _last_placed_label(idx)
            if last_placed in _NON_TEXT_LABELS:
                gap_before = min(gap_before, _max_nontext_gap)
            pending_title_det = det
            pending_title_gap = gap_before
            continue

        is_text = label in _TEXT_CLASSES
        block_w = det.x2 - det.x1
        is_narrow = (
            label in ("plain text", "titled_block_body")
            and block_w < median_plain_w * 0.65
        )

        if not is_text or is_narrow:
            _det = det
            _gap_before = gap_before
            _gap_after = gap_after
            _ptd = pending_title_det
            _ptg = pending_title_gap
            pending_title_det = None
            pending_title_gap = 0

            def _make_nontext_unit(_d, _gb, _ga, _ptd, _ptg):
                def _run():
                    crop = img_bgr[_d.y1:_d.y2, _d.x1:_d.x2].copy()
                    actual_gap = _gb
                    if _d.label == "titled_block_body" and _ptd is not None:
                        title_img = img_bgr[_ptd.y1:_ptd.y2, _ptd.x1:_ptd.x2].copy()
                        t_h, t_w = title_img.shape[:2]
                        b_h, b_w = crop.shape[:2]
                        combined_w = max(t_w, b_w)
                        canvas_t = np.empty((t_h, combined_w, 3), dtype=np.uint8)
                        canvas_t[:] = bg
                        canvas_t[:, :t_w] = title_img
                        canvas_b = np.empty((b_h, combined_w, 3), dtype=np.uint8)
                        canvas_b[:] = bg
                        canvas_b[:, :b_w] = crop
                        crop = np.vstack([canvas_t, canvas_b])
                        actual_gap = _ptg
                    if crop.size == 0:
                        return None
                    return _zoom_crop(crop), actual_gap, _ga
                return _run

            work_units.append(_make_nontext_unit(_det, _gap_before, _gap_after, _ptd, _ptg))
            continue

        # ---- text block (not narrow) ----
        # titled_block_body (not narrow): emit pending title first as its own unit
        if label == "titled_block_body" and pending_title_det is not None:
            _ptd = pending_title_det
            _ptg = pending_title_gap
            _body_gap = max(min_gap, int((det.y1 - all_dets[idx - 1].y2) * zoom_factor))
            pending_title_det = None
            pending_title_gap = 0

            def _make_title_unit(_ptd, _ptg, _bga):
                def _run():
                    title_img = img_bgr[_ptd.y1:_ptd.y2, _ptd.x1:_ptd.x2].copy()
                    return _zoom_crop(title_img), _ptg, _bga
                return _run

            work_units.append(_make_title_unit(_ptd, _ptg, _body_gap))
            gap_before = _body_gap
        else:
            pending_title_det = None
            pending_title_gap = 0

        _det = det
        _gap_before = gap_before
        _gap_after = gap_after

        def _make_text_unit(_d, _gb, _ga):
            def _run():
                crop = _masked_crop(_d)
                strip = _reflow_text(_d, crop)
                if strip is None:
                    return None
                return strip, _gb, _ga
            return _run

        work_units.append(_make_text_unit(_det, _gap_before, _gap_after))

    # ------------------------------------------------------------------
    # Phase 2: execute work units in parallel, assemble canvas in order
    # ------------------------------------------------------------------
    canvas_h = max(int(page_h * zoom_factor * 2), 2000)
    canvas = np.empty((canvas_h, new_page_width, 3), dtype=np.uint8)
    canvas[:] = bg
    current_y = 0

    def _ensure_space(needed: int):
        nonlocal canvas, canvas_h
        if current_y + needed > canvas_h:
            new_h = max(current_y + needed + 1000, canvas_h + 2000)
            new_canvas = np.empty((new_h, new_page_width, 3), dtype=np.uint8)
            new_canvas[:] = bg
            new_canvas[:canvas_h] = canvas
            canvas = new_canvas
            canvas_h = new_h

    def _place(strip: np.ndarray, gap_before: int):
        nonlocal current_y
        sh = strip.shape[0]
        _ensure_space(gap_before + sh)
        current_y += gap_before
        canvas[current_y:current_y + sh] = strip
        current_y += sh

    last_gap_after = min_gap
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fn) for fn in work_units]
        for fut in futures:
            result = fut.result()  # preserves submission order
            if result is None:
                continue
            strip, gap_before, gap_after = result
            _place(strip, gap_before)
            last_gap_after = gap_after

    # Crop canvas to actual content (add bottom margin from last block to page bottom)
    current_y += last_gap_after
    final_h = max(1, current_y)
    result_bgr = canvas[:final_h]
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


# ---------------------------------------------------------------------------
# Flet app
# ---------------------------------------------------------------------------

async def main(page: ft.Page):
    page.title = "Doc Layout"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = ft.Padding(top=48, left=0, right=0, bottom=10)

    state = {
        "doc_path": None,
        "doc_type": None,               # "djvu" or "pdf"
        "pdf_doc": None,                # open pypdfium2.PdfDocument handle
        "page_index": 0,
        "total_pages": 0,
        "page_image": None,         # PIL Image — original render
        "net": None,                # DocLayout-YOLO model
        "doctr_net": None,          # fast_base ONNX model
        "lang_detector": None,      # LangDetector (CommonLingua + CRNN)
        "local_reflow_image": None,  # PIL Image — reflowed by local models
        "local_dets": None,         # cached layout detections (local)
        "local_word_boxes": None,   # cached word boxes (local)
        "server_reflow_image": None, # PIL Image — reflowed by server
        "server_dets": None,        # cached layout detections (server)
        "server_word_boxes": None,  # cached word boxes (server)
        "server_left_margin": None, # content-aware left margin from server
        "server_right_margin": None, # content-aware right margin from server
        "server_page_img": None,    # 300 DPI image used for server reflow
        "last_reflow_mode": None,   # "local" or "server" — for cycle_zoom
        "base_zoom": None,           # auto-computed from median word width
        "zoom_step": 0,             # number of ×1.2 increments on top of base_zoom
        "container_w": 0,
        "lang": "en",               # user-selected ISO 639-1 code
        "pdf_lock": asyncio.Lock(), # serializes pypdfium2 access (not thread-safe)
        "doctr_lock": asyncio.Lock(), # serializes cv2.dnn inference (not thread-safe)
        "server_url": "http://localhost:8000",  # remote server URL
        "skew_angle": 0.0,          # last skew angle returned by server
    }

    # Load persisted settings
    _settings = _load_settings()
    if "server_url" in _settings:
        state["server_url"] = str(_settings["server_url"])

    # ---- controls ----
    status_text = ft.Text("Loading models…", color=ft.Colors.GREY_400, size=13)

    _BLANK = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9'
    img_control = ft.Image(src=_BLANK, fit=ft.BoxFit.FILL)
    spinner = ft.ProgressRing(width=96, height=96, stroke_width=8, visible=False)

    img_container = ft.Container(
        content=ft.Stack(
            controls=[
                img_control,
                ft.Container(
                    content=spinner,
                    alignment=ft.Alignment(0, 0),
                ),
            ],
        ),
    )

    scroll_wrapper = ft.Container(
        content=ft.Column(
            controls=[img_container],
            scroll=ft.ScrollMode.ALWAYS,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
        expand=True,
    )

    page_label = ft.Text("", size=16, weight=ft.FontWeight.W_500)

    btn_prev = ft.IconButton(
        icon=ft.Icons.ARROW_BACK_IOS,
        on_click=lambda _: asyncio.ensure_future(navigate(-1)),
        disabled=True,
    )
    btn_next = ft.IconButton(
        icon=ft.Icons.ARROW_FORWARD_IOS,
        on_click=lambda _: asyncio.ensure_future(navigate(1)),
        disabled=True,
    )
    btn_open = ft.ElevatedButton(
        "Open file",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=lambda e: asyncio.ensure_future(open_file(e)),
    )
    btn_zoom_out = ft.IconButton(
        icon=ft.Icons.TEXT_DECREASE,
        tooltip="Zoom out (fewer words per line)",
        on_click=lambda _: asyncio.ensure_future(do_zoom(-1)),
        disabled=True,
    )
    txt_zoom = ft.Text("—", width=48, text_align=ft.TextAlign.CENTER)
    btn_zoom_in = ft.IconButton(
        icon=ft.Icons.TEXT_INCREASE,
        tooltip="Zoom in (more words per line)",
        on_click=lambda _: asyncio.ensure_future(do_zoom(+1)),
        disabled=True,
    )
    btn_fix_skew = ft.IconButton(
        icon=ft.Icons.STRAIGHTEN,
        tooltip="Fix skew",
        on_click=lambda _: asyncio.ensure_future(fix_skew()),
        disabled=True,
    )
    dd_lang = ft.Dropdown(
        value="en",
        width=160,
        options=[
            ft.DropdownOption(key=code, text=f"{name} ({code.upper()})")
            for code, name in LANGUAGES
        ],
        on_select=lambda e: _on_lang_select(e.control.value),
     )
    btn_reflow_local = ft.ElevatedButton(
        "Local",
        icon=ft.Icons.COMPUTER,
        disabled=True,
        on_click=lambda _: asyncio.ensure_future(run_reflow("local")),
    )
    btn_reflow_server = ft.ElevatedButton(
        "Server",
        icon=ft.Icons.CLOUD,
        disabled=not bool(state["server_url"]),
        on_click=lambda _: asyncio.ensure_future(run_reflow("server")),
    )
    tf_server_url = ft.TextField(
        value=state["server_url"],
        label="Server URL",
        width=260,
        on_submit=lambda e: _on_server_url_change(e.control.value),
        on_blur=lambda e: _on_server_url_change(e.control.value),
    )

    # ---- helpers ----
    def set_status(msg: str):
        status_text.value = msg
        status_text.update()

    def _on_lang_select(code: str):
        if code and code != state["lang"]:
            state["lang"] = code
            state["local_reflow_image"] = None
            state["server_reflow_image"] = None

    def _on_server_url_change(url: str):
        url = url.strip()
        if url != state["server_url"]:
            state["server_url"] = url
            btn_reflow_server.disabled = not bool(url)
            btn_reflow_server.update()
            _save_settings({"server_url": url})

    def refresh_image():
        # Show the most recently computed reflow image, or the original page
        reflow_img = None
        if state["last_reflow_mode"] == "server" and state["server_reflow_image"] is not None:
            reflow_img = state["server_reflow_image"]
        elif state["last_reflow_mode"] == "local" and state["local_reflow_image"] is not None:
            reflow_img = state["local_reflow_image"]
        img = reflow_img if reflow_img is not None else state["page_image"]
        if img is None:
            return
        w = state["container_w"]
        if w <= 0:
            return
        orig_w, orig_h = img.size
        display_w = min(int(w), orig_w)
        display_h = int(display_w * orig_h / orig_w)
        resized = img.resize((display_w, display_h), Image.LANCZOS) if display_w != orig_w else img
        img_control.src = pil_to_bytes(resized)
        img_control.width = display_w
        img_control.height = display_h
        img_container.update()

    def on_container_resize(e):
        w = int(e.width)
        if w <= 0:
            return
        old_w = state["container_w"]
        state["container_w"] = w
        if abs(w - old_w) > 20 and state["base_zoom"] is not None:
            # Width changed significantly (e.g. rotation) — invalidate reflow
            # images so they are re-rendered at the new width. Keep base_zoom,
            # zoom_step, dets and word_boxes — all still valid.
            state["local_reflow_image"] = None
            state["server_reflow_image"] = None
            mode = state["last_reflow_mode"]
            if mode is not None:
                asyncio.ensure_future(run_reflow(mode))
                return
        refresh_image()

    scroll_wrapper.on_size_change = on_container_resize

    # ---- model loading ----
    async def load_model_bg():
        await asyncio.sleep(0)
        try:
            models_d = _models_dir()
            onnx_path = models_d / "doclayout.onnx"
            doctr_path = models_d / "fast_base.onnx"
            loop = asyncio.get_event_loop()
            state["net"] = await loop.run_in_executor(
                None, lambda: inference.load_model(str(onnx_path))
            )
            set_status("Layout model ready. Loading word model…")
            state["doctr_net"] = await loop.run_in_executor(
                None, lambda: doctr_inference.load_model(str(doctr_path))
            )
            # Load language detector from downloaded models dir
            set_status("Loading language models…")
            try:
                state["lang_detector"] = await loop.run_in_executor(
                    None,
                    lambda: lang_detect.load_models(str(models_d)),
                )
                if state["lang_detector"] is None:
                    set_status("Models ready. Lang detector returned None.")
                else:
                    set_status("Models ready. Open a DjVu or PDF file.")
            except Exception as _e:
                set_status(f"Models ready. Lang load error: {_e}")
                return
            if state["page_image"] is not None:
                btn_reflow_local.disabled = False
                btn_reflow_local.update()
        except Exception as e:
            set_status(f"Model load failed: {e}\n{traceback.format_exc()}")

    # ---- page navigation ----
    async def load_page(index: int):
        if state["doc_path"] is None:
            return
        state["page_index"] = index
        state["local_reflow_image"] = None
        state["local_dets"] = None
        state["local_word_boxes"] = None
        state["server_reflow_image"] = None
        state["server_dets"] = None
        state["server_word_boxes"] = None
        state["server_left_margin"] = None
        state["server_right_margin"] = None
        state["server_page_img"] = None
        state["skew_angle"] = 0.0
        state["last_reflow_mode"] = None
        state["base_zoom"] = None
        # zoom_step intentionally NOT reset here — carries over across pages.
        # It is reset only when a new document is opened.
        btn_reflow_local.disabled = True
        btn_reflow_server.disabled = True
        btn_zoom_out.disabled = True
        btn_zoom_in.disabled = True
        btn_fix_skew.disabled = True
        page.update()
        set_status(f"Rendering page {index + 1}…")
        await asyncio.sleep(0)
        try:
            loop = asyncio.get_event_loop()
            if state["doc_type"] == "pdf":
                async with state["pdf_lock"]:
                    state["page_image"] = await loop.run_in_executor(
                        None, lambda: render_pdf_page(state["pdf_doc"], index)
                    )
            else:
                state["page_image"] = await loop.run_in_executor(
                    None, lambda: render_djvu_page(state["doc_path"], index)
                )
        except Exception as e:
            set_status(f"Error rendering page: {e}\n{traceback.format_exc()}")
            return
        page_label.value = f"Page {index + 1} / {state['total_pages']}"
        btn_prev.disabled = index == 0
        btn_next.disabled = index >= state["total_pages"] - 1
        models_ready = state["net"] is not None and state["doctr_net"] is not None
        btn_reflow_local.disabled = not models_ready
        btn_reflow_server.disabled = not bool(state["server_url"])
        btn_fix_skew.disabled = not (models_ready or bool(state["server_url"]))
        page.update()
        refresh_image()
        set_status("Ready.")

    async def navigate(delta: int):
        new_idx = state["page_index"] + delta
        if 0 <= new_idx < state["total_pages"]:
            await load_page(new_idx)

    async def detect_language_bg():
        """
        Sample pages from the middle of the book, run CRNN + CommonLingua,
        and auto-set the language dropdown.
        """
        detector = state.get("lang_detector")
        if detector is None:
            print("[detect_language_bg] lang_detector is None, skipping")
            set_status("Lang detector not loaded.")
            return
        total = state.get("total_pages", 0)
        if total == 0 or state["doc_path"] is None:
            return

        # Wait for doctr_net to be ready (it may still be loading)
        for _ in range(60):
            if state.get("doctr_net") is not None:
                break
            await asyncio.sleep(1.0)
        else:
            print("[detect_language_bg] doctr_net never became ready, skipping")
            return

        set_status("Detecting language…")
        await asyncio.sleep(0)

        # Pick up to 3 pages from the middle quarter of the book
        mid = total // 2
        quarter = max(1, total // 4)
        candidates = list(range(max(0, mid - quarter), min(total, mid + quarter)))
        step = max(1, len(candidates) // 3)
        sample_pages = candidates[::step][:3]

        loop = asyncio.get_event_loop()

        def _render_pages():
            images = []
            for pg_idx in sample_pages:
                try:
                    if state["doc_type"] == "pdf":
                        img = render_pdf_page(state["pdf_doc"], pg_idx)
                    else:
                        img = render_djvu_page(state["doc_path"], pg_idx)
                    if img is not None:
                        images.append(img)
                except Exception:
                    continue
            return images

        def _detect_words_single(img):
            doctr_net = state.get("doctr_net")
            if doctr_net is None:
                return None
            return doctr_inference.detect_words(doctr_net, img)

        try:
            async with state["pdf_lock"]:
                images = await loop.run_in_executor(None, _render_pages)
            # Detect words one page at a time, releasing the lock between pages
            # so run_reflow can acquire it without waiting for all 3 pages.
            word_boxes_list = []
            for img in images:
                try:
                    async with state["doctr_lock"]:
                        boxes = await loop.run_in_executor(None, lambda i=img: _detect_words_single(i))
                    if boxes is not None:
                        word_boxes_list.append(boxes)
                except Exception:
                    continue
            if word_boxes_list:
                lang_code, conf = detector.detect(word_boxes_list, images[:len(word_boxes_list)])
            else:
                lang_code, conf = None, 0.0
            if lang_code:
                state["lang"] = lang_code
                dd_lang.value = lang_code
                dd_lang.update()
                conf_pct = int(conf * 100)
                set_status(f"Language detected: {lang_code} ({conf_pct}%)")
            else:
                set_status("Ready.")
        except Exception as e:
            set_status(f"Lang detect error: {e}")

    # ---- reflow ----
    async def run_reflow(mode: str):
        """mode is "local" or "server"."""
        page_img = state["page_image"]
        if page_img is None:
            return

        dets_key   = f"{mode}_dets"
        boxes_key  = f"{mode}_word_boxes"
        image_key  = f"{mode}_reflow_image"
        btn        = btn_reflow_local if mode == "local" else btn_reflow_server

        # If cached reflow image exists, just display it
        if state[image_key] is not None:
            state["last_reflow_mode"] = mode
            btn_zoom_out.disabled = False
            btn_zoom_in.disabled = False
            page.update()
            refresh_image()
            return

        # Show spinner, disable the pressed button
        spinner.visible = True
        btn.disabled = True
        btn_zoom_out.disabled = True
        btn_zoom_in.disabled = True
        page.update()

        try:
            loop = asyncio.get_event_loop()

            if mode == "server":
                # ---- Server mode: send image to remote server ----
                if state[dets_key] is None:
                    set_status("Sending page to server…")
                    # Render at 300 DPI for server mode to match CLI quality
                    if state["doc_type"] == "pdf":
                        async with state["pdf_lock"]:
                            server_page_img = await loop.run_in_executor(
                                None, lambda: render_pdf_page(state["pdf_doc"], state["page_index"], dpi=300)
                            )
                    else:
                        server_page_img = await loop.run_in_executor(
                            None, lambda: render_djvu_page(state["doc_path"], state["page_index"], dpi=300)
                        )
                    img_bytes = io.BytesIO()
                    server_page_img.save(img_bytes, format="JPEG", quality=95)
                    img_bytes = img_bytes.getvalue()

                    new_page_width = max(state["container_w"], 300)

                    srv_result = await loop.run_in_executor(
                        None,
                        lambda: server_inference.analyze_page_bytes(
                            image_bytes=img_bytes,
                            filename="page.jpg",
                            server_url=state["server_url"],
                            page_width=new_page_width,
                            zoom_factor=2.5,
                            lang=state["lang"] if state["lang"] != "en" else None,
                            toc_algorithm="none",
                        ),
                    )

                    state["skew_angle"] = srv_result.skew_angle
                    state[dets_key]  = srv_result.dets
                    state[boxes_key] = srv_result.word_boxes
                    state["server_left_margin"]  = srv_result.left_margin
                    state["server_right_margin"] = srv_result.right_margin
                    state["server_page_img"] = server_page_img

                dets = state[dets_key]
                word_boxes = state[boxes_key]

            else:
                # ---- Local mode: run ONNX models on device ----
                if state[dets_key] is None:
                    set_status("Detecting layout…")
                    dets = await loop.run_in_executor(
                        None, lambda: inference.detect(state["net"], page_img)
                    )
                    state[dets_key] = dets
                else:
                    dets = state[dets_key]

                if state[boxes_key] is None:
                    set_status("Detecting words…")
                    async with state["doctr_lock"]:
                        word_boxes = await loop.run_in_executor(
                            None, lambda: doctr_inference.detect_words(state["doctr_net"], page_img)
                        )
                    state[boxes_key] = word_boxes
                else:
                    word_boxes = state[boxes_key]

            # Step 3: per-region reflow
            set_status("Reflowing…")
            # Compute base_zoom from actual word boxes every reflow.
            # Raw word widths are used directly — reflow_words applies
            # zoom_factor to those same widths, so no DPI conversion needed.
            # zoom_step is NOT reset here — carries user's manual preference.
            margin = max(10, int(20 / 2))
            available_w = max(state["container_w"], 300) - 2 * margin
            ws = [x2 - x1 for x1, y1, x2, y2 in word_boxes if x2 > x1]
            median_word_w = float(np.median(ws)) if ws else 50.0
            base = available_w / (median_word_w * 4.0)
            state["base_zoom"] = max(0.1, base)
            zoom_factor = state["base_zoom"] * (ZOOM_INCREMENT ** state["zoom_step"])
            txt_zoom.value = f"{zoom_factor:.2g}×"
            new_page_width = max(state["container_w"], 300)

            if mode == "server":
                _lm = state.get("server_left_margin")
                _rm = state.get("server_right_margin")
                _srv_img = state.get("server_page_img") or page_img
                reflow_img = await loop.run_in_executor(
                    None,
                    lambda: _do_reflow(dets, word_boxes, _srv_img, zoom_factor, new_page_width, state["lang"], _lm, _rm),
                )
            else:
                reflow_img = await loop.run_in_executor(
                    None,
                    lambda: _do_reflow(dets, word_boxes, page_img, zoom_factor, new_page_width, state["lang"]),
                )

            state[image_key] = reflow_img
            state["last_reflow_mode"] = mode
            btn_zoom_out.disabled = False
            btn_zoom_in.disabled = False
            set_status("Done.")

        except Exception as e:
            set_status(f"Reflow error: {e}\n{traceback.format_exc()}")
        finally:
            spinner.visible = False
            btn.disabled = False
            page.update()

        refresh_image()

    async def do_zoom(delta: int):
        state["zoom_step"] += delta
        zoom_factor = state["base_zoom"] * (ZOOM_INCREMENT ** state["zoom_step"])
        txt_zoom.value = f"{zoom_factor:.2g}×"
        txt_zoom.update()
        # Invalidate reflow image only — dets/word_boxes kept, no server call
        mode = state["last_reflow_mode"]
        if mode == "local":
            state["local_reflow_image"] = None
        elif mode == "server":
            state["server_reflow_image"] = None
        if mode is not None:
            await run_reflow(mode)

    async def fix_skew():
        page_img = state.get("page_image")
        if page_img is None:
            return
        btn_fix_skew.disabled = True
        btn_fix_skew.icon = ft.Icons.HOURGLASS_TOP
        btn_fix_skew.update()
        try:
            last_mode = state.get("last_reflow_mode")
            if last_mode == "server":
                # Use angle already returned by server
                angle = state.get("skew_angle", 0.0)
                if abs(angle) <= 0.1:
                    set_status("No significant skew detected")
                    return
                img_bgr = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2BGR),
                )
            else:
                dets = state.get("local_dets")
                if dets is None:
                    set_status("Run reflow first to detect layout")
                    return
                img_bgr = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2BGR),
                )
                angle = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: skew_detection.detect_skew_in_text_regions(img_bgr, dets),
                )
                if abs(angle) <= 0.1:
                    set_status("No significant skew detected")
                    return
            corrected_bgr = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: skew_detection.rotate_image(img_bgr, angle),
            )
            corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
            state["page_image"] = Image.fromarray(corrected_rgb)
            # Invalidate all caches for this page so reflow uses corrected image
            state["local_dets"] = None
            state["local_word_boxes"] = None
            state["local_reflow_image"] = None
            state["server_dets"] = None
            state["server_word_boxes"] = None
            state["server_reflow_image"] = None
            state["server_left_margin"] = None
            state["server_right_margin"] = None
            state["server_page_img"] = None
            state["last_reflow_mode"] = None
            set_status(f"Skew corrected ({angle:+.2f}°)")
        except Exception as exc:
            set_status(f"Skew fix failed: {exc}")
        finally:
            btn_fix_skew.disabled = False
            btn_fix_skew.icon = ft.Icons.STRAIGHTEN
            btn_fix_skew.update()

    # ---- file picker ----
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def open_file(_):
        files = await file_picker.pick_files(
            dialog_title="Open file",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["djvu", "djv", "pdf"],
        )
        if not files:
            return
        path = files[0].path
        if not path:
            set_status("Could not access file path.")
            return

        ext = Path(path).suffix.lower().lstrip(".")
        doc_type = "pdf" if ext == "pdf" else "djvu"

        # Close any previously open PDF document
        if state["pdf_doc"] is not None:
            try:
                state["pdf_doc"].close()
            except Exception:
                pass
            state["pdf_doc"] = None

        state["doc_path"] = path
        state["doc_type"] = doc_type
        state["page_index"] = 0
        state["zoom_step"] = 0   # reset zoom on new document
        set_status("Opening document…")
        await asyncio.sleep(0)
        try:
            loop = asyncio.get_event_loop()
            if doc_type == "pdf":
                import pypdfium2 as pdfium
                def _open_pdf():
                    doc = pdfium.PdfDocument(path)
                    return doc, len(doc)
                pdf_doc, total = await loop.run_in_executor(None, _open_pdf)
                state["pdf_doc"] = pdf_doc
                state["total_pages"] = total
            else:
                import djvu.decode as djvu_decode
                def _open_djvu():
                    ctx = djvu_decode.Context()
                    doc = ctx.new_document(djvu_decode.FileURI(path))
                    doc.decoding_job.wait()
                    return len(doc.pages)
                state["total_pages"] = await loop.run_in_executor(None, _open_djvu)
        except Exception as ex:
            set_status(f"Error opening file: {ex}\n{traceback.format_exc()}")
            return
        await load_page(0)
        # Detect language in background (samples middle pages)
        # asyncio.ensure_future(detect_language_bg())  # temporarily disabled

    # ---- layout ----
    # ---- normal UI layout ----
    normal_ui = ft.Column(
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        controls=[
            ft.Row([btn_open, btn_prev, page_label, btn_next], spacing=8),
            scroll_wrapper,
            ft.Row(
                [btn_reflow_local, btn_reflow_server,
                 btn_zoom_out, txt_zoom, btn_zoom_in,
                 btn_fix_skew],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            ft.Row(
                [dd_lang, tf_server_url],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=12,
            ),
            status_text,
        ],
        spacing=8,
    )

    # ---- download UI ----
    dl_title    = ft.Text("Flow Reader", size=24, weight=ft.FontWeight.BOLD)
    dl_label    = ft.Text("Preparing…")
    dl_bar      = ft.ProgressBar(value=0, width=300)
    dl_bytes    = ft.Text("")
    dl_error    = ft.Text("", color="red")
    dl_retry    = ft.ElevatedButton("Retry", visible=False)

    download_ui = ft.Column(
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[dl_title, dl_label, dl_bar, dl_bytes, dl_error, dl_retry],
        spacing=16,
    )

    async def do_download():
        dl_error.value = ""
        dl_retry.visible = False
        models_d = _models_dir()
        models_d.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()

        for name, (url, expected_size) in MODEL_URLS.items():
            dest = models_d / name
            mb = expected_size / 1024 / 1024
            dl_label.value = f"Downloading {name} ({mb:.0f} MB)…"
            dl_bar.value = 0
            dl_bytes.value = f"0 / {mb:.0f} MB"
            page.update()

            def _download(url=url, dest=dest, expected_size=expected_size,
                          name=name, mb=mb):
                req = urllib.request.urlopen(url, timeout=60)
                received = 0
                chunk = 65536
                with open(dest, "wb") as f:
                    while True:
                        data = req.read(chunk)
                        if not data:
                            break
                        f.write(data)
                        received += len(data)
                        # post progress back to event loop
                        asyncio.run_coroutine_threadsafe(
                            _update_progress(received, expected_size, mb),
                            loop,
                        )
                if dest.stat().st_size != expected_size:
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{name}: size mismatch "
                        f"(got {dest.stat().st_size if dest.exists() else 0}, "
                        f"expected {expected_size})"
                    )

            try:
                await loop.run_in_executor(None, _download)
            except Exception as e:
                dl_error.value = f"Download failed: {e}"
                dl_retry.visible = True
                page.update()
                return

        # All downloaded — switch to normal UI
        page.controls.clear()
        page.add(normal_ui)
        page.update()
        asyncio.ensure_future(load_model_bg())

    async def _update_progress(received: int, total: int, mb: float):
        dl_bar.value = received / total
        dl_bytes.value = f"{received/1024/1024:.1f} / {mb:.0f} MB"
        page.update()

    async def on_retry(_):
        asyncio.ensure_future(do_download())

    dl_retry.on_click = on_retry

    # ---- startup ----
    if _models_ok():
        page.add(normal_ui)
        asyncio.ensure_future(load_model_bg())
    else:
        page.add(download_ui)
        asyncio.ensure_future(do_download())


if __name__ == "__main__":
    ft.app(main)
