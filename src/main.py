"""
doc-layout: DjVu/PDF viewer with word-level reflow.
"""

from __future__ import annotations

import asyncio
import io
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import flet as ft
from PIL import Image

import inference
import doctr_inference
import line_grouping
import reflow_words

ASSETS_DIR = Path(__file__).parent / "assets"
ONNX_PATH       = ASSETS_DIR / "doclayout.onnx"
DOCTR_ONNX_PATH = ASSETS_DIR / "fast_base.onnx"

_TEXT_CLASSES    = {"plain text", "title", "titled_block_body"}
_NON_TEXT_LABELS = {
    "figure", "figure_and_caption", "figure_caption",
    "table", "table_and_caption", "table_caption", "table_footnote",
    "isolate_formula", "isolate_formula_and_caption", "formula_caption",
}
ZOOM_STEPS = [2.0, 1.5, 1.0]

LANGUAGES = [
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
]


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
    # Main sequential pass (titled_block_title is stateful)
    # ------------------------------------------------------------------
    # Allocate a tall canvas; we'll crop at the end
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

    # Single line height estimate at 150dpi (12pt body text ≈ 25px).
    # Used to cap the gap after non-text blocks (figure/table/formula → text).
    _LINE_H_PX = 25  # original-space pixels
    _max_nontext_gap = max(min_gap, int(_LINE_H_PX * zoom_factor * 2))

    def _last_placed_label(idx: int) -> Optional[str]:
        """Return label of last detection that was actually placed (skip titled_block_title / abandon)."""
        for i in range(idx - 1, -1, -1):
            if all_dets[i].label not in ("titled_block_title", "abandon"):
                return all_dets[i].label
        return None

    pending_title_img: Optional[np.ndarray] = None
    pending_title_gap: int = 0

    for idx, det in enumerate(all_dets):
        prev_y2 = all_dets[idx - 1].y2 if idx > 0 else 0
        next_y1 = all_dets[idx + 1].y1 if idx < len(all_dets) - 1 else page_h
        gap_before = max(min_gap, int((det.y1 - prev_y2) * zoom_factor))
        gap_after  = max(min_gap, int((next_y1 - det.y2) * zoom_factor))

        label = det.label

        # Cap gap_before when transitioning from a non-text block to a text block
        # (direct: figure→text, or indirect: figure→titled_block_title→titled_block_body)
        if label in _TEXT_CLASSES:
            last_placed = _last_placed_label(idx)
            if last_placed in _NON_TEXT_LABELS:
                gap_before = min(gap_before, _max_nontext_gap)

        # Skip noise
        if label == "abandon":
            continue

        # ---- titled_block_title: buffer and skip ----
        if label == "titled_block_title":
            pending_title_img = img_bgr[det.y1:det.y2, det.x1:det.x2].copy()
            # Also cap the buffered gap using the same rule
            last_placed = _last_placed_label(idx)
            if last_placed in _NON_TEXT_LABELS:
                gap_before = min(gap_before, _max_nontext_gap)
            pending_title_gap = gap_before
            continue

        # ---- non-text blocks (and titled_block_body narrow path) ----
        is_text = label in _TEXT_CLASSES
        block_w = det.x2 - det.x1
        is_narrow = (
            label in ("plain text", "titled_block_body")
            and block_w < median_plain_w * 0.65
        )

        if not is_text or is_narrow:
            crop = img_bgr[det.y1:det.y2, det.x1:det.x2].copy()

            if label == "titled_block_body" and pending_title_img is not None:
                # Stack title on top of body, zoom as one image
                t_h, t_w = pending_title_img.shape[:2]
                b_h, b_w = crop.shape[:2]
                combined_w = max(t_w, b_w)
                canvas_t = np.empty((t_h, combined_w, 3), dtype=np.uint8)
                canvas_t[:] = bg
                canvas_t[:, :t_w] = pending_title_img
                canvas_b = np.empty((b_h, combined_w, 3), dtype=np.uint8)
                canvas_b[:] = bg
                canvas_b[:, :b_w] = crop
                crop = np.vstack([canvas_t, canvas_b])
                gap_before = pending_title_gap
                pending_title_img = None
                pending_title_gap = 0

            if crop.size == 0:
                continue

            strip = _zoom_crop(crop)
            _place(strip, gap_before)
            continue

        # ---- text block (not narrow) ----

        # titled_block_body (not narrow): first emit pending title as zoomed crop
        if label == "titled_block_body" and pending_title_img is not None:
            title_strip = _zoom_crop(pending_title_img)
            _place(title_strip, pending_title_gap)
            # gap between title strip and body is the original title→body gap
            gap_before = max(min_gap, int((det.y1 - all_dets[idx - 1].y2) * zoom_factor))
            pending_title_img = None
            pending_title_gap = 0

        crop = _masked_crop(det)
        strip = _reflow_text(det, crop)
        if strip is None:
            continue
        _place(strip, gap_before)

    # Crop canvas to actual content (add bottom margin from last block to page bottom)
    current_y += gap_after  # gap_after of last processed block
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
        "reflow_image": None,       # PIL Image — reflowed (None until computed)
        "reflow_dets": None,        # cached layout detections for current page
        "reflow_word_boxes": None,  # cached full-page word boxes for current page
        "show_reflow": False,
        "zoom_level": 0,            # index into ZOOM_STEPS
        "container_w": 0,
        "lang": "en",               # user-selected ISO 639-1 code
    }

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
                    expand=True,
                ),
            ],
            expand=True,
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
    btn_reflow = ft.ElevatedButton(
        "Reflow",
        icon=ft.Icons.WRAP_TEXT,
        on_click=lambda _: asyncio.ensure_future(run_reflow()),
        disabled=True,
    )
    btn_zoom = ft.ElevatedButton(
        f"{ZOOM_STEPS[0]:g}×",
        icon=ft.Icons.ZOOM_IN,
        on_click=lambda _: asyncio.ensure_future(cycle_zoom()),
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

    # ---- helpers ----
    def set_status(msg: str):
        status_text.value = msg
        status_text.update()

    def _on_lang_select(code: str):
        if code and code != state["lang"]:
            state["lang"] = code
            state["reflow_image"] = None
            state["show_reflow"] = False

    def refresh_image():
        if state["show_reflow"] and state["reflow_image"] is not None:
            img = state["reflow_image"]
        else:
            img = state["page_image"]
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
        state["container_w"] = w
        refresh_image()

    img_container.on_size_change = on_container_resize

    # ---- model loading ----
    async def load_model_bg():
        await asyncio.sleep(0)
        try:
            loop = asyncio.get_event_loop()
            state["net"] = await loop.run_in_executor(
                None, lambda: inference.load_model(str(ONNX_PATH))
            )
            set_status("Layout model ready. Loading word model…")
            state["doctr_net"] = await loop.run_in_executor(
                None, lambda: doctr_inference.load_model(str(DOCTR_ONNX_PATH))
            )
            set_status("Models ready. Open a DjVu or PDF file.")
            if state["page_image"] is not None:
                btn_reflow.disabled = False
                btn_reflow.update()
        except Exception as e:
            set_status(f"Model load failed: {e}\n{traceback.format_exc()}")

    # ---- page navigation ----
    async def load_page(index: int):
        if state["doc_path"] is None:
            return
        state["page_index"] = index
        state["reflow_image"] = None
        state["reflow_dets"] = None
        state["reflow_word_boxes"] = None
        state["show_reflow"] = False
        state["zoom_level"] = 0
        btn_reflow.text = "Reflow"
        btn_reflow.disabled = True
        btn_zoom.disabled = True
        btn_zoom.text = f"{ZOOM_STEPS[0]:g}×"
        page.update()
        set_status(f"Rendering page {index + 1}…")
        await asyncio.sleep(0)
        try:
            loop = asyncio.get_event_loop()
            if state["doc_type"] == "pdf":
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
        btn_reflow.disabled = not models_ready
        page.update()
        refresh_image()
        set_status("Ready.")

    async def navigate(delta: int):
        new_idx = state["page_index"] + delta
        if 0 <= new_idx < state["total_pages"]:
            await load_page(new_idx)

    # ---- reflow ----
    async def run_reflow():
        # Toggle back to original if currently showing reflow
        if state["show_reflow"]:
            state["show_reflow"] = False
            btn_reflow.text = "Reflow"
            btn_zoom.disabled = True
            page.update()
            refresh_image()
            return

        # If cached reflow exists at current zoom, just show it
        if state["reflow_image"] is not None:
            state["show_reflow"] = True
            btn_reflow.text = "Original"
            btn_zoom.disabled = False
            page.update()
            refresh_image()
            return

        page_img = state["page_image"]
        if page_img is None:
            return

        # Show spinner, disable buttons
        spinner.visible = True
        page.update()
        btn_reflow.disabled = True
        btn_zoom.disabled = True
        page.update()

        try:
            loop = asyncio.get_event_loop()

            # Step 1: layout detection (cached across zoom changes)
            if state["reflow_dets"] is None:
                set_status("Detecting layout…")
                dets = await loop.run_in_executor(
                    None, lambda: inference.detect(state["net"], page_img)
                )
                state["reflow_dets"] = dets
            else:
                dets = state["reflow_dets"]

            # Step 2: word detection — one call on full page (cached across zoom changes)
            if state["reflow_word_boxes"] is None:
                set_status("Detecting words…")
                word_boxes = await loop.run_in_executor(
                    None, lambda: doctr_inference.detect_words(state["doctr_net"], page_img)
                )
                state["reflow_word_boxes"] = word_boxes
            else:
                word_boxes = state["reflow_word_boxes"]

            # Step 3: per-region reflow (parallel inside _do_reflow)
            set_status("Reflowing…")
            zoom_factor = ZOOM_STEPS[state["zoom_level"]]
            new_page_width = max(state["container_w"], 300)

            reflow_img = await loop.run_in_executor(
                None,
                lambda: _do_reflow(dets, word_boxes, page_img, zoom_factor, new_page_width, state["lang"]),
            )
            state["reflow_image"] = reflow_img
            state["show_reflow"] = True
            btn_reflow.text = "Original"
            btn_zoom.disabled = False
            set_status("Done.")

        except Exception as e:
            set_status(f"Reflow error: {e}\n{traceback.format_exc()}")
        finally:
            spinner.visible = False
            btn_reflow.disabled = False
            page.update()

        refresh_image()

    async def cycle_zoom():
        state["zoom_level"] = (state["zoom_level"] + 1) % len(ZOOM_STEPS)
        z = ZOOM_STEPS[state["zoom_level"]]
        btn_zoom.text = f"{z:g}×"
        btn_zoom.update()
        # Invalidate reflow cache (dets and word_boxes are kept)
        state["reflow_image"] = None
        state["show_reflow"] = False
        await run_reflow()

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

    # ---- layout ----
    page.add(
        ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                ft.Row([btn_open, btn_prev, page_label, btn_next], spacing=8),
                img_container,
                ft.Row(
                    [btn_reflow, btn_zoom, dd_lang],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=12,
                ),
                status_text,
            ],
            spacing=8,
        )
    )

    asyncio.ensure_future(load_model_bg())


if __name__ == "__main__":
    ft.app(main)
