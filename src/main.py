"""
doc-layout: DjVu viewer with DocLayout-YOLO layout detection.
"""

from __future__ import annotations

import asyncio
import io
import traceback
from pathlib import Path
from typing import List

import flet as ft
from PIL import Image, ImageDraw, ImageFont

import inference
from inference import Detection

ASSETS_DIR = Path(__file__).parent / "assets"
ONNX_PATH = ASSETS_DIR / "doclayout.onnx"


def pil_to_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def draw_detections(img: Image.Image, detections: List[Detection]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    lw = max(2, w // 300)
    for det in detections:
        r, g, b = det.color_rgb
        draw.rectangle([det.x1, det.y1, det.x2, det.y2],
                       fill=(r, g, b, 40), outline=(r, g, b, 220), width=lw)
        label = f"{det.label} {det.conf:.0%}"
        tx, ty = det.x1 + lw, det.y1 + lw
        try:
            font = ImageFont.load_default(size=max(12, w // 60))
        except TypeError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=(r, g, b, 180))
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
    return out


def render_djvu_page(doc_path: str, page_index: int, dpi: int = 150) -> Image.Image:
    import djvu.decode as djvu_decode
    ctx = djvu_decode.Context()
    doc = ctx.new_document(djvu_decode.FileURI(doc_path))
    doc.decoding_job.wait()
    page = doc.pages[page_index]
    job = page.decode(wait=True)
    native_w, native_h = job.size
    native_dpi = page.dpi
    scale = dpi / native_dpi
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


async def main(page: ft.Page):
    page.title = "Doc Layout"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = ft.padding.only(top=48, left=0, right=0, bottom=10)

    state = {
        "doc_path": None,
        "page_index": 0,
        "total_pages": 0,
        "page_image": None,
        "detections": [],
        "net": None,
    }

    # ---- controls ----
    status_text = ft.Text("Loading model…", color=ft.Colors.GREY_400, size=13)

    _BLANK = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\x1b\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9'
    img_control = ft.Image(src=_BLANK, fit=ft.BoxFit.FILL)
    img_container = ft.Container(content=img_control, expand=True)

    page_label = ft.Text("", size=16, weight=ft.FontWeight.W_500)

    btn_prev = ft.IconButton(icon=ft.Icons.ARROW_BACK_IOS,
                             on_click=lambda _: asyncio.ensure_future(navigate(-1)),
                             disabled=True)
    btn_next = ft.IconButton(icon=ft.Icons.ARROW_FORWARD_IOS,
                             on_click=lambda _: asyncio.ensure_future(navigate(1)),
                             disabled=True)

    btn_open = ft.ElevatedButton("Open DjVu", icon=ft.Icons.FOLDER_OPEN,
                                 on_click=lambda e: asyncio.ensure_future(open_file(e)))
    btn_detect = ft.ElevatedButton("Detect Layout", icon=ft.Icons.SEARCH,
                                   on_click=lambda _: asyncio.ensure_future(run_detection()),
                                   disabled=True)

    legend_row = ft.Row(wrap=True, visible=False)

    # ---- helpers ----
    def set_status(msg: str):
        status_text.value = msg
        status_text.update()

    def refresh_image():
        img = state["page_image"]
        if img is None:
            return
        w = state.get("container_w", 0)
        if w <= 0:
            return  # wait for on_size_change
        if state["detections"]:
            img = draw_detections(img, state["detections"])
        orig_w, orig_h = img.size
        display_w = min(int(w), orig_w)  # never upscale beyond original
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


    async def load_model_bg():
        await asyncio.sleep(0)
        try:
            loop = asyncio.get_event_loop()
            state["net"] = await loop.run_in_executor(
                None, lambda: inference.load_model(str(ONNX_PATH))
            )
            set_status("Model ready. Open a DjVu file.")
            if state["page_image"] is not None:
                btn_detect.disabled = False
                page.update()
        except Exception as e:
            set_status(f"Model load failed: {e}\n{traceback.format_exc()}")

    async def load_page(index: int):
        if state["doc_path"] is None:
            return
        state["page_index"] = index
        state["detections"] = []
        legend_row.visible = False
        legend_row.update()
        set_status(f"Rendering page {index + 1}…")
        await asyncio.sleep(0)
        try:
            loop = asyncio.get_event_loop()
            state["page_image"] = await loop.run_in_executor(
                None, lambda: render_djvu_page(state["doc_path"], index)
            )
        except Exception as e:
            set_status(f"Error rendering page: {e}\n{traceback.format_exc()}")
            return
        page_label.value = f"Page {index + 1} / {state['total_pages']}"
        page_label.update()
        btn_prev.disabled = index == 0
        btn_next.disabled = index >= state["total_pages"] - 1
        btn_detect.disabled = state["net"] is None
        page.update()
        refresh_image()

    async def navigate(delta: int):
        new_idx = state["page_index"] + delta
        if 0 <= new_idx < state["total_pages"]:
            await load_page(new_idx)

    async def run_detection():
        if state["page_image"] is None or state["net"] is None:
            return
        set_status("Running layout detection…")
        btn_detect.disabled = True
        page.update()
        try:
            loop = asyncio.get_event_loop()
            dets = await loop.run_in_executor(
                None, lambda: inference.detect(state["net"], state["page_image"])
            )
            state["detections"] = dets
            _build_legend(dets)
            set_status(f"Found {len(dets)} region(s).")
        except Exception as e:
            set_status(f"Detection error: {e}\n{traceback.format_exc()}")
        finally:
            btn_detect.disabled = False
        refresh_image()

    def _build_legend(detections: List[Detection]):
        seen = {}
        for d in detections:
            if d.class_id not in seen:
                seen[d.class_id] = d
        legend_row.controls.clear()
        for d in seen.values():
            r, g, b = d.color_rgb
            legend_row.controls.append(
                ft.Container(
                    content=ft.Text(d.label, size=12, color=ft.Colors.WHITE),
                    bgcolor=f"#{r:02x}{g:02x}{b:02x}",
                    border_radius=12,
                    padding=ft.padding.symmetric(horizontal=8, vertical=4),
                    margin=ft.margin.only(right=4, bottom=4),
                )
            )
        legend_row.visible = bool(legend_row.controls)
        page.update()

    # ---- file picker ----
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def open_file(_):
        files = await file_picker.pick_files(
            dialog_title="Open DjVu file",
            allowed_extensions=["djvu", "djv"],
        )
        if not files:
            return
        path = files[0].path
        if not path:
            set_status("Could not access file path.")
            return
        state["doc_path"] = path
        state["page_index"] = 0
        state["detections"] = []
        set_status("Opening document…")
        await asyncio.sleep(0)
        try:
            import djvu.decode as djvu_decode
            loop = asyncio.get_event_loop()
            def _open():
                ctx = djvu_decode.Context()
                doc = ctx.new_document(djvu_decode.FileURI(path))
                doc.decoding_job.wait()
                return len(doc.pages)
            state["total_pages"] = await loop.run_in_executor(None, _open)
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
                ft.Row([btn_detect], alignment=ft.MainAxisAlignment.CENTER),
                legend_row,
                status_text,
            ],
            spacing=8,
        )
    )

    async def auto_open():
        await asyncio.sleep(0.5)
        test_path = "/home/sergey/Downloads/books/dvurog.djvu"
        state["doc_path"] = test_path
        import djvu.decode as djvu_decode
        loop = asyncio.get_event_loop()
        def _open():
            ctx = djvu_decode.Context()
            doc = ctx.new_document(djvu_decode.FileURI(test_path))
            doc.decoding_job.wait()
            return len(doc.pages)
        state["total_pages"] = await loop.run_in_executor(None, _open)
        await load_page(10)

    asyncio.ensure_future(load_model_bg())
    asyncio.ensure_future(auto_open())


if __name__ == "__main__":
    ft.app(main)
