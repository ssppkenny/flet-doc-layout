# doc-layout

Android app for DjVu document viewing with automatic layout detection.
Built with [Flet](https://flet.dev) (Flutter/Python), running
[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) inference
via ONNX on-device.

---

## Features

- Open and navigate multi-page DjVu files
- Full-page rendering at screen width with vertical scroll
- One-tap layout detection — detects titles, text blocks, figures, tables,
  formulas and more
- Colour-coded bounding box overlay with confidence scores
- Runs entirely on-device (no network required after install)

---

## Repository layout

```
doc-layout/
├── src/
│   ├── main.py              # Flet UI — page rendering, navigation, detection overlay
│   ├── inference.py         # cv2.dnn ONNX wrapper + Python NMS
│   └── assets/
│       └── doclayout.onnx   # model file (not in git — see below)
├── export_onnx.py           # script to export .pt weights → patched ONNX
├── demo.sh                  # ADB script: launch app, open file, detect, record
├── pixi.toml                # pixi workspace (desktop dev env + build tasks)
└── pyproject.toml           # flet build config + Android dependencies
```

---

## Model

### What it is

The app uses **DocLayout-YOLO DocStructBench** — a YOLOv10-based model
trained on document layout analysis.  It detects 10 classes:

| id | class |
|----|-------|
| 0 | title |
| 1 | plain text |
| 2 | abandon |
| 3 | figure |
| 4 | figure_caption |
| 5 | table |
| 6 | table_caption |
| 7 | table_footnote |
| 8 | isolate_formula |
| 9 | formula_caption |

The model is **not stored in git** (75 MB).  You need to generate it once
and place it at `src/assets/doclayout.onnx`.

### Generating the ONNX file

You need a Python environment that has `doclayout-yolo==0.0.4` and `torch`
installed.  The easiest way is to use the companion `segmentation` pixi
workspace if you have it, or create a plain venv:

```bash
pip install doclayout-yolo==0.0.4
```

Then run:

```bash
python export_onnx.py
```

The script will:

1. Download the `.pt` weights from Hugging Face on first run
   (`juliozhao/DocLayout-YOLO-DocStructBench`, ~100 MB, cached in
   `~/.cache/huggingface/`).
2. Export to ONNX with `imgsz=1024, simplify=True`.
3. **Patch the ONNX graph** — the raw export includes a post-processing
   subgraph (TopK, NMS) that uses operators not supported by `cv2.dnn`.
   The script strips everything after the final Conv layer and transposes
   the output to `[1, 21504, 14]` (boxes × attributes), which `cv2.dnn`
   can run on Android.
4. Copy the result to `src/assets/doclayout.onnx`.

### ONNX model spec

| Property | Value |
|----------|-------|
| Input name | `images` |
| Input shape | `[1, 3, 1024, 1024]` |
| Input dtype | `float32`, normalised `[0, 1]`, RGB, NCHW |
| Output name | `output0` |
| Output shape | `[1, 21504, 14]` |
| Output layout | cols 0–3: `x1 y1 x2 y2` (letterbox coords); cols 4–13: raw class scores (pre-sigmoid) |

Post-processing (top-K filtering, sigmoid, threshold, NMS) is done in
Python in `src/inference.py`.

---

## Building the APK

### Prerequisites

- [pixi](https://prefix.dev/docs/pixi/overview) installed
- Android NDK + SDK (configured for Flet builds)
- The companion `mobile-forge` repo checked out at `../mobile-forge`
  with the following wheels already built in `../mobile-forge/dist/`:
  - `python_djvulibre-0.8.8-cp312-cp312-android_24_arm64_v8a.whl`
  - `flet_libdjvulibre-3.5.29-0-py3-none-android_24_arm64_v8a.whl`
  - `flet_libcpp_shared-27.3.13750724-0-py3-none-android_24_arm64_v8a.whl`
  - `flet_libjpeg-3.0.90-1-py3-none-android_24_arm64_v8a.whl`

  See the `mobile-forge` README for how to build these wheels.

- `src/assets/doclayout.onnx` present (see above).

### Build

```bash
pixi run build-apk
```

This runs `flet build apk --arch arm64-v8a --yes`.  The output APK is at
`build/apk/doc-layout.apk`.

### Install

```bash
adb install -r build/apk/doc-layout.apk
```

---

## Running the desktop app (for development)

```bash
pixi run python3 src/main.py
```

Requires `python-djvulibre` installed in the pixi env (desktop build of
the library, not the Android wheel).  The app auto-opens
`~/Downloads/books/dvurog.djvu` at page 10 on startup for quick testing —
remove the `auto_open()` call in `main.py` before building the final APK.

---

## Demo script

`demo.sh` automates a full demo on a connected Android device via ADB:

1. Pushes `dvurog.djvu` to `/sdcard/Download/` on the device
2. Starts a screen recording
3. Launches the app
4. Opens the DjVu file through the file picker
5. Navigates to page 4
6. Taps **Detect Layout** and waits for results
7. Stops the recording and pulls it to `/tmp/demo.mp4`

```bash
bash demo.sh
```

Requires `adb` in `PATH` and a device connected with USB debugging enabled.

---

## Architecture notes

### DjVu rendering

`render_djvu_page()` in `main.py` uses `python-djvulibre` (Cython bindings
for DjVuLibre).  The key detail: `job.render()` takes a `page_rect` that
defines the **output canvas size**, not the native page size.  Both
`page_rect` and `render_rect` must be `(0, 0, out_w, out_h)`:

```python
data = job.render(
    djvu_decode.RENDER_COLOR,
    (0, 0, out_w, out_h),   # page_rect  — output canvas
    (0, 0, out_w, out_h),   # render_rect — region to copy
    fmt,
)
```

Passing native dimensions as `page_rect` while using scaled dimensions as
`render_rect` causes only the top-left corner of the page to be rendered.

### Inference pipeline

1. Original PIL image is passed to `inference.detect()` (never the
   display-resized copy, to preserve detection accuracy).
2. Image is letterboxed to 1024×1024, converted to `float32 [0,1]` NCHW.
3. `cv2.dnn.readNetFromONNX` runs the forward pass.
4. Output `[1, 21504, 14]` is filtered: top-2000 by max class score,
   sigmoid applied, threshold 0.51, then per-class NMS (IoU 0.45).
5. Box coordinates are scaled back from letterbox space to original image
   pixels and returned as `Detection` dataclass instances.
