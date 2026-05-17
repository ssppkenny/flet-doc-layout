"""
server_inference.py — HTTP client for the OCR reflow server.

Sends a page image to POST /page and returns parsed results that can be
fed directly into _do_reflow() in main.py.

Usage:
    import server_inference
    result = server_inference.analyze_page(
        image_path="/path/to/page.png",
        server_url="http://192.168.1.10:8000",
        page_width=2000,
        zoom_factor=2.5,
        lang="ru",
        bin=False,
        toc_algorithm="layoutlm",
    )
    # result.dets        : List[Detection]
    # result.word_boxes  : List[Tuple[int,int,int,int]]
    # result.skew_angle  : float  (degrees, counter-clockwise)
    # result.is_toc      : bool
    # result.left_margin : int    (in page_width units)
    # result.right_margin: int
    # result.bg_color    : Tuple[int,int,int]  (R,G,B)
    # result.image_width : int    (skew-corrected image width in pixels)
    # result.image_height: int
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from inference import Detection


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ServerPageResult:
    dets: List[Detection]
    word_boxes: List[Tuple[int, int, int, int]]
    skew_angle: float
    is_toc: bool
    left_margin: int
    right_margin: int
    bg_color: Tuple[int, int, int]
    image_width: int
    image_height: int


# ---------------------------------------------------------------------------
# Block type → Detection class_id mapping
# (must match CLASS_NAMES in inference.py)
# ---------------------------------------------------------------------------

_BLOCK_TYPE_TO_CLASS_ID = {
    "title":              0,
    "plain text":         1,
    "abandon":            2,
    "figure":             3,
    "figure_caption":     4,
    "table":              5,
    "table_caption":      6,
    "table_footnote":     7,
    "isolate_formula":    8,
    "formula_caption":    9,
    # server-side composite types
    "titled_block_title": 0,   # treated as title
    "titled_block_body":  1,   # treated as plain text
    "figure_and_caption": 3,   # treated as figure
}


def _block_type_to_class_id(block_type: str) -> int:
    return _BLOCK_TYPE_TO_CLASS_ID.get(block_type, 1)


# ---------------------------------------------------------------------------
# Multipart POST (stdlib only — no httpx/requests dependency)
# ---------------------------------------------------------------------------

def _post_image(
    url: str,
    image_bytes: bytes,
    filename: str,
    timeout: int = 120,
) -> dict:
    boundary = "----ServerInferenceBoundary"
    ext = filename.rsplit(".", 1)[-1].lower()
    content_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_page(
    image_path: str,
    server_url: str = "http://localhost:8000",
    page_width: int = 2000,
    zoom_factor: float = 2.5,
    lang: Optional[str] = None,
    bin: bool = False,
    toc_algorithm: str = "layoutlm",
    timeout: int = 120,
) -> ServerPageResult:
    """
    POST the image to the server and return parsed results.

    Raises:
        urllib.error.URLError  — network error or server unreachable
        urllib.error.HTTPError — server returned 4xx/5xx
        ValueError             — response missing expected fields
    """
    path = Path(image_path)
    image_bytes = path.read_bytes()

    params: dict = {
        "page_width": page_width,
        "zoom_factor": zoom_factor,
        "bin": "true" if bin else "false",
        "toc_algorithm": toc_algorithm,
    }
    if lang:
        params["lang"] = lang

    query = urllib.parse.urlencode(params)
    endpoint = f"{server_url.rstrip('/')}/page?{query}"

    data = _post_image(endpoint, image_bytes, path.name, timeout=timeout)

    return _parse_response(data)


def analyze_page_bytes(
    image_bytes: bytes,
    filename: str = "page.png",
    server_url: str = "http://localhost:8000",
    page_width: int = 2000,
    zoom_factor: float = 2.5,
    lang: Optional[str] = None,
    bin: bool = False,
    toc_algorithm: str = "layoutlm",
    timeout: int = 120,
) -> ServerPageResult:
    """
    Same as analyze_page() but accepts raw image bytes instead of a file path.
    Useful when the image is already in memory (e.g. rendered PDF page).
    """
    params: dict = {
        "page_width": page_width,
        "zoom_factor": zoom_factor,
        "bin": "true" if bin else "false",
        "toc_algorithm": toc_algorithm,
    }
    if lang:
        params["lang"] = lang

    query = urllib.parse.urlencode(params)
    endpoint = f"{server_url.rstrip('/')}/page?{query}"

    data = _post_image(endpoint, image_bytes, filename, timeout=timeout)

    return _parse_response(data)


def ocr_page(
    image_bytes: bytes,
    filename: str = "page.jpg",
    server_url: str = "http://localhost:8000",
    no_pix2tex: bool = False,
    cache_key: str = "",
    timeout: int = 300,
) -> str:
    """
    POST image bytes to /ocr_page and return the HTML string.

    If cache_key is provided, the server stores the result under that key
    so it can later be fetched via GET /ocr_result/{cache_key}.

    Raises:
        urllib.error.URLError  — network error or server unreachable
        urllib.error.HTTPError — server returned 4xx/5xx
    """
    boundary = "----ServerInferenceBoundary"
    ext = filename.rsplit(".", 1)[-1].lower()
    content_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    params = urllib.parse.urlencode({
        "no_pix2tex": "true" if no_pix2tex else "false",
        "cache_key": cache_key,
    })
    endpoint = f"{server_url.rstrip('/')}/ocr_page?{params}"

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def ocr_page_stream(
    image_bytes: bytes,
    filename: str = "page.jpg",
    server_url: str = "http://localhost:8000",
    timeout: int = 30,
) -> tuple[str, str]:
    """POST image bytes to /ocr_page_stream and return (stream_url, token).

    The server starts OCR in the background and returns immediately.
    The caller should navigate a WebView to stream_url to see results
    appear progressively.

    Returns:
        (stream_url, token) — stream_url is the full URL to load in WebView.

    Raises:
        urllib.error.URLError  — network error or server unreachable
        urllib.error.HTTPError — server returned 4xx/5xx
        ValueError             — response missing expected fields
    """
    boundary = "----ServerInferenceBoundary"
    ext = filename.rsplit(".", 1)[-1].lower()
    content_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    endpoint = f"{server_url.rstrip('/')}/ocr_page_stream"
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    stream_url = data.get("stream_url")
    token = data.get("token")
    if not stream_url or not token:
        raise ValueError(f"Server response missing stream_url/token: {data}")
    return stream_url, token


def check_health(server_url: str = "http://localhost:8000", timeout: int = 5) -> bool:
    """Return True if the server is reachable and healthy."""
    try:
        url = f"{server_url.rstrip('/')}/health"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return body.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(data: dict) -> ServerPageResult:
    """Convert raw JSON dict from server into ServerPageResult."""
    required = [
        "image_width", "image_height", "background_color",
        "skew_angle", "is_toc", "left_margin", "right_margin", "blocks",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Server response missing fields: {missing}")

    dets: List[Detection] = []
    word_boxes: List[Tuple[int, int, int, int]] = []

    for block in data["blocks"]:
        bbox = block["bbox"]          # [xmin, ymin, xmax, ymax]
        block_type = block["block_type"]
        class_id = _block_type_to_class_id(block_type)

        det = Detection(
            label=block_type,
            class_id=class_id,
            conf=1.0,
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3]),
        )
        dets.append(det)

    # Prefer top-level word_boxes (all words on page, no line-grouping filter).
    # Fall back to walking blocks[].lines[].words for older server versions.
    if "word_boxes" in data:
        for box in data["word_boxes"]:
            word_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    else:
        seen_words: set = set()
        for block in data["blocks"]:
            for line in block.get("lines", []):
                for w in line.get("words", []):
                    box = (int(w["xmin"]), int(w["ymin"]), int(w["xmax"]), int(w["ymax"]))
                    if box not in seen_words:
                        seen_words.add(box)
                        word_boxes.append(box)

    bg = data["background_color"]
    bg_color = (int(bg[0]), int(bg[1]), int(bg[2]))

    return ServerPageResult(
        dets=dets,
        word_boxes=word_boxes,
        skew_angle=float(data["skew_angle"]),
        is_toc=bool(data["is_toc"]),
        left_margin=int(data["left_margin"]),
        right_margin=int(data["right_margin"]),
        bg_color=bg_color,
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
    )
