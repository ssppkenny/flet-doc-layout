"""
Export DocLayout-YOLO .pt weights to ONNX format.

Run this from the segmentation pixi environment which has doclayout-yolo==0.0.4 and torch:

    cd /home/sergey/code/python/doc-layout
    pixi run -e default --manifest-path ../segmentation/pixi.toml python export_onnx.py

The output ONNX file will be copied to src/assets/doclayout.onnx.
"""

import shutil
from pathlib import Path

from doclayout_yolo import YOLOv10

PT_PATH = Path.home() / (
    ".cache/huggingface/hub/"
    "models--juliozhao--DocLayout-YOLO-DocStructBench/"
    "snapshots/8c3299a30b8ff29a1503c4431b035b93220f7b11/"
    "doclayout_yolo_docstructbench_imgsz1024.pt"
)

OUT_DIR = Path(__file__).parent / "src" / "assets"
OUT_PATH = OUT_DIR / "doclayout.onnx"

print(f"Loading model from {PT_PATH} ...")
model = YOLOv10(str(PT_PATH))

print("Exporting to ONNX (imgsz=1024, simplify=True) ...")
exported = model.export(format="onnx", imgsz=1024, simplify=True)
# exported is the path to the generated .onnx file (next to the .pt)
exported = Path(exported)
print(f"Exported to {exported}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(exported, OUT_PATH)
print(f"Copied to {OUT_PATH}")
