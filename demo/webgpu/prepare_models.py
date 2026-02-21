#!/usr/bin/env python3
"""Copy demo/models/*.onnx to demo/webgpu/models/ for the browser demo.
person_detector: onnxruntime-web에서 'array length mismatch' 방지를 위해 dynamic=False로 별도 export."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "demo" / "models"
DST = ROOT / "demo" / "webgpu" / "models"

DST.mkdir(parents=True, exist_ok=True)

# pose (external data .data 파일도 함께 복사 - onnx가 pose_3d.onnx.data 참조)
if (SRC / "pose_3d.onnx").exists():
    shutil.copy2(SRC / "pose_3d.onnx", DST / "pose_3d.onnx")
    if (SRC / "pose_3d.onnx.data").exists():
        shutil.copy2(SRC / "pose_3d.onnx.data", DST / "pose_3d.onnx.data")
    print("OK:", DST / "pose_3d.onnx")
else:
    print("Skip: pose_3d.onnx not found in demo/models/")

# person_detector: 브라우저용 dynamic=False export (onnxruntime-web 호환)
web_det = DST / "person_detector.onnx"
try:
    from ultralytics import YOLO
    print("Exporting person_detector for web (dynamic=False)...")
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, opset=14, simplify=True, dynamic=False)
    cwd_onnx = Path.cwd() / "yolov8n.onnx"
    if cwd_onnx.exists():
        shutil.move(str(cwd_onnx), str(web_det))
        print("OK:", web_det)
except Exception as e:
    if (SRC / "person_detector.onnx").exists():
        shutil.copy2(SRC / "person_detector.onnx", web_det)
        print("OK (copied from demo/models):", web_det)
        print("  Note: if browser shows 'array length mismatch', run: pip install ultralytics && python prepare_models.py")
    else:
        print("Skip: person_detector not available:", e)
