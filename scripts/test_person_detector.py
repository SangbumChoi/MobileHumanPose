#!/usr/bin/env python3
"""
Person detector 검증: input.jpg 기준
1) PyTorch(Ultralytics) detection
2) ONNX 변환 및 ONNX inference
3) PyTorch vs ONNX raw 출력 값 비교
4) demo 최종 테스트
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGE = ROOT / "assets" / "input.jpg"
DEMO_MODELS = ROOT / "demo" / "models"
PERSON_DETECTOR_ONNX = DEMO_MODELS / "person_detector.onnx"


def load_image():
    try:
        import cv2
    except ImportError:
        print("ERROR: pip install opencv-python")
        sys.exit(1)
    if not INPUT_IMAGE.exists():
        print(f"ERROR: {INPUT_IMAGE} not found")
        sys.exit(1)
    img = cv2.imread(str(INPUT_IMAGE))
    if img is None:
        print(f"ERROR: Failed to load {INPUT_IMAGE}")
        sys.exit(1)
    print(f"[OK] Loaded {INPUT_IMAGE} shape {img.shape}")
    return img


def step1_pytorch_detection(frame_bgr):
    """1. PyTorch(Ultralytics) person detector 테스트."""
    print("\n=== 1. PyTorch (Ultralytics) person detector ===")
    try:
        from src.box_detector.detection import get_bboxes_ultralytics
    except ImportError:
        sys.path.insert(0, str(ROOT))
        from src.box_detector.detection import get_bboxes_ultralytics
    boxes = get_bboxes_ultralytics(frame_bgr, conf_thres=0.5)
    if boxes is None:
        print("FAIL: Ultralytics not installed. pip install ultralytics")
        return None
    print(f"  bboxes (person): {len(boxes)}")
    for i, b in enumerate(boxes):
        print(f"    [{i}] x={b[0]:.1f} y={b[1]:.1f} w={b[2]:.1f} h={b[3]:.1f}")
    return boxes


def preprocess_640(frame_bgr):
    """ONNX/demo와 동일한 전처리: letterbox 640x640, 0~1."""
    import numpy as np
    from PIL import Image
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    w, h = img.size
    target = 640
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    pad_w = (target - nw) / 2
    pad_h = (target - nh) / 2
    padded = np.zeros((target, target, 3), dtype=np.uint8)
    padded[int(pad_h) : int(pad_h) + nh, int(pad_w) : int(pad_w) + nw] = np.array(img)
    inp = padded.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[np.newaxis]
    return inp, scale, pad_w, pad_h, w, h


def step2_onnx_export_and_run(frame_bgr):
    """2. ONNX 변환 후 inference."""
    print("\n=== 2. ONNX person_detector ===")
    if not PERSON_DETECTOR_ONNX.exists():
        print("  Exporting person_detector.onnx...")
        try:
            from ultralytics import YOLO
        except ImportError:
            print("  FAIL: pip install ultralytics")
            return None, None
        DEMO_MODELS.mkdir(parents=True, exist_ok=True)
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640, opset=14, simplify=True, dynamic=True)
        import shutil
        cwd_onnx = Path.cwd() / "yolov8n.onnx"
        if cwd_onnx.exists():
            shutil.move(str(cwd_onnx), str(PERSON_DETECTOR_ONNX))
        print("  Saved", PERSON_DETECTOR_ONNX)
    if not PERSON_DETECTOR_ONNX.exists():
        print("  FAIL: ONNX file not found")
        return None, None

    import onnxruntime as ort
    import numpy as np
    inp, scale, pad_w, pad_h, img_w, img_h = preprocess_640(frame_bgr)
    session = ort.InferenceSession(str(PERSON_DETECTOR_ONNX), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out = session.run(None, {in_name: inp})[0]
    # Postprocess
    if out.shape[1] == 84:
        out_t = out[0].T
        scores = out_t[:, 4:].max(axis=1)
        class_ids = out_t[:, 4:].argmax(axis=1)
        boxes_raw = out_t[:, :4]
        keep = (class_ids == 0) & (scores >= 0.5)
        boxes_raw = boxes_raw[keep]
        scores = scores[keep]
        boxes = []
        for b in boxes_raw:
            cx, cy, bw, bh = b
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            boxes.append([max(0, x1), max(0, y1), bw / scale, bh / scale])
        # NMS (demo와 동일)
        if len(boxes) > 1:
            try:
                import cv2
                idx = cv2.dnn.NMSBoxes(boxes, scores.tolist(), 0.5, 0.45)
                idx = np.array(idx).flatten()
                boxes = [boxes[i] for i in idx]
            except Exception:
                pass
        boxes = boxes[:10]
        print(f"  bboxes (person): {len(boxes)}")
        for i, b in enumerate(boxes):
            print(f"    [{i}] x={b[0]:.1f} y={b[1]:.1f} w={b[2]:.1f} h={b[3]:.1f}")
        return out, boxes
    print("  FAIL: unexpected output shape", out.shape)
    return None, None


def step3_compare_values(frame_bgr):
    """3. 동일 입력에 대해 PyTorch raw vs ONNX raw 값 비교."""
    print("\n=== 3. PyTorch vs ONNX raw 출력 값 비교 ===")
    inp, scale, pad_w, pad_h, img_w, img_h = preprocess_640(frame_bgr)
    import numpy as np

    # ONNX
    if not PERSON_DETECTOR_ONNX.exists():
        print("  SKIP: person_detector.onnx not found")
        return
    import onnxruntime as ort
    session = ort.InferenceSession(str(PERSON_DETECTOR_ONNX), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_onnx = session.run(None, {in_name: inp})[0]

    # PyTorch 같은 입력으로 forward
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("  SKIP: torch/ultralytics not available")
        return
    model = YOLO("yolov8n.pt")
    inp_pt = torch.from_numpy(inp).float()
    with torch.no_grad():
        out_pt = model.model(inp_pt)
    # YOLOv8: out_pt can be tuple; we need (1, 84, 8400)
    if isinstance(out_pt, (list, tuple)):
        out_pt = out_pt[0]
    out_pt_np = out_pt.cpu().numpy()

    # Shape
    print(f"  ONNX shape: {out_onnx.shape}, PyTorch shape: {out_pt_np.shape}")
    if out_onnx.shape != out_pt_np.shape:
        print("  WARN: shape mismatch")
        return
    diff = np.abs(out_onnx.astype(np.float64) - out_pt_np.astype(np.float64))
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    print(f"  max |ONNX - PyTorch|: {max_diff:.6f}")
    print(f"  mean |ONNX - PyTorch|: {mean_diff:.6f}")
    if max_diff < 1e-4:
        print("  OK: 값 차이 없음 (동일)")
    elif max_diff < 1e-2:
        print("  OK: 미세 차이 (수치 오차 범위)")
    else:
        print("  WARN: 차이 있음 (ONNX 변환/연산 확인)")


def step4_final_demo(frame_bgr):
    """4. demo/demo.py 최종 테스트 (이미지 한 장)."""
    print("\n=== 4. 최종 테스트 (demo) ===")
    pose_onnx = DEMO_MODELS / "pose_3d.onnx"
    if not pose_onnx.exists():
        print("  SKIP: demo/models/pose_3d.onnx not found. Run export first.")
        return
    sys.path.insert(0, str(ROOT))
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "demo.demo", "-m", str(pose_onnx), "-i", str(INPUT_IMAGE), "--headless"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode == 0:
        print("  OK: demo ran successfully")
        if r.stdout:
            for line in r.stdout.strip().split("\n")[-5:]:
                print("   ", line)
    else:
        print("  FAIL: demo exit code", r.returncode)
        if r.stderr:
            print(r.stderr[:500])


def main():
    sys.path.insert(0, str(ROOT))
    frame = load_image()

    pt_boxes = step1_pytorch_detection(frame)
    out_onnx, onnx_boxes = step2_onnx_export_and_run(frame)
    step3_compare_values(frame)
    step4_final_demo(frame)

    # 요약
    print("\n=== 요약 ===")
    if pt_boxes is not None:
        print(f"  PyTorch bboxes: {len(pt_boxes)}")
    if onnx_boxes is not None:
        print(f"  ONNX bboxes: {len(onnx_boxes)}")
    print("  Done.")


if __name__ == "__main__":
    main()
