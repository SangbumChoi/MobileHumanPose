"""
Human detection for bbox input. Supports Ultralytics YOLO (ONNX/native) or Hugging Face.
Use latest YOLOv8/v11 for person detection; optional HF fallback.
"""
import os
import os.path as osp

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEMO_MODELS = osp.join(ROOT, 'demo', 'models')
os.makedirs(DEMO_MODELS, exist_ok=True)

# COCO person class id
PERSON_CLASS_ID = 0


def get_bboxes_ultralytics(frame_bgr, conf_thres=0.5, iou_thres=0.45):
    """YOLOv8/v11 person detection via Ultralytics. Returns list of [x, y, w, h]."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    model_path = osp.join(DEMO_MODELS, 'yolov8n.pt')
    if not osp.isfile(model_path):
        model_path = 'yolov8n.pt'  # downloads on first use
    model = YOLO(model_path)
    results = model(frame_bgr, conf=conf_thres, iou=iou_thres, verbose=False)[0]
    boxes = []
    if results.boxes is None:
        return boxes
    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS_ID:
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    return boxes


def get_bboxes_hf(frame_bgr, conf_thres=0.5):
    """Hugging Face object detection (person). Returns list of [x, y, w, h]."""
    try:
        from transformers import pipeline
    except ImportError:
        return None

    detector = pipeline(
        "object-detection",
        model="Xenova/detr-resnet-50",
        revision="no_timm",
    )
    # DETR uses RGB, COCO person id 1
    frame_rgb = frame_bgr[:, :, ::-1]
    out = detector(frame_rgb)
    boxes = []
    for d in out:
        if d['label'].lower() != 'person':
            continue
        if d['score'] < conf_thres:
            continue
        x1 = d['box']['xmin']
        y1 = d['box']['ymin']
        x2 = d['box']['xmax']
        y2 = d['box']['ymax']
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    return boxes


def get_bboxes_onnx(frame_bgr, onnx_path, conf_thres=0.5):
    """Run person detection using a YOLO ONNX model. Returns list of [x, y, w, h]."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not osp.isfile(onnx_path):
        return None

    # YOLOv8 preprocessing: letterbox 640x640, normalize 0-1
    from PIL import Image
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    w, h = img.size
    target = 640
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    pad_w = (target - nw) // 2
    pad_h = (target - nh) // 2
    padded = np.zeros((target, target, 3), dtype=np.uint8)
    padded[pad_h:pad_h + nh, pad_w:pad_w + nw] = np.array(img)
    inp = padded.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[np.newaxis]

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    out = session.run(None, {session.get_inputs()[0].name: inp})[0]

    # YOLOv8 output: (1, 84, 8400) for 80 classes + 4 bbox
    # Postprocess: transpose, filter person (class 0), scale back to frame
    if out.shape[1] == 84:
        out = out[0].T  # (8400, 84)
        scores = out[:, 4:].max(axis=1)
        class_ids = out[:, 4:].argmax(axis=1)
        boxes_raw = out[:, :4]
        keep = (class_ids == PERSON_CLASS_ID) & (scores >= conf_thres)
        boxes_raw = boxes_raw[keep]
        scores = scores[keep]
        if len(boxes_raw) == 0:
            return []
        # cx cy w h (normalized 0..640) -> x1 y1 w h (frame coords)
        boxes = []
        for b in boxes_raw:
            cx, cy, bw, bh = b
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            bw_f = bw / scale
            bh_f = bh / scale
            boxes.append([max(0, x1), max(0, y1), bw_f, bh_f])
        return boxes[:10]
    return []


def get_person_bboxes(frame_bgr, backend='ultralytics', conf_thres=0.5, **kwargs):
    """
    Get person bboxes from BGR frame.
    backend: 'ultralytics' | 'hf' | 'onnx'
    For 'onnx', pass onnx_path=... in kwargs.
    """
    if backend == 'ultralytics':
        return get_bboxes_ultralytics(frame_bgr, conf_thres=conf_thres, **kwargs)
    if backend == 'hf':
        return get_bboxes_hf(frame_bgr, conf_thres=conf_thres, **kwargs)
    if backend == 'onnx':
        return get_bboxes_onnx(frame_bgr, kwargs.get('onnx_path', ''), conf_thres=conf_thres)
    return []
