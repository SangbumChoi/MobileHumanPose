"""
Person detection via ONNX inference. Uses demo/models/person_detector.onnx.
Run scripts/download_and_export_demo_models.py to generate the ONNX file.
"""
import os.path as osp

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
MODELS_DIR = osp.join(ROOT, "demo", "models")
PERSON_DETECTOR_ONNX = osp.join(MODELS_DIR, "person_detector.onnx")
PERSON_CLASS_ID = 0


def get_person_bboxes(frame_bgr, backend="onnx", conf_thres=0.5, **kwargs):
    """
    Get person bboxes via ONNX inference (demo/models/person_detector.onnx).
    backend: ignored (always ONNX). kwargs['onnx_path'] overrides default path.
    """
    onnx_path = kwargs.get("onnx_path", PERSON_DETECTOR_ONNX)
    return _get_bboxes_onnx(frame_bgr, onnx_path, conf_thres)


def _get_bboxes_onnx(frame_bgr, onnx_path, conf_thres=0.5):
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    if not osp.isfile(onnx_path):
        return []

    import numpy as np
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
    padded[pad_h : pad_h + nh, pad_w : pad_w + nw] = np.array(img)
    inp = padded.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[np.newaxis]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = session.run(None, {session.get_inputs()[0].name: inp})[0]

    if out.shape[1] == 84:
        out = out[0].T
        scores = out[:, 4:].max(axis=1)
        class_ids = out[:, 4:].argmax(axis=1)
        boxes_raw = out[:, :4]
        keep = (class_ids == PERSON_CLASS_ID) & (scores >= conf_thres)
        boxes_raw = boxes_raw[keep]
        scores = scores[keep]
        if len(boxes_raw) == 0:
            return []
        # cx,cy,w,h (640 space) -> x1,y1,w,h (frame space)
        boxes = []
        for b in boxes_raw:
            cx, cy, bw, bh = b
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            bw_f = bw / scale
            bh_f = bh / scale
            boxes.append([max(0, x1), max(0, y1), bw_f, bh_f])
        scores = scores.tolist()
        # NMS (PyTorch Ultralytics와 동일하게 iou_thres=0.45)
        try:
            import cv2
            xywh = np.array(boxes, dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(
                xywh.tolist(), scores, conf_thres, 0.45
            )
            idx = np.array(indices).flatten()
            boxes = [boxes[i] for i in idx]
        except Exception:
            pass
        return boxes[:10]
    return []
