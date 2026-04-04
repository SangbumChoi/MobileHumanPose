"""
Root depth estimation via ONNX inference. Uses demo/models/rootnet.onnx.
Run scripts/download_and_export_demo_models.py to download ckpt and export ONNX.
"""
import math
import os.path as osp

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
ROOTNET_ONNX = osp.join(ROOT, "demo", "models", "rootnet.onnx")
BBOX_REAL = (2000, 2000)
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def _heuristic_root_depth(bbox, img_w, img_h):
    area = bbox[2] * bbox[3]
    img_area = img_w * img_h
    scale = math.sqrt(img_area / max(area, 1))
    depth = 1500.0 * scale
    return max(500.0, min(30000.0, depth))


def _preprocess_patch(img_patch_rgb):
    x = img_patch_rgb.astype(np.float32) / 255.0
    x = (x - np.array(PIXEL_MEAN).reshape(1, 1, 3)) / np.array(PIXEL_STD).reshape(1, 1, 3)
    x = x.transpose(2, 0, 1).astype(np.float32)
    return x[np.newaxis]


def get_root_depths(img_bgr, bbox_list, focal, princpt, process_bbox_fn, generate_patch_fn):
    """
    Get root depth (mm) per bbox via ONNX (demo/models/rootnet.onnx).
    Falls back to heuristic if ONNX file is missing.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return _heuristic_fallback(img_bgr, bbox_list, process_bbox_fn)
    if not osp.isfile(ROOTNET_ONNX):
        return _heuristic_fallback(img_bgr, bbox_list, process_bbox_fn)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(ROOTNET_ONNX, providers=providers)
    except Exception:
        session = ort.InferenceSession(ROOTNET_ONNX, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in session.get_inputs()]
    h, w = img_bgr.shape[:2]
    depths = []
    for bbox in bbox_list:
        bbox = process_bbox_fn(np.array(bbox), w, h)
        if bbox is None:
            depths.append(1500.0)
            continue
        img_patch, _ = generate_patch_fn(img_bgr, bbox, False, 1.0, 0.0, False)
        img_patch_rgb = img_patch[:, :, ::-1]
        img_tensor = _preprocess_patch(img_patch_rgb)
        k_value = np.array([
            math.sqrt(BBOX_REAL[0] * BBOX_REAL[1] * focal[0] * focal[1] / (bbox[2] * bbox[3]))
        ], dtype=np.float32).reshape(1, 1)
        out = session.run(None, {in_names[0]: img_tensor, in_names[1]: k_value})[0]
        root_3d = out[0]
        depths.append(float(root_3d[2]))
    return depths


def _heuristic_fallback(img_bgr, bbox_list, process_bbox_fn):
    h, w = img_bgr.shape[:2]
    depths = []
    for bbox in bbox_list:
        b = process_bbox_fn(np.array(bbox), w, h)
        if b is not None:
            depths.append(_heuristic_root_depth(b, w, h))
        else:
            depths.append(1500.0)
    return depths
