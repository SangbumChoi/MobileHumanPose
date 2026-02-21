"""
RootNet inference for root depth estimation.
3DMPPE_ROOTNET_RELEASE: https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
Checkpoint: https://drive.google.com/file/d/1ZHoXNFxBBsmis-5Xzu7dfXYGNxjpntgt/view (snapshot_18.pth.tar)

Use demo/export_rootnet_onnx.py to export ONNX, or heuristic fallback when unavailable.
"""
import math
import os.path as osp

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
ROOTNET_ONNX = osp.join(ROOT, 'demo', 'models', 'rootnet.onnx')
ROOTNET_CKPT = osp.join(ROOT, 'demo', 'models', 'snapshot_18.pth.tar')
BBOX_REAL = (2000, 2000)
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def _heuristic_root_depth(bbox, img_w, img_h):
    """Heuristic root depth when RootNet unavailable. ~1500mm for typical bbox."""
    area = bbox[2] * bbox[3]
    img_area = img_w * img_h
    # Larger bbox -> closer (smaller depth)
    scale = math.sqrt(img_area / max(area, 1))
    depth = 1500.0 * scale
    return max(500.0, min(30000.0, depth))


def _preprocess_patch(img_patch_rgb):
    """Normalize patch for RootNet: HWC float [0,1] -> CHW float, normalized."""
    x = img_patch_rgb.astype(np.float32) / 255.0
    x = (x - np.array(PIXEL_MEAN).reshape(1, 1, 3)) / np.array(PIXEL_STD).reshape(1, 1, 3)
    x = x.transpose(2, 0, 1).astype(np.float32)
    return x[np.newaxis]


def get_root_depths_onnx(img_bgr, bbox_list, focal, princpt, process_bbox_fn, generate_patch_fn):
    """Get root depths using RootNet ONNX. Returns list of float (mm) or None on failure."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not osp.isfile(ROOTNET_ONNX):
        return None

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(ROOTNET_ONNX, providers=providers)
    except Exception:
        session = ort.InferenceSession(ROOTNET_ONNX, providers=['CPUExecutionProvider'])
    in_names = [i.name for i in session.get_inputs()]
    h, w = img_bgr.shape[:2]
    depths = []
    for bbox in bbox_list:
        bbox = process_bbox_fn(np.array(bbox), w, h)
        if bbox is None:
            depths.append(1500.0)
            continue
        img_patch, _ = generate_patch_fn(img_bgr, bbox, False, 1.0, 0.0, False)
        img_patch_rgb = img_patch[:, :, ::-1]  # BGR -> RGB
        img_tensor = _preprocess_patch(img_patch_rgb)
        k_value = np.array([
            math.sqrt(BBOX_REAL[0] * BBOX_REAL[1] * focal[0] * focal[1] / (bbox[2] * bbox[3]))
        ], dtype=np.float32).reshape(1, 1)
        out = session.run(None, {in_names[0]: img_tensor, in_names[1]: k_value})[0]
        root_3d = out[0]
        depths.append(float(root_3d[2]))
    return depths


def get_root_depths_torch(img_bgr, bbox_list, focal, princpt, process_bbox_fn, generate_patch_fn):
    """Get root depths using RootNet PyTorch (3DMPPE). Returns list of float or None on failure."""
    if not osp.isfile(ROOTNET_CKPT):
        return None
    try:
        import torch
        import torchvision.transforms as transforms
    except ImportError:
        return None

    # Import from 3DMPPE clone if available
    clone_dir = osp.join(ROOT, 'demo', 'models', '3dmppe_rootnet_release')
    if osp.isdir(clone_dir):
        import sys
        main_dir = osp.join(clone_dir, 'main')
        nets_dir = osp.join(clone_dir, 'nets')
        if main_dir not in sys.path:
            sys.path.insert(0, osp.join(clone_dir, 'common'))
            sys.path.insert(0, main_dir)
            if osp.isdir(nets_dir):
                sys.path.insert(0, clone_dir)
    else:
        return None

    try:
        from config import cfg as root_cfg
        from model import get_pose_net
    except ImportError:
        return None

    model = get_pose_net(root_cfg, False)
    ckpt = torch.load(ROOTNET_CKPT, map_location='cpu')
    state = ckpt.get('network', ckpt)
    if next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(PIXEL_MEAN), std=list(PIXEL_STD)),
    ])
    h, w = img_bgr.shape[:2]
    depths = []
    with torch.no_grad():
        for bbox in bbox_list:
            bbox = process_bbox_fn(np.array(bbox), w, h)
            if bbox is None:
                depths.append(1500.0)
                continue
            img_patch, _ = generate_patch_fn(img_bgr, bbox, False, 1.0, 0.0, False)
            img_t = transform(img_patch).float().unsqueeze(0)
            k_val = np.array([
                math.sqrt(BBOX_REAL[0] * BBOX_REAL[1] * focal[0] * focal[1] / (bbox[2] * bbox[3]))
            ], dtype=np.float32)
            k_t = torch.FloatTensor(k_val).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                img_t = img_t.cuda()
                k_t = k_t.cuda()
            root_3d = model(img_t, k_t)[0].cpu().numpy()
            depths.append(float(root_3d[2]))
    return depths


def get_root_depths(img_bgr, bbox_list, focal, princpt, process_bbox_fn, generate_patch_fn):
    """
    Get root depth (mm) for each bbox. Prefers ONNX, then PyTorch, then heuristic.
    Returns list of float.
    """
    depths = get_root_depths_onnx(img_bgr, bbox_list, focal, princpt, process_bbox_fn, generate_patch_fn)
    if depths is not None:
        return depths
    depths = get_root_depths_torch(img_bgr, bbox_list, focal, princpt, process_bbox_fn, generate_patch_fn)
    if depths is not None:
        return depths
    h, w = img_bgr.shape[:2]
    return [_heuristic_root_depth(np.array(b), w, h) for b in bbox_list]
