"""Shared utilities for the MobileHumanPose end-to-end pipeline.

This module is intentionally self-contained (it does NOT import the repo's
``main/config.py``) so that every pipeline stage runs on a plain CPU box
without the multi-GPU / CUDA assumptions baked into the original training code.

It reuses the *real* network architecture from ``common/backbone`` (LpNet*),
so the model we train and deploy is the genuine MobileHumanPose backbone.
"""

import os
import os.path as osp
import sys
import json
import logging

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = osp.dirname(osp.abspath(__file__))
REPO_DIR = osp.dirname(PIPELINE_DIR)
WORK_DIR = osp.join(PIPELINE_DIR, "work")
ASSETS_DIR = osp.join(PIPELINE_DIR, "assets")
FALLBACK_DIR = osp.join(ASSETS_DIR, "fallback")

RAW_DIR = osp.join(WORK_DIR, "01_raw")
CURATED_DIR = osp.join(WORK_DIR, "02_curated")
ANNOTATED_DIR = osp.join(WORK_DIR, "03_annotated")
BALANCED_DIR = osp.join(WORK_DIR, "04_balanced")
MODELS_DIR = osp.join(WORK_DIR, "05_models")
DEMO_DIR = osp.join(WORK_DIR, "06_demo")

# Make the repo backbone importable (common/backbone/__init__.py).
sys.path.insert(0, osp.join(REPO_DIR, "common"))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s] %(message)s",
                                                datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Keypoint definition. We use the repo's canonical 19-joint COCO convention
# (17 COCO joints + Thorax + Pelvis) -- identical to data/MSCOCO 'train' and to
# data/CrawlPipeline -- so the model trained by the original repo Trainer is
# consumed unchanged by inference / ONNX / demo.
# ---------------------------------------------------------------------------
JOINTS_NAME = (
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "Thorax", "Pelvis",
)
JOINT_NUM = len(JOINTS_NAME)  # 19

# 0-indexed skeleton edges (matches data/MSCOCO 'train').
SKELETON = (
    (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9),
    (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12),
    (17, 5), (17, 6), (18, 11), (18, 12),
)
FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

# ---------------------------------------------------------------------------
# Model geometry (matches the original config.py defaults).
# ---------------------------------------------------------------------------
INPUT_SHAPE = (256, 256)            # (h, w)
OUTPUT_SHAPE = (32, 32)             # input // 8
DEPTH_DIM = 32
EMBEDDING_SIZE = 2048
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(joint_num=JOINT_NUM, init_weights=False):
    """Build the genuine LpNetSkiConcat backbone on CPU."""
    from backbone import LpNetSkiConcat  # repo architecture
    model = LpNetSkiConcat(INPUT_SHAPE, joint_num=joint_num,
                           embedding_size=EMBEDDING_SIZE, width_mult=1.0)
    if init_weights:
        model.init_weights()
    return model


def soft_argmax(heatmaps, joint_num):
    """Differentiable soft-argmax over the (depth, H, W) 3D heatmap.

    Mirrors ``main/model.py`` exactly so trained weights stay compatible.
    Returns coords in output-grid units: x in [0,W], y in [0,H], z in [0,depth].
    """
    hm = heatmaps.reshape(-1, joint_num, DEPTH_DIM * OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])
    hm = F.softmax(hm, 2)
    hm = hm.reshape(-1, joint_num, DEPTH_DIM, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])

    accu_x = hm.sum(dim=(2, 3))  # -> (N, J, W)
    accu_y = hm.sum(dim=(2, 4))  # -> (N, J, H)
    accu_z = hm.sum(dim=(3, 4))  # -> (N, J, depth)

    device = heatmaps.device
    accu_x = accu_x * torch.arange(1, OUTPUT_SHAPE[1] + 1, device=device, dtype=accu_x.dtype)
    accu_y = accu_y * torch.arange(1, OUTPUT_SHAPE[0] + 1, device=device, dtype=accu_y.dtype)
    accu_z = accu_z * torch.arange(1, DEPTH_DIM + 1, device=device, dtype=accu_z.dtype)

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1
    return torch.cat((accu_x, accu_y, accu_z), dim=2)  # (N, J, 3)


# ---------------------------------------------------------------------------
# Image patch generation (self-contained version of data/dataset.py helpers).
# ---------------------------------------------------------------------------
def _rotate_2d(pt, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_w, src_h, dst_w, dst_h, scale, rot, inv=False):
    src_w, src_h = src_w * scale, src_h * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_downdir = _rotate_2d(np.array([0, src_h * 0.5], np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0], np.float32), rot_rad)

    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], np.float32)

    src = np.stack([src_center, src_center + src_downdir, src_center + src_rightdir])
    dst = np.stack([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def generate_patch_image(cvimg, bbox, do_flip=False, scale=1.0, rot=0.0):
    """Crop a (xmin,ymin,w,h) bbox into a 256x256 patch. Returns (patch_rgb_float, trans)."""
    img = cvimg.copy()
    img_h, img_w = img.shape[:2]
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_w, bb_h = float(bbox[2]), float(bbox[3])
    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_w - bb_c_x - 1
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_w, bb_h,
                                    INPUT_SHAPE[1], INPUT_SHAPE[0], scale, rot)
    patch = cv2.warpAffine(img, trans, (INPUT_SHAPE[1], INPUT_SHAPE[0]), flags=cv2.INTER_LINEAR)
    patch = patch[:, :, ::-1].astype(np.float32)  # BGR->RGB
    return patch, trans


def trans_point2d(pt, trans):
    src = np.array([pt[0], pt[1], 1.0]).T
    return np.dot(trans, src)[:2]


def process_bbox(bbox, img_w, img_h, aspect_ratio=1.0, expand=1.25):
    """Sanitize a (x,y,w,h) bbox to a fixed aspect ratio with margin."""
    x, y, w, h = bbox
    c_x, c_y = x + w / 2.0, y + h / 2.0
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    w *= expand
    h *= expand
    return np.array([c_x - w / 2.0, c_y - h / 2.0, w, h], dtype=np.float32)


def normalize_patch(patch_rgb_float):
    """HWC RGB float[0..255] -> CHW normalized tensor."""
    t = torch.from_numpy(np.ascontiguousarray(patch_rgb_float.transpose(2, 0, 1))) / 255.0
    mean = torch.tensor(PIXEL_MEAN).view(3, 1, 1)
    std = torch.tensor(PIXEL_STD).view(3, 1, 1)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# Small JSON helpers
# ---------------------------------------------------------------------------
def save_json(obj, path):
    ensure_dir(osp.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
