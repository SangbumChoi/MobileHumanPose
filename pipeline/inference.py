"""Shared inference: detect persons -> run trained LpNet -> 2D pose.

Used by the Gradio demo and any CLI. Detector boxes come from the fast
MobileNetV3 Faster-RCNN; pose comes from our trained MobileHumanPose model.
"""

import os.path as osp

import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

from common import (MODELS_DIR, JOINT_NUM, SKELETON, INPUT_SHAPE, OUTPUT_SHAPE,
                    build_model, soft_argmax, generate_patch_image, process_bbox,
                    normalize_patch, get_logger)

log = get_logger("infer")
COCO_PERSON = 1
_DETECTOR = None


def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        w = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        _DETECTOR = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=w)
        _DETECTOR.eval()
    return _DETECTOR


def load_pose_model(ckpt_path=None):
    ckpt_path = ckpt_path or osp.join(MODELS_DIR, "pose_model.pth")
    model = build_model(JOINT_NUM, init_weights=False)
    if osp.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("network", ckpt.get("state_dict"))
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state)
        log.info("Loaded trained model from %s", ckpt_path)
    else:
        model.init_weights()
        log.info("No checkpoint at %s; using randomly-initialised model.", ckpt_path)
    model.eval()
    return model


@torch.no_grad()
def detect_persons(img_bgr, det_score=0.7, max_persons=5):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = get_detector()([to_tensor(rgb)])[0]
    boxes = []
    for i in range(len(out["scores"])):
        if int(out["labels"][i]) == COCO_PERSON and float(out["scores"][i]) >= det_score:
            x1, y1, x2, y2 = out["boxes"][i].numpy()
            boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_persons]


@torch.no_grad()
def estimate_pose(model, img_bgr, bbox):
    """Return (17,2) keypoints in original-image pixel coordinates."""
    h, w = img_bgr.shape[:2]
    pbox = process_bbox(np.array(bbox, dtype=np.float32), w, h)
    patch, trans = generate_patch_image(img_bgr, pbox)
    x = normalize_patch(patch).unsqueeze(0)
    coord = soft_argmax(model(x), JOINT_NUM)[0].numpy()  # (17,3) output-grid units
    coord[:, 0] = coord[:, 0] / OUTPUT_SHAPE[1] * INPUT_SHAPE[1]
    coord[:, 1] = coord[:, 1] / OUTPUT_SHAPE[0] * INPUT_SHAPE[0]
    xy1 = np.concatenate([coord[:, :2], np.ones((JOINT_NUM, 1))], axis=1)
    inv = np.concatenate([trans, [[0, 0, 1]]], axis=0)
    orig = (np.linalg.inv(inv) @ xy1.T).T[:, :2]
    return orig


def draw_pose(img_bgr, keypoints_list):
    vis = img_bgr.copy()
    palette = [(0, 255, 0), (0, 200, 255), (255, 120, 0), (255, 0, 200), (180, 255, 0)]
    for idx, kps in enumerate(keypoints_list):
        col = palette[idx % len(palette)]
        for a, b in SKELETON:
            cv2.line(vis, tuple(kps[a].astype(int)), tuple(kps[b].astype(int)), col, 2)
        for j in range(len(kps)):
            cv2.circle(vis, tuple(kps[j].astype(int)), 3, (0, 0, 255), -1)
    return vis


def run_image(model, img_bgr, det_score=0.7):
    boxes = detect_persons(img_bgr, det_score=det_score)
    kps_list = [estimate_pose(model, img_bgr, b) for b in boxes]
    return draw_pose(img_bgr, kps_list), boxes, kps_list
