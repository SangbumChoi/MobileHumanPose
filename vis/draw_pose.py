#!/usr/bin/env python3
"""
Python replacement for MATLAB draw_3Dpose_coco / draw_3Dpose_mupots.
Reads preds_2d_kpt_*.mat, preds_3d_kpt_*.mat and *_img_name.txt, draws 2D/3D
skeletons and saves images (no MATLAB required).

Usage:
  python vis/draw_pose.py --dataset coco --result_dir output/result --root_path /path/to/coco/val2017 --save_dir vis/out
  python vis/draw_pose.py --dataset mupots --result_dir output/result --root_path /path/to/MultiPersonTestSet --save_dir vis/out
  python vis/draw_pose.py --dataset coco --img_name_txt vis/coco_img_name.txt --mat_2d output/result/preds_2d_kpt_coco.mat --mat_3d output/result/preds_3d_kpt_coco.mat --root_path . --save_dir vis/out
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import scipy.io as sio

# MuPoTS/COCO 17 joints, 16 edges (0-indexed, same as MATLAB skeleton)
MUPTOS_SKELETON = [
    (0, 16), (1, 16), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10),
    (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
]
# Demo/COCO 18 joints (same as demo.py skeleton)
DEMO_SKELETON_18 = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
]

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.utils.vis import draw_bboxes, vis_keypoints, vis_3d_skeleton_to_file


def _get_skeleton_for_joints(num_joints: int):
    if num_joints == 18:
        return DEMO_SKELETON_18
    return MUPTOS_SKELETON


def _ensure_list_of_preds(pred_2d, pred_3d):
    """Normalize .mat entry to list of (J,2) and (J,3). Accepts 17 or 18 joints."""
    if pred_2d.ndim == 2:
        pred_2d = np.asarray(pred_2d, dtype=np.float64)
        pred_3d = np.asarray(pred_3d, dtype=np.float64)
        if pred_2d.shape[1] == 2 and pred_2d.shape[0] not in (17, 18):
            pred_2d = pred_2d.T
        if pred_3d.shape[1] == 3 and pred_3d.shape[0] not in (17, 18):
            pred_3d = pred_3d.T
        return [(pred_2d, pred_3d)]
    out = []
    for i in range(pred_2d.shape[0]):
        p2 = np.asarray(pred_2d[i], dtype=np.float64)
        p3 = np.asarray(pred_3d[i], dtype=np.float64)
        if p2.ndim == 2 and p2.shape[1] == 2 and p2.shape[0] not in (17, 18):
            p2 = p2.T
        if p3.ndim == 2 and p3.shape[1] == 3 and p3.shape[0] not in (17, 18):
            p3 = p3.T
        out.append((p2, p3))
    return out


def draw_2d_and_save(img_bgr, pred_2d, kps_lines, save_path_2d, bbox=None):
    """Overlay 2D skeleton (and optional bbox) and save. pred_2d: (J, 2)."""
    out = np.asarray(img_bgr, dtype=np.uint8)
    if bbox is not None:
        draw_bboxes(out, [bbox], color=(0, 255, 0), thickness=2)
    kps = np.zeros((3, pred_2d.shape[0]), dtype=np.float32)
    kps[0, :] = pred_2d[:, 0]
    kps[1, :] = pred_2d[:, 1]
    kps[2, :] = 1.0
    out = vis_keypoints(out, kps, kps_lines, kp_thresh=0.01, alpha=1)
    os.makedirs(osp.dirname(save_path_2d) or ".", exist_ok=True)
    cv2.imwrite(save_path_2d, out)


def draw_3d_and_save(pred_3d, kps_lines, save_path_3d):
    """Draw 3D skeleton and save to file. pred_3d: (17, 3)."""
    vis = np.ones((pred_3d.shape[0], 1), dtype=np.float64)
    os.makedirs(osp.dirname(save_path_3d) or ".", exist_ok=True)
    vis_3d_skeleton_to_file(pred_3d, vis, kps_lines, save_path_3d)


def run_coco(
    img_name_txt: str,
    root_path: str,
    save_dir: str,
    mat_2d_path: str | None = None,
    mat_3d_path: str | None = None,
    result_dir: str | None = None,
) -> None:
    mat_2d_path = mat_2d_path or (osp.join(result_dir, "preds_2d_kpt_coco.mat") if result_dir else None)
    mat_3d_path = mat_3d_path or (osp.join(result_dir, "preds_3d_kpt_coco.mat") if result_dir else None)
    if not mat_2d_path or not mat_3d_path or not osp.isfile(mat_2d_path) or not osp.isfile(mat_3d_path):
        raise FileNotFoundError("Need preds_2d_kpt_coco.mat and preds_3d_kpt_coco.mat (--result_dir or --mat_2d/--mat_3d)")
    preds_2d = sio.loadmat(mat_2d_path)
    preds_3d = sio.loadmat(mat_3d_path)
    keys = [k for k in preds_2d if not k.startswith("__")]
    with open(img_name_txt) as f:
        lines = [line.strip() for line in f if line.strip()]
    os.makedirs(save_dir, exist_ok=True)
    for line in lines:
        if line not in keys:
            continue
        pred_2d_all = preds_2d[line]
        pred_3d_all = preds_3d[line]
        # COCO: key is coco_00000000 -> image file 00000000.jpg
        parts = line.split("_", 1)
        img_file = parts[1] + ".jpg" if len(parts) > 1 else line + ".jpg"
        img_path = osp.join(root_path, img_file)
        if not osp.isfile(img_path):
            print("Skip (image not found):", img_path)
            continue
        img = cv2.imread(img_path)
        if img is None:
            print("Skip (read failed):", img_path)
            continue
        list_preds = _ensure_list_of_preds(pred_2d_all, pred_3d_all)
        kps_lines = _get_skeleton_for_joints(list_preds[0][0].shape[0])
        for i, (pred_2d, pred_3d) in enumerate(list_preds):
            base = osp.join(save_dir, osp.splitext(img_file)[0] + f"_{i}")
            x_min, x_max = pred_2d[:, 0].min(), pred_2d[:, 0].max()
            y_min, y_max = pred_2d[:, 1].min(), pred_2d[:, 1].max()
            pad = max(20, (x_max - x_min + y_max - y_min) * 0.1)
            bbox_from_kpt = [x_min - pad, y_min - pad, x_max - x_min + 2 * pad, y_max - y_min + 2 * pad]
            draw_2d_and_save(img, pred_2d, kps_lines, base + "_2d.jpg", bbox=bbox_from_kpt)
            draw_3d_and_save(pred_3d, kps_lines, base + "_3d.jpg")
        print("Saved", len(list_preds), "persons →", save_dir)


def run_mupots(
    img_name_txt: str,
    root_path: str,
    save_dir: str,
    mat_2d_path: str | None = None,
    mat_3d_path: str | None = None,
    result_dir: str | None = None,
) -> None:
    mat_2d_path = mat_2d_path or (osp.join(result_dir, "preds_2d_kpt_mupots.mat") if result_dir else None)
    mat_3d_path = mat_3d_path or (osp.join(result_dir, "preds_3d_kpt_mupots.mat") if result_dir else None)
    if not mat_2d_path or not mat_3d_path or not osp.isfile(mat_2d_path) or not osp.isfile(mat_3d_path):
        raise FileNotFoundError("Need preds_2d_kpt_mupots.mat and preds_3d_kpt_mupots.mat (--result_dir or --mat_2d/--mat_3d)")
    preds_2d = sio.loadmat(mat_2d_path)
    preds_3d = sio.loadmat(mat_3d_path)
    with open(img_name_txt) as f:
        lines = [line.strip() for line in f if line.strip()]
    os.makedirs(save_dir, exist_ok=True)
    for line in lines:
        toks = line.split()
        if len(toks) < 2:
            continue
        folder_id = int(toks[0])
        frame_id = int(toks[1])
        key = f"TS{folder_id}_img_{frame_id:06d}"
        if key not in preds_2d or key not in preds_3d:
            continue
        pred_2d_all = preds_2d[key]
        pred_3d_all = preds_3d[key]
        img_name = f"TS{folder_id}/img_{frame_id:06d}.jpg"
        img_path = osp.join(root_path, img_name)
        if not osp.isfile(img_path):
            print("Skip (image not found):", img_path)
            continue
        img = cv2.imread(img_path)
        if img is None:
            print("Skip (read failed):", img_path)
            continue
        sub_save = osp.join(save_dir, f"TS{folder_id}")
        os.makedirs(sub_save, exist_ok=True)
        for i, (pred_2d, pred_3d) in enumerate(_ensure_list_of_preds(pred_2d_all, pred_3d_all)):
            base = osp.join(sub_save, f"img_{frame_id:06d}_{i}")
            draw_2d_and_save(img, pred_2d, MUPTOS_SKELETON, base + "_2d.jpg")
            draw_3d_and_save(pred_3d, MUPTOS_SKELETON, base + "_3d.jpg")
        print("Saved:", base)


def main():
    p = argparse.ArgumentParser(description="Draw 2D/3D pose from .mat results (Python replacement for MATLAB)")
    p.add_argument("--dataset", choices=["coco", "mupots"], required=True)
    p.add_argument("--result_dir", default=None, help="Directory containing preds_2d_kpt_*.mat, preds_3d_kpt_*.mat")
    p.add_argument("--img_name_txt", default=None, help="Path to coco_img_name.txt or mupots_img_name.txt")
    p.add_argument("--mat_2d", default=None, help="Override: path to preds_2d_kpt_*.mat")
    p.add_argument("--mat_3d", default=None, help="Override: path to preds_3d_kpt_*.mat")
    p.add_argument("--root_path", required=True, help="Root directory of input images (e.g. COCO val2017, MuPoTS MultiPersonTestSet)")
    p.add_argument("--save_dir", default="vis/out", help="Output directory for _2d.jpg and _3d.jpg")
    args = p.parse_args()

    result_dir = args.result_dir or osp.join(ROOT, "output", "result")
    txt_coco = args.img_name_txt or osp.join(ROOT, "vis", "coco_img_name.txt")
    txt_mupots = args.img_name_txt or osp.join(ROOT, "vis", "mupots_img_name.txt")
    if args.dataset == "coco":
        run_coco(
            txt_coco,
            args.root_path,
            args.save_dir,
            mat_2d_path=args.mat_2d,
            mat_3d_path=args.mat_3d,
            result_dir=result_dir,
        )
    else:
        run_mupots(
            txt_mupots,
            args.root_path,
            args.save_dir,
            mat_2d_path=args.mat_2d,
            mat_3d_path=args.mat_3d,
            result_dir=result_dir,
        )


if __name__ == "__main__":
    main()
