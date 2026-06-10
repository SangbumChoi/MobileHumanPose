"""Diagnose pose-model quality: predictions vs auto-label ground truth.

Draws GT (green) vs prediction (red) on annotated person crops and reports the
mean per-joint pixel error in the 256x256 patch space, split into train-style
crops. Writes a side-by-side sheet to /tmp/model_diag/.

Usage: python diagnose_model.py [--ckpt path] [--n 12]
"""

import argparse
import os
import os.path as osp

import numpy as np
import cv2
import torch

from common import (CURATED_DIR, ANNOTATED_DIR, MODELS_DIR, JOINT_NUM, SKELETON,
                    INPUT_SHAPE, OUTPUT_SHAPE, build_model, soft_argmax,
                    generate_patch_image, process_bbox, trans_point2d,
                    normalize_patch, load_json, ensure_dir, get_logger)

log = get_logger("diag")
OUT = "/tmp/model_diag"

# 17 COCO -> 19 (Thorax=mid shoulders, Pelvis=mid hips), same as CrawlPipeline.
def to19(k17):
    thorax = (k17[5] + k17[6]) / 2.0
    pelvis = (k17[11] + k17[12]) / 2.0
    return np.vstack([k17, thorax[None], pelvis[None]])


def main(ckpt=None, n=12):
    ensure_dir(OUT)
    ckpt = ckpt or osp.join(MODELS_DIR, "pose_model.pth")
    model = build_model(JOINT_NUM, init_weights=False)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("network", state.get("state_dict"))
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    coco = load_json(osp.join(ANNOTATED_DIR, "annotations.json"))
    id2f = {im["id"]: im["file_name"] for im in coco["images"]}
    anns = coco["annotations"]
    step = max(1, len(anns) // n)
    picked = anns[::step][:n]

    tiles, errs = [], []
    for a in picked:
        img = cv2.imread(osp.join(CURATED_DIR, id2f[a["image_id"]]))
        bbox = process_bbox(np.array(a["bbox"], np.float32), img.shape[1], img.shape[0])
        patch, t = generate_patch_image(img, bbox)
        with torch.no_grad():
            coord = soft_argmax(model(normalize_patch(patch)[None]), JOINT_NUM)[0].numpy()
        pred = coord[:, :2].copy()
        pred[:, 0] *= INPUT_SHAPE[1] / OUTPUT_SHAPE[1]
        pred[:, 1] *= INPUT_SHAPE[0] / OUTPUT_SHAPE[0]

        k17 = np.array(a["keypoints"], np.float32).reshape(-1, 3)
        vis17 = k17[:, 2] > 0
        gt19 = to19(k17[:, :2])
        vis19 = np.concatenate([vis17, [vis17[5] & vis17[6], vis17[11] & vis17[12]]])
        gt_p = np.array([trans_point2d(p, t) for p in gt19])

        err = np.linalg.norm(pred[vis19] - gt_p[vis19], axis=1)
        errs.append(err.mean())

        tile = patch[:, :, ::-1].astype(np.uint8).copy()
        for x, y in gt_p[vis19]:
            cv2.circle(tile, (int(x), int(y)), 4, (0, 255, 0), -1)
        for a_, b_ in SKELETON:
            cv2.line(tile, tuple(np.int32(pred[a_])), tuple(np.int32(pred[b_])),
                     (0, 0, 255), 2)
        cv2.putText(tile, "%.1fpx" % err.mean(), (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(tile)

    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    cv2.imwrite(osp.join(OUT, "pred_vs_gt.jpg"), sheet)
    log.info("mean per-joint error over %d crops: %.1f px (256x256 patch; GT=green dots, pred=red skeleton)",
             len(errs), float(np.mean(errs)))
    log.info("sheet -> %s/pred_vs_gt.jpg", OUT)
    return float(np.mean(errs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    main(args.ckpt, args.n)
