"""Stage 3 - Annotate images with a pretrained 2D keypoint model.

Leverages torchvision's ``keypointrcnn_resnet50_fpn`` (COCO-17 keypoints) to
auto-label every curated image, emitting a standard COCO-format
``annotations.json`` (the format the original repo expects for custom data).

Output: ``work/03_annotated/annotations.json`` + a few overlay previews.
"""

import argparse
import os
import os.path as osp

import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

from common import (CURATED_DIR, ANNOTATED_DIR, JOINTS_NAME, SKELETON,
                    ensure_dir, save_json, get_logger)

log = get_logger("annotate")
COCO_PERSON = 1


def _load_kp_model():
    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model


def _draw(img_bgr, kpts, kp_thresh=2.0):
    for (a, b) in SKELETON:
        if kpts[a, 2] > 0 and kpts[b, 2] > 0:
            cv2.line(img_bgr, tuple(kpts[a, :2].astype(int)), tuple(kpts[b, :2].astype(int)),
                     (0, 255, 0), 2)
    for j in range(len(kpts)):
        if kpts[j, 2] > 0:
            cv2.circle(img_bgr, tuple(kpts[j, :2].astype(int)), 3, (0, 0, 255), -1)
    return img_bgr


def annotate(in_dir=CURATED_DIR, out_dir=ANNOTATED_DIR, det_score=0.8,
             kp_score=3.0, n_preview=6):
    ensure_dir(out_dir)
    prev_dir = ensure_dir(osp.join(out_dir, "previews"))
    model = _load_kp_model()

    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    images, annotations = [], []
    img_id, ann_id = 0, 0
    for fn in files:
        path = osp.join(in_dir, fn)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            out = model([to_tensor(rgb)])[0]

        kept_anns = []
        for i in range(len(out["scores"])):
            if int(out["labels"][i]) != COCO_PERSON or float(out["scores"][i]) < det_score:
                continue
            kps = out["keypoints"][i].numpy()          # (17,3)
            kps_score = out["keypoints_scores"][i].numpy()  # (17,)
            vis = (kps_score >= kp_score).astype(np.float32) * 2.0  # COCO v=2 visible / 0
            kps[:, 2] = vis
            if vis.sum() / 2.0 < 5:                     # need enough visible joints
                continue
            x1, y1, x2, y2 = out["boxes"][i].numpy()
            kept_anns.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "num_keypoints": int(vis.sum() / 2),
                "keypoints": [round(float(v), 2) for v in kps.reshape(-1)],
                "det_score": round(float(out["scores"][i]), 3),
            })
            ann_id += 1
        if not kept_anns:
            continue
        images.append({"id": img_id, "file_name": fn, "width": w, "height": h})
        annotations += kept_anns

        if img_id < n_preview:
            vis_img = bgr.copy()
            for a in kept_anns:
                _draw(vis_img, np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3))
            cv2.imwrite(osp.join(prev_dir, fn), vis_img)
        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person",
                        "keypoints": list(JOINTS_NAME),
                        "skeleton": [[a + 1, b + 1] for (a, b) in SKELETON]}],
    }
    save_json(coco, osp.join(out_dir, "annotations.json"))
    log.info("Annotated %d images with %d person instances", len(images), len(annotations))
    return len(annotations)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-annotate curated images.")
    ap.add_argument("--det_score", type=float, default=0.8)
    ap.add_argument("--kp_score", type=float, default=3.0)
    args = ap.parse_args()
    annotate(det_score=args.det_score, kp_score=args.kp_score)
