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

from common import (CURATED_DIR, ANNOTATED_DIR, ensure_dir, save_json, get_logger)

log = get_logger("annotate")
COCO_PERSON = 1

# Raw COCO-17 keypoint names + skeleton (what this stage emits, independent of
# the 19-joint convention the model is later trained on).
COCO17_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle")
COCO17_SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                   (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                   (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
COCO17_FLIP = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def _flip_perm():
    perm = list(range(17))
    for a, b in COCO17_FLIP:
        perm[a], perm[b] = b, a
    return perm


FLIP_PERM = _flip_perm()


def _load_kp_model():
    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model


def _draw(img_bgr, kpts, kp_thresh=2.0):
    for (a, b) in COCO17_SKELETON:
        if kpts[a, 2] > 0 and kpts[b, 2] > 0:
            cv2.line(img_bgr, tuple(kpts[a, :2].astype(int)), tuple(kpts[b, :2].astype(int)),
                     (0, 255, 0), 2)
    for j in range(len(kpts)):
        if kpts[j, 2] > 0:
            cv2.circle(img_bgr, tuple(kpts[j, :2].astype(int)), 3, (0, 0, 255), -1)
    return img_bgr


@torch.no_grad()
def _kps_on_crop(model, crop_rgb):
    """Run the detector on a single crop; return (kps(17,3), scores(17)) for the
    largest person, or None."""
    out = model([to_tensor(crop_rgb)])[0]
    cand = [i for i in range(len(out["scores"]))
            if int(out["labels"][i]) == COCO_PERSON and float(out["scores"][i]) >= 0.5]
    if not cand:
        return None
    i = max(cand, key=lambda i: float((out["boxes"][i][2] - out["boxes"][i][0]) *
                                      (out["boxes"][i][3] - out["boxes"][i][1])))
    return out["keypoints"][i].numpy().copy(), out["keypoints_scores"][i].numpy().copy()


@torch.no_grad()
def _refine_person(model, rgb, box, pad=0.25):
    """High-res per-person crop + horizontal-flip TTA. Returns
    (keypoints_img(17,3 in full-image px), avg_scores(17), flip_consistency_px)
    or None. Sharper than whole-image detection because the person fills the
    detector's frame; flip-averaging cancels left/right asymmetric error."""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1e, y1e = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    x2e, y2e = min(W, int(x2 + pad * bw)), min(H, int(y2 + pad * bh))
    crop = rgb[y1e:y2e, x1e:x2e]
    ch, cw = crop.shape[:2]
    if ch < 16 or cw < 16:
        return None
    r = _kps_on_crop(model, crop)
    if r is None:
        return None
    kp, sc = r
    rf = _kps_on_crop(model, crop[:, ::-1, :].copy())
    consist = float("nan")
    if rf is not None:
        kpf, scf = rf
        kpf[:, 0] = cw - 1 - kpf[:, 0]
        kpf, scf = kpf[FLIP_PERM], scf[FLIP_PERM]
        consist = float(np.linalg.norm(kp[:, :2] - kpf[:, :2], axis=1).mean())
        kp[:, :2] = (kp[:, :2] + kpf[:, :2]) / 2.0
        sc = (sc + scf) / 2.0
    kp[:, 0] += x1e
    kp[:, 1] += y1e
    return kp, sc, consist


def annotate(in_dir=CURATED_DIR, out_dir=ANNOTATED_DIR, det_score=0.8,
             kp_score=3.0, n_preview=6, refine=True):
    ensure_dir(out_dir)
    prev_dir = ensure_dir(osp.join(out_dir, "previews"))
    model = _load_kp_model()

    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    images, annotations = [], []
    img_id, ann_id = 0, 0
    consistencies = []
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
            box = out["boxes"][i].numpy()
            kps = out["keypoints"][i].numpy()          # (17,3) whole-image
            kps_score = out["keypoints_scores"][i].numpy()
            if refine:                                  # sharpen via crop + flip TTA
                ref = _refine_person(model, rgb, box)
                if ref is not None:
                    kps, kps_score, consist = ref
                    if consist == consist:              # not nan
                        consistencies.append(consist)
            vis = (kps_score >= kp_score).astype(np.float32) * 2.0  # COCO v=2 / 0
            kps[:, 2] = vis
            if vis.sum() / 2.0 < 5:                     # need enough visible joints
                continue
            x1, y1, x2, y2 = box
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
                        "keypoints": list(COCO17_NAMES),
                        "skeleton": [[a + 1, b + 1] for (a, b) in COCO17_SKELETON]}],
    }
    save_json(coco, osp.join(out_dir, "annotations.json"))
    note = ""
    if consistencies:
        note = "  flip-consistency=%.2fpx (lower=cleaner labels)" % np.mean(consistencies)
    log.info("Annotated %d images with %d person instances%s%s",
             len(images), len(annotations), "  [refined]" if refine else "", note)
    return len(annotations)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-annotate curated images.")
    ap.add_argument("--det_score", type=float, default=0.8)
    ap.add_argument("--kp_score", type=float, default=3.0)
    ap.add_argument("--no_refine", action="store_true",
                    help="disable per-person crop + flip-TTA refinement")
    args = ap.parse_args()
    annotate(det_score=args.det_score, kp_score=args.kp_score, refine=not args.no_refine)
