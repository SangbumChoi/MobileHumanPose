"""Stage 2 - Curate images (quality / dedup / person-presence).

Removes unusable images so downstream annotation is not wasted:
  * un-decodable or too-small images
  * near-duplicates (perceptual hash)
  * images with no confident person detection (fast MobileNetV3 detector)

Output: ``work/02_curated/`` with kept images + ``curate_report.json``.
"""

import argparse
import os
import os.path as osp
import shutil

import torch
from PIL import Image
import imagehash
import torchvision
from torchvision.transforms.functional import to_tensor

from common import (RAW_DIR, CURATED_DIR, ensure_dir, save_json, load_json, get_logger)

log = get_logger("curate")
COCO_PERSON = 1


def _load_detector():
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()
    return model


def curate(in_dir=RAW_DIR, out_dir=CURATED_DIR, min_side=160,
           det_score=0.7, hash_thresh=6):
    ensure_dir(out_dir)
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    log.info("Curating %d raw images", len(files))

    detector = _load_detector()

    kept, removed, seen_hashes = [], [], []
    for fn in files:
        path = osp.join(in_dir, fn)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            removed.append({"file": fn, "reason": "undecodable"}); continue
        if min(img.size) < min_side:
            removed.append({"file": fn, "reason": "too_small"}); continue

        ph = imagehash.phash(img)
        if any((ph - h) <= hash_thresh for h in seen_hashes):
            removed.append({"file": fn, "reason": "duplicate"}); continue

        with torch.no_grad():
            out = detector([to_tensor(img)])[0]
        person_scores = [float(s) for l, s in zip(out["labels"], out["scores"])
                         if int(l) == COCO_PERSON and float(s) >= det_score]
        if not person_scores:
            removed.append({"file": fn, "reason": "no_person"}); continue

        seen_hashes.append(ph)
        dst = osp.join(out_dir, fn)
        shutil.copy2(path, dst)
        kept.append({"file": fn, "num_persons": len(person_scores),
                     "max_score": round(max(person_scores), 3)})

    save_json({"kept": kept, "removed": removed,
               "summary": {"in": len(files), "kept": len(kept), "removed": len(removed)}},
              osp.join(out_dir, "curate_report.json"))
    log.info("Kept %d / %d images (removed %d)", len(kept), len(files), len(removed))
    return len(kept)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Curate crawled images.")
    ap.add_argument("--det_score", type=float, default=0.7)
    ap.add_argument("--hash_thresh", type=int, default=6)
    args = ap.parse_args()
    curate(det_score=args.det_score, hash_thresh=args.hash_thresh)
