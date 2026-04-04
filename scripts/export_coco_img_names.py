#!/usr/bin/env python3
"""
Export COCO test image names to a .txt file for 3D visualization (MATLAB).
Format: one line per image, key matching preds_2d_kpt_coco.mat (e.g. coco_000000000139).
Usage:
  python scripts/export_coco_img_names.py
  python scripts/export_coco_img_names.py --output vis/coco_img_name.txt
  python scripts/export_coco_img_names.py --mat output/result/preds_2d_kpt_coco.mat  # from existing .mat
Output is written to vis/coco_img_name.txt by default so that draw_3Dpose_coco.m
(from vis/single or vis/multi) finds it via fopen('../coco_img_name.txt').
"""
import argparse
import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def from_dataset(out_path: str) -> None:
    """Use MSCOCO test set order (same as evaluate()) to produce coco_XXXXXXXX lines."""
    import os
    from data.MSCOCO.MSCOCO import MSCOCO

    dataset = MSCOCO("test")
    os.makedirs(osp.dirname(osp.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for gt in dataset.data:
            # Same key as in MSCOCO.evaluate(): coco_ + filename without extension
            parts = gt["img_path"].replace("\\", "/").split("/")
            name = parts[-1].rsplit(".", 1)[0]  # e.g. 000000000139
            key = "coco_" + name
            f.write(key + "\n")
    print(f"Wrote {len(dataset.data)} lines to {out_path}")


def from_mat(mat_path: str, out_path: str) -> None:
    """Dump keys from preds_2d_kpt_coco.mat so .txt matches the .mat exactly."""
    import os
    import scipy.io as sio

    m = sio.loadmat(mat_path)
    keys = [k for k in m if not k.startswith("__")]
    os.makedirs(osp.dirname(osp.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for k in keys:
            f.write(k + "\n")
    print(f"Wrote {len(keys)} lines from {mat_path} to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Export COCO image names for 3D vis")
    p.add_argument("--output", "-o", default=osp.join(ROOT, "vis", "coco_img_name.txt"), help="Output .txt path")
    p.add_argument("--mat", default="", help="Optional: use keys from this preds_2d_kpt_coco.mat")
    args = p.parse_args()
    if args.mat and osp.isfile(args.mat):
        from_mat(args.mat, args.output)
    else:
        from_dataset(args.output)


if __name__ == "__main__":
    main()
