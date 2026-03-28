#!/usr/bin/env python3
"""
Export MuPoTS test image names to a .txt file for 3D visualization (MATLAB).
Format: one line per image, "folder_id frame_id" (e.g. "1 1" for TS1/img_000001.jpg)
so that draw_3Dpose_mupots.m can parse with strsplit and build TS%d/img_%06d.jpg.
Usage:
  python scripts/export_mupots_img_names.py
  python scripts/export_mupots_img_names.py --output vis/mupots_img_name.txt
  python scripts/export_mupots_img_names.py --mat output/result/preds_2d_kpt_mupots.mat  # from existing .mat
Output is written to vis/mupots_img_name.txt by default so that draw_3Dpose_mupots.m
(from vis/single or vis/multi) finds it via fopen('../mupots_img_name.txt').
"""
import argparse
import os
import os.path as osp
import re
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def from_dataset(out_path: str) -> None:
    """Use MuPoTS test set order (same as evaluate()) to produce 'folder_id frame_id' lines."""
    from data.MuPoTS.MuPoTS import MuPoTS

    dataset = MuPoTS("test")
    os.makedirs(osp.dirname(osp.abspath(out_path)) or ".", exist_ok=True)
    # img_path is like .../MultiPersonTestSet/TS1/img_000001.jpg -> TS1, 1
    pattern = re.compile(r"TS(\d+)[/\\]img_(\d+)\.jpg$", re.I)
    with open(out_path, "w") as f:
        for gt in dataset.data:
            path = gt["img_path"].replace("\\", "/")
            m = pattern.search(path)
            if m:
                folder_id = int(m.group(1))
                frame_id = int(m.group(2))
                f.write(f"{folder_id} {frame_id}\n")
            else:
                # Fallback: same key as .mat (TS1_img_000001) -> "1 1"
                parts = path.split("/")[-2:]
                if len(parts) == 2 and parts[0].startswith("TS") and parts[1].startswith("img_"):
                    folder_id = int(parts[0][2:])
                    frame_id = int(parts[1].replace("img_", "").replace(".jpg", ""))
                    f.write(f"{folder_id} {frame_id}\n")
    print(f"Wrote {len(dataset.data)} lines to {out_path}")


def from_mat(mat_path: str, out_path: str) -> None:
    """Dump keys from preds_2d_kpt_mupots.mat as 'folder_id frame_id' (e.g. TS1_img_000001 -> 1 1)."""
    import scipy.io as sio

    m = sio.loadmat(mat_path)
    keys = [k for k in m if not k.startswith("__")]
    os.makedirs(osp.dirname(osp.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for k in keys:
            # TS1_img_000001 -> 1 1
            if k.startswith("TS") and "_img_" in k:
                rest = k[2:]
                a, b = rest.split("_img_", 1)
                f.write(f"{int(a)} {int(b)}\n")
            else:
                f.write(k + "\n")
    print(f"Wrote {len(keys)} lines from {mat_path} to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Export MuPoTS image names for 3D vis")
    p.add_argument("--output", "-o", default=osp.join(ROOT, "vis", "mupots_img_name.txt"), help="Output .txt path")
    p.add_argument("--mat", default="", help="Optional: use keys from this preds_2d_kpt_mupots.mat")
    args = p.parse_args()
    if args.mat and osp.isfile(args.mat):
        from_mat(args.mat, args.output)
    else:
        from_dataset(args.output)


if __name__ == "__main__":
    main()
