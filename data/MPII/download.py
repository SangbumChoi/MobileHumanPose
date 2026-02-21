#!/usr/bin/env python3
"""
Download MPII Human Pose Dataset (images + annotations) and optionally convert mat to COCO train.json.
Official: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download
"""
import json
import os
import os.path as osp
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = SCRIPT_DIR / "_download"
# Direct URLs (working as of 2026)
URL_IMAGES = "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
URL_ANNOT = "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip"


def download_file(url: str, dest: Path, desc: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def extract_tar(tar_path: Path, out_dir: Path):
    import tarfile
    print("Extracting images...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)


def extract_zip(zip_path: Path, out_dir: Path):
    import zipfile
    print("Extracting annotations...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _mat_val(x):
    import numpy as np
    if x is None:
        return 0
    try:
        return float(np.asarray(x).flat[0])
    except Exception:
        return 0


def mat_to_coco(mpii_dir: Path):
    """Convert MPII mpii_human_pose_v1_u12_1.mat to COCO train.json (16 joints)."""
    import numpy as np
    import scipy.io as sio
    mat_path = mpii_dir / "mpii_human_pose_v1_u12_2" / "mpii_human_pose_v1_u12_1.mat"
    if not mat_path.exists():
        for p in mpii_dir.glob("**/*.mat"):
            if "u12" in p.name:
                mat_path = p
                break
    if not mat_path.exists():
        print("MPII .mat not found. Extract annotations zip first.")
        return
    release = sio.loadmat(str(mat_path), struct_as_record=False)["RELEASE"][0, 0]
    annolist = release.annolist
    img_train = release.img_train
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    for idx in range(annolist.shape[1]):
        if not img_train[0, idx]:
            continue
        a = annolist[0, idx]
        name_obj = a.image[0, 0].name
        img_name = name_obj.flat[0] if hasattr(name_obj, "flat") else str(name_obj)
        img_name = str(img_name).strip()
        if not img_name.endswith(".jpg"):
            img_name = img_name + ".jpg"
        file_name = "images/" + img_name  # tar extracts to ./images/
        img_id += 1
        images.append({"id": img_id, "file_name": file_name, "width": 0, "height": 0})
        for ri in range(a.annorect.shape[1]):
            re = a.annorect[0, ri]
            if not hasattr(re, "annopoints") or re.annopoints.size == 0:
                continue
            pts = re.annopoints[0, 0].point
            kp = [0.0] * 48
            xs, ys = [], []
            for ji in range(pts.shape[1]):
                p = pts[0, ji]
                pid = int(_mat_val(getattr(p, "id", 0)))
                x, y = _mat_val(getattr(p, "x", 0)), _mat_val(getattr(p, "y", 0))
                vis = 1.0 if _mat_val(getattr(p, "is_visible", 1)) > 0.5 else 0.0
                if 0 <= pid < 16:
                    kp[pid * 3] = x
                    kp[pid * 3 + 1] = y
                    kp[pid * 3 + 2] = vis
                xs.append(x)
                ys.append(y)
            if not xs:
                continue
            x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
            w = max(xs) - min(xs) + 40
            h = max(ys) - min(ys) + 40
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "num_keypoints": 16,
                "bbox": [x1, y1, w, h],
                "keypoints": kp,
            })
    out_path = mpii_dir / "annotations" / "train.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"images": images, "annotations": annotations}, f, indent=2)
    print("Wrote", out_path)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true", help="Only convert mat to train.json")
    p.add_argument("--skip-convert", action="store_true", help="Only download and extract")
    args = p.parse_args()
    os.chdir(SCRIPT_DIR)
    if not args.skip_download:
        download_file(URL_IMAGES, DOWNLOAD_DIR / "mpii_human_pose_v1.tar.gz", "MPII images")
        download_file(URL_ANNOT, DOWNLOAD_DIR / "mpii_human_pose_v1_u12_2.zip", "MPII annotations")
        extract_tar(DOWNLOAD_DIR / "mpii_human_pose_v1.tar.gz", SCRIPT_DIR)
        extract_zip(DOWNLOAD_DIR / "mpii_human_pose_v1_u12_2.zip", SCRIPT_DIR)
    if not args.skip_convert:
        mat_to_coco(SCRIPT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
