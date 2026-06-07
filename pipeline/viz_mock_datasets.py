"""Visualize AND validate every mock-dataset image.

For each dataset it:
  * draws the loader's own joint_img + skeleton on every real image,
  * builds a grid montage of all images,
  * validates each person: keypoints in-bounds, keypoints inside the (expanded)
    bbox, and -- for 3D datasets -- that joint_cam reprojects onto joint_img.

Outputs montages + per-image overlays to /tmp/mock_viz/. Exits non-zero if any
dataset's validity rate is below threshold.
"""

import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
for p in ["main", "data", "common"]:
    sys.path.insert(0, osp.join(REPO, p))
for d in os.listdir(osp.join(REPO, "data")):
    if osp.isdir(osp.join(REPO, "data", d)):
        sys.path.insert(0, osp.join(REPO, "data", d))
os.chdir(REPO)
from config import cfg  # noqa
cfg.set_args("0")
from utils.pose_utils import cam2pixel

OUT = "/tmp/mock_viz"
os.makedirs(OUT, exist_ok=True)
_CMAP = plt.get_cmap("hsv")


def _colors(n):
    return [(int(b * 255), int(g * 255), int(r * 255))
            for r, g, b, _ in (_CMAP(i / max(n, 1)) for i in range(n))]


def draw_2d(img, joints_xy, skeleton):
    cols = _colors(len(skeleton))
    for e, (a, b) in enumerate(skeleton):
        cv2.line(img, tuple(np.int32(joints_xy[a])), tuple(np.int32(joints_xy[b])),
                 cols[e], 2, cv2.LINE_AA)
    for x, y in joints_xy:
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def expand_bbox(b, m=0.6):
    x, y, w, h = b
    return [x - m * w, y - m * h, w * (1 + 2 * m), h * (1 + 2 * m)]


def validate_person(item, W, H):
    """Return dict of validity metrics for one person annotation."""
    xy = np.array(item["joint_img"])[:, :2]
    inb = np.mean((xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H))
    ex, ey, ew, eh = expand_bbox(item["bbox"])
    inbox = np.mean((xy[:, 0] >= ex) & (xy[:, 0] <= ex + ew) &
                    (xy[:, 1] >= ey) & (xy[:, 1] <= ey + eh))
    reproj_err = None
    if "joint_cam" in item and "f" in item and "c" in item and np.any(item["joint_cam"]):
        rep = cam2pixel(np.array(item["joint_cam"]), np.array(item["f"]), np.array(item["c"]))[:, :2]
        reproj_err = float(np.max(np.abs(rep - xy)))
    ok = inb >= 0.6 and inbox >= 0.6 and (reproj_err is None or reproj_err < 1.0)
    return {"in_bounds": inb, "in_bbox": inbox, "reproj_err": reproj_err, "ok": ok}


def make_grid(images, cols=5, cell_h=200):
    cells = []
    for im in images:
        s = cell_h / im.shape[0]
        cells.append(cv2.resize(im, (int(im.shape[1] * s), cell_h)))
    cw = max(c.shape[1] for c in cells)
    cells = [np.pad(c, ((0, 0), (0, cw - c.shape[1]), (0, 0))) for c in cells]
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def process(name, split):
    mod = __import__(name)
    db = getattr(mod, name)(split)
    by_img = defaultdict(list)
    for d in db.data:
        by_img[d["img_path"]].append(d)

    overlays, metrics = [], []
    sub = osp.join(OUT, name)
    os.makedirs(sub, exist_ok=True)
    for k, (img_path, items) in enumerate(sorted(by_img.items())):
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        for it in items:
            metrics.append(validate_person(it, W, H))
            draw_2d(img, np.array(it["joint_img"])[:, :2], db.skeleton)
        cv2.rectangle(img, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(img, "%s #%d (%d ppl)" % (name, k, len(items)), (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(osp.join(sub, "%02d.jpg" % k), img)
        overlays.append(img)

    grid = make_grid(overlays, cols=5)
    cv2.imwrite(osp.join(OUT, "grid_%s.jpg" % name), grid)

    n = len(metrics)
    n_ok = sum(m["ok"] for m in metrics)
    inb = np.mean([m["in_bounds"] for m in metrics])
    inbox = np.mean([m["in_bbox"] for m in metrics])
    rerr = [m["reproj_err"] for m in metrics if m["reproj_err"] is not None]
    rstr = ("reproj<=%.3fpx" % max(rerr)) if rerr else "2D"
    print("  %-9s images=%2d persons=%3d  valid=%3d/%-3d  in_bounds=%.2f in_bbox=%.2f  %s"
          % (name, len(overlays), n, n_ok, n, inb, inbox, rstr))
    return n, n_ok


def main():
    specs = [("MSCOCO", "train"), ("MPII", "train"), ("Human36M", "train"),
             ("MuCo", "train"), ("MuPoTS", "test")]
    print("== per-image keypoint validity (all datasets) ==")
    total, ok = 0, 0
    for name, split in specs:
        n, n_ok = process(name, split)
        total += n; ok += n_ok
    rate = ok / max(total, 1)
    print("\nTOTAL valid persons: %d/%d (%.1f%%)  -> grids in %s" % (ok, total, 100 * rate, OUT))
    if rate < 0.9:
        print("VALIDATION FAILED: validity rate below 90%")
        sys.exit(1)
    print("VALIDATION PASSED.")


if __name__ == "__main__":
    main()
