"""Visualize each mock dataset's keypoints on its real human images.

Draws, per dataset, the loader's own ``joint_img`` (the 2D pixel coords the
model trains on) + the dataset's skeleton on the real image, and a 3D skeleton
plot from ``joint_cam`` for the 3D datasets. Outputs to /tmp/mock_viz/.
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

OUT = "/tmp/mock_viz"
os.makedirs(OUT, exist_ok=True)
_CMAP = plt.get_cmap("hsv")


def _colors(n):
    return [(int(b * 255), int(g * 255), int(r * 255))
            for r, g, b, _ in (_CMAP(i / max(n, 1)) for i in range(n))]


def draw_2d(img, joints_xy, skeleton, joints_name):
    cols = _colors(len(skeleton))
    for e, (a, b) in enumerate(skeleton):
        pa, pb = joints_xy[a], joints_xy[b]
        cv2.line(img, tuple(np.int32(pa)), tuple(np.int32(pb)), cols[e], 3, cv2.LINE_AA)
    for j, (x, y) in enumerate(joints_xy):
        cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def render_3d(joints_cam, skeleton, title, path):
    fig = plt.figure(figsize=(4.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = joints_cam[:, 0], joints_cam[:, 1], joints_cam[:, 2]
    cols = _CMAP(np.linspace(0, 1, len(skeleton)))
    for e, (a, b) in enumerate(skeleton):
        ax.plot([X[a], X[b]], [Z[a], Z[b]], [-Y[a], -Y[b]], c=cols[e], lw=2)
    ax.scatter(X, Z, -Y, c="k", s=12)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Z (depth)"); ax.set_zlabel("-Y")
    ax.view_init(elev=12, azim=-72)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def banner(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def visualize(name, split, tag):
    mod = __import__(name)
    db = getattr(mod, name)(split)
    by_img = defaultdict(list)
    for d in db.data:
        by_img[d["img_path"]].append(d)
    img_path, items = max(by_img.items(), key=lambda kv: len(kv[1]))  # most people
    img = cv2.imread(img_path)
    for d in items:
        draw_2d(img, np.array(d["joint_img"])[:, :2], db.skeleton, db.joints_name)
    img = banner(img, "%s  %s  %dj  %d person(s)" % (name, tag, db.joint_num, len(items)))
    h = 360
    img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    p2d = osp.join(OUT, "%s_2d.jpg" % name)
    cv2.imwrite(p2d, img)
    p3d = None
    if "joint_cam" in items[0] and db.joints_have_depth:
        p3d = osp.join(OUT, "%s_3d.png" % name)
        render_3d(np.array(items[0]["joint_cam"]), db.skeleton,
                  "%s 3D (camera coords)" % name, p3d)
    print("[%s] %s | joints: %s" % (name, tag, ", ".join(db.joints_name)))
    return p2d, p3d


def main():
    specs = [("MSCOCO", "train", "2D"), ("MPII", "train", "2D"),
             ("Human36M", "train", "3D"), ("MuCo", "train", "3D"),
             ("MuPoTS", "test", "3D")]
    twod, threed = [], []
    for name, split, tag in specs:
        p2d, p3d = visualize(name, split, tag)
        twod.append(p2d)
        if p3d:
            threed.append(p3d)

    # montage of the 2D overlays (rows of 2, normalized to common height)
    H = 320
    imgs = [cv2.imread(p) for p in twod]
    imgs = [cv2.resize(i, (int(i.shape[1] * H / i.shape[0]), H)) for i in imgs]
    W = max(i.shape[1] for i in imgs)
    imgs = [np.pad(i, ((0, 0), (0, W - i.shape[1]), (0, 0))) for i in imgs]
    if len(imgs) % 2:
        imgs.append(np.zeros_like(imgs[0]))
    rows = [np.hstack(imgs[i:i + 2]) for i in range(0, len(imgs), 2)]
    cv2.imwrite(osp.join(OUT, "montage_2d.jpg"), np.vstack(rows))
    print("wrote montage_2d.jpg and per-dataset overlays to", OUT)


if __name__ == "__main__":
    main()
