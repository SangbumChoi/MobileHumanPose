"""Stage 5 - Train the MobileHumanPose (LpNet) model on curated data.

Trains the genuine ``LpNetSkiConcat`` backbone with the original soft-argmax +
L1 coordinate loss. Because the auto-labels are 2D (COCO keypoints), the depth
term is masked out via ``have_depth=0`` -- a mode the original loss already
supports. Sampling is rebalanced with the Stage-4 weights.

Output: ``work/05_models/pose_model.pth`` + ``train_curve.png``.

NOTE: on a CPU sandbox with a tiny crawled set this is a *functional* training
run (loss decreases, checkpoint is valid), not a research-grade model. Scale
``--epochs`` and the crawl size on a GPU for real accuracy.
"""

import argparse
import os.path as osp
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (CURATED_DIR, ANNOTATED_DIR, BALANCED_DIR, MODELS_DIR,
                    JOINT_NUM, JOINTS_NAME, SKELETON, FLIP_PAIRS,
                    INPUT_SHAPE, OUTPUT_SHAPE, DEPTH_DIM,
                    build_model, soft_argmax, generate_patch_image,
                    process_bbox, trans_point2d, normalize_patch,
                    ensure_dir, save_json, load_json, get_logger)

log = get_logger("train")


class PoseDataset(Dataset):
    def __init__(self, curated_dir, ann_dir, augment=True):
        coco = load_json(osp.join(ann_dir, "annotations.json"))
        self.id2file = {im["id"]: im["file_name"] for im in coco["images"]}
        self.anns = coco["annotations"]
        self.curated_dir = curated_dir
        self.augment = augment
        self._cache = {}

    def __len__(self):
        return len(self.anns)

    def _img(self, fn):
        if fn not in self._cache:
            self._cache[fn] = cv2.imread(osp.join(self.curated_dir, fn))
        return self._cache[fn]

    def __getitem__(self, idx):
        a = self.anns[idx]
        img = self._img(self.id2file[a["image_id"]])
        img_h, img_w = img.shape[:2]
        kps = np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3)  # (17, x/y/v)
        bbox = process_bbox(np.array(a["bbox"], dtype=np.float32), img_w, img_h)

        if self.augment:
            scale = np.clip(np.random.randn() * 0.25 + 1.0, 0.7, 1.3)
            rot = np.clip(np.random.randn() * 2, -2, 2) * 30 if random.random() < 0.6 else 0.0
            do_flip = random.random() < 0.5
        else:
            scale, rot, do_flip = 1.0, 0.0, False

        joint_xy = kps[:, :2].copy()
        vis = (kps[:, 2] > 0).astype(np.float32)
        if do_flip:
            joint_xy[:, 0] = img_w - joint_xy[:, 0] - 1
            for p, q in FLIP_PAIRS:
                joint_xy[[p, q]] = joint_xy[[q, p]]
                vis[[p, q]] = vis[[q, p]]

        patch, trans = generate_patch_image(img, bbox, do_flip, scale, rot)
        for c in range(3):
            patch[:, :, c] = np.clip(patch[:, :, c] * random.uniform(0.8, 1.2), 0, 255) \
                if self.augment else patch[:, :, c]

        target = np.zeros((JOINT_NUM, 3), dtype=np.float32)
        for j in range(JOINT_NUM):
            xy = trans_point2d(joint_xy[j], trans)
            target[j, 0] = xy[0] / INPUT_SHAPE[1] * OUTPUT_SHAPE[1]
            target[j, 1] = xy[1] / INPUT_SHAPE[0] * OUTPUT_SHAPE[0]
            in_bounds = 0 <= xy[0] < INPUT_SHAPE[1] and 0 <= xy[1] < INPUT_SHAPE[0]
            vis[j] *= float(in_bounds)
        target[:, 2] = DEPTH_DIM / 2.0  # placeholder; masked by have_depth=0

        img_t = normalize_patch(patch)
        target_t = torch.from_numpy(target)
        vis_t = torch.from_numpy(vis[:, None].repeat(3, axis=1))  # (17,3)
        have_depth = torch.zeros(1)  # 2D-only supervision
        return img_t, target_t, vis_t, have_depth


def train(epochs=12, batch_size=8, lr=1e-3, num_workers=0, seed=0,
          out_dir=MODELS_DIR):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_grad_enabled(True)  # robust if a prior stage disabled grad globally
    ensure_dir(out_dir)
    ds = PoseDataset(CURATED_DIR, ANNOTATED_DIR, augment=True)
    log.info("Training samples: %d", len(ds))

    # Stage-4 rebalanced sampling.
    weights_path = osp.join(BALANCED_DIR, "sample_weights.json")
    if osp.exists(weights_path):
        w = load_json(weights_path)
        sample_w = [w.get(str(a["id"]), 1.0) for a in ds.anns]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        log.info("Using Stage-4 rebalanced WeightedRandomSampler.")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = build_model(JOINT_NUM, init_weights=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.7),
                                                                   int(epochs * 0.9)], gamma=0.1)

    history = []
    for ep in range(epochs):
        ep_loss, nb = 0.0, 0
        for img_t, target_t, vis_t, have_depth in loader:
            fm = model(img_t)
            coord = soft_argmax(fm, JOINT_NUM)
            loss_c = torch.abs(coord - target_t) * vis_t
            loss = ((loss_c[:, :, 0] + loss_c[:, :, 1]
                     + loss_c[:, :, 2] * have_depth) / 3.0).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        avg = ep_loss / max(nb, 1)
        history.append(avg)
        log.info("epoch %2d/%d  loss=%.4f  lr=%.1e", ep + 1, epochs, avg, sched.get_last_lr()[0])

    ckpt = {
        "state_dict": model.state_dict(),
        "meta": {"joint_num": JOINT_NUM, "joints_name": list(JOINTS_NAME),
                 "skeleton": [list(s) for s in SKELETON],
                 "input_shape": list(INPUT_SHAPE), "output_shape": list(OUTPUT_SHAPE),
                 "depth_dim": DEPTH_DIM, "backbone": "LPSKI"},
    }
    out_path = osp.join(out_dir, "pose_model.pth")
    torch.save(ckpt, out_path)
    save_json({"loss_history": history, "epochs": epochs}, osp.join(out_dir, "train_log.json"))

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), history, marker="o")
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.title("LpNet training (CPU)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(osp.join(out_dir, "train_curve.png"), dpi=110); plt.close()

    log.info("Saved checkpoint -> %s", out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train LpNet pose model.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          num_workers=args.num_workers)
