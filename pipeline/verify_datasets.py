"""Verify every dataset loader works on its mock data (2D and 3D), through the
repo's own DatasetLoader + Trainer. Run from anywhere:

    python pipeline/verify_datasets.py

Exits non-zero on any failure (used as a CI check via tests/test_datasets.py).
"""

import os
import os.path as osp
import sys

import numpy as np
import torch
import torchvision.transforms as transforms

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
for p in ["main", "data", "common"]:
    sys.path.insert(0, osp.join(REPO, p))
for d in os.listdir(osp.join(REPO, "data")):
    sub = osp.join(REPO, "data", d)
    if osp.isdir(sub):
        sys.path.insert(0, sub)
os.chdir(REPO)

from config import cfg
cfg.set_args(gpu_ids="0", continue_train=False)
# Configure the mixed 2D+3D training the original repo is designed for, BEFORE
# importing base (base.py exec-imports the datasets named here at import time).
cfg.trainset_3d = ["Human36M"]
cfg.trainset_2d = ["MPII"]
cfg.testset = "MuPoTS"
cfg.batch_size = 2
cfg.num_thread = 0
cfg.end_epoch = 1
cfg.lr_dec_epoch = [1]

TF = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])

from dataset import DatasetLoader


def check_dataset(name, split, is_train, dims):
    mod = __import__(name)
    db = getattr(mod, name)(split)
    assert len(db.data) >= 1, "%s: empty" % name
    loader = DatasetLoader(db, None, is_train, TF)
    sample = loader[0]
    if is_train:
        img, joint_img, joint_vis, have_depth = sample
        assert img.shape == (3, 256, 256)
        assert joint_img.shape == (db.joint_num, 3)
        assert joint_vis.shape == (db.joint_num, 1)
        depth = "3D" if db.joints_have_depth else "2D"
        print("  [OK] %-9s %-5s  %2d joints  %s  %d samples  have_depth=%s"
              % (name, split, db.joint_num, depth, len(db.data), bool(have_depth[0])))
    else:
        assert sample.shape == (3, 256, 256)
        print("  [OK] %-9s %-5s  %2d joints  (test)  %d samples"
              % (name, split, db.joint_num, len(db.data)))
    return db


def main():
    print("== per-dataset load + DatasetLoader sample ==")
    check_dataset("MSCOCO", "train", True, 19)    # 2D
    check_dataset("MPII", "train", True, 16)       # 2D
    check_dataset("Human36M", "train", True, 18)   # 3D
    check_dataset("MuCo", "train", True, 21)       # 3D
    check_dataset("MuPoTS", "test", False, 21)     # 3D (test)

    print("== mixed 2D+3D training iteration via the original repo Trainer ==")
    from base import Trainer
    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()
    trainer.set_lr(0)
    img, joint_img, joint_vis, have_depth = next(iter(trainer.batch_generator))
    trainer.optimizer.zero_grad()
    target = {"coord": joint_img, "vis": joint_vis, "have_depth": have_depth}
    loss = trainer.model(img, target).mean()
    loss.backward()
    trainer.optimizer.step()
    assert torch.isfinite(loss), "loss not finite"
    print("  [OK] Trainer step on Human36M(3D)+MPII(2D); loss=%.4f" % float(loss))

    print("== test-set loading via the original repo Tester ==")
    from base import Tester
    tester = Tester(cfg.backbone)
    tester._make_batch_generator()
    batch = next(iter(tester.batch_generator))
    assert batch.shape[1:] == (3, 256, 256)
    print("  [OK] Tester batch_generator on MuPoTS; batch=%s" % (tuple(batch.shape),))

    print("\nALL DATASET CHECKS PASSED (2D + 3D).")


if __name__ == "__main__":
    main()
