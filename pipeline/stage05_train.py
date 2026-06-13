"""Stage 5 - Train MobileHumanPose on the curated data using the ORIGINAL repo
training machinery (common/base.py ``Trainer``), not a re-implementation.

It configures the global ``cfg`` to use the ``CrawlPipeline`` dataset (a 2D
COCO loader that reads the pipeline's Stage 1-4 outputs), then runs the repo's
own Trainer + soft-argmax + L1 loss on CPU/GPU. The Stage-4 embedding weights
are honoured via dataset oversampling inside CrawlPipeline.

Outputs:
  * output/model_dump/snapshot_{epoch}.pth.tar  (repo format; usable by test.py)
  * pipeline/work/05_models/pose_model.pth      (network + meta, for demo/onnx)
  * pipeline/work/05_models/train_curve.png / train_log.json
"""

import argparse
import os.path as osp
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (REPO_DIR, MODELS_DIR, JOINTS_NAME, SKELETON, INPUT_SHAPE,
                    OUTPUT_SHAPE, DEPTH_DIM, ensure_dir, save_json, get_logger)

log = get_logger("train")


def _configure_cfg(backbone, epochs, batch_size, lr, num_thread, resume=False):
    """Set up the repo's global cfg to train on CrawlPipeline (before importing base)."""
    sys.path.insert(0, osp.join(REPO_DIR, "main"))
    sys.path.insert(0, osp.join(REPO_DIR, "data"))
    sys.path.insert(0, osp.join(REPO_DIR, "data", "CrawlPipeline"))
    from config import cfg
    cfg.set_args(gpu_ids="0", continue_train=resume)  # auto-falls back to CPU
    cfg.backbone = backbone
    cfg.trainset_3d = ["CrawlPipeline"]   # sole training set (2D; depth masked)
    cfg.trainset_2d = []
    cfg.testset = "CrawlPipeline"
    cfg.end_epoch = epochs
    cfg.batch_size = batch_size
    cfg.lr = lr
    cfg.lr_dec_epoch = [max(1, int(epochs * 0.7)), max(2, int(epochs * 0.9))]
    cfg.num_thread = num_thread
    return cfg


def train(epochs=20, batch_size=8, lr=1e-3, num_thread=0, backbone="LPSKI",
          out_dir=MODELS_DIR, save_every=15, resume=False):
    ensure_dir(out_dir)
    cfg = _configure_cfg(backbone, epochs, batch_size, lr, num_thread, resume=resume)

    # Import the repo Trainer only AFTER cfg is configured (base.py dynamically
    # imports the datasets named in cfg at import time).
    from base import Trainer

    # Mild augmentation profile: the repo defaults (rot +-60, occlusion 50%
    # covering up to 70% of the person) are tuned for million-sample datasets
    # and cause mean-pose collapse at our scale (verified: with them the model
    # cannot even memorize a fixed batch's distribution; without them a fixed
    # batch overfits to loss ~0.05).
    import dataset as repo_dataset
    repo_dataset.set_aug_config(rot_factor=15, rot_prob=0.4, scale_factor=0.15,
                                occlusion_prob=0.15, occlusion_area_max=0.3,
                                color_factor=0.1)

    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    def _save_convenience(n_done):
        """Persist the demo/ONNX checkpoint (backbone weights + meta). Called
        periodically so a killed long run still leaves a usable, recent model."""
        net = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        ckpt = {"network": net.backbone.state_dict(),
                "meta": {"joint_num": trainer.joint_num, "joints_name": list(JOINTS_NAME),
                         "skeleton": [list(s) for s in SKELETON], "backbone": backbone,
                         "input_shape": list(INPUT_SHAPE), "output_shape": list(OUTPUT_SHAPE),
                         "depth_dim": DEPTH_DIM, "epochs_done": n_done}}
        torch.save(ckpt, osp.join(out_dir, "pose_model.pth"))

    history = []
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.set_lr(epoch)
        ep_loss, nb = 0.0, 0
        for input_img, joint_img, joint_vis, joints_have_depth in trainer.batch_generator:
            trainer.optimizer.zero_grad()
            target = {"coord": joint_img, "vis": joint_vis, "have_depth": joints_have_depth}
            loss = trainer.model(input_img, target).mean()
            loss.backward()
            trainer.optimizer.step()
            ep_loss += loss.item(); nb += 1
        avg = ep_loss / max(nb, 1)
        history.append(avg)
        log.info("epoch %2d/%d  loss=%.4f  lr=%.1e", epoch + 1, cfg.end_epoch, avg, trainer.get_lr())

        # Periodic checkpoint: survives container reclaim / process kill, and
        # lets a re-launch resume via the repo Trainer's continue_train.
        if save_every and (epoch + 1) % save_every == 0 and (epoch + 1) < cfg.end_epoch:
            trainer.save_model({"epoch": epoch, "network": trainer.model.state_dict(),
                                "optimizer": trainer.optimizer.state_dict()}, epoch)
            _save_convenience(epoch + 1)
            log.info("  checkpoint saved at epoch %d", epoch + 1)

    # Repo-format checkpoint (loadable by main/test.py).
    trainer.save_model({"epoch": cfg.end_epoch - 1,
                        "network": trainer.model.state_dict(),
                        "optimizer": trainer.optimizer.state_dict()}, cfg.end_epoch - 1)

    _save_convenience(cfg.end_epoch)
    out_path = osp.join(out_dir, "pose_model.pth")
    save_json({"loss_history": history, "epochs": epochs, "joint_num": trainer.joint_num},
              osp.join(out_dir, "train_log.json"))

    if history:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(history) + 1), history, marker="o")
        plt.xlabel("epoch"); plt.ylabel("train loss")
        plt.title("LpNet training via repo Trainer (CrawlPipeline)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(osp.join(out_dir, "train_curve.png"), dpi=110); plt.close()

    log.info("Saved checkpoint -> %s", out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train via the original repo Trainer.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", default="LPSKI")
    ap.add_argument("--num_thread", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=15)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest snapshot in output/model_dump")
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          backbone=args.backbone, num_thread=args.num_thread,
          save_every=args.save_every, resume=args.resume)
