#!/usr/bin/env python3
"""
Video inference pipeline for MobileHumanPose.
Usage: python demo/video_inference.py --input video.mp4 [--output out.mp4] [--gpu 0]
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

from common.utils.pose_utils import process_bbox
from common.utils.vis import vis_keypoints
from data.dataset import generate_patch_image
from src.config import cfg
from src.model import get_pose_net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input video path')
    p.add_argument('--output', '-o', default=None, help='Output video path')
    p.add_argument('--model_path', '-m', required=True, help='Snapshot path')
    p.add_argument('--joint_num', type=int, default=18)
    p.add_argument('--bbox', nargs=4, type=float, default=None, help='x y w h (None=full frame)')
    p.add_argument('--root_depth', type=float, default=1000.0, help='Root depth (mm) for 3D')
    return p.parse_args()


def main():
    args = parse_args()
    model = get_pose_net(cfg.backbone, False, args.joint_num)
    if torch.cuda.is_available():
        model = DataParallel(model).cuda()
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['network'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
    ])

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.output or args.input.replace('.', '_out.')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Skeleton for 18 joints (Dummy/H36M)
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))

    frame_idx = 0
    focal = [1500, 1500]
    princpt = [w / 2, h / 2]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.bbox:
            bbox = np.array(args.bbox)
        else:
            # Use full frame as person bbox
            bbox = np.array([0, 0, w, h])

        bbox = process_bbox(bbox, w, h)
        if bbox is None:
            writer.write(frame)
            frame_idx += 1
            continue

        img_patch, _ = generate_patch_image(frame, bbox, False, 1.0, 0.0, False)
        img_tensor = transform(img_patch)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pose = model(img_tensor)

        pose = pose[0].cpu().numpy()
        pose[:, 0] = pose[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
        pose[:, 1] = pose[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]

        vis_kps = np.zeros((3, args.joint_num))
        vis_kps[0], vis_kps[1] = pose[:, 0], pose[:, 1]
        vis_kps[2] = 1
        vis_frame = vis_keypoints(frame.copy(), vis_kps, skeleton)
        writer.write(vis_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total or '?'} frames")

    cap.release()
    writer.release()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
