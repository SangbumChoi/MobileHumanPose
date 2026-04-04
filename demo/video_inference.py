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

from common.utils.pose_utils import pixel2cam, process_bbox
from common.utils.vis import vis_3d_skeleton_to_file, vis_keypoints
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
    p.add_argument('--save_3d_every', type=int, default=0, help='Save 3D figure every N frames (0=disabled)')
    p.add_argument('--save_3d_dir', default='vis/video_3d', help='Directory for 3D outputs when --save_3d_every > 0')
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

    # Skeleton for 18 joints (Dummy/H36M), kps_lines for vis
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    kps_lines = list(skeleton)

    if args.save_3d_every > 0:
        os.makedirs(args.save_3d_dir, exist_ok=True)

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
        pose_2d = pose.copy()
        pose_2d[:, 0] = pose[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
        pose_2d[:, 1] = pose[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]

        # 3D: convert depth to camera space and pixel2cam
        pose_3d_2d = pose_2d.copy()
        pose_3d_2d[:, 2] = (pose[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + args.root_depth
        pose_3d_cam = pixel2cam(pose_3d_2d, focal, princpt)

        vis_kps = np.zeros((3, args.joint_num))
        vis_kps[0], vis_kps[1] = pose_2d[:, 0], pose_2d[:, 1]
        vis_kps[2] = 1
        vis_frame = vis_keypoints(frame.copy(), vis_kps, skeleton)

        if args.save_3d_every > 0 and frame_idx % args.save_3d_every == 0:
            kpt_3d_vis = np.ones((args.joint_num, 1), dtype=np.float64)
            out_3d = osp.join(args.save_3d_dir, f"frame_{frame_idx:06d}_3d.jpg")
            vis_3d_skeleton_to_file(pose_3d_cam, kpt_3d_vis, kps_lines, out_3d)
        writer.write(vis_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total or '?'} frames")

    cap.release()
    writer.release()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
