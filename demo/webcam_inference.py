#!/usr/bin/env python3
"""
Local webcam inference: human detection (YOLO/HF) -> bbox -> pose -> 2D overlay + 3D visualization.
Usage: python demo/webcam_inference.py [--detection ultralytics|hf|onnx] [--no-3d]
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
from common.utils.vis import vis_keypoints
from data.dataset import generate_patch_image
from src.config import cfg
from src.model import get_pose_net

try:
    from .detection import get_person_bboxes
except ImportError:
    from detection import get_person_bboxes

SKELETON = (
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
)
JOINT_NUM = 18


def run_pose(model, transform, frame, bbox, focal, princpt):
    """Run pose on one bbox; return 2D kpts (frame coords) and 3D kpts (camera mm)."""
    bbox = process_bbox(np.array(bbox), frame.shape[1], frame.shape[0])
    if bbox is None:
        return None, None
    img_patch, _ = generate_patch_image(frame, bbox, False, 1.0, 0.0, False)
    img_tensor = transform(img_patch).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    with torch.no_grad():
        pose = model(img_tensor)[0].cpu().numpy()
    # to frame 2D
    pose_2d = pose.copy()
    pose_2d[:, 0] = pose[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
    pose_2d[:, 1] = pose[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]
    # depth: root-relative -> absolute (use bbox center depth proxy)
    root_depth = 1500.0
    pose_2d[:, 2] = (pose[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + root_depth
    pose_3d = pixel2cam(pose_2d, focal, princpt)
    return pose_2d, pose_3d


def draw_3d_skeleton(ax, kpt_3d, color='cyan'):
    """Draw 3D skeleton on matplotlib ax."""
    for (i1, i2) in SKELETON:
        x = [kpt_3d[i1, 0], kpt_3d[i2, 0]]
        y = [kpt_3d[i1, 1], kpt_3d[i2, 1]]
        z = [kpt_3d[i1, 2], kpt_3d[i2, 2]]
        ax.plot(x, z, -np.array(y), c=color, linewidth=2)
    ax.scatter(kpt_3d[:, 0], kpt_3d[:, 2], -kpt_3d[:, 1], c=color, s=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model_path', '-m', default=None)
    parser.add_argument('--detection', choices=['ultralytics', 'hf', 'onnx'], default='ultralytics')
    parser.add_argument('--onnx_path', default='', help='For detection=onnx')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--no-3d', action='store_true', help='Disable 3D visualization window')
    args = parser.parse_args()

    model_path = args.model_path or osp.join(osp.dirname(__file__), '..', 'output', 'model_dump', 'snapshot_0.pth.tar')
    if not osp.isfile(model_path):
        print('Pose model not found:', model_path)
        print('Train first: pip install -e . && python -m src.train')
        return

    # Pose model
    model = get_pose_net(cfg.backbone, False, JOINT_NUM)
    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt['network']
    if next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    if torch.cuda.is_available():
        model = DataParallel(model).cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Cannot open camera', args.camera)
        return

    # 3D window
    ax_3d = None
    if not args.no_3d:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6))
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Z'); ax_3d.set_zlabel('Y')
        plt.ion()

    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    focal = [1500, 1500]
    princpt = [w / 2, h / 2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Human detection
            bboxes = get_person_bboxes(
                frame,
                backend=args.detection,
                conf_thres=args.conf,
                onnx_path=args.onnx_path or osp.join(osp.dirname(__file__), 'models', 'yolov8n.onnx'),
            )
            if not bboxes:
                bboxes = [[0, 0, w, h]]

            all_2d = []
            all_3d = []
            vis_frame = frame.copy()
            for i, bbox in enumerate(bboxes[:4]):
                pose_2d, pose_3d = run_pose(model, transform, frame, bbox, focal, princpt)
                if pose_2d is None:
                    continue
                all_2d.append(pose_2d)
                all_3d.append(pose_3d)
                vis_kps = np.zeros((3, JOINT_NUM))
                vis_kps[0], vis_kps[1] = pose_2d[:, 0], pose_2d[:, 1]
                vis_kps[2] = 1
                vis_frame = vis_keypoints(vis_frame, vis_kps, SKELETON)

            # 3D plot
            if ax_3d is not None and all_3d:
                ax_3d.cla()
                ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Z'); ax_3d.set_zlabel('Y')
                for i, kpt in enumerate(all_3d):
                    draw_3d_skeleton(ax_3d, kpt, color=colors[i % len(colors)])
                ax_3d.set_box_aspect([1, 1, 1])
                plt.pause(0.001)

            cv2.imshow('MobileHumanPose Webcam', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ax_3d is not None:
            plt.ioff()
            plt.close()


if __name__ == '__main__':
    main()
