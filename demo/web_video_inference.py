#!/usr/bin/env python3
"""
Web-based video inference for MobileHumanPose.
Input: short video; Output: pose-overlaid video saved to assets/videos/
Usage: python demo/web_video_inference.py
"""
import os
import os.path as osp
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
ASSETS_VIDEOS = osp.join(ROOT, 'assets', 'videos')
os.makedirs(ASSETS_VIDEOS, exist_ok=True)

from common.utils.pose_utils import process_bbox
from common.utils.vis import vis_keypoints
from data.dataset import generate_patch_image
from src.config import cfg
from src.model import get_pose_net

SKELETON = (
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
)
JOINT_NUM = 18


def load_model(model_path):
    model = get_pose_net(cfg.backbone, False, JOINT_NUM)
    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt['network']
    if next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    if torch.cuda.is_available():
        model = DataParallel(model).cuda()
    model.eval()
    return model


def run_inference(video_path, model_path):
    if video_path is None:
        return None, "Please upload a short video."

    model_path = model_path or osp.join(ROOT, 'output', 'model_dump', 'snapshot_0.pth.tar')
    if not osp.isfile(model_path):
        return None, f"Model not found: {model_path}"

    model = load_model(model_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"pose_{timestamp}.mp4"
    out_path = osp.join(ASSETS_VIDEOS, out_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = np.array([0, 0, w, h])
        bbox = process_bbox(bbox, w, h)
        if bbox is not None:
            img_patch, _ = generate_patch_image(frame, bbox, False, 1.0, 0.0, False)
            img_tensor = transform(img_patch).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            with torch.no_grad():
                pose = model(img_tensor)[0].cpu().numpy()

            pose[:, 0] = pose[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pose[:, 1] = pose[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            vis_kps = np.zeros((3, JOINT_NUM))
            vis_kps[0], vis_kps[1] = pose[:, 0], pose[:, 1]
            vis_kps[2] = 1
            frame = vis_keypoints(frame.copy(), vis_kps, SKELETON)

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    return out_path, f"Done. Saved to assets/videos/{out_name} ({frame_count} frames)"


def build_ui():
    default_model = osp.join(ROOT, 'output', 'model_dump', 'snapshot_0.pth.tar')

    with gr.Blocks(title="MobileHumanPose Video Inference") as demo:
        gr.Markdown("# MobileHumanPose: Web Video Inference")
        gr.Markdown("Upload a short video to run pose estimation. Output is saved to `assets/videos/`.")

        with gr.Row():
            video_in = gr.Video(label="Input Video", sources=["upload"])
            video_out = gr.Video(label="Output Video")

        with gr.Row():
            model_in = gr.Textbox(
                label="Model path",
                value=default_model,
                placeholder="path/to/snapshot_X.pth.tar",
            )
            run_btn = gr.Button("Run Inference")

        msg = gr.Textbox(label="Log", interactive=False)

        def infer(video, model):
            if video is None:
                return None, "Upload a video."
            path = video
            if isinstance(video, dict) and 'path' in video:
                path = video['path']
            elif hasattr(video, 'name'):
                path = video.name
            return run_inference(path, model or default_model)

        run_btn.click(infer, inputs=[video_in, model_in], outputs=[video_out, msg])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
