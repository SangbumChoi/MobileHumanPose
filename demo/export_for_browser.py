#!/usr/bin/env python3
"""
Export pose model and person detector to ONNX for browser (WebGPU).
Output: demo/webgpu/models/pose.onnx, person_detector.onnx
"""
import os
import os.path as osp

import torch

OUT_DIR = osp.join(osp.dirname(__file__), 'webgpu', 'models')
os.makedirs(OUT_DIR, exist_ok=True)


def export_pose_onnx():
    from src.config import cfg
    from src.model import get_pose_net

    path = osp.join(OUT_DIR, 'pose.onnx')
    model = get_pose_net(cfg.backbone, False, 18)
    ckpt = torch.load(
        osp.join(osp.dirname(__file__), '..', 'output', 'model_dump', 'snapshot_0.pth.tar'),
        map_location='cpu',
    )
    state = ckpt['network']
    if next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )
    print('Exported', path)
    return path


def export_detector_onnx():
    try:
        from ultralytics import YOLO
    except ImportError:
        print('Install ultralytics: pip install ultralytics')
        return None
    path = osp.join(OUT_DIR, 'person_detector.onnx')
    model = YOLO('yolov8n.pt')
    # Ultralytics saves to cwd as yolov8n.onnx; export then move
    model.export(format='onnx', imgsz=640, opset=14, simplify=True, dynamic=True)
    import shutil
    cwd_onnx = osp.join(os.getcwd(), 'yolov8n.onnx')
    if osp.isfile(cwd_onnx):
        shutil.move(cwd_onnx, path)
        print('Exported', path)
    elif osp.isfile(path):
        print('Exported', path)
    return path


if __name__ == '__main__':
    export_pose_onnx()
    export_detector_onnx()
