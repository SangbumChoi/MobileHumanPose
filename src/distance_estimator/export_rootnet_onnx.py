#!/usr/bin/env python3
"""
Export 3DMPPE RootNet to ONNX.
1. Clones https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
2. Downloads checkpoint from https://drive.google.com/file/d/1ZHoXNFxBBsmis-5Xzu7dfXYGNxjpntgt/view
3. Exports rootnet.onnx to demo/models/

Usage: python demo/export_rootnet_onnx.py
"""
import os
import os.path as osp
import subprocess
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
MODELS_DIR = osp.join(ROOT, 'demo', 'models')
CLONE_DIR = osp.join(MODELS_DIR, '3dmppe_rootnet_release')
CKPT_FILE = osp.join(MODELS_DIR, 'snapshot_18.pth.tar')
OUT_ONNX = osp.join(MODELS_DIR, 'rootnet.onnx')
GDRIVE_ID = '1ZHoXNFxBBsmis-5Xzu7dfXYGNxjpntgt'
REPO_URL = 'https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE'


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Clone 3DMPPE_ROOTNET_RELEASE if needed
    if not osp.isdir(CLONE_DIR):
        print('Cloning 3DMPPE_ROOTNET_RELEASE...')
        subprocess.run(
            ['git', 'clone', '--depth', '1', REPO_URL, CLONE_DIR],
            check=True,
            cwd=ROOT,
        )

    # Download checkpoint via gdown
    if not osp.isfile(CKPT_FILE):
        try:
            import gdown
        except ImportError:
            print('Install gdown: pip install gdown')
            sys.exit(1)
        print('Downloading snapshot_18.pth.tar...')
        url = f'https://drive.google.com/uc?id={GDRIVE_ID}'
        gdown.download(url, CKPT_FILE, quiet=False)

    # Export ONNX
    sys.path.insert(0, osp.join(CLONE_DIR, 'common'))
    sys.path.insert(0, osp.join(CLONE_DIR, 'main'))
    sys.path.insert(0, CLONE_DIR)

    import torch

    from config import cfg
    from model import get_pose_net

    print('Loading RootNet...')
    model = get_pose_net(cfg, False)
    ckpt = torch.load(CKPT_FILE, map_location='cpu')
    state = ckpt.get('network', ckpt)
    if next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    dummy_img = torch.randn(1, 3, 256, 256)
    dummy_k = torch.randn(1, 1)

    print('Exporting to ONNX...')
    torch.onnx.export(
        model,
        (dummy_img, dummy_k),
        OUT_ONNX,
        input_names=['input_img', 'k_value'],
        output_names=['root_3d'],
        opset_version=14,
        dynamic_axes={
            'input_img': {0: 'batch'},
            'k_value': {0: 'batch'},
            'root_3d': {0: 'batch'},
        },
    )
    print('Saved', OUT_ONNX)


if __name__ == '__main__':
    main()
