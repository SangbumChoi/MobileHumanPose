#!/usr/bin/env python3
"""
Download pretrained weights and export ONNX models to demo/models/.
1. RootNet: snapshot_18.pth.tar from Google Drive -> rootnet.onnx
2. Person detector: YOLOv8 -> person_detector.onnx
3. Pose 3D: snapshot from output/model_dump -> pose_3d.onnx

Usage: python scripts/download_and_export_demo_models.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "demo" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
GDRIVE_ROOTNET_ID = "1ZHoXNFxBBsmis-5Xzu7dfXYGNxjpntgt"
ROOTNET_CKPT = MODELS_DIR / "snapshot_18.pth.tar"
ROOTNET_REPO = "https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE"
CLONE_DIR = MODELS_DIR / "3dmppe_rootnet_release"


def download_rootnet():
    if ROOTNET_CKPT.exists():
        print("RootNet ckpt exists:", ROOTNET_CKPT)
        return True
    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown")
        return False
    print("Downloading RootNet snapshot_18.pth.tar...")
    url = f"https://drive.google.com/uc?id={GDRIVE_ROOTNET_ID}"
    gdown.download(url, str(ROOTNET_CKPT), quiet=False)
    return ROOTNET_CKPT.exists()


def export_rootnet_onnx():
    out_onnx = MODELS_DIR / "rootnet.onnx"
    if out_onnx.exists():
        print("rootnet.onnx exists:", out_onnx)
        return True
    if not CLONE_DIR.exists():
        print("Cloning 3DMPPE_ROOTNET_RELEASE...")
        subprocess.run(["git", "clone", "--depth", "1", ROOTNET_REPO, str(CLONE_DIR)], check=True, cwd=str(ROOT))
    sys.path.insert(0, str(CLONE_DIR / "common"))
    sys.path.insert(0, str(CLONE_DIR / "main"))
    sys.path.insert(0, str(CLONE_DIR))
    import torch
    from config import cfg
    from model import get_pose_net

    print("Loading RootNet...")
    model = get_pose_net(cfg, False)
    ckpt = torch.load(str(ROOTNET_CKPT), map_location="cpu")
    state = ckpt.get("network", ckpt)
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    dummy_img = torch.randn(1, 3, 256, 256)
    dummy_k = torch.randn(1, 1)
    print("Exporting rootnet.onnx...")
    torch.onnx.export(
        model,
        (dummy_img, dummy_k),
        str(out_onnx),
        input_names=["input_img", "k_value"],
        output_names=["root_3d"],
        opset_version=14,
        dynamic_axes={"input_img": {0: "batch"}, "k_value": {0: "batch"}, "root_3d": {0: "batch"}},
    )
    print("Saved", out_onnx)
    return True


def export_person_detector_onnx():
    out_onnx = MODELS_DIR / "person_detector.onnx"
    if out_onnx.exists():
        print("person_detector.onnx exists:", out_onnx)
        return True
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics")
        return False
    print("Exporting YOLOv8 -> person_detector.onnx...")
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, opset=14, simplify=True, dynamic=True)
    cwd_onnx = Path.cwd() / "yolov8n.onnx"
    if cwd_onnx.exists():
        shutil.move(str(cwd_onnx), str(out_onnx))
    print("Saved", out_onnx)
    return out_onnx.exists()


def export_pose_3d_onnx():
    out_onnx = MODELS_DIR / "pose_3d.onnx"
    pose_ckpt = ROOT / "output" / "model_dump" / "snapshot_0.pth.tar"
    if out_onnx.exists():
        print("pose_3d.onnx exists:", out_onnx)
        return True
    if not pose_ckpt.exists():
        print("Pose ckpt not found:", pose_ckpt)
        print("Run: python -m src.train first")
        return False
    print("Exporting pose_3d.onnx...")
    ret = subprocess.run(
        [sys.executable, "-m", "src.3dpose_estimator.export", "-f", "onnx", "-m", str(pose_ckpt), "-o", str(out_onnx)],
        cwd=str(ROOT),
    )
    return ret.returncode == 0 and out_onnx.exists()


def main():
    sys.path.insert(0, str(ROOT))
    print("=== Download & Export demo/models ===\n")
    download_rootnet()
    export_rootnet_onnx()
    export_person_detector_onnx()
    export_pose_3d_onnx()
    print("\nDone. Check demo/models/")


if __name__ == "__main__":
    main()
