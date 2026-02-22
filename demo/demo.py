"""
Image demo for 3D human pose estimation.
Uses ONNX for detection (person_detector.onnx) and rootnet (rootnet.onnx).
Pose model: PyTorch (.pth.tar) or ONNX (pose_3d.onnx).

Usage:
  python demo/demo.py -m demo/models/pose_3d.onnx -i image.jpg
  python demo/demo.py -m output/model_dump/snapshot_0.pth.tar -i image.jpg [--bbox x y w h]
"""
import argparse
import os.path as osp

import cv2
import numpy as np

from common.utils.pose_utils import pixel2cam, process_bbox
from common.utils.vis import vis_3d_with_image_plane, vis_keypoints
from data.dataset import generate_patch_image
from src.config import cfg

try:
    from .detection import get_person_bboxes
    from .rootnet import get_root_depths
except ImportError:
    from detection import get_person_bboxes
    from rootnet import get_root_depths

joint_num = 18
skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))


def load_pose_onnx(model_path):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def load_pose_torch(model_path):
    import torch
    import torchvision.transforms as transforms
    from torch.nn.parallel.data_parallel import DataParallel

    from src.model import get_pose_net

    model = get_pose_net(cfg.backbone, False, joint_num)
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["network"]
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        model = DataParallel(model).cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])
    return model, transform


def infer_pose_onnx(session, img_bgr):
    x = img_bgr.astype(np.float32) / 255.0
    x = (x - np.array(cfg.pixel_mean, dtype=np.float32).reshape(1, 1, 3)) / np.array(cfg.pixel_std, dtype=np.float32).reshape(1, 1, 3)
    x = x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    return out


def infer_pose_torch(model, transform, img_bgr):
    import torch
    img_t = transform(img_bgr)
    if torch.cuda.is_available():
        img_t = img_t.cuda()
    with torch.no_grad():
        out = model(img_t[None])
    return out[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True, help="pose_3d.onnx or snapshot_*.pth.tar")
    parser.add_argument("--input_image", "-i", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X", "Y", "W", "H"), help="Optional single bbox")
    parser.add_argument("--headless", action="store_true", help="Skip 3D vis window (for CI/loop)")
    args = parser.parse_args()

    use_onnx = args.model_path.endswith(".onnx")
    if use_onnx:
        pose_session = load_pose_onnx(args.model_path)
        pose_fn = lambda img: infer_pose_onnx(pose_session, img)
    else:
        pose_model, transform = load_pose_torch(args.model_path)
        pose_fn = lambda img: infer_pose_torch(pose_model, transform, img)

    img_path = args.input_image
    assert osp.exists(img_path), "Cannot find image at " + img_path
    original_img = cv2.imread(img_path)
    original_img_height, original_img_width = original_img.shape[:2]
    focal = [1500, 1500]
    princpt = [original_img_width / 2, original_img_height / 2]

    if args.bbox is not None:
        bbox_list = [list(args.bbox)]
    else:
        bbox_list = get_person_bboxes(original_img, conf_thres=args.conf)
    if not bbox_list:
        bbox_list = [[0, 0, original_img_width, original_img_height]]

    root_depth_list = get_root_depths(
        original_img, bbox_list, focal, princpt, process_bbox, generate_patch_image
    )
    person_num = len(bbox_list)
    print(f"Focal: {focal}, princpt: {princpt}, persons: {person_num}")

    output_pose_2d_list = []
    output_pose_3d_list = []
    for n in range(person_num):
        bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)

        if use_onnx:
            pose_3d = pose_fn(img)
        else:
            pose_3d = pose_fn(img)

        pose_3d = np.asarray(pose_3d)
        if pose_3d.ndim == 3:
            pose_3d = pose_3d[0]
        pose_3d = pose_3d.astype(np.float64)

        pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
        pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
        pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
        img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
        pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.T).T[:, :2]
        output_pose_2d_list.append(pose_3d[:, :2].copy())

        pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + root_depth_list[n]
        pose_3d = pixel2cam(pose_3d, focal, princpt)
        output_pose_3d_list.append(pose_3d.copy())

    vis_img = original_img.copy()
    for n in range(person_num):
        vis_kps = np.zeros((3, joint_num))
        vis_kps[0, :] = output_pose_2d_list[n][:, 0]
        vis_kps[1, :] = output_pose_2d_list[n][:, 1]
        vis_kps[2, :] = 1
        vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
    out_2d = osp.join(osp.dirname(osp.abspath(img_path)) or ".", "output_pose_2d.jpg")
    cv2.imwrite(out_2d, vis_img)
    print("Saved", out_2d)

    if not args.headless:
        vis_kps = np.array(output_pose_3d_list)
        vis_3d_with_image_plane(original_img, vis_kps, np.ones_like(vis_kps), skeleton, "3D Pose (camera space)")


if __name__ == "__main__":
    main()
