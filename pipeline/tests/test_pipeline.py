"""Small end-to-end smoke test for the MobileHumanPose pipeline.

One test per stage (crawl ~10 images -> ... -> 3D-keypoint model -> ONNX),
kept tiny so it runs in CI on CPU. Tests run in file order and share
``pipeline/work/``; each asserts the previous stage's artifacts exist, so a
failure points at the exact stage that broke.
"""

import os
import os.path as osp

import numpy as np
import pytest

import common
from common import (RAW_DIR, CURATED_DIR, ANNOTATED_DIR, BALANCED_DIR,
                    MODELS_DIR, JOINT_NUM, OUTPUT_SHAPE, DEPTH_DIM,
                    build_model, soft_argmax, load_json)

CRAWL_LIMIT = 10
EPOCHS = 2


def test_1_crawl():
    import stage01_crawl
    n = stage01_crawl.crawl("person full body standing", limit=CRAWL_LIMIT)
    assert n >= 1, "crawl produced no images (network + fallback both empty)"
    assert osp.exists(osp.join(RAW_DIR, "manifest.json"))


def test_2_curate():
    import stage02_curate
    n = stage02_curate.curate()
    assert n >= 1, "curate kept no images"
    report = load_json(osp.join(CURATED_DIR, "curate_report.json"))
    assert report["summary"]["kept"] == n


def test_3_annotate():
    import stage03_annotate
    n = stage03_annotate.annotate()
    coco = load_json(osp.join(ANNOTATED_DIR, "annotations.json"))
    assert n == len(coco["annotations"]) >= 1
    # COCO-17 keypoints: 17 joints x (x, y, v) = 51 values per instance.
    assert len(coco["annotations"][0]["keypoints"]) == JOINT_NUM * 3
    assert coco["categories"][0]["keypoints"] == list(common.JOINTS_NAME)


def test_4_embed_curate():
    import stage04_embed_curate
    n = stage04_embed_curate.embed_curate()
    feats = np.load(osp.join(BALANCED_DIR, "embeddings.npy"))
    assert feats.shape[0] == n
    weights = load_json(osp.join(BALANCED_DIR, "sample_weights.json"))
    assert len(weights) == n and all(w > 0 for w in weights.values())
    assert osp.exists(osp.join(BALANCED_DIR, "distribution.png"))


def test_5_train():
    import stage05_train
    out = stage05_train.train(epochs=EPOCHS, batch_size=4)
    assert osp.exists(out)
    log = load_json(osp.join(MODELS_DIR, "train_log.json"))
    assert len(log["loss_history"]) == EPOCHS
    assert all(np.isfinite(log["loss_history"])), "loss diverged to nan/inf"


def test_6_model_3d_keypoint_output():
    """The model must emit a full 3D keypoint per joint (x, y, depth)."""
    import torch
    model = build_model(JOINT_NUM, init_weights=False)
    ckpt = osp.join(MODELS_DIR, "pose_model.pth")
    if osp.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    model.eval()
    with torch.no_grad():
        coords = soft_argmax(model(torch.randn(2, 3, 256, 256)), JOINT_NUM)
    assert coords.shape == (2, JOINT_NUM, 3), "expected (N, 17, 3) 3D keypoints"
    z = coords[..., 2]
    assert float(z.min()) >= -1 and float(z.max()) <= DEPTH_DIM + 1, \
        "depth (3rd) coordinate outside the discretized depth range"
    x, y = coords[..., 0], coords[..., 1]
    assert float(x.max()) <= OUTPUT_SHAPE[1] + 1 and float(y.max()) <= OUTPUT_SHAPE[0] + 1


def test_7_export_onnx():
    import export_onnx
    out = export_onnx.export()
    assert osp.exists(out)
    onnx_size = os.path.getsize(out)
    assert onnx_size > 1_000_000, "ONNX file suspiciously small (weights not embedded?)"

    # Standalone onnxruntime parity vs torch.
    import torch
    import onnxruntime as ort
    wrapper = export_onnx.PoseNetONNX(build_model(JOINT_NUM, init_weights=True)).eval()
    dummy = torch.randn(1, 3, 256, 256)
    # re-export this exact wrapper so the comparison is apples-to-apples
    tmp = osp.join(MODELS_DIR, "parity_check.onnx")
    torch.onnx.export(wrapper, dummy, tmp, input_names=["input"],
                      output_names=["coords"], opset_version=12, dynamo=False)
    with torch.no_grad():
        ref = wrapper(dummy).numpy()
    got = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"]).run(
        ["coords"], {"input": dummy.numpy()})[0]
    assert np.abs(ref - got).max() < 1e-3, "torch vs onnxruntime mismatch"


def test_8_inference_pipeline():
    """Full inference path: detect persons -> trained pose model -> keypoints."""
    import cv2
    from inference import load_pose_model, run_image
    files = [f for f in os.listdir(CURATED_DIR) if f.endswith(".jpg")]
    assert files, "no curated images to run inference on"
    model = load_pose_model()
    img = cv2.imread(osp.join(CURATED_DIR, files[0]))
    vis, boxes, kps = run_image(model, img, det_score=0.5)
    assert vis.shape == img.shape
    if kps:
        assert kps[0].shape == (JOINT_NUM, 2)
