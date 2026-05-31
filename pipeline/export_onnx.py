"""Stage 6b (part 1) - Export the trained LpNet to ONNX for in-browser use.

Wraps backbone + soft-argmax so the exported graph takes a normalised
1x3x256x256 image and returns (1, 17, 3) coords in output-grid units. Verifies
torch vs onnxruntime parity, then copies the model where the static web demo
and the github.io site expect it.
"""

import argparse
import os.path as osp
import shutil

import numpy as np
import torch

from common import (MODELS_DIR, PIPELINE_DIR, JOINT_NUM, INPUT_SHAPE,
                    build_model, soft_argmax, get_logger, ensure_dir)

log = get_logger("onnx")


class PoseNetONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return soft_argmax(self.model(x), JOINT_NUM)  # (N, 17, 3)


def export(ckpt_path=None, out_path=None, web_dir=None):
    ckpt_path = ckpt_path or osp.join(MODELS_DIR, "pose_model.pth")
    out_path = out_path or osp.join(MODELS_DIR, "pose_model.onnx")
    model = build_model(JOINT_NUM, init_weights=False)
    if osp.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    else:
        model.init_weights()
        log.info("No checkpoint; exporting random-init model.")
    model.eval()
    wrapper = PoseNetONNX(model).eval()

    dummy = torch.randn(1, 3, *INPUT_SHAPE)
    # dynamo=False -> legacy exporter, which embeds weights inline (single
    # self-contained .onnx file, required for onnxruntime-web in the browser).
    torch.onnx.export(
        wrapper, dummy, out_path, input_names=["input"], output_names=["coords"],
        opset_version=12, dynamic_axes=None, dynamo=False)
    log.info("Exported ONNX -> %s", out_path)

    # Parity check against onnxruntime.
    try:
        import onnxruntime as ort
        with torch.no_grad():
            ref = wrapper(dummy).numpy()
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        got = sess.run(["coords"], {"input": dummy.numpy()})[0]
        max_err = float(np.abs(ref - got).max())
        log.info("torch vs onnxruntime max abs diff: %.3e", max_err)
        assert max_err < 1e-3, "ONNX parity check failed"
    except ImportError:
        log.info("onnxruntime not installed; skipped parity check.")

    if web_dir:
        ensure_dir(web_dir)
        shutil.copy2(out_path, osp.join(web_dir, "pose_model.onnx"))
        log.info("Copied ONNX -> %s", web_dir)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--web_dir", default=osp.join(PIPELINE_DIR, "web"))
    args = ap.parse_args()
    export(web_dir=args.web_dir)
