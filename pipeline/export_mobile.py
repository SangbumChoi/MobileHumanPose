"""Export the trained MobileHumanPose model to mobile formats:

  * TFLite  (ONNX -> onnx2tf -> saved_model -> .tflite float32/float16),
    verified against PyTorch with the TFLite interpreter.
  * Core ML (.mlpackage via coremltools ML Program). Conversion runs on Linux;
    *prediction* requires macOS, so we validate the spec, not outputs.

Run:  python export_mobile.py          (expects work/05_models/pose_model.pth)
Outputs land in work/05_models/mobile/.
"""

import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys

import numpy as np
import torch

from common import (MODELS_DIR, JOINT_NUM, INPUT_SHAPE, build_model, soft_argmax,
                    ensure_dir, get_logger)
from export_onnx import PoseNetONNX, export as export_onnx_model

log = get_logger("mobile")


def _load_wrapper(ckpt_path):
    model = build_model(JOINT_NUM, init_weights=False)
    if osp.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("network", ckpt.get("state_dict"))
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
        model.load_state_dict(state)
        log.info("Loaded %s", ckpt_path)
    else:
        model.init_weights()
        log.info("No checkpoint; exporting random-init model.")
    return PoseNetONNX(model).eval()


def export_tflite(onnx_path, out_dir, wrapper):
    """ONNX -> TFLite via onnx2tf; parity-check float32 against torch."""
    sm_dir = osp.join(out_dir, "saved_model")
    if osp.isdir(sm_dir):
        shutil.rmtree(sm_dir)
    # onnx2tf loads this calibration file from cwd; its remote copy is an
    # old pickled .npy that new numpy refuses -- pre-create a valid one.
    calib = osp.join(os.getcwd(),
                     "calibration_image_sample_data_20x128x128x3_float32.npy")
    if not osp.exists(calib):
        np.save(calib, np.random.rand(20, 128, 128, 3).astype(np.float32))
    cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", sm_dir,
           "--non_verbose", "--disable_group_convolution"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # fall back without the optional flag (older onnx2tf versions)
        res = subprocess.run([sys.executable, "-m", "onnx2tf", "-i", onnx_path,
                              "-o", sm_dir, "--non_verbose"],
                             capture_output=True, text=True)
    assert res.returncode == 0, "onnx2tf failed:\n" + res.stdout[-2000:] + res.stderr[-2000:]

    produced = [f for f in os.listdir(sm_dir) if f.endswith(".tflite")]
    assert produced, "onnx2tf produced no .tflite files"
    outs = {}
    for f in produced:
        kind = "float16" if "float16" in f else "float32"
        dst = osp.join(out_dir, "pose_model_%s.tflite" % kind)
        shutil.copy2(osp.join(sm_dir, f), dst)
        outs[kind] = dst
        log.info("TFLite %s -> %s (%.1f MB)", kind, dst, os.path.getsize(dst) / 1e6)

    # Parity check float32 vs torch. onnx2tf converts to NHWC inputs.
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=outs["float32"])
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = torch.randn(1, 3, *INPUT_SHAPE)
    feed = x.numpy().transpose(0, 2, 3, 1) if list(inp["shape"]) == [1, *INPUT_SHAPE, 3] \
        else x.numpy()
    interp.set_tensor(inp["index"], feed.astype(np.float32))
    interp.invoke()
    got = interp.get_tensor(out["index"]).reshape(1, JOINT_NUM, 3)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    err = float(np.abs(ref - got).max())
    log.info("TFLite float32 vs torch max abs diff: %.4f (coords in 32-grid units)", err)
    assert err < 0.1, "TFLite parity check failed"
    return outs


def export_coreml(wrapper, out_dir):
    import coremltools as ct
    example = torch.randn(1, 3, *INPUT_SHAPE)
    traced = torch.jit.trace(wrapper, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name="coords")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    dst = osp.join(out_dir, "PoseModel.mlpackage")
    if osp.isdir(dst):
        shutil.rmtree(dst)
    mlmodel.save(dst)
    spec = mlmodel.get_spec()
    log.info("CoreML mlprogram -> %s (inputs=%s outputs=%s)", dst,
             [i.name for i in spec.description.input],
             [o.name for o in spec.description.output])
    log.info("NOTE: CoreML prediction requires macOS; spec validated, outputs not run.")
    return dst


def main(ckpt=None):
    ckpt = ckpt or osp.join(MODELS_DIR, "pose_model.pth")
    out_dir = ensure_dir(osp.join(MODELS_DIR, "mobile"))
    wrapper = _load_wrapper(ckpt)
    onnx_path = osp.join(MODELS_DIR, "pose_model.onnx")
    if not osp.exists(onnx_path):
        export_onnx_model(ckpt_path=ckpt, out_path=onnx_path)
    export_tflite(onnx_path, out_dir, wrapper)
    export_coreml(wrapper, out_dir)
    log.info("Mobile exports complete in %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    main(args.ckpt)
