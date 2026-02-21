"""
Export 3D pose model to ONNX or CoreML.
Usage:
  python -m src.3dpose_estimator.export --format onnx --modelpath output/model_dump/snapshot_25.pth.tar
  python -m src.3dpose_estimator.export --format coreml --modelpath output/model_dump/snapshot_25.pth.tar [--output output/pose_3d.mlpackage]
"""
import argparse
import os
import os.path as osp

import numpy as np
import torch

from common.base import Transformer

try:
    from src.config import cfg
except ImportError:
    from .config import cfg

# ONNX
try:
    import onnx
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

# CoreML
try:
    import coremltools as ct
    _COREML_AVAILABLE = True
except ImportError:
    _COREML_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Export 3D pose model to ONNX or CoreML")
    p.add_argument("--format", "-f", choices=["onnx", "coreml"], required=True, help="Export format")
    p.add_argument("--modelpath", "-m", required=True, help="Path to .pth.tar checkpoint")
    p.add_argument("--output", "-o", default="", help="Output path (default: output_dir/pose_3d.<ext>)")
    p.add_argument("--joint", type=int, default=18, help="Number of joints (default: 18)")
    p.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")
    p.add_argument("--check", action="store_true", help="Run ONNX check and compare PyTorch vs ONNX output")
    return p.parse_args()


def get_output_path(fmt, user_path):
    if user_path:
        return user_path
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    return osp.join(out_dir, "pose_3d.onnx" if fmt == "onnx" else "pose_3d.mlpackage")


def export_onnx(transformer, output_path, opset, run_check):
    model = transformer.model
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, *cfg.input_shape, device=device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
    )

    if run_check and _ONNX_AVAILABLE:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        ref = model(dummy).cpu().detach().numpy()
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        onnx_out = session.run(None, {"input": dummy.cpu().numpy()})[0]
        diff = np.linalg.norm(ref - onnx_out)
        print(f"ONNX check: PyTorch vs ONNX L2 diff = {diff}")
        if diff > 1e-4:
            print("Warning: ONNX output differs from PyTorch.")

    print(f"ONNX saved: {output_path}")
    return output_path


def export_coreml(transformer, output_path):
    if not _COREML_AVAILABLE:
        raise RuntimeError("coremltools not installed. pip install coremltools")

    model = transformer.model
    model.eval()
    # Trace and convert on CPU for stable CoreML conversion
    model_cpu = model.cpu()
    dummy = torch.randn(1, 3, *cfg.input_shape)

    # TorchScript trace (required for CoreML converter)
    with torch.no_grad():
        traced = torch.jit.trace(model_cpu, dummy)

    # TensorType: model expects normalized float (NCHW). Client preprocesses with same mean/std.
    input_shape = (1, 3) + tuple(cfg.input_shape)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT32,
    )

    # mlprogram must be saved as .mlpackage (directory)
    if not output_path.endswith(".mlpackage"):
        output_path = output_path.rstrip("/") + ".mlpackage"
    mlmodel.save(output_path)
    print(f"CoreML saved: {output_path}")
    return output_path


def main():
    args = parse_args()
    if args.format == "onnx" and not _ONNX_AVAILABLE:
        raise RuntimeError("onnx and onnxruntime required for ONNX export. pip install onnx onnxruntime")
    if args.format == "coreml" and not _COREML_AVAILABLE:
        raise RuntimeError("coremltools required for CoreML export. pip install coremltools")

    transformer = Transformer(args.joint, args.modelpath)
    transformer._make_model()
    output_path = get_output_path(args.format, args.output)

    if args.format == "onnx":
        export_onnx(transformer, output_path, args.opset, args.check)
    else:
        export_coreml(transformer, output_path)


if __name__ == "__main__":
    main()
