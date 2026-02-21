#!/usr/bin/env python3
"""
RAPHL Loop: Run All Pipeline / Human Loop
20회 반복하여 전체 파이프라인 테스트.

Usage: python scripts/raphl_loop.py [--iter 20]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, timeout=60, cwd=None, env=None):
    e = {**os.environ.copy(), **(env or {})}
    if "MPLBACKEND" not in e:
        e["MPLBACKEND"] = "Agg"
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=e,
        )
        return r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", "-n", type=int, default=20, help="반복 횟수")
    args = ap.parse_args()
    n = args.iter

    results = {k: {"pass": 0, "fail": 0} for k in range(n)}
    tests = []

    # 1. generate_dummy_data (1회만)
    ok, out = run(f"{sys.executable} scripts/generate_dummy_data.py", timeout=30)
    if not ok:
        print("[FAIL] generate_dummy_data.py")
        print(out[:500] if out else "")
        sys.exit(1)
    print("[OK] generate_dummy_data.py")

    dummy_img = ROOT / "data" / "Dummy" / "images" / "s_2_act_01_subact_01_ca_01" / "s_2_act_01_subact_01_ca_01_000001.jpg"
    pose_onnx = ROOT / "demo" / "models" / "pose_3d.onnx"
    pose_ckpt = ROOT / "output" / "model_dump" / "snapshot_0.pth.tar"
    rootnet_onnx = ROOT / "demo" / "models" / "rootnet.onnx"
    person_onnx = ROOT / "demo" / "models" / "person_detector.onnx"
    bbox = "0 0 1000 1002"

    model_path = str(pose_onnx) if pose_onnx.exists() else (str(pose_ckpt) if pose_ckpt.exists() else None)

    if not dummy_img.exists():
        print("[FAIL] Dummy image not found:", dummy_img)
        sys.exit(1)

    # 2~5. n회 반복 테스트
    for i in range(n):
        row = {"demo": False, "export": False, "test": False, "config": False}
        tests.append(row)

        # demo (headless: no 3D vis window)
        if model_path:
            ok, _ = run(
                f'{sys.executable} demo/demo.py -m "{model_path}" -i "{dummy_img}" --bbox {bbox} --headless',
                timeout=30,
            )
            row["demo"] = ok

        # export (pose ONNX 존재 시 검증만)
        if pose_ckpt.exists():
            ok, _ = run(
                f'{sys.executable} -m src.3dpose_estimator.export -f onnx -m "{pose_ckpt}" -o output/pose_test.onnx',
                timeout=60,
            )
            row["export"] = ok

        # src.test
        if pose_ckpt.exists():
            ok, _ = run(
                f"{sys.executable} -m src.test --test_epoch 0",
                timeout=120,
            )
            row["test"] = ok

        # test_config_combinations
        ok, _ = run(
            f"{sys.executable} src/3dpose_estimator/test_config_combinations.py",
            timeout=30,
        )
        row["config"] = ok

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n}] done")

    # 6. ruff (pyproject excludes scripts, demo/models)
    ok, out = run(
        f"{sys.executable} -m ruff check src common data demo --output-format=concise",
        timeout=20,
    )
    ruff_ok = ok

    # Report
    print("\n" + "=" * 50)
    print("RAPHL Loop Report")
    print("=" * 50)
    demo_ok = sum(1 for t in tests if t["demo"])
    export_ok = sum(1 for t in tests if t["export"])
    test_ok = sum(1 for t in tests if t["test"])
    config_ok = sum(1 for t in tests if t["config"])
    print(f"generate_dummy_data: 1/1 OK")
    print(f"demo:                {demo_ok}/{n} OK" + (f" (model={model_path})" if model_path else " (no model)"))
    print(f"export:              {export_ok}/{n} OK")
    print(f"src.test:            {test_ok}/{n} OK")
    print(f"test_config_combos:  {config_ok}/{n} OK")
    print(f"ruff:                {'OK' if ruff_ok else 'FAIL'}")
    print("=" * 50)
    all_ok = (demo_ok == n or not model_path) and config_ok == n and ruff_ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
