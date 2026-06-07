"""CI check: every dataset loader (2D + 3D) works on its committed mock data,
through the repo's own DatasetLoader / Trainer / Tester.

Runs pipeline/verify_datasets.py in a subprocess so the repo's import-time
dataset `exec` wiring stays isolated from the other tests in the session.
"""

import os.path as osp
import subprocess
import sys

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def test_all_dataset_formats_load_2d_and_3d():
    script = osp.join(REPO, "pipeline", "verify_datasets.py")
    res = subprocess.run([sys.executable, script], cwd=REPO,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, "verify_datasets failed:\n" + res.stdout + res.stderr
    out = res.stdout
    for name in ("MSCOCO", "MPII", "Human36M", "MuCo", "MuPoTS"):
        assert ("[OK] %-9s" % name)[:14] in out or name in out, "missing %s in output" % name
    assert "ALL DATASET CHECKS PASSED" in out


def test_mock_keypoints_are_valid_on_every_image():
    """Every mock annotation must be in-bounds, in-bbox, and (3D) reproject."""
    script = osp.join(REPO, "pipeline", "viz_mock_datasets.py")
    res = subprocess.run([sys.executable, script], cwd=REPO,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, "validation failed:\n" + res.stdout + res.stderr
    assert "VALIDATION PASSED" in res.stdout
