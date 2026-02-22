#!/usr/bin/env python3
"""Generate dummy datasets. Delegates to data/Dummy/generate_dummy_data.py
Usage: python scripts/generate_dummy_data.py [--mscoco-only]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from importlib.util import spec_from_file_location, module_from_spec

_mod_path = ROOT / "data" / "Dummy" / "generate_dummy_data.py"
_spec = spec_from_file_location("_dummy_gen", _mod_path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mscoco-only", action="store_true", help="Only generate MSCOCO from Roboflow")
    parser.add_argument("--workspace", default="tennis-court-segmentation", help="Roboflow workspace")
    parser.add_argument("--project", default="tennis-player-pose", help="Roboflow project")
    parser.add_argument("--version", type=int, default=1, help="Roboflow version")
    args = parser.parse_args()
    if args.mscoco_only:
        _mod.generate_dummy_mscoco(
            workspace=args.workspace,
            project_name=args.project,
            version_num=args.version,
        )
        print("Done (MSCOCO only).")
    else:
        _mod.main()
