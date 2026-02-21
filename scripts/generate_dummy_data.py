#!/usr/bin/env python3
"""Generate dummy datasets. Delegates to data/Dummy/generate_dummy_data.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from importlib.util import spec_from_file_location, module_from_spec

_mod_path = ROOT / "data" / "Dummy" / "generate_dummy_data.py"
_spec = spec_from_file_location("_dummy_gen", _mod_path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.main()
