"""Re-export config from 3dpose_estimator for 'from src.config import cfg'."""
import importlib.util
import os.path as osp

_this_dir = osp.dirname(osp.abspath(__file__))
_config_path = osp.join(_this_dir, "3dpose_estimator", "config.py")
_spec = importlib.util.spec_from_file_location("_pose_config", _config_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
cfg = _mod.cfg

__all__ = ["cfg"]
