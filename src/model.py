"""Re-export get_pose_net from 3dpose_estimator for 'from src.model import get_pose_net'."""
import importlib.util
import os.path as osp

_this_dir = osp.dirname(osp.abspath(__file__))
_model_path = osp.join(_this_dir, "3dpose_estimator", "model.py")
_spec = importlib.util.spec_from_file_location("_pose_model", _model_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_pose_net = _mod.get_pose_net

__all__ = ["get_pose_net"]
