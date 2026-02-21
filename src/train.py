"""Entry point for training. Delegates to 3dpose_estimator.train."""
import importlib.util
import os.path as osp

_this_dir = osp.dirname(osp.abspath(__file__))
_train_path = osp.join(_this_dir, "3dpose_estimator", "train.py")
_spec = importlib.util.spec_from_file_location("_pose_train", _train_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
