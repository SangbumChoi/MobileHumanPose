"""
Dataset registry. Avoid exec(); use explicit mapping.
"""
import os.path as osp
import sys


def _add_data_paths():
    from config import cfg
    from utils.dir_utils import add_pypath
    for ds in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
        add_pypath(osp.join(cfg.data_dir, ds))

def get_dataset(name, split):
    """Load dataset by name: Dummy, Human36M, MuCo, MSCOCO, MPII, MuPoTS."""
    _add_data_paths()
    mod = __import__(name, fromlist=[name])
    cls = getattr(mod, name)
    return cls(split)
