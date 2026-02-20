#!/usr/bin/env python3
"""
Test dataset combinations in main/config.py.
Usage: python scripts/test_config_combinations.py
"""
import sys
import os.path as osp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'main'))
sys.path.insert(0, str(ROOT / 'common'))
sys.path.insert(0, str(ROOT / 'data'))

COMBOS = [
    {'trainset_3d': ['Dummy'], 'trainset_2d': [], 'testset': 'Dummy'},
    {'trainset_3d': ['Dummy'], 'trainset_2d': ['MPII'], 'testset': 'Dummy'},
    {'trainset_3d': ['Human36M'], 'trainset_2d': [], 'testset': 'Human36M'},
    {'trainset_3d': ['MuCo'], 'trainset_2d': [], 'testset': 'MuPoTS'},
    {'trainset_3d': ['MuCo'], 'trainset_2d': ['MSCOCO'], 'testset': 'MuPoTS'},
]


def main():
    import config
    from utils.dir_utils import add_pypath

    cfg = config.cfg
    failed = []

    for combo in COMBOS:
        try:
            cfg.trainset_3d = combo['trainset_3d']
            cfg.trainset_2d = combo['trainset_2d']
            cfg.testset = combo['testset']
            for ds in cfg.trainset_3d + cfg.trainset_2d:
                add_pypath(osp.join(cfg.data_dir, ds))
            add_pypath(osp.join(cfg.data_dir, cfg.testset))

            for ds in cfg.trainset_3d:
                mod = __import__(ds, fromlist=[ds])
                getattr(mod, ds)('train')
            for ds in cfg.trainset_2d:
                mod = __import__(ds, fromlist=[ds])
                getattr(mod, ds)('train')
            mod = __import__(cfg.testset, fromlist=[cfg.testset])
            getattr(mod, cfg.testset)('test')
            print(f"OK: {combo}")
        except Exception as e:
            print(f"FAIL: {combo} -> {e}")
            failed.append((combo, str(e)))

    if failed:
        print(f"\n{len(failed)} failed")
        sys.exit(1)
    print("\nAll OK")


if __name__ == '__main__':
    main()
